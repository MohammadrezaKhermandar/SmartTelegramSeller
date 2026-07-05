"""Conversation memory backed by SQLite.

Tables:
- user_sessions:          one row per chat — requirements, stage, profile,
                          conversation summary, focus product, purchase status,
                          last activity timestamp.
- recommended_products:   full snapshot of every product the bot has shown,
                          with its position (گزینه اول/دوم/...) so questions
                          about previous options are answered from memory,
                          NOT by re-querying CSV/RAG/Pandas.
- followups:              follow-up scheduling / delivery state.

This module is intentionally synchronous (sqlite3) — calls are short and the
bot wraps graph execution in a thread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_sessions (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    requirements TEXT NOT NULL DEFAULT '{}',
    user_profile TEXT NOT NULL DEFAULT '{}',
    conversation_stage TEXT NOT NULL DEFAULT 'new',
    memory_summary TEXT NOT NULL DEFAULT '',
    focus_product_id TEXT,
    purchase_status TEXT NOT NULL DEFAULT 'browsing',
    last_message_at REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recommended_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    product_json TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    recommended_at REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_rec_chat ON recommended_products (chat_id, is_active);

CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    kind TEXT NOT NULL,               -- 'idle_1h' | 'purchase_2d'
    due_at REAL NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_follow_due ON followups (sent, due_at);
"""


class MemoryService:
    """Thread-safe SQLite persistence for sessions, recommendations, follow-ups."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        path = db_path or str(settings.memory_db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        logger.info("Memory DB ready at %s", path)

    # ------------------------------------------------------------- sessions

    def get_or_create_session(self, chat_id: str, user_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user_sessions WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if row is None:
                now = time.time()
                self._conn.execute(
                    "INSERT INTO user_sessions (chat_id, user_id, last_message_at, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (chat_id, user_id, now, now),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM user_sessions WHERE chat_id = ?", (chat_id,)
                ).fetchone()
        session = dict(row)
        session["requirements"] = json.loads(session["requirements"])
        session["user_profile"] = json.loads(session["user_profile"])
        return session

    def update_session(self, chat_id: str, **fields: Any) -> None:
        allowed = {
            "requirements", "user_profile", "conversation_stage", "memory_summary",
            "focus_product_id", "purchase_status", "last_message_at",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        for key in ("requirements", "user_profile"):
            if key in updates and isinstance(updates[key], dict):
                updates[key] = json.dumps(updates[key], ensure_ascii=False)
        sets = ", ".join(f"{k} = ?" for k in updates)
        with self._lock:
            self._conn.execute(
                f"UPDATE user_sessions SET {sets} WHERE chat_id = ?",
                (*updates.values(), chat_id),
            )
            self._conn.commit()

    def touch(self, chat_id: str) -> None:
        self.update_session(chat_id, last_message_at=time.time())

    # ------------------------------------------------- recommended products

    def save_recommendations(
        self, chat_id: str, products: list[dict[str, Any]], reasons: Optional[list[str]] = None
    ) -> None:
        """Store a new active recommendation set; previous set is archived."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE recommended_products SET is_active = 0 WHERE chat_id = ?", (chat_id,)
            )
            for i, product in enumerate(products):
                reason = (reasons or [])[i] if reasons and i < len(reasons) else ""
                self._conn.execute(
                    "INSERT INTO recommended_products "
                    "(chat_id, product_id, position, product_json, reason, recommended_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chat_id,
                        str(product.get("product_id", "")),
                        i + 1,
                        json.dumps(product, ensure_ascii=False, default=str),
                        reason,
                        now,
                    ),
                )
            self._conn.commit()
        logger.info("Saved %d active recommendations for chat %s", len(products), chat_id)

    def deactivate_over_budget_recommendations(
        self, chat_id: str, max_price: float
    ) -> int:
        """Archive active recommendations priced above max_price."""
        removed = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, product_json FROM recommended_products "
                "WHERE chat_id = ? AND is_active = 1",
                (chat_id,),
            ).fetchall()
            for row in rows:
                product = json.loads(row["product_json"])
                price = float(product.get("effective_price") or product.get("price") or 0)
                if price > max_price:
                    self._conn.execute(
                        "UPDATE recommended_products SET is_active = 0 WHERE id = ?",
                        (row["id"],),
                    )
                    removed += 1
            self._conn.commit()
        if removed:
            logger.info(
                "Deactivated %d over-budget recommendations for chat %s (max=%s)",
                removed, chat_id, max_price,
            )
        return removed

    def get_active_recommendations(self, chat_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM recommended_products "
                "WHERE chat_id = ? AND is_active = 1 ORDER BY position",
                (chat_id,),
            ).fetchall()
        result = []
        for row in rows:
            product = json.loads(row["product_json"])
            product["_position"] = row["position"]
            product["_reason"] = row["reason"]
            result.append(product)
        return result

    def get_recommendation_by_position(
        self, chat_id: str, position: int
    ) -> Optional[dict[str, Any]]:
        recs = self.get_active_recommendations(chat_id)
        for rec in recs:
            if rec["_position"] == position:
                return rec
        return None

    def find_product_by_name_query(
        self, chat_id: str, product_name_query: str, min_score: float = 0.35
    ) -> Optional[dict[str, Any]]:
        """Match active recommendations by partial / fuzzy product name."""
        from app.utils.text_normalizer import match_product_name_query

        recs = self.get_active_recommendations(chat_id)
        return match_product_name_query(product_name_query, recs, min_score=min_score)

    # ------------------------------------------------------------ followups

    def schedule_followups(self, chat_id: str) -> None:
        """(Re)schedule both follow-ups relative to now. Called after every
        bot reply so timers restart when the user is active."""
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM followups WHERE chat_id = ? AND sent = 0", (chat_id,))
            self._conn.execute(
                "INSERT INTO followups (chat_id, kind, due_at, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, "idle_1h", now + settings.followup_idle_seconds, now),
            )
            self._conn.execute(
                "INSERT INTO followups (chat_id, kind, due_at, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, "purchase_2d", now + settings.followup_purchase_seconds, now),
            )
            self._conn.commit()

    def cancel_followups(self, chat_id: str, kind: Optional[str] = None) -> None:
        with self._lock:
            if kind:
                self._conn.execute(
                    "DELETE FROM followups WHERE chat_id = ? AND kind = ? AND sent = 0",
                    (chat_id, kind),
                )
            else:
                self._conn.execute(
                    "DELETE FROM followups WHERE chat_id = ? AND sent = 0", (chat_id,)
                )
            self._conn.commit()

    def due_followups(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM followups WHERE sent = 0 AND due_at <= ?", (now,)
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_followup_sent(self, followup_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE followups SET sent = 1 WHERE id = ?", (followup_id,))
            self._conn.commit()

    def get_purchase_status(self, chat_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT purchase_status FROM user_sessions WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return row["purchase_status"] if row else "browsing"

    def reset_chat(self, chat_id: str) -> None:
        """Delete all persisted state for a chat (session, recommendations, follow-ups)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM recommended_products WHERE chat_id = ?", (chat_id,)
            )
            self._conn.execute("DELETE FROM followups WHERE chat_id = ?", (chat_id,))
            self._conn.execute("DELETE FROM user_sessions WHERE chat_id = ?", (chat_id,))
            self._conn.commit()
        logger.info("Cleared memory for chat %s", chat_id)


_memory: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _memory
    if _memory is None:
        _memory = MemoryService()
    return _memory
