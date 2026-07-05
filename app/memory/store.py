"""Additional memory store for follow-up jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class UserSession:
    user_id: str
    last_message_at: datetime | None = None
    last_recommendation_at: datetime | None = None
    purchase_status: str = "none"
    recommended_products: list[dict[str, Any]] = field(default_factory=list)
    followup_1h_scheduled: bool = False
    discount_2d_scheduled: bool = False


class SessionStore:
    """In-memory session store for scheduled jobs."""

    def __init__(self) -> None:
        self._sessions: dict[str, UserSession] = {}

    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]

    def update_from_state(self, user_id: str, state: dict[str, Any]) -> None:
        session = self.get_or_create(user_id)
        session.last_message_at = datetime.now(timezone.utc)
        if state.get("recommended_products"):
            session.recommended_products = state["recommended_products"]
            session.last_recommendation_at = datetime.now(timezone.utc)
        if state.get("purchase_status"):
            session.purchase_status = state["purchase_status"]

    def mark_purchased(self, user_id: str) -> None:
        session = self.get_or_create(user_id)
        session.purchase_status = "purchased"

    def reset(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    def all_sessions(self) -> dict[str, UserSession]:
        return self._sessions


session_store = SessionStore()
