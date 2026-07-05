"""LangGraph checkpointer for conversation memory."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

_checkpointer: MemorySaver | None = None


def get_checkpointer() -> MemorySaver:
    """Return in-memory checkpointer – thread_id = Telegram user ID."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def reset_checkpointer() -> None:
    """Reset checkpointer (for tests)."""
    global _checkpointer
    _checkpointer = None


def delete_user_thread(user_id: str) -> None:
    """Delete all LangGraph checkpoints for a Telegram user (/reset)."""
    thread_id = str(user_id)
    get_checkpointer().delete_thread(thread_id)
