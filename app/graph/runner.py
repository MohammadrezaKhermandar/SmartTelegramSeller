"""Invoke LangGraph for a user message."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from app.graph.graph_builder import get_graph
from app.graph.state import SalesAssistantState
from app.memory.checkpointer import delete_user_thread
from app.memory.store import session_store
from app.utils.logging import logger


def invoke_graph(
    user_id: str,
    text: str = "",
    image_input: dict[str, Any] | None = None,
    url_input: str | None = None,
) -> dict[str, Any]:
    """
    Run the sales assistant graph for a user.
    thread_id = user_id for independent per-user memory.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": str(user_id)}}

    # Pass only new input; checkpointer merges prior state (Memory)
    partial: SalesAssistantState = {"user_id": str(user_id)}
    if text:
        partial["messages"] = [HumanMessage(content=text)]
    if image_input:
        partial["image_input"] = image_input
    if url_input:
        partial["url_input"] = url_input

    result = graph.invoke(partial, config)

    session_store.update_from_state(str(user_id), result)
    logger.info("Graph completed for user %s, stage=%s", user_id, result.get("conversation_stage"))
    return result


def reset_user_state(user_id: str) -> None:
    """Reset user conversation and LangGraph checkpointer thread."""
    uid = str(user_id)
    session_store.reset(uid)
    delete_user_thread(uid)
    logger.info("Reset state and checkpointer for user %s", uid)
