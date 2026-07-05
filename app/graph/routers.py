"""Conditional routing for LangGraph."""

from __future__ import annotations

from typing import Literal

from app.graph.state import SalesAssistantState


def route_after_classifier(
    state: SalesAssistantState,
) -> Literal[
    "image_similarity",
    "url_similarity",
    "intent_detector",
]:
    """Route based on input type."""
    if state.get("image_input"):
        return "image_similarity"
    if state.get("url_input"):
        return "url_similarity"
    return "intent_detector"


def route_after_intent(
    state: SalesAssistantState,
) -> Literal[
    "requirement_extractor",
    "answer_from_memory",
    "compare_products",
    "update_requirements",
    "final_response",
    "url_similarity",
]:
    """Route based on detected intent."""
    intent = state.get("current_intent", "unknown")

    if intent == "url_input":
        return "url_similarity"
    if intent == "followup_question":
        return "answer_from_memory"
    if intent == "compare":
        return "compare_products"
    if intent == "change_preferences":
        return "update_requirements"
    if intent in ("greeting", "general_chat", "company_question", "purchase", "unknown"):
        return "final_response"
    return "requirement_extractor"


def route_after_requirements(
    state: SalesAssistantState,
) -> Literal["enough_info_checker"]:
    return "enough_info_checker"


def route_after_info_check(
    state: SalesAssistantState,
) -> Literal["ask_clarifying", "tool_agent"]:
    """Route to clarifying question or tool-calling search agent."""
    missing = state.get("missing_slots") or []
    if missing:
        return "ask_clarifying"
    return "tool_agent"


def route_after_update(
    state: SalesAssistantState,
) -> Literal["enough_info_checker", "tool_agent"]:
    """After preference change, check info or search directly."""
    missing = state.get("missing_slots") or []
    req = state.get("requirements") or {}
    has_category = bool(req.get("category"))
    if missing and not has_category:
        return "enough_info_checker"
    return "tool_agent"


def route_after_hybrid_search(
    state: SalesAssistantState,
) -> Literal["recommend_products", "error_handler"]:
    """Route on search success or failure."""
    if state.get("errors") and not state.get("last_search_result"):
        return "error_handler"
    return "recommend_products"


def route_after_error(
    state: SalesAssistantState,
) -> Literal["hybrid_search", "save_memory"]:
    """Retry search or give up."""
    retry = state.get("retry_count") or 0
    if retry < 3:
        return "hybrid_search"
    return "save_memory"
