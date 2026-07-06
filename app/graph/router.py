"""Conditional routing logic for the sales graph."""

from __future__ import annotations

from typing import Literal

from app.graph.state import HARD_SLOTS, SalesState
from app.utils.logger import get_logger
from app.utils.text_normalizer import is_memory_reference

logger = get_logger(__name__)

Route = Literal[
    "needs_clarification",
    "answer_from_memory",
    "product_search",
    "compare_products",
    "image_similarity",
    "link_similarity",
    "smalltalk",
    "error",
]


def conditional_router(state: SalesState) -> Route:
    """Main decision point of the graph.

    Priority order:
    1. error            -> retry_or_fallback
    2. image / link     -> similarity paths
    3. purchase request  -> smalltalk (checkout coming soon)
    4. compare intent    -> comparison from memory
    5. product images    -> send from memory
    6. explicit search   -> product_search (budget/brand updates)
    7. memory question   -> answer from memory (even if slots missing)
    8. product request   -> clarify if hard slots missing, else hybrid search
    9. everything else   -> small talk
    """
    route = _decide(state)
    logger.info(
        "Route=%s (intent=%s, type=%s, missing=%s, changed=%s, memory=%s)",
        route,
        state.get("intent"),
        state.get("message_type"),
        state.get("missing_slots"),
        state.get("requirements_changed"),
        state.get("should_use_memory"),
    )
    return route


def _decide(state: SalesState) -> Route:
    if state.get("error"):
        return "error"

    message_type = state.get("message_type", "text")
    if message_type == "image":
        return "image_similarity"
    if message_type == "link":
        return "link_similarity"

    intent = state.get("intent", "chitchat")
    recs = state.get("last_recommended_products") or []

    if intent == "purchase_requested":
        return "smalltalk"

    if intent == "compare_products" and recs:
        return "compare_products"

    if intent == "request_product_images" and recs:
        return "answer_from_memory"

    if intent == "change_preferences":
        missing_hard = [
            s for s in HARD_SLOTS if s not in state.get("requirements", {})
        ]
        if missing_hard:
            return "needs_clarification"
        return "product_search"

    if state.get("should_search_products"):
        missing_hard = [
            s for s in HARD_SLOTS if s not in state.get("requirements", {})
        ]
        if missing_hard:
            return "needs_clarification"
        # Partial updates (budget/brand/…) with active recs keep prior soft slots.
        if (state.get("explicit_requirement_update") or state.get("requirements_changed")) and recs:
            return "product_search"
        if state.get("missing_slots"):
            return "needs_clarification"
        return "product_search"

    if state.get("should_use_memory") or (
        recs and is_memory_reference(state.get("current_message", ""), recs)
    ):
        return "answer_from_memory"

    if intent == "product_request":
        missing_hard = [
            s for s in HARD_SLOTS if s not in state.get("requirements", {})
        ]
        if missing_hard or state.get("missing_slots"):
            return "needs_clarification"
        return "product_search"

    return "smalltalk"
