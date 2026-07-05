"""LangGraph state definition for the sales assistant."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


class SalesState(TypedDict, total=False):
    """Single source of truth flowing through the graph.

    `total=False` lets nodes return partial updates (LangGraph merges them),
    while `initial_state()` guarantees every key exists at runtime.
    """

    # --- identity / input ---
    user_id: str
    chat_id: str
    current_message: str
    message_type: Literal["text", "image", "link"]
    image_caption: Optional[str]
    image_file_name: Optional[str]

    # --- user knowledge (persisted via memory_service) ---
    user_profile: dict
    requirements: dict          # budget / category / brands / use_case / priorities / constraints
    missing_slots: list[str]    # requirement slots still unknown
    conversation_stage: str     # new | clarifying | recommended | comparing | ...

    # --- conversation context ---
    intent: str                 # detected intent for this message
    last_recommended_products: list[dict]
    selected_product_ids: list[str]
    last_query_result: list[dict]
    comparison_context: dict
    memory_summary: str

    # --- routing flags ---
    should_search_products: bool
    should_use_memory: bool
    requirements_changed: bool  # user changed budget/brand/... -> partial re-run
    explicit_requirement_update: bool  # this message carries new/changed slots

    # --- error handling ---
    error: Optional[str]
    retry_count: int

    # --- output ---
    final_response: str
    send_product_images: list[dict]  # explicit photo send from memory (no re-search)


# Requirement slots the bot tries to fill before recommending.
# `budget` and `category` are hard requirements; the rest are nice-to-have
# and asked at most once.
HARD_SLOTS = ("category", "budget")
SOFT_SLOTS = ("use_case", "brands")
ALL_SLOTS = HARD_SLOTS + SOFT_SLOTS


def initial_state(
    user_id: str,
    chat_id: str,
    message: str,
    message_type: str = "text",
    image_caption: Optional[str] = None,
    image_file_name: Optional[str] = None,
) -> SalesState:
    """Fresh per-message state; session fields are hydrated by the
    load_or_create_session node."""
    return SalesState(
        user_id=user_id,
        chat_id=chat_id,
        current_message=message,
        message_type=message_type,  # type: ignore[typeddict-item]
        image_caption=image_caption,
        image_file_name=image_file_name,
        user_profile={},
        requirements={},
        missing_slots=[],
        conversation_stage="new",
        intent="",
        last_recommended_products=[],
        selected_product_ids=[],
        last_query_result=[],
        comparison_context={},
        memory_summary="",
        should_search_products=False,
        should_use_memory=False,
        requirements_changed=False,
        explicit_requirement_update=False,
        error=None,
        retry_count=0,
        final_response="",
        send_product_images=[],
    )
