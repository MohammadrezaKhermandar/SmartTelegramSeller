"""LangGraph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class UserRequirements(TypedDict, total=False):
    category: str | None
    brand: str | None
    min_price: float | None
    max_price: float | None
    usage: str | None
    features: list[str]
    raw_query: str | None


class UserProfile(TypedDict, total=False):
    name: str | None
    preferences: dict[str, Any]


ConversationStage = Literal[
    "greeting",
    "general_chat",
    "company_info",
    "gathering_requirements",
    "recommending",
    "answering_followup",
    "comparing",
    "image_search",
    "url_search",
    "error",
    "completed",
]

IntentType = Literal[
    "new_product_request",
    "followup_question",
    "compare",
    "change_preferences",
    "image_input",
    "url_input",
    "greeting",
    "general_chat",
    "company_question",
    "purchase",
    "unknown",
]


class SalesAssistantState(TypedDict, total=False):
    """LangGraph State – persisted per user via checkpointer."""

    user_id: str
    messages: Annotated[list, add_messages]
    user_profile: UserProfile
    current_intent: IntentType
    product_category: str | None
    requirements: UserRequirements
    missing_slots: list[str]
    recommended_products: list[dict[str, Any]]
    selected_product_ids: list[str]
    last_search_query: str | None
    last_search_result: list[dict[str, Any]]
    conversation_stage: ConversationStage
    image_input: dict[str, Any] | None
    url_input: str | None
    errors: list[str]
    retry_count: int
    last_user_message_at: str | None
    purchase_status: Literal["none", "pending", "purchased"]
    response_text: str | None
    search_note: str | None
    compare_result: dict[str, Any] | None
    from_memory: bool
    from_tool_agent: bool
