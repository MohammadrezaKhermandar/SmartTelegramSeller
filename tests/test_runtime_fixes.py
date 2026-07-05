"""Tests for budget routing and LLM token limits."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.graph import nodes
from app.graph.builder import run_turn
from app.graph.router import conditional_router
from app.graph.state import initial_state
from app.services import llm_service
from app.services.memory_service import get_memory_service


@pytest.fixture()
def chat_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def test_budget_constraint_overrides_memory_question(chat_id):
    memory = get_memory_service()
    memory.get_or_create_session(chat_id, "u")
    memory.update_session(
        chat_id,
        requirements={
            "category": "لپ‌تاپ",
            "budget": 50_000_000,
            "use_case": "برنامه‌نویسی",
            "brands": [],
            "brands_asked": True,
        },
        conversation_stage="recommended",
    )
    memory.save_recommendations(
        chat_id,
        [
            {"product_id": "1", "name": "A", "effective_price": 45_000_000},
            {"product_id": "2", "name": "B", "effective_price": 48_000_000},
            {"product_id": "3", "name": "C", "effective_price": 49_000_000},
        ],
    )

    state = initial_state("u", chat_id, "نه ببین من بیشتر از 40 میلیون ندارم واقعا")
    state["last_recommended_products"] = memory.get_active_recommendations(chat_id)
    state["requirements"] = memory.get_or_create_session(chat_id, "u")["requirements"]
    state["conversation_stage"] = "recommended"

    extracted = nodes.extract_user_intent_and_requirements(state)
    state.update(extracted)
    relevance = nodes.check_memory_relevance(state)
    state.update(relevance)

    assert extracted["intent"] == "product_request"
    assert extracted["explicit_requirement_update"] is True
    assert relevance["should_use_memory"] is False
    assert relevance["should_search_products"] is True
    assert conditional_router(state) == "product_search"


@pytest.mark.integration
def test_budget_constraint_refreshes_recommendations(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    before = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    result = run_turn("u", chat_id, "نه ببین من بیشتر از 40 میلیون ندارم واقعا")
    after = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    session = get_memory_service().get_or_create_session(chat_id, "u")
    assert session["requirements"]["budget"] == 40_000_000
    assert session["requirements"].get("category")
    assert "گزینه" in result["final_response"] or len(after) >= 3


def test_llm_max_tokens_is_capped(monkeypatch):
    captured: dict = {}

    class FakeBound:
        def invoke(self, messages):
            return MagicMock(content="ok")

    class FakeLLM:
        def bind(self, **kwargs):
            captured.update(kwargs)
            return FakeBound()

    monkeypatch.setattr(llm_service, "get_llm", lambda temperature=0.4: FakeLLM())
    monkeypatch.setattr(llm_service, "clamp_max_tokens", lambda n: min(n, 800))
    llm_service.chat("sys", "user")
    assert captured.get("max_tokens", 0) <= 1200
    assert captured.get("max_tokens") == 800


def test_llm_402_returns_none_without_breaking_flow(monkeypatch):
    class FakeLLM:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            raise Exception("Error code: 402 - insufficient credits")

    monkeypatch.setattr(llm_service, "get_llm", lambda temperature=0.4: FakeLLM())
    assert llm_service.chat("sys", "user") is None
    assert llm_service.chat_json("sys", "user") is None
