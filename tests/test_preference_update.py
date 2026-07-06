"""Tests for change_preferences / allow_budget_overflow updates."""

import uuid

import pytest

from app.graph import nodes
from app.graph.builder import run_turn
from app.graph.router import conditional_router
from app.graph.state import initial_state
from app.services.memory_service import get_memory_service
from app.utils.text_normalizer import is_change_preferences, is_reject_budget_overflow


@pytest.fixture()
def chat_id() -> str:
    return f"pref-{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize(
    "message",
    [
        "نه قیمت بالاتر نمیتونم",
        "بیشتر از این نمی‌تونم هزینه کنم",
        "بودجه‌ام همینه",
        "نمی‌خوام بیشتر هزینه کنم",
        "سقف بودجه همینه",
        "گرون‌تر نمی‌خوام",
        "فقط داخل همین بودجه",
    ],
)
def test_reject_budget_overflow_phrases(message: str):
    assert is_reject_budget_overflow(message)
    assert is_change_preferences(message)


def _seed_session_with_three_laptops(chat_id: str) -> None:
    memory = get_memory_service()
    memory.get_or_create_session(chat_id, "u")
    memory.update_session(
        chat_id,
        requirements={
            "category": "لپ‌تاپ",
            "budget": 40_000_000,
            "use_case": "برنامه‌نویسی",
            "brands": [],
            "brands_asked": True,
            "allow_budget_overflow": True,
        },
        conversation_stage="recommended",
    )
    memory.save_recommendations(
        chat_id,
        [
            {
                "product_id": "p35",
                "name": "لپ‌تاپ ۳۵M",
                "category": "لپ‌تاپ",
                "effective_price": 35_000_000,
                "rating": 4.5,
            },
            {
                "product_id": "p37",
                "name": "لپ‌تاپ ۳۷M",
                "category": "لپ‌تاپ",
                "effective_price": 37_000_000,
                "rating": 4.4,
            },
            {
                "product_id": "p52",
                "name": "لپ‌تاپ ۵۲M",
                "category": "لپ‌تاپ",
                "effective_price": 52_000_000,
                "rating": 4.6,
                "is_alternative": True,
                "over_budget": True,
            },
        ],
    )


def test_reject_overflow_not_treated_as_memory_question(chat_id):
    _seed_session_with_three_laptops(chat_id)
    memory = get_memory_service()

    state = initial_state("u", chat_id, "نه قیمت بالاتر نمیتونم")
    state["last_recommended_products"] = memory.get_active_recommendations(chat_id)
    state["requirements"] = memory.get_or_create_session(chat_id, "u")["requirements"]
    state["conversation_stage"] = "recommended"

    extracted = nodes.extract_user_intent_and_requirements(state)
    state.update(extracted)
    relevance = nodes.check_memory_relevance(state)
    state.update(relevance)

    assert extracted["intent"] == "change_preferences"
    assert state["requirements"]["allow_budget_overflow"] is False
    assert state["requirements"]["hard_max_price"] is True
    assert relevance["should_use_memory"] is False
    assert relevance["should_search_products"] is True
    assert conditional_router(state) == "product_search"


def test_reject_overflow_removes_52m_and_reruns(chat_id):
    _seed_session_with_three_laptops(chat_id)
    memory = get_memory_service()

    state = initial_state("u", chat_id, "نه قیمت بالاتر نمیتونم")
    state["last_recommended_products"] = memory.get_active_recommendations(chat_id)
    state["requirements"] = memory.get_or_create_session(chat_id, "u")["requirements"]
    state["conversation_stage"] = "recommended"

    state.update(nodes.extract_user_intent_and_requirements(state))
    state.update(nodes.check_memory_relevance(state))

    search = nodes.hybrid_product_search(state)
    state.update(search)
    state.update(nodes.rank_recommendations(state))
    out = nodes.generate_sales_response(state)

    products = out["last_recommended_products"]
    response = out["final_response"]
    active = memory.get_active_recommendations(chat_id)

    assert all(p["effective_price"] <= 40_000_000 for p in products)
    assert all(p["effective_price"] <= 40_000_000 for p in active)
    assert not any(p["effective_price"] == 52_000_000 for p in products)
    assert "۵۲" not in response and "52" not in response
    assert "لپ‌تاپ ۵۲M" not in response
    assert len(products) == 2
    assert "نزدیک‌ترین جایگزین" not in response


@pytest.mark.integration
def test_reject_overflow_end_to_end(chat_id):
    run_turn(
        "u",
        chat_id,
        "لپ‌تاپ تا ۴۰ میلیون برای برنامه‌نویسی، برند فرقی نداره",
    )
    memory = get_memory_service()
    memory.save_recommendations(
        chat_id,
        [
            *memory.get_active_recommendations(chat_id)[:2],
            {
                "product_id": "p52",
                "name": "لپ‌تاپ ۵۲M",
                "category": "لپ‌تاپ",
                "effective_price": 52_000_000,
                "is_alternative": True,
                "over_budget": True,
            },
        ],
    )

    result = run_turn("u", chat_id, "نه قیمت بالاتر نمیتونم")
    active = memory.get_active_recommendations(chat_id)
    response = result.get("final_response", "")

    assert result.get("requirements", memory.get_or_create_session(chat_id, "u")["requirements"])[
        "allow_budget_overflow"
    ] is False
    assert all(p["effective_price"] <= 40_000_000 for p in active)
    assert "لپ‌تاپ ۵۲M" not in response
