"""Hard max-price budget constraint tests."""

import uuid
from unittest.mock import patch

import pytest

from app.graph import nodes
from app.graph.builder import run_turn
from app.graph.router import conditional_router
from app.graph.state import initial_state
from app.services import llm_service
from app.services.memory_service import get_memory_service
from app.services.recommendation_service import (
    filter_products_by_max_price,
    get_recommendation_service,
)
from app.utils.text_normalizer import extract_budget, is_hard_max_budget


@pytest.fixture()
def chat_id() -> str:
    return f"hard-budget-{uuid.uuid4().hex[:8]}"


def test_is_hard_max_budget_phrases():
    assert is_hard_max_budget("نه بیشتر از 40 میلیون ندارم")
    assert is_hard_max_budget("نهایتا ۴۰ میلیون")
    assert is_hard_max_budget("سقف بودجه ۳۵ میلیون")
    assert is_hard_max_budget("بالاتر از ۵۰ میلیون نمی‌خوام")
    assert not is_hard_max_budget("لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی")


def test_filter_products_by_max_price():
    products = [
        {"name": "A", "effective_price": 30_000_000},
        {"name": "B", "effective_price": 52_555_500},
        {"name": "C", "effective_price": 40_000_000},
    ]
    filtered = filter_products_by_max_price(products, 40_000_000)
    assert len(filtered) == 2
    assert all(p["effective_price"] <= 40_000_000 for p in filtered)


def test_recommend_strict_budget_no_padding():
    service = get_recommendation_service()
    result = service.recommend(
        "لپ‌تاپ برنامه‌نویسی",
        {
            "budget": 40_000_000,
            "category": "لپ‌تاپ",
            "use_case": "برنامه‌نویسی",
            "hard_max_price": True,
        },
        top_k=3,
        strict_budget=True,
    )
    assert len(result["products"]) <= 3
    assert all(p["effective_price"] <= 40_000_000 for p in result["products"])
    assert not any(p.get("over_budget") for p in result["products"])
    # Catalog has only 2 in-stock laptops under 40M — must not backfill a 3rd.
    assert len(result["products"]) == 2


def test_budget_constraint_routes_to_product_search(chat_id):
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
            {"product_id": "3", "name": "C", "effective_price": 52_555_500},
        ],
    )

    state = initial_state("u", chat_id, "نه بیشتر از 40 میلیون ندارم")
    state["last_recommended_products"] = memory.get_active_recommendations(chat_id)
    state["requirements"] = memory.get_or_create_session(chat_id, "u")["requirements"]
    state["conversation_stage"] = "recommended"

    extracted = nodes.extract_user_intent_and_requirements(state)
    state.update(extracted)
    relevance = nodes.check_memory_relevance(state)
    state.update(relevance)

    assert extracted["intent"] == "product_request"
    assert extracted["explicit_requirement_update"] is True
    assert state["requirements"]["budget"] == 40_000_000
    assert state["requirements"].get("hard_max_price") is True
    assert relevance["should_use_memory"] is False
    assert relevance["should_search_products"] is True
    assert conditional_router(state) == "product_search"


def test_memory_purges_over_budget_on_explicit_update(chat_id):
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
            {"product_id": "1", "name": "A", "effective_price": 35_000_000},
            {"product_id": "2", "name": "B", "effective_price": 52_555_500},
        ],
    )

    state = initial_state("u", chat_id, "نه بیشتر از 40 میلیون ندارم")
    state["last_recommended_products"] = memory.get_active_recommendations(chat_id)
    state["requirements"] = memory.get_or_create_session(chat_id, "u")["requirements"]
    state["conversation_stage"] = "recommended"

    extracted = nodes.extract_user_intent_and_requirements(state)
    state.update(extracted)
    nodes.check_memory_relevance(state)

    active = memory.get_active_recommendations(chat_id)
    assert all(p["effective_price"] <= 40_000_000 for p in active)
    assert not any(p["effective_price"] > 40_000_000 for p in active)


@pytest.mark.integration
def test_hard_budget_end_to_end(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    result = run_turn("u", chat_id, "نه بیشتر از 40 میلیون ندارم")

    memory = get_memory_service()
    session = memory.get_or_create_session(chat_id, "u")
    active = memory.get_active_recommendations(chat_id)
    response = result.get("final_response", "")

    assert session["requirements"]["budget"] == 40_000_000
    assert session["requirements"].get("hard_max_price") is True
    assert all(p["effective_price"] <= 40_000_000 for p in active)
    assert len(active) == 2
    assert "52" not in response and "۵۲" not in response
    assert "نزدیک‌ترین جایگزین" not in response


@pytest.mark.integration
def test_generate_sales_response_guard(chat_id):
    state = initial_state("u", chat_id, "نه بیشتر از 40 میلیون ندارم")
    state["requirements"] = {
        "category": "لپ‌تاپ",
        "budget": 40_000_000,
        "use_case": "برنامه‌نویسی",
        "hard_max_price": True,
    }
    state["last_recommended_products"] = [
        {
            "product_id": "ok",
            "name": "لپ‌تاپ ارزان",
            "effective_price": 35_000_000,
            "rating": 4.5,
            "score_components": {"budget": 1.0},
            "_reason": "توی بودجه‌ات جا می‌شه",
        },
        {
            "product_id": "bad",
            "name": "لپ‌تاپ گران",
            "effective_price": 52_555_500,
            "rating": 4.6,
            "is_alternative": True,
            "over_budget": True,
            "_reason": "نزدیک‌ترین جایگزین",
        },
    ]
    state["comparison_context"] = {"strict_budget": True}

    with patch.object(nodes.llm_service, "chat", return_value=None):
        out = nodes.generate_sales_response(state)

    assert "لپ‌تاپ گران" not in out["final_response"]
    assert "52" not in out["final_response"]
    assert len(out["last_recommended_products"]) == 1
    assert out["last_recommended_products"][0]["effective_price"] <= 40_000_000


def test_validate_polish_rejects_forbidden_over_budget_name():
    """Polish must not keep over-budget products even if raw draft included them."""
    draft = (
        "1. لپ‌تاپ ارزان — ۳۵,۰۰۰,۰۰۰ تومان\n"
        "   توی بودجه‌ات جا می‌شه\n"
        "2. لپ‌تاپ گران — ۵۲,۵۵۵,۵۰۰ تومان\n"
        "   نزدیک‌ترین جایگزین"
    )
    polished = (
        "با توجه به بودجه‌ات:\n"
        "1. لپ‌تاپ گران — ۵۲,۵۵۵,۵۰۰ تومان\n"
        "2. لپ‌تاپ ارزان — ۳۵,۰۰۰,۰۰۰ تومان"
    )
    result = llm_service.validate_polish(
        draft,
        polished,
        ["لپ‌تاپ ارزان"],
        forbidden_names=["لپ‌تاپ گران"],
    )
    assert "لپ‌تاپ گران" not in result
    assert "لپ‌تاپ ارزان" in result
    assert "نزدیک‌ترین جایگزین" not in result


def test_build_recommendation_text_polish_guard(chat_id):
    """End-to-end polish guard when upstream list still contains an over-budget item."""
    state = initial_state("u", chat_id, "نهایتا ۴۰ میلیون")
    requirements = {
        "category": "لپ‌تاپ",
        "budget": 40_000_000,
        "use_case": "برنامه‌نویسی",
        "hard_max_price": True,
    }
    products = [
        {
            "name": "لپ‌تاپ ارزان",
            "effective_price": 35_000_000,
            "rating": 4.5,
            "_reason": "توی بودجه‌ات جا می‌شه",
        },
        {
            "name": "لپ‌تاپ گران",
            "effective_price": 52_555_500,
            "rating": 4.6,
            "is_alternative": True,
            "_reason": "نزدیک‌ترین جایگزین",
        },
    ]
    hallucinated = (
        "پیشنهاد من:\n"
        "1. لپ‌تاپ گران — ۵۲,۵۵۵,۵۰۰ تومان\n"
        "2. لپ‌تاپ ارزان — ۳۵,۰۰۰,۰۰۰ تومان"
    )
    with patch.object(nodes.llm_service, "chat", return_value=hallucinated):
        text = nodes._build_recommendation_text(state, products, requirements, strict=True)

    assert "لپ‌تاپ گران" not in text
    assert "۵۲" not in text and "52" not in text
    assert "نزدیک‌ترین جایگزین" not in text
    assert "لپ‌تاپ ارزان" in text


def test_recommend_strict_empty_returns_clear_reason():
    service = get_recommendation_service()
    result = service.recommend(
        "لپ‌تاپ برنامه‌نویسی",
        {
            "budget": 1_000_000,
            "category": "لپ‌تاپ",
            "use_case": "برنامه‌نویسی",
            "hard_max_price": True,
        },
        top_k=3,
        strict_budget=True,
    )
    assert result["products"] == []
    assert nodes.HARD_BUDGET_EMPTY_MESSAGE in result["empty_reason"]


def test_hard_budget_empty_response_message(chat_id):
    state = initial_state("u", chat_id, "نهایتا ۵ میلیون")
    state["requirements"] = {
        "category": "لپ‌تاپ",
        "budget": 5_000_000,
        "use_case": "برنامه‌نویسی",
        "hard_max_price": True,
    }
    state["last_recommended_products"] = []
    state["comparison_context"] = {
        "strict_budget": True,
        "empty_reason": nodes.HARD_BUDGET_EMPTY_MESSAGE,
    }

    out = nodes.generate_sales_response(state)

    assert nodes.HARD_BUDGET_EMPTY_MESSAGE in out["final_response"]
    assert out["last_recommended_products"] == []
    assert "نزدیک‌ترین" not in out["final_response"]
    assert "بالاتر از بودجه" not in out["final_response"]


@pytest.mark.integration
def test_hard_budget_empty_end_to_end(chat_id):
    result = run_turn(
        "u",
        chat_id,
        "لپ‌تاپ برای برنامه‌نویسی، نهایتا ۵ میلیون، برند فرقی نداره",
    )
    response = result.get("final_response", "")
    active = get_memory_service().get_active_recommendations(chat_id)

    assert nodes.HARD_BUDGET_EMPTY_MESSAGE in response
    assert active == []
    assert "نزدیک‌ترین" not in response
