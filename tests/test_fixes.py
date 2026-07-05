"""Regression tests for reported Telegram / graph bugs."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from app.graph import nodes
from app.graph.builder import run_turn
from app.graph.router import conditional_router
from app.graph.state import initial_state
from app.services.memory_service import get_memory_service
from app.services.recommendation_service import get_recommendation_service
from app.utils.text_normalizer import detect_category, extract_budget, is_memory_reference


@pytest.fixture()
def chat_id() -> str:
    return f"fix-{uuid.uuid4().hex[:8]}"


def _sample_recs() -> list[dict]:
    return [
        {
            "product_id": "1",
            "name": "لپ‌تاپ اپل Pavilion x360 Lite",
            "category": "لپ‌تاپ",
            "brand": "اپل",
            "effective_price": 45_000_000,
            "price": 45_000_000,
            "stock": 5,
            "rating": 4.5,
            "features": "رم 16GB، SSD 512GB",
            "_position": 1,
        },
        {
            "product_id": "2",
            "name": "لپ‌تاپ ایسوس VivoBook 15",
            "category": "لپ‌تاپ",
            "brand": "ایسوس",
            "effective_price": 38_000_000,
            "price": 38_000_000,
            "stock": 3,
            "rating": 4.2,
            "features": "رم 8GB، SSD 256GB",
            "_position": 2,
        },
    ]


def test_no_budget_from_plain_laptop_request():
    assert extract_budget("یه لپ‌تاپ برای برنامه‌نویسی می‌خوام") is None


def test_budget_50_million_extraction():
    assert extract_budget("بودجه‌ام ۵۰ میلیونه") == 50_000_000


def test_budget_40_million_ta():
    assert extract_budget("تا ۴۰ میلیون") == 40_000_000


def test_budget_30_toman_shorthand():
    assert extract_budget("زیر ۳۰ تومن") == 30_000_000


def test_laptop_typo_maps_to_category():
    from app.services.pandas_query_service import get_pandas_service

    cats = get_pandas_service().list_categories()
    assert detect_category("یه لبتاب میخوام", cats) == "لپ‌تاپ"


def test_memory_question_domi_ram_routes_to_memory():
    state = initial_state("u", "c1", "دومی رمش چنده")
    state["last_recommended_products"] = _sample_recs()
    state["intent"] = nodes._rule_based_intent(state)
    mem = nodes.check_memory_relevance(state)
    assert mem["should_use_memory"] is True
    assert mem["should_search_products"] is False
    assert mem["intent"] == "memory_question"
    assert conditional_router({**state, **mem}) == "answer_from_memory"


def test_product_name_memory_reference_routes_to_memory():
    recs = _sample_recs()
    query = "رم اپل Pavilion x360 Lite چنده"
    state = initial_state("u", "c1", query)
    state["last_recommended_products"] = recs
    state["intent"] = nodes._rule_based_intent(state)
    assert is_memory_reference(query, recs)
    mem = nodes.check_memory_relevance(state)
    assert conditional_router({**state, **mem}) == "answer_from_memory"


def test_no_non_laptop_in_laptop_recommendations():
    from app.services.pandas_query_service import ProductFilter, get_pandas_service
    from app.services.recommendation_service import _in_category

    pandas = get_pandas_service()
    result = pandas.query(
        ProductFilter(
            categories=["لپ‌تاپ"],
            max_price=50_000_000,
            in_stock_only=True,
            limit=10,
        )
    )
    assert result["products"]
    for product in result["products"][:3]:
        assert _in_category(product, "لپ‌تاپ")


@pytest.mark.integration
def test_clarifying_no_brand_then_search(chat_id):
    run_turn("u", chat_id, "یه لبتاب میخوام")
    run_turn("u", chat_id, "بودجه ۴۰ میلیون\nلبتاب\nبرای برنامه‌نویسی")
    result = run_turn("u", chat_id, "نه فرقی نداره")
    recs = get_memory_service().get_active_recommendations(chat_id)
    assert len(recs) >= 3
    assert "مقایسه" not in result["final_response"][:80]


def test_single_response_per_turn():
    """Handler sends one text reply per graph turn (photos are separate)."""
    from app.telegram.handlers import _reply_with_result

    message = AsyncMock()
    message.reply_text = AsyncMock()
    message.reply_photo = AsyncMock()

    result = {
        "final_response": "سلام — این یک پاسخ تست است.",
        "last_recommended_products": [
            {"name": "X", "effective_price": 1000, "image_url": "http://example.com/x.jpg"}
        ],
    }
    asyncio.run(_reply_with_result(message, result))
    assert message.reply_text.await_count == 1


@pytest.mark.integration
def test_compare_all_current_recommendations(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    result = run_turn("u", chat_id, "میشه با هم مقایسشون کنی")
    assert "قیمت" in result["final_response"]
    assert "گزینه" in result["final_response"]


@pytest.mark.integration
def test_budget_change_preserves_category_and_use_case(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    run_turn("u", chat_id, "بودجم ۵۰ میلیونه")
    session = get_memory_service().get_or_create_session(chat_id, "u")
    assert session["requirements"]["budget"] == 50_000_000
    assert session["requirements"].get("category")
    assert session["requirements"].get("use_case")
