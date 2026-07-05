"""Tests for memory behavior in follow-up questions."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph, reset_graph
from app.graph.runner import invoke_graph
from app.main import initialize_app
from app.memory.checkpointer import reset_checkpointer

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_app():
    reset_graph()
    reset_checkpointer()
    initialize_app(CSV_PATH)
    yield
    reset_graph()
    reset_checkpointer()


def test_followup_uses_memory_not_full_search():
    user_id = "memory_test_user"
    graph = build_graph()
    config = {"configurable": {"thread_id": user_id}}

    # First: get recommendations
    result1 = graph.invoke(
        {
            "user_id": user_id,
            "messages": [
                HumanMessage(content="لپ‌تاپ برای برنامه‌نویسی با بودجه ۱۰۰ میلیون")
            ],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )
    assert len(result1.get("recommended_products", [])) >= 1
    products = result1["recommended_products"]
    second_product = products[1] if len(products) > 1 else products[0]

    # Follow-up about second product
    with patch("app.graph.nodes.tool_agent_node") as mock_search:
        result2 = graph.invoke(
            {
                "messages": [HumanMessage(content="اون دومی قیمتش چقدره؟")],
                "user_id": user_id,
            },
            config,
        )
        mock_search.assert_not_called()

    assert result2.get("from_memory") is True
    assert result2.get("response_text")
    assert str(int(second_product.get("price", 0))) in result2["response_text"].replace(",", "")


def test_changed_budget_refreshes_recommendations():
    user_id = "budget_change_user"
    config = {"configurable": {"thread_id": user_id}}
    graph = build_graph()

    graph.invoke(
        {
            "user_id": user_id,
            "messages": [HumanMessage(content="لپ‌تاپ برای اداری با بودجه ۱۰۰ میلیون")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="بودجه‌ام رو به ۵۰ میلیون تغییر دادم")],
            "user_id": user_id,
        },
        config,
    )
    assert result.get("recommended_products")
    req = result.get("requirements", {})
    assert req.get("max_price") == 50_000_000
    assert req.get("category") == "لپ‌تاپ"


def test_budget_shod_preserves_category():
    user_id = "budget_shod_user"
    config = {"configurable": {"thread_id": user_id}}
    graph = build_graph()

    graph.invoke(
        {
            "user_id": user_id,
            "messages": [HumanMessage(content="لپ‌تاپ برای اداری با بودجه ۱۰۰ میلیون")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="بودجه شد ۳۰ میلیون")],
            "user_id": user_id,
        },
        config,
    )
    req = result.get("requirements", {})
    assert req.get("category") == "لپ‌تاپ"
    assert req.get("max_price") == 30_000_000


def test_budget_update_keeps_same_category_recommendations():
    user_id = "budget_category_user"
    config = {"configurable": {"thread_id": user_id}}
    graph = build_graph()

    graph.invoke(
        {
            "user_id": user_id,
            "messages": [HumanMessage(content="لپ‌تاپ برای اداری با بودجه ۱۰۰ میلیون")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="بودجه شد ۵۰ میلیون")],
            "user_id": user_id,
        },
        config,
    )
    products = result.get("recommended_products") or []
    assert products, "Expected refreshed recommendations after budget change"
    for product in products[:5]:
        category = str(product.get("category", ""))
        assert "لپ" in category or "لپ‌تاپ" in category


def test_laptop_gathering_conversation_all_recommendations_are_laptops():
    """یه لپ‌تاپ می‌خوام → بودجه ۱۰۰ میلیون برای برنامه‌نویسی must stay in laptop category."""
    user_id = "laptop_gathering_user"
    config = {"configurable": {"thread_id": user_id}}
    graph = build_graph()

    turn1 = graph.invoke(
        {
            "user_id": user_id,
            "messages": [HumanMessage(content="یه لپ‌تاپ می‌خوام")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )
    assert turn1.get("requirements", {}).get("category") == "لپ‌تاپ"
    assert turn1.get("conversation_stage") == "gathering_requirements"

    turn2 = graph.invoke(
        {
            "messages": [HumanMessage(content="بودجه ۱۰۰ میلیون برای برنامه‌نویسی")],
            "user_id": user_id,
        },
        config,
    )
    assert turn2.get("current_intent") == "new_product_request"
    req = turn2.get("requirements", {})
    assert req.get("category") == "لپ‌تاپ"
    assert req.get("usage") == "برنامه‌نویسی"
    assert req.get("max_price") == 100_000_000

    products = turn2.get("recommended_products") or []
    assert products, "Expected laptop recommendations after completing requirements"
    for product in products:
        category = str(product.get("category", ""))
        assert category == "لپ‌تاپ" or "لپ" in category, (
            f"Expected laptop, got category={category!r} title={product.get('title')!r}"
        )
