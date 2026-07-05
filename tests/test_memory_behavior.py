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
            "messages": [HumanMessage(content="بودجه‌ام رو به ۳۰ میلیون تغییر دادم")],
            "user_id": user_id,
        },
        config,
    )
    assert result.get("recommended_products")
    req = result.get("requirements", {})
    assert req.get("max_price") is not None
