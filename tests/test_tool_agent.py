"""Tests for tool agent node."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph, reset_graph
from app.graph.nodes import tool_agent_node
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


def test_tool_agent_falls_back_without_llm():
    state = {
        "requirements": {
            "category": "لپ‌تاپ",
            "usage": "برنامه‌نویسی",
            "max_price": 80_000_000,
            "raw_query": "لپ‌تاپ برنامه‌نویسی",
        },
        "last_search_query": "لپ‌تاپ برنامه‌نویسی",
        "retry_count": 0,
        "errors": [],
    }

    with patch("app.graph.tool_agent.run_search_tool_agent", return_value=None):
        result = tool_agent_node(state)

    assert result.get("last_search_result")
    assert len(result["last_search_result"]) >= 3
    assert result.get("from_tool_agent") is False


def test_tool_agent_uses_llm_tool_calls():
    fake_products = [
        {"product_id": "1", "title": "A", "price": 1000, "score": 1.0},
        {"product_id": "2", "title": "B", "price": 2000, "score": 0.9},
        {"product_id": "3", "title": "C", "price": 3000, "score": 0.8},
    ]
    state = {
        "requirements": {"category": "لپ‌تاپ", "raw_query": "لپ‌تاپ"},
        "last_search_query": "لپ‌تاپ",
        "retry_count": 0,
        "errors": [],
    }

    with patch(
        "app.graph.nodes.run_search_tool_agent",
        return_value=(fake_products, "agent ok"),
    ):
        result = tool_agent_node(state)

    assert result.get("from_tool_agent") is True
    assert len(result["last_search_result"]) == 3


def test_graph_uses_tool_agent_for_full_requirements():
    graph = build_graph()
    config = {"configurable": {"thread_id": "tool_agent_graph_test"}}

    with patch("app.graph.tool_agent.run_search_tool_agent", return_value=None):
        result = graph.invoke(
            {
                "user_id": "tool_agent_graph_test",
                "messages": [
                    HumanMessage(
                        content="لپ‌تاپ برای برنامه‌نویسی با بودجه ۸۰ میلیون تومان می‌خوام"
                    )
                ],
                "retry_count": 0,
                "errors": [],
            },
            config,
        )

    assert result.get("recommended_products")
    assert len(result["recommended_products"]) >= 3
