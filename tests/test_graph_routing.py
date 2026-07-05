"""Tests for LangGraph routing."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph, reset_graph
from app.graph.routers import route_after_info_check, route_after_intent
from app.graph.state import SalesAssistantState
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


def test_incomplete_request_routes_to_clarifying():
    state: SalesAssistantState = {
        "missing_slots": ["budget", "usage"],
        "requirements": {"category": "لپ‌تاپ"},
    }
    assert route_after_info_check(state) == "ask_clarifying"


def test_enough_info_routes_to_tool_agent():
    state: SalesAssistantState = {
        "missing_slots": [],
        "requirements": {
            "category": "لپ‌تاپ",
            "usage": "برنامه‌نویسی",
            "max_price": 50_000_000,
        },
    }
    assert route_after_info_check(state) == "tool_agent"


def test_followup_intent_routes_to_memory():
    state: SalesAssistantState = {
        "current_intent": "followup_question",
        "recommended_products": [{"product_id": "1", "title": "test"}],
    }
    assert route_after_intent(state) == "answer_from_memory"


def test_company_question_intent_routes_to_final_response():
    state: SalesAssistantState = {"current_intent": "company_question"}
    assert route_after_intent(state) == "final_response"


def test_general_chat_intent_routes_to_final_response():
    state: SalesAssistantState = {"current_intent": "general_chat"}
    assert route_after_intent(state) == "final_response"


def test_graph_clarifying_question_for_laptop_request():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_routing_1"}}
    result = graph.invoke(
        {
            "user_id": "test_routing_1",
            "messages": [HumanMessage(content="یه لپ‌تاپ می‌خوام")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )
    assert result.get("response_text")
    assert "بودجه" in result["response_text"] or "کاری" in result["response_text"]
    assert result.get("conversation_stage") == "gathering_requirements"


def test_graph_hybrid_search_with_full_requirements():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_routing_2"}}
    result = graph.invoke(
        {
            "user_id": "test_routing_2",
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
