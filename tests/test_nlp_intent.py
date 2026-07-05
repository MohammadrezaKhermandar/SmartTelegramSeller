"""Tests for rule-based intent detection."""

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nlp
from app.graph.graph_builder import build_graph, reset_graph
from app.graph.prompts import BUSINESS_PROFILE, answer_company_question
from app.graph.routers import route_after_intent
from app.main import initialize_app
from app.memory.checkpointer import reset_checkpointer
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("سلام", "greeting"),
        ("چطوری خوبی مدیر مجموعه شما کیه؟", "company_question"),
        ("یه لپ تاپ میخوام", "new_product_request"),
        ("عطر زنونه پیشنهاد بده", "new_product_request"),
        ("مدیرتون کیه؟", "company_question"),
        ("گوشی دارید؟", "new_product_request"),
        ("چطوری خوبی؟", "general_chat"),
        ("ساعت کاریتون چیه؟", "company_question"),
        ("شما کی هستید؟", "company_question"),
        ("آدرس فروشگاه کجاست؟", "company_question"),
        ("فقط یه حرف", "unknown"),
    ],
)
def test_detect_intent(message: str, expected: str) -> None:
    assert nlp.detect_intent(message) == expected


@pytest.mark.parametrize(
    "text",
    [
        "یه لب تاب میخوام",
        "یه لبتاب میخوام",
        "لپ تاب برای برنامه‌نویسی",
        "لپتاب دارید؟",
        "یه لپ‌تاپ می‌خوام",
    ],
)
def test_laptop_misspellings_detected_as_laptop_category(text: str) -> None:
    req = nlp.extract_requirements(text)
    assert req["category"] == "لپ‌تاپ", f"{text!r} must map to لپ‌تاپ"


def test_company_question_answers_manager_from_profile() -> None:
    response = answer_company_question("مدیرتون کیه؟")
    assert BUSINESS_PROFILE["manager"] in response
    assert "بودجه" not in response
    assert "کاری" not in response


def test_company_question_routes_to_final_response() -> None:
    state = {"current_intent": "company_question"}
    assert route_after_intent(state) == "final_response"


def test_general_chat_routes_to_final_response() -> None:
    state = {"current_intent": "general_chat"}
    assert route_after_intent(state) == "final_response"


@pytest.fixture(autouse=True)
def setup_app():
    reset_graph()
    reset_checkpointer()
    initialize_app(CSV_PATH)
    yield
    reset_graph()
    reset_checkpointer()


def test_graph_general_chat_does_not_ask_budget() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_intent_general"}}
    result = graph.invoke(
        {
            "user_id": "test_intent_general",
            "messages": [HumanMessage(content="چطوری خوبی مدیر مجموعه شما کیه؟")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )
    response = result.get("response_text", "")
    assert BUSINESS_PROFILE["manager"] in response
    assert "بودجه" not in response
    assert result.get("conversation_stage") in ("company_info", "general_chat")
