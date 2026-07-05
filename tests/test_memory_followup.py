"""Tests for answering product-attribute follow-ups from memory."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.graph import nlp
from app.graph.nodes import answer_from_memory_node

LAPTOP = {
    "product_id": "42",
    "title": "لنوو IdeaPad 5",
    "category": "لپ‌تاپ",
    "brand": "لنوو",
    "price": 39325000.0,
    "rating": 4.2,
    "features": "پورت Thunderbolt 4 | پردازنده Core i7 | باتری ۱۰ ساعته | رم ۱۶ گیگ",
    "description": "لنوو IdeaPad 5 مناسب برنامه‌نویسی.",
}

SECOND_LAPTOP = {
    "product_id": "43",
    "title": "ایسوس VivoBook",
    "category": "لپ‌تاپ",
    "brand": "ایسوس",
    "price": 35000000.0,
    "rating": 4.0,
    "features": "نمایشگر Full HD | رم ۸ گیگ | SSD 512 گیگ",
    "description": "ایسوس VivoBook.",
}


def _state(text: str, recommended: list[dict]) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "recommended_products": recommended,
    }


def test_single_product_ram_question_answers_from_memory():
    result = answer_from_memory_node(_state("رمش چقدره؟", [LAPTOP]))
    assert result["from_memory"] is True
    assert "۱۶ گیگ" in result["response_text"] or "16 گیگ" in result["response_text"]
    assert "رم" in result["response_text"]


def test_single_product_price_question_answers_from_memory():
    result = answer_from_memory_node(_state("قیمتش چنده؟", [LAPTOP]))
    assert result["from_memory"] is True
    assert "39,325,000" in result["response_text"]


def test_multiple_products_ordinal_ram_question_resolves_second():
    result = answer_from_memory_node(
        _state("دومی رمش چقدره؟", [LAPTOP, SECOND_LAPTOP])
    )
    assert "۸ گیگ" in result["response_text"] or "8 گیگ" in result["response_text"]


def test_option_number_reference_resolves_product():
    product = nlp.resolve_ordinal_reference("گزینه ۲ رمش چقدره؟", [LAPTOP, SECOND_LAPTOP])
    assert product is SECOND_LAPTOP


def test_battery_question_answers_from_features():
    result = answer_from_memory_node(_state("باتریش چقدره؟", [LAPTOP]))
    assert "۱۰ ساعته" in result["response_text"]


def test_missing_spec_returns_not_recorded_message():
    no_ram = {**LAPTOP, "features": "پورت USB-C", "description": "بدون جزئیات"}
    result = answer_from_memory_node(_state("رمش چقدره؟", [no_ram]))
    assert result["response_text"] == "این مشخصه در اطلاعات محصول ثبت نشده."


def test_budget_phrase_is_not_ordinal_reference():
    assert nlp.resolve_ordinal_reference("بودجه شد ۳۰ میلیون", [LAPTOP, SECOND_LAPTOP]) is None


def test_budget_phrase_routes_to_change_preferences_not_followup():
    intent = nlp.detect_intent(
        "بودجه شد ۳۰ میلیون",
        has_recommendations=True,
        existing_requirements={"category": "لپ‌تاپ"},
        conversation_stage="recommending",
    )
    assert intent == "change_preferences"


def test_ram_question_detected_as_ram_not_price():
    assert nlp.detect_field_question("رمش چقدره؟") == "ram"
    assert nlp.detect_field_question("قیمتش چنده؟") == "price"
    assert nlp.detect_field_question("حافظه‌اش چقدره؟") == "storage"
    assert nlp.detect_field_question("باتریش چقدره؟") == "battery"
    assert nlp.detect_field_question("امتیازش چنده؟") == "rating"
    assert nlp.detect_field_question("برندش چیه؟") == "brand"
