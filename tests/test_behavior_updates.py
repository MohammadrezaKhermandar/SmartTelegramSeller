"""Tests for purchase, compare, images, and Telegram command updates."""

import uuid

import pytest

from app.graph.builder import run_turn
from app.services.memory_service import get_memory_service
from app.telegram.formatters import (
    HELP_TEXT,
    PURCHASE_COMING_SOON_TEXT,
    build_categories_message,
)
from app.utils.text_normalizer import is_product_image_request, is_purchase_request


@pytest.fixture()
def chat_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def test_purchase_request_says_coming_soon(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    result = run_turn("u", chat_id, "می‌خوام بخرم")
    assert PURCHASE_COMING_SOON_TEXT in result["final_response"]
    assert get_memory_service().get_purchase_status(chat_id) == "browsing"
    session = get_memory_service().get_or_create_session(chat_id, "u")
    assert session["conversation_stage"] == "purchase_requested"


def test_mark_purchased_command_says_coming_soon():
    assert "/mark_purchased" in HELP_TEXT
    assert "به‌زودی" in PURCHASE_COMING_SOON_TEXT


@pytest.mark.integration
def test_compare_two_selected_products_only(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    recs = get_memory_service().get_active_recommendations(chat_id)
    assert len(recs) >= 3
    third_name = recs[2]["name"]
    result = run_turn("u", chat_id, "گزینه اول و سوم رو مقایسه کن")
    assert "گزینه" in result["final_response"]
    assert "جمع‌بندی" in result["final_response"]
    assert recs[0]["name"] in result["final_response"] or "گزینه ۱" in result["final_response"]
    assert third_name in result["final_response"] or "گزینه ۳" in result["final_response"]
    # Middle option should not be the focus of a 1-vs-3 compare
    if len(recs) >= 3 and recs[1]["name"] not in (recs[0]["name"], recs[2]["name"]):
        assert recs[1]["name"] not in result["final_response"]


@pytest.mark.integration
def test_compare_without_selection_compares_all_current_recommendations(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    recs = get_memory_service().get_active_recommendations(chat_id)
    result = run_turn("u", chat_id, "میشه با هم مقایسشون کنی")
    assert "قیمت" in result["final_response"]
    for rec in recs[:3]:
        assert rec["name"] in result["final_response"]


@pytest.mark.integration
def test_request_product_images_from_memory(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    result = run_turn("u", chat_id, "عکس گزینه اول رو بفرست")
    assert result.get("send_product_images")
    assert len(result["send_product_images"]) == 1
    assert result["send_product_images"][0]["_position"] == 1
    before = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    run_turn("u", chat_id, "عکس گزینه‌ها رو بفرست")
    after = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    assert before == after


def test_categories_command_lists_catalog_categories():
    text = build_categories_message()
    assert "دسته‌بندی" in text
    assert "لپ" in text or "گوشی" in text or len(text) > 50


def test_help_lists_existing_commands():
    for cmd in ("/start", "/help", "/categories", "/mark_purchased", "/reset"):
        assert cmd in HELP_TEXT


def test_purchase_and_image_helpers():
    assert is_purchase_request("می‌خوام بخرم")
    assert is_purchase_request("ثبت سفارش")
    assert is_product_image_request("عکس گزینه اول رو بفرست")
    assert not is_product_image_request("یه لپ‌تاپ می‌خوام")
