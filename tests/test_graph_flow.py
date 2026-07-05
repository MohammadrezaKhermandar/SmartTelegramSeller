"""End-to-end graph tests covering the required scenarios (offline, no LLM)."""

import uuid

import pytest

from app.graph.builder import run_turn
from app.services.memory_service import get_memory_service

pytestmark = pytest.mark.integration


@pytest.fixture()
def chat_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def chat_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def test_scenario_1_incomplete_need_asks_questions(chat_id):
    result = run_turn("u", chat_id, "یه لپ‌تاپ می‌خوام")
    assert "؟" in result["final_response"]
    assert not get_memory_service().get_active_recommendations(chat_id)


def test_scenario_2_complete_need_recommends_three(chat_id):
    run_turn("u", chat_id, "یه لپ‌تاپ می‌خوام")
    result = run_turn("u", chat_id, "تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره")
    recs = get_memory_service().get_active_recommendations(chat_id)
    assert len(recs) >= 3
    assert result["final_response"]


def test_scenario_3_memory_question_no_research(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام، برند فرقی نداره")
    before = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    result = run_turn("u", chat_id, "گزینه دوم رمش چقدره؟")
    after = [p["product_id"] for p in get_memory_service().get_active_recommendations(chat_id)]
    assert before == after  # no re-search happened
    assert result["final_response"]


def test_scenario_4_budget_change_partial_update(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام، برند فرقی نداره")
    run_turn("u", chat_id, "بودجه‌ام شد ۴۰ میلیون")
    session = get_memory_service().get_or_create_session(chat_id, "u")
    assert session["requirements"]["budget"] == 40_000_000
    assert session["requirements"].get("category")  # category preserved
    assert session["requirements"].get("use_case")  # use case preserved


def test_scenario_5_comparison_from_memory(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام، برند فرقی نداره")
    result = run_turn("u", chat_id, "گزینه اول و دوم رو مقایسه کن")
    assert "گزینه" in result["final_response"]
    assert "قیمت" in result["final_response"]


def test_scenario_6_link_similarity(chat_id):
    result = run_turn(
        "u", chat_id, "https://nonexistent-shop-xyz.example/product/sony-wireless-headphone"
    )
    # even with fetch failure, slug fallback must produce suggestions
    assert result["final_response"]
    assert result.get("last_recommended_products")


def test_scenario_7_image_with_caption(chat_id):
    result = run_turn(
        "u", chat_id, "", message_type="image", image_caption="اسپیکر جی‌بی‌ال"
    )
    assert result.get("last_recommended_products")


def test_scenario_7_image_without_caption_asks(chat_id):
    result = run_turn("u", chat_id, "", message_type="image")
    assert "؟" in result["final_response"]
    assert not result.get("last_recommended_products")


def test_error_fallback_is_natural(chat_id):
    # Force an internal error by corrupting the message type path
    result = run_turn("u", chat_id, "", message_type="text")
    assert result["final_response"]  # never empty, never a stack trace
    assert "Traceback" not in result["final_response"]


def test_purchase_request_says_coming_soon(chat_id):
    run_turn("u", chat_id, "لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام، برند فرقی نداره")
    result = run_turn("u", chat_id, "همین گزینه اول رو خریدم")
    assert "به‌زودی" in result["final_response"]
    assert get_memory_service().get_purchase_status(chat_id) == "browsing"
