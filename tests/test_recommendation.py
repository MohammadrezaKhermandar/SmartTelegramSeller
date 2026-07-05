import pytest

from app.services.recommendation_service import (
    _availability_score,
    _brand_score,
    _budget_score,
    _in_category,
)
from app.utils.text_normalizer import brands_match

def test_budget_score_curve():
    assert _budget_score(50, 100) == 0.85     # much cheaper
    assert _budget_score(90, 100) == 1.0      # inside budget
    assert _budget_score(110, 100) == 0.5     # slightly over
    assert _budget_score(200, 100) == 0.0     # way over
    assert _budget_score(50, None) == 0.6     # unknown budget -> neutral


def test_brand_score():
    assert _brand_score("سونی", ["سونی"]) == 1.0
    assert _brand_score("سونی", ["اپل"]) == 0.2
    assert _brand_score("سونی", []) == 0.6


def test_brands_match_no_substring_trap():
    assert not brands_match("دل", "دلسی")
    assert brands_match("دل", "دل")


def test_in_category_laptop():
    assert _in_category({"category": "لپ‌تاپ"}, "لپ‌تاپ")


def test_availability_score():
    assert _availability_score(0) == 0.0
    assert _availability_score(5) == 0.7
    assert _availability_score(100) == 1.0


@pytest.mark.integration
def test_hybrid_returns_at_least_three():
    from app.services.recommendation_service import get_recommendation_service

    service = get_recommendation_service()
    result = service.recommend(
        "لپ‌تاپ برای برنامه‌نویسی",
        {"budget": 50_000_000, "category": "لپ‌تاپ", "use_case": "برنامه‌نویسی"},
        top_k=3,
    )
    assert len(result["products"]) >= 3
    scores = [p["score"] for p in result["products"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_hybrid_respects_budget_mostly():
    from app.services.recommendation_service import get_recommendation_service

    service = get_recommendation_service()
    budget = 20_000_000
    result = service.recommend("گوشی موبایل", {"budget": budget, "category": "گوشی"}, top_k=3)
    in_budget = [p for p in result["products"] if not p.get("is_alternative")]
    assert all(p["effective_price"] <= budget for p in in_budget)
