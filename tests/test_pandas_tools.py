"""Tests for Pandas product tools."""

from pathlib import Path

import pytest

from app.data.product_loader import load_products
from app.data.product_repository import ProductRepository
from app.tools.pandas_tools import (
    filter_by_brand_tool,
    filter_by_category_tool,
    filter_by_price_range_tool,
    get_product_by_id_tool,
    hybrid_recommend,
    init_pandas_tools,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_repo():
    df, _ = load_products(CSV_PATH)
    repo = ProductRepository(df)
    init_pandas_tools(repo)
    return repo


def test_filter_by_category():
    results = filter_by_category_tool.invoke({"category": "لپ‌تاپ"})
    assert len(results) > 0
    assert all("لپ‌تاپ" in str(r.get("category", "")) for r in results)


def test_filter_by_price_range():
    results = filter_by_price_range_tool.invoke({"min_price": 1000000, "max_price": 5000000})
    assert len(results) > 0
    for r in results:
        assert 1000000 <= r["price"] <= 5000000


def test_filter_by_brand():
    results = filter_by_brand_tool.invoke({"brand": "لنوو"})
    assert len(results) > 0


def test_get_product_by_id():
    product = get_product_by_id_tool.invoke({"product_id": "1"})
    assert product is not None
    assert product["product_id"] == "1" or str(product.get("product_id")) == "1"


def test_hybrid_without_category_never_searches_full_catalog():
    products, note = hybrid_recommend(
        query="بودجه ۱۰۰ میلیون برای برنامه‌نویسی",
        category=None,
        max_price=100_000_000,
        rag_results=None,
    )
    assert products == []
    assert "دسته" in note


def test_hybrid_never_returns_cross_category():
    from app.tools.rag_tools import semantic_search

    rag = semantic_search("بودجه ۱۰۰ میلیون برای برنامه‌نویسی")
    products, _ = hybrid_recommend(
        query="بودجه ۱۰۰ میلیون برای برنامه‌نویسی",
        category="لپ‌تاپ",
        max_price=100_000_000,
        rag_results=rag,
    )
    assert products
    for product in products:
        assert "لپ" in str(product.get("category", ""))


def test_hybrid_recommend_returns_at_least_three():
    products, note = hybrid_recommend(
        query="لپ‌تاپ برای برنامه‌نویسی",
        category="لپ‌تاپ",
        min_price=None,
        max_price=100_000_000,
        rag_results=None,
        min_count=3,
    )
    assert len(products) >= 3


def _init_repo_with_laptops(count: int):
    """Init pandas tools with a repo containing exactly `count` laptops."""
    df, _ = load_products(CSV_PATH)
    laptops = df[df["category"].astype(str).str.contains("لپ", na=False)]
    stock_col = "availability" if "availability" in laptops.columns else "stock"
    laptops = laptops[laptops[stock_col].astype(float) > 0].head(count)
    assert len(laptops) == count
    init_pandas_tools(ProductRepository(laptops))


def test_note_zero_products_no_match_and_no_recommendations():
    from app.graph.prompts import SEARCH_NO_MATCH_MESSAGE, format_product_recommendation

    products, note = hybrid_recommend(
        query="لپ‌تاپ",
        category="لپ‌تاپ",
        max_price=1,  # impossible budget → zero matches
        rag_results=[],
    )
    assert products == []
    assert note == SEARCH_NO_MATCH_MESSAGE
    response = format_product_recommendation(products, note)
    assert "پیدا نکردم" in response
    assert "1." not in response  # no product list rendered


def test_note_exactly_one_product():
    _init_repo_with_laptops(1)
    products, note = hybrid_recommend(query="لپ‌تاپ", category="لپ‌تاپ", rag_results=[])
    assert len(products) == 1
    assert note == "فقط یک گزینه نزدیک به شرایطت پیدا کردم."
    from app.graph.prompts import format_product_recommendation

    response = format_product_recommendation(products, note)
    assert "پیدا نکردم" not in response
    assert note in response


def test_note_exactly_two_products():
    _init_repo_with_laptops(2)
    products, note = hybrid_recommend(query="لپ‌تاپ", category="لپ‌تاپ", rag_results=[])
    assert len(products) == 2
    assert note == "فقط دو گزینه نزدیک به شرایطت پیدا کردم."
    from app.graph.prompts import format_product_recommendation

    response = format_product_recommendation(products, note)
    assert "پیدا نکردم" not in response


def test_note_three_or_more_products_normal_intro():
    products, note = hybrid_recommend(
        query="لپ‌تاپ برای برنامه‌نویسی",
        category="لپ‌تاپ",
        max_price=100_000_000,
        rag_results=[],
    )
    assert len(products) >= 3
    assert note == ""
    from app.graph.prompts import format_product_recommendation

    response = format_product_recommendation(products, note)
    assert response.startswith("بر اساس نیازت")
    assert "پیدا نکردم" not in response


def test_filter_missing_column_graceful():
    df, _ = load_products(CSV_PATH)
    df2 = df.drop(columns=["brand"], errors="ignore")
    repo = ProductRepository(df2)
    init_pandas_tools(repo)
    results = filter_by_brand_tool.invoke({"brand": "لنوو"})
    assert isinstance(results, list)
