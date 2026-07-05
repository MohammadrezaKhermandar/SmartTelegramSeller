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


def test_filter_missing_column_graceful():
    df, _ = load_products(CSV_PATH)
    df2 = df.drop(columns=["brand"], errors="ignore")
    repo = ProductRepository(df2)
    init_pandas_tools(repo)
    results = filter_by_brand_tool.invoke({"brand": "لنوو"})
    assert isinstance(results, list)
