"""Tests for RAG tools with TF-IDF fallback."""

from pathlib import Path

import pytest

from app.data.product_loader import load_products
from app.data.vector_store import KeywordVectorStore, reset_vector_store
from app.tools.rag_tools import init_rag_tools, semantic_search

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_rag():
    reset_vector_store()
    df, _ = load_products(CSV_PATH)
    init_rag_tools(df)


def test_keyword_vector_store_search():
    df, _ = load_products(CSV_PATH)
    store = KeywordVectorStore(df)
    results = store.search("لپ‌تاپ گیمینگ", top_k=5)
    assert len(results) > 0
    assert "product_id" in results[0]
    assert "score" in results[0]


def test_semantic_search_fallback():
    results = semantic_search("هدفون بی‌سیم", top_k=5)
    assert len(results) > 0
    assert results[0]["title"]


def test_semantic_search_returns_required_fields():
    results = semantic_search("تلویزیون", top_k=3)
    for r in results:
        assert "product_id" in r
        assert "title" in r
        assert "price" in r
        assert "brand" in r
