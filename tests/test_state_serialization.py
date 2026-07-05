"""Tests that LangGraph state payloads are msgpack/JSON serializable."""

import json
from pathlib import Path

import ormsgpack
import pytest

from app.graph import nlp
from app.main import initialize_app
from app.tools.pandas_tools import hybrid_recommend
from app.tools.rag_tools import semantic_search
from app.utils.json_safe import to_json_safe

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_app():
    initialize_app(CSV_PATH)


def test_product_tool_results_are_msgpack_serializable():
    rag_results = semantic_search("لپ‌تاپ اداری")
    products, _ = hybrid_recommend(
        query="لپ‌تاپ اداری",
        category="لپ‌تاپ",
        max_price=100_000_000,
        rag_results=rag_results,
    )
    requirements = to_json_safe(
        {"category": "لپ‌تاپ", "max_price": 100_000_000, "usage": "اداری"}
    )
    payload = {
        "recommended_products": products,
        "last_search_result": products,
        "selected_product_ids": [str(p["product_id"]) for p in products[:3]],
        "requirements": requirements,
    }
    packed = ormsgpack.packb(payload)
    assert packed
    json.dumps(payload)


def test_merge_requirements_preserves_category_on_budget_only_update():
    existing = {
        "category": "لپ‌تاپ",
        "max_price": 50_000_000,
        "raw_query": "لپ‌تاپ برای اداری با بودجه ۵۰ میلیون",
        "usage": "اداری",
    }
    new = nlp.extract_requirements("بودجه شد ۳۰ میلیون")
    merged = nlp.merge_requirements(existing, new)

    assert merged["category"] == "لپ‌تاپ"
    assert merged["max_price"] == 30_000_000
    assert merged["raw_query"] == existing["raw_query"]
    assert merged["usage"] == "اداری"
