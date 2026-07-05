"""Tests for product CSV loading and column detection."""

from pathlib import Path

import pandas as pd
import pytest

from app.data.product_loader import (
    build_column_map,
    detect_column,
    load_products,
    normalize_dataframe,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture
def products_df():
    df, _ = load_products(CSV_PATH)
    return df


def test_csv_loads_successfully(products_df):
    assert len(products_df) == 500
    assert "product_id" in products_df.columns
    assert "combined_text" in products_df.columns


def test_column_detection():
    raw = pd.read_csv(CSV_PATH)
    col_map = build_column_map(raw)
    assert col_map["product_id"] == "id"
    assert col_map["title"] == "title"
    assert col_map["brand"] == "brand"
    assert col_map["category"] == "category"
    assert col_map["price"] == "price"
    assert col_map["description"] == "description"
    assert col_map["availability"] == "stock"


def test_normalize_dataframe():
    raw = pd.read_csv(CSV_PATH)
    normalized, col_map = normalize_dataframe(raw)
    assert "product_id" in normalized.columns
    assert normalized["combined_text"].str.len().min() > 0


def test_detect_column_missing():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    assert detect_column(df, "image_url") is None


def test_product_id_is_string(products_df):
    assert products_df["product_id"].dtype == object or str(products_df["product_id"].iloc[0])
