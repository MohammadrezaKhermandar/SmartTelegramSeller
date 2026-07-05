"""Pandas-based product repository for structured queries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.data.product_loader import product_to_dict


class ProductRepository:
    """Structured product access via Pandas."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def get_by_id(self, product_id: str | int) -> dict[str, Any] | None:
        pid = str(product_id)
        matches = self.df[self.df["product_id"].astype(str) == pid]
        if matches.empty:
            return None
        return product_to_dict(matches.iloc[0])

    def filter_by_category(self, category: str) -> pd.DataFrame:
        if "category" not in self.df.columns:
            return self.df.iloc[0:0]
        mask = self.df["category"].astype(str).str.contains(category, case=False, na=False)
        return self.df[mask]

    def filter_by_brand(self, brand: str) -> pd.DataFrame:
        if "brand" not in self.df.columns:
            return self.df.iloc[0:0]
        mask = self.df["brand"].astype(str).str.contains(brand, case=False, na=False)
        return self.df[mask]

    def filter_by_price_range(
        self, min_price: float | None = None, max_price: float | None = None
    ) -> pd.DataFrame:
        if "price" not in self.df.columns:
            return self.df
        result = self.df.copy()
        if min_price is not None:
            result = result[result["price"] >= min_price]
        if max_price is not None:
            result = result[result["price"] <= max_price]
        return result

    def filter_by_availability(self, in_stock_only: bool = True) -> pd.DataFrame:
        col = "availability" if "availability" in self.df.columns else "stock"
        if col not in self.df.columns:
            return self.df
        if in_stock_only:
            return self.df[self.df[col].astype(float) > 0]
        return self.df

    def sort_by(
        self, df: pd.DataFrame, field: str = "price", ascending: bool = True
    ) -> pd.DataFrame:
        if field not in df.columns:
            return df
        return df.sort_values(by=field, ascending=ascending)

    def apply_filters(
        self,
        category: str | None = None,
        brand: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = True,
    ) -> pd.DataFrame:
        """Apply multiple hard filters sequentially."""
        result = self.df
        if category:
            result = self.filter_by_category(category)
        if brand:
            result = self.filter_by_brand(brand)
        if min_price is not None or max_price is not None:
            result = self.filter_by_price_range(min_price, max_price)
        if in_stock_only:
            result = self.filter_by_availability(True)
        return result

    def to_product_list(self, df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
        return [product_to_dict(row) for _, row in df.head(limit).iterrows()]

    def get_multiple_by_ids(self, product_ids: list[str | int]) -> list[dict[str, Any]]:
        ids = {str(i) for i in product_ids}
        matches = self.df[self.df["product_id"].astype(str).isin(ids)]
        return [product_to_dict(row) for _, row in matches.iterrows()]

    def search_text(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Simple text search fallback when vector store unavailable."""
        if "combined_text" not in self.df.columns:
            return self.df.iloc[0:0]
        mask = self.df["combined_text"].astype(str).str.contains(
            query, case=False, na=False
        )
        return self.df[mask].head(limit)

    def list_categories(self, limit: int = 30) -> list[tuple[str, int]]:
        """Return (category, count) pairs sorted by popularity."""
        if "category" not in self.df.columns:
            return []
        counts = self.df["category"].value_counts()
        return [(str(cat), int(cnt)) for cat, cnt in counts.head(limit).items()]
