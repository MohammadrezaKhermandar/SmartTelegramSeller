"""Structured (exact) product queries over the catalog with Pandas.

Supports price / brand / category / stock / rating filters, sorting, and a
relaxation strategy that suggests close alternatives when the exact query
returns nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

from app.services.product_loader import ProductCatalog, get_catalog
from app.utils.logger import get_logger
from app.utils.text_normalizer import brands_match, normalize

logger = get_logger(__name__)

SortKey = Literal["price_asc", "price_desc", "rating", "reviews", "discount"]


@dataclass
class ProductFilter:
    """Declarative filter passed from requirement extraction to Pandas."""

    max_price: Optional[float] = None
    min_price: Optional[float] = None
    brands: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    in_stock_only: bool = True
    min_rating: Optional[float] = None
    sort_by: SortKey = "rating"
    limit: int = 20
    strict_budget: bool = False  # when True, never relax max_price


class PandasQueryService:
    """Exact filtering engine over the cleaned catalog DataFrame."""

    def __init__(self, catalog: Optional[ProductCatalog] = None) -> None:
        self.catalog = catalog or get_catalog()

    # -------------------------------------------------------------- filters

    def _apply(self, f: ProductFilter, relax_stock: bool = False) -> pd.DataFrame:
        df = self.catalog.df.copy()

        if f.in_stock_only and not relax_stock:
            df = df[df["stock"] > 0]
        if f.max_price is not None:
            df = df[df["effective_price"] <= f.max_price]
        if f.min_price is not None:
            df = df[df["effective_price"] >= f.min_price]
        if f.min_rating is not None:
            df = df[df["rating"] >= f.min_rating]
        if f.brands and not df.empty:
            df = df[
                df["brand"].apply(
                    lambda b: any(brands_match(pref, str(b)) for pref in f.brands)
                )
            ]
        if f.categories and not df.empty:
            patterns = [normalize(c) for c in f.categories]
            df = df[
                df["category"].apply(
                    lambda c: any(
                        pat in normalize(str(c)) or normalize(str(c)) in pat
                        for pat in patterns
                    )
                )
            ]
        if f.keywords and not df.empty:
            pattern = "|".join(normalize(k) for k in f.keywords)
            df = df[df["search_text"].apply(normalize).str.contains(pattern, case=False, na=False)]
        return df

    @staticmethod
    def _sort(df: pd.DataFrame, key: SortKey) -> pd.DataFrame:
        mapping: dict[str, tuple[str, bool]] = {
            "price_asc": ("effective_price", True),
            "price_desc": ("effective_price", False),
            "rating": ("rating", False),
            "reviews": ("review_count", False),
            "discount": ("discount", False),
        }
        col, asc = mapping.get(key, ("rating", False))
        return df.sort_values(col, ascending=asc)

    # ---------------------------------------------------------------- query

    def query(self, f: ProductFilter) -> dict[str, Any]:
        """Run the filter. If nothing matches, progressively relax constraints
        within the same category before giving up.
        """
        df = self._apply(f)
        relaxed: list[str] = []
        has_category = bool(f.categories)

        if df.empty and f.brands:
            f2 = ProductFilter(**{**f.__dict__, "brands": []})
            df = self._apply(f2)
            if not df.empty:
                relaxed.append("brand")

        if df.empty and f.max_price is not None and not f.strict_budget:
            f3 = ProductFilter(**{**f.__dict__, "brands": [], "max_price": f.max_price * 1.25})
            df = self._apply(f3)
            if not df.empty:
                relaxed.append("budget+25%")

        if df.empty and f.in_stock_only:
            df = self._apply(f, relax_stock=True)
            if not df.empty:
                relaxed.append("stock")

        if df.empty and f.categories and f.max_price is not None and not f.strict_budget:
            f_cat = ProductFilter(**{**f.__dict__, "max_price": None, "brands": []})
            df = self._apply(f_cat)
            if not df.empty:
                relaxed.append("budget->category_only")

        if df.empty and f.keywords and not has_category:
            f4 = ProductFilter(keywords=f.keywords, in_stock_only=False)
            df = self._apply(f4)
            if not df.empty:
                relaxed.append("keywords")

        if df.empty:
            return {
                "products": [],
                "relaxed": relaxed,
                "exact_match": False,
            }

        df = self._sort(df, f.sort_by).head(f.limit)
        logger.info(
            "Pandas query: %d results (relaxed=%s) filter=%s",
            len(df), relaxed or "none", f,
        )
        return {
            "products": df.to_dict(orient="records"),
            "relaxed": relaxed,
            "exact_match": not relaxed,
        }

    def category_has_products(self, category: str, in_stock_only: bool = True) -> bool:
        f = ProductFilter(categories=[category], in_stock_only=in_stock_only, limit=1)
        return not self._apply(f).empty

    # -------------------------------------------------------------- helpers

    def list_categories(self) -> list[str]:
        return sorted(self.catalog.df["category"].dropna().unique().tolist())

    def list_brands(self) -> list[str]:
        return sorted(self.catalog.df["brand"].dropna().unique().tolist())


_service: Optional[PandasQueryService] = None


def get_pandas_service() -> PandasQueryService:
    global _service
    if _service is None:
        _service = PandasQueryService()
    return _service
