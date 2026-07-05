"""CSV product catalog loader.

Responsibilities:
- Load the CSV at runtime with retry.
- Detect columns and map arbitrary column names to a canonical schema
  via COLUMN_ALIASES (configurable mapping).
- Clean broken rows: null names, invalid prices, missing descriptions.
- Build a searchable RAG text per product.
- Expose the file hash so the vector index is only rebuilt when the CSV changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.config import settings
from app.utils.logger import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

# Canonical column -> accepted aliases in the source CSV (case-insensitive).
# Extend this mapping if the shop ships a CSV with different headers.
COLUMN_ALIASES: dict[str, list[str]] = {
    "product_id": ["id", "product_id", "sku", "code"],
    "category": ["category", "cat", "دسته", "دسته_بندی"],
    "brand": ["brand", "make", "برند"],
    "model": ["model", "مدل"],
    "name": ["title", "name", "product_name", "نام", "عنوان"],
    "description": ["description", "desc", "توضیحات"],
    "price": ["price", "قیمت", "amount"],
    "discount": ["discount", "off", "تخفیف"],
    "stock": ["stock", "inventory", "availability", "موجودی"],
    "rating": ["rating", "score", "امتیاز"],
    "review_count": ["review_count", "reviews", "نظرات"],
    "color": ["color", "رنگ"],
    "warranty": ["warranty", "گارانتی"],
    "features": ["features", "specs", "ویژگی‌ها", "ویژگی ها"],
    "tags": ["tags", "برچسب‌ها", "keywords"],
    "image_url": ["image_url", "image", "img", "تصویر"],
    "product_url": ["product_url", "url", "link", "لینک"],
}

REQUIRED_COLUMNS = ("product_id", "name", "price")


class ProductCatalog:
    """In-memory, cleaned view of the product CSV."""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = csv_path or settings.products_csv_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.csv_hash: str = ""
        self._load()

    # ------------------------------------------------------------------ load

    @retry(max_attempts=3, exceptions=(OSError, pd.errors.ParserError))
    def _read_csv(self) -> pd.DataFrame:
        logger.info("Loading products CSV from %s", self.csv_path)
        return pd.read_csv(self.csv_path, encoding="utf-8-sig", on_bad_lines="skip")

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Products CSV not found: {self.csv_path}")
        raw = self._read_csv()
        self.csv_hash = self._compute_hash(self.csv_path)
        self.df = self._normalize_columns(raw)
        self.df = self._clean(self.df)
        self.df["search_text"] = self.df.apply(self.build_search_text, axis=1)
        logger.info("Catalog ready: %d valid products (hash=%s)", len(self.df), self.csv_hash[:10])

    @staticmethod
    def _compute_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------- column mapping

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename: dict[str, str] = {}
        lower_cols = {c.strip().lower(): c for c in df.columns}
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias.lower() in lower_cols:
                    rename[lower_cols[alias.lower()]] = canonical
                    break
        df = df.rename(columns=rename)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns {missing}. "
                f"Detected columns: {list(df.columns)}. "
                "Add aliases to COLUMN_ALIASES in product_loader.py."
            )
        # Ensure every canonical column exists so downstream code never KeyErrors
        for canonical in COLUMN_ALIASES:
            if canonical not in df.columns:
                df[canonical] = None
        return df

    # --------------------------------------------------------------- clean

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.copy()

        df["name"] = df["name"].astype(str).str.strip()
        df = df[df["name"].notna() & (df["name"] != "") & (df["name"].str.lower() != "nan")]

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["price"].notna() & (df["price"] > 0)]

        df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0).clip(0, 100)
        df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).clip(0, 5)
        df["review_count"] = (
            pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
        )

        for col in ("category", "brand", "model", "description", "color",
                    "warranty", "features", "tags", "image_url", "product_url"):
            df[col] = df[col].fillna("").astype(str).str.strip()
            df.loc[df[col].str.lower() == "nan", col] = ""

        df["product_id"] = df["product_id"].astype(str).str.strip()
        df = df[df["product_id"] != ""]
        df = df.drop_duplicates(subset="product_id", keep="first")

        # Effective price after discount — used for budget filters
        df["effective_price"] = (df["price"] * (1 - df["discount"] / 100)).round(0)

        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d invalid rows while cleaning CSV", dropped)
        return df.reset_index(drop=True)

    # ---------------------------------------------------------- RAG text

    @staticmethod
    def build_search_text(row: pd.Series) -> str:
        """Combine name, category, brand, description, features, tags and price
        into one searchable document per product."""
        parts = [
            str(row.get("name", "")),
            f"دسته: {row.get('category', '')}",
            f"برند: {row.get('brand', '')}",
            f"مدل: {row.get('model', '')}",
            str(row.get("description", "")),
            f"ویژگی‌ها: {row.get('features', '')}",
            f"برچسب‌ها: {row.get('tags', '')}",
            f"رنگ: {row.get('color', '')}",
            f"قیمت: {int(row.get('price', 0))} تومان",
        ]
        return "\n".join(p for p in parts if p and not p.endswith(": "))

    # ------------------------------------------------------------- helpers

    def get_by_id(self, product_id: str) -> Optional[dict[str, Any]]:
        rows = self.df[self.df["product_id"] == str(product_id)]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def to_records(self) -> list[dict[str, Any]]:
        return self.df.to_dict(orient="records")


_catalog: Optional[ProductCatalog] = None


def get_catalog() -> ProductCatalog:
    """Singleton accessor so the CSV is parsed once per process."""
    global _catalog
    if _catalog is None:
        _catalog = ProductCatalog()
    return _catalog
