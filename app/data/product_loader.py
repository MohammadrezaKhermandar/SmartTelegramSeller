"""Load and normalize product CSV/Excel with robust column detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.config import CSV_DOWNLOAD_URL, CSV_PATH, PROJECT_ROOT
from app.utils.errors import ProductLoadError
from app.utils.logging import logger
from app.utils.retry import with_retry

LOCAL_PRODUCTS_DIR_NAME = "500-پروداکتس"
LOCAL_PRODUCT_BASENAME = "products_500"
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Column alias patterns for dynamic detection
COLUMN_ALIASES: dict[str, list[str]] = {
    "product_id": ["id", "product_id", "productid", "sku", "item_id"],
    "title": ["title", "name", "product_name", "product_title"],
    "brand": ["brand", "manufacturer", "maker"],
    "category": ["category", "cat", "product_category", "type"],
    "price": ["price", "cost", "amount", "unit_price"],
    "description": ["description", "desc", "details", "summary"],
    "image_url": ["image_url", "image", "img", "picture", "photo_url", "thumbnail"],
    "availability": ["stock", "availability", "in_stock", "qty", "quantity"],
    "rating": ["rating", "stars", "score"],
    "discount": ["discount", "discount_percent", "sale"],
    "model": ["model", "variant"],
    "features": ["features", "specs", "specifications"],
    "tags": ["tags", "keywords"],
    "color": ["color", "colour"],
    "warranty": ["warranty", "guarantee"],
}

_UNAVAILABLE_TEXT = (
    "ناموجود",
    "not available",
    "out of stock",
    "unavailable",
    "false",
    "no",
    "off",
)
_AVAILABLE_TEXT = (
    "موجود",
    "available",
    "in stock",
    "true",
    "yes",
    "on",
)


def parse_availability(value: Any) -> int:
    """
    Normalize availability/stock to a non-negative integer count.
    Unavailable values are checked before available ones (e.g. «ناموجود» vs «موجود»).
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, float) and pd.isna(value):
        return 0

    if isinstance(value, (int, float)):
        return max(0, int(value))

    text = str(value).strip()
    if text == "":
        return 0

    lowered = text.lower()
    try:
        return max(0, int(float(lowered.replace(",", ""))))
    except ValueError:
        pass

    for token in _UNAVAILABLE_TEXT:
        if token in lowered or lowered == token:
            return 0

    for token in _AVAILABLE_TEXT:
        if token in lowered or lowered == token:
            return 1

    return 0


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[\s\-_]+", "_", str(name).strip().lower())


def detect_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Map a canonical column name to an actual CSV column."""
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    normalized = {_normalize_col_name(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize_col_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def build_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Build mapping from canonical names to actual column names."""
    return {canonical: detect_column(df, canonical) for canonical in COLUMN_ALIASES}


def discover_local_product_file(project_root: Path | None = None) -> Path | None:
    """
    Discover local product file under 500-پروداکتs (recursive) or project root.
    Matches basename products_500 with .csv/.xlsx/.xls extensions.
    """
    root = project_root or PROJECT_ROOT
    candidates: list[Path] = []

    products_dir = root / LOCAL_PRODUCTS_DIR_NAME
    if products_dir.is_dir():
        for path in products_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.stem.lower() == LOCAL_PRODUCT_BASENAME.lower():
                candidates.append(path)

    for ext in SUPPORTED_EXTENSIONS:
        root_candidate = root / f"{LOCAL_PRODUCT_BASENAME}{ext}"
        if root_candidate.is_file():
            candidates.append(root_candidate)

    if not candidates:
        return None

    # Prefer exact basename in the local folder, then shortest path.
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    chosen = candidates[0]
    logger.info("Discovered local product file: %s", chosen)
    return chosen


def read_product_file(path: Path) -> pd.DataFrame:
    """Read products from CSV or Excel."""
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
    except ImportError as exc:
        raise ProductLoadError(
            f"Excel support requires openpyxl/xlrd for {path.name}: {exc}"
        ) from exc
    except Exception as exc:
        raise ProductLoadError(f"Failed to read product file {path}: {exc}") from exc

    raise ProductLoadError(f"Unsupported product file extension: {ext}")


@with_retry(exceptions=(requests.RequestException, OSError))
def download_csv(url: str, dest: Path) -> None:
    """Download product CSV from URL."""
    logger.info("Downloading product CSV from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info("CSV saved to %s", dest)


def ensure_product_file_exists(
    path: Path = CSV_PATH,
    url: str = CSV_DOWNLOAD_URL,
    *,
    force_url: bool = False,
) -> Path:
    """
    Resolve product file path.
    force_url=True: download from URL only; raise RuntimeError on failure.
    force_url=False: local discovery, then existing path, then download fallback.
    """
    if force_url:
        try:
            download_csv(url, path)
        except Exception as exc:
            raise RuntimeError(f"Forced URL download failed: {exc}") from exc
        if not path.exists():
            raise RuntimeError("Forced URL download failed: file not created")
        return path

    local = discover_local_product_file(PROJECT_ROOT)
    if local is not None:
        return local

    if path.exists():
        return path

    try:
        download_csv(url, path)
    except Exception as exc:
        raise ProductLoadError(f"Could not download product file: {exc}") from exc
    return path


def ensure_csv_exists(path: Path = CSV_PATH, url: str = CSV_DOWNLOAD_URL) -> Path:
    """Backward-compatible alias for ensure_product_file_exists(force_url=False)."""
    return ensure_product_file_exists(path=path, url=url, force_url=False)


def _build_combined_text(row: pd.Series, col_map: dict[str, str | None]) -> str:
    """Create RAG text from product fields."""
    parts: list[str] = []
    for field in ("title", "brand", "category", "description", "model", "features", "tags"):
        col = col_map.get(field)
        if col and col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " | ".join(parts)


def _normalize_availability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse availability/stock columns into integer counts."""
    result = df.copy()
    for col in ("availability", "stock"):
        if col in result.columns:
            result[col] = result[col].apply(parse_availability)
    return result


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Normalize DataFrame columns to canonical names where possible."""
    col_map = build_column_map(df)
    normalized = df.copy()

    rename: dict[str, str] = {}
    for canonical, actual in col_map.items():
        if actual and actual != canonical:
            rename[actual] = canonical

    normalized = normalized.rename(columns=rename)

    # Ensure product_id exists
    if "product_id" not in normalized.columns:
        if "id" in normalized.columns:
            normalized = normalized.rename(columns={"id": "product_id"})
        else:
            normalized["product_id"] = range(1, len(normalized) + 1)

    normalized["product_id"] = normalized["product_id"].astype(str)
    normalized = _normalize_availability_columns(normalized)

    # Combined text for RAG
    col_map_after = build_column_map(normalized)
    normalized["combined_text"] = normalized.apply(
        lambda row: _build_combined_text(row, col_map_after), axis=1
    )

    return normalized, col_map_after


def load_products(
    path: Path | None = None,
    *,
    force_url: bool = False,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load products from CSV/Excel with normalization."""
    file_path = path or ensure_product_file_exists(force_url=force_url)
    try:
        df = read_product_file(file_path)
    except ProductLoadError:
        raise
    except Exception as exc:
        raise ProductLoadError(f"Failed to read product file: {exc}") from exc

    if df.empty:
        raise ProductLoadError("Product file is empty")

    normalized, col_map = normalize_dataframe(df)
    logger.info(
        "Loaded %d products from %s, columns: %s",
        len(normalized),
        file_path.name,
        list(normalized.columns),
    )
    return normalized, col_map


def product_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a product row to a dictionary for API responses."""
    result: dict[str, Any] = {}
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        if col in ("price", "rating", "discount"):
            try:
                result[col] = float(val)
            except (ValueError, TypeError):
                result[col] = val
        elif col in ("stock", "availability"):
            result[col] = parse_availability(val)
        else:
            result[col] = val
    return result
