"""Convert pandas/numpy values to plain JSON/msgpack-safe Python types."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Recursively convert values to LangGraph-checkpoint-safe Python types."""
    if value is None:
        return None

    if isinstance(value, (str, bool)):
        return value

    if isinstance(value, int) and not isinstance(value, np.integer):
        return value

    if isinstance(value, float) and not isinstance(value, np.floating):
        if value != value:  # NaN
            return None
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Series):
        return to_json_safe(value.to_dict())

    if isinstance(value, pd.DataFrame):
        return [to_json_safe(row.to_dict()) for _, row in value.iterrows()]

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return to_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    return str(value)


def products_to_json_safe(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize a list of product dicts for LangGraph state/checkpointer."""
    return [to_json_safe(product) for product in products]
