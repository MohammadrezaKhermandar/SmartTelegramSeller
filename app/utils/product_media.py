"""Product image URL helpers for Telegram media."""

from __future__ import annotations

from typing import Any


def product_image_url(product: dict[str, Any]) -> str | None:
    """
    Return a usable image URL for Telegram send_photo.
    Uses CSV image_url when present; otherwise a stable placeholder per product.
    """
    raw = product.get("image_url") or product.get("image") or product.get("thumbnail")
    if raw and str(raw).strip().lower().startswith(("http://", "https://")):
        return str(raw).strip()

    pid = product.get("product_id")
    if pid is None:
        return None

    # Stable placeholder for demo when CSV has no image column
    return f"https://picsum.photos/seed/sales-product-{pid}/480/360"
