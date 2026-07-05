"""Tests for Telegram product photo replies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.telegram.media import reply_with_recommendations
from app.utils.product_media import product_image_url


def test_product_image_url_placeholder_when_missing():
    url = product_image_url({"product_id": "42"})
    assert url
    assert url.startswith("https://")
    assert "42" in url


def test_product_image_url_uses_csv_value():
    url = product_image_url({"product_id": "1", "image_url": "https://example.com/p.jpg"})
    assert url == "https://example.com/p.jpg"


@pytest.mark.asyncio
async def test_reply_with_recommendations_sends_photos():
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()

    products = [
        {"product_id": "1", "title": "لپ‌تاپ A", "price": 50_000_000},
        {"product_id": "2", "title": "لپ‌تاپ B", "price": 40_000_000},
    ]

    await reply_with_recommendations(update, "پیشنهادها:", products, max_photos=2)

    update.message.reply_text.assert_called_once()
    assert update.message.reply_photo.await_count == 2
