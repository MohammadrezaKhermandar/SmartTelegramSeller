"""URL-based product similarity tools."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.tools.rag_tools import find_similar_products
from app.utils.errors import URLFetchError
from app.utils.logging import logger
from app.utils.retry import with_retry


@with_retry(exceptions=(requests.RequestException, URLFetchError))
def fetch_url_metadata(url: str) -> dict[str, str]:
    """Fetch page title and meta description."""
    headers = {"User-Agent": "SalesAssistantBot/1.0"}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = title or og_title["content"].strip()

    return {"title": title, "description": description, "url": url}


def extract_product_text(metadata: dict[str, str]) -> str:
    """Build search query from URL metadata."""
    parts = [metadata.get("title", ""), metadata.get("description", "")]
    text = " ".join(p for p in parts if p)
    # Remove common URL noise
    text = re.sub(r"\s+", " ", text).strip()
    return text or metadata.get("url", "")


def process_url_input(url: str) -> tuple[str, list[dict[str, Any]], str | None]:
    """
    Process external product URL.
    Returns (query, similar_products, error_message).
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "", [], "لینک معتبر نیست. لطفاً آدرس کامل محصول را بفرست."

    try:
        metadata = fetch_url_metadata(url)
        query = extract_product_text(metadata)
        if not query or len(query) < 5:
            return "", [], "نتوانستم اطلاعات محصول را از لینک بخوانم. لطفاً نام یا ویژگی‌های اصلی محصول را بنویس."
        similar = find_similar_products(query, top_k=5)
        logger.info("URL query: %s -> %d results", query[:80], len(similar))
        return query, similar, None
    except Exception as exc:
        logger.warning("URL fetch failed: %s", exc)
        return "", [], "در خواندن لینک مشکلی پیش آمد. لطفاً نام محصول یا ویژگی‌های اصلیش را مستقیم بنویس."
