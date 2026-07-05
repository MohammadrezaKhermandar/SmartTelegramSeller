"""Similar-product lookup from an external product URL.

Pipeline:
1. Fetch the page (with retry + timeout).
2. Extract title / meta description / og:title.
3. If fetching fails, fall back to keywords found in the URL slug itself.
4. Feed the extracted text into the hybrid recommender to find in-shop
   alternatives.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from app.utils.logger import get_logger
from app.utils.retry import retry
from app.utils.text_normalizer import enrich_english_keywords

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
}


@retry(max_attempts=2, base_delay=1.0, exceptions=(httpx.HTTPError,))
def _fetch(url: str) -> str:
    with httpx.Client(timeout=10, follow_redirects=True, headers=_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _slug_keywords(url: str) -> str:
    """Extract human-readable words from the URL path as a fallback query."""
    path = unquote(urlparse(url).path)
    words = re.split(r"[/\-_+.%]+", path)
    keep = [w for w in words if len(w) > 2 and not w.isdigit()]
    return " ".join(keep)


def extract_product_text(url: str) -> dict[str, Any]:
    """Return {'query': str, 'source': 'page'|'url', 'title': str|None}."""
    try:
        html = _fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        title: Optional[str] = None
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = str(og["content"]).strip()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = str(meta_desc["content"]).strip()

        h1 = soup.find("h1")
        h1_text = h1.get_text(strip=True) if h1 else ""

        query = " ".join(p for p in (title or "", h1_text, description) if p)[:500]
        if query.strip():
            logger.info("Link extraction OK for %s: %.80s", url, query)
            return {"query": enrich_english_keywords(query), "source": "page", "title": title}
    except Exception as exc:
        logger.warning("Fetching %s failed (%s) — falling back to URL slug", url, exc)

    slug = _slug_keywords(url)
    return {"query": enrich_english_keywords(slug) or url, "source": "url", "title": None}
