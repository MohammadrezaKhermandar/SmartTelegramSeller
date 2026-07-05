"""LLM access layer.

- Provider-agnostic (OpenRouter / xAI / Groq via OpenAI-compatible APIs).
- JSON-mode helper for structured extraction.
- Graceful degradation: if USE_LLM=false or the key is missing, callers
  receive None and fall back to rule-based behavior.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

ABSOLUTE_MAX_TOKENS = 1200
JSON_MAX_TOKENS = 500
CHAT_MAX_TOKENS = 800

_llm: Optional[ChatOpenAI] = None


def clamp_max_tokens(requested: int) -> int:
    """Cap output tokens to project limits and env default."""
    return min(requested, settings.llm_max_tokens, ABSOLUTE_MAX_TOKENS)


def get_llm(temperature: float = 0.4) -> Optional[ChatOpenAI]:
    """Return a cached ChatOpenAI client, or None when LLM usage is disabled."""
    global _llm
    if not settings.use_llm or not settings.llm_api_key:
        return None
    if _llm is None:
        headers = {}
        if settings.llm_provider == "openrouter":
            headers = {"X-Title": settings.openrouter_app_name}
        _llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=temperature,
            timeout=45,
            max_retries=0,
            max_tokens=clamp_max_tokens(settings.llm_max_tokens),
            default_headers=headers,
        )
        logger.info("LLM ready: provider=%s model=%s", settings.llm_provider, settings.llm_model)
    return _llm


def _is_non_retryable_llm_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (401, 402, 403, 429):
        return True
    msg = str(exc).lower()
    return any(token in msg for token in ("error code: 401", "error code: 402", "error code: 403", "error code: 429", "insufficient", "credits", "afford"))


def _invoke(system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:
    llm = get_llm()
    assert llm is not None
    capped = clamp_max_tokens(max_tokens)
    bound = llm.bind(max_tokens=capped)
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = bound.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return str(result.content)
        except Exception as exc:
            last_exc = exc
            if _is_non_retryable_llm_error(exc):
                logger.warning("LLM non-retryable error: %s", exc)
                raise
            if attempt == 0:
                logger.warning("LLM invoke failed (attempt 1/2): %s — retrying", exc)
                time.sleep(0.8)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def validate_polish(
    draft: str,
    polished: Optional[str],
    product_names: list[str],
    *,
    max_product_count: Optional[int] = None,
    forbidden_names: Optional[list[str]] = None,
) -> str:
    """Return polished text only if every product name from draft is preserved."""
    draft = draft.strip()
    blocked = [n for n in (forbidden_names or []) if n]
    if blocked:
        draft = _strip_forbidden_names(draft, blocked)
    if not polished:
        return draft
    polished = polished.strip()
    if blocked and any(name in polished for name in blocked):
        logger.warning("Polish rejected — forbidden over-budget product kept")
        return draft
    for name in product_names:
        if name and name in draft and name not in polished:
            logger.warning("Polish discarded — missing product name: %s", name)
            return draft
    for name in product_names:
        if name and name not in draft and name in polished:
            logger.warning("Polish discarded — hallucinated product: %s", name)
            return draft
    if max_product_count is not None:
        listed = sum(1 for name in product_names if name and name in polished)
        if listed > max_product_count:
            logger.warning(
                "Polish discarded — too many products (%d > %d)",
                listed, max_product_count,
            )
            return draft
    return polished


def _strip_forbidden_names(text: str, names: list[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return text.strip()
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if any(name in line for name in names):
            if i + 1 < len(lines) and lines[i + 1].startswith("   "):
                i += 2
            else:
                i += 1
            continue
        out.append(line)
        i += 1
    alt_phrases = ("نزدیک‌ترین جایگزین", "کمی بالاتر از بودجه")
    out = [ln for ln in out if not any(p in ln for p in alt_phrases)]
    return "\n".join(out).strip()


def chat(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Plain chat completion. Returns None if LLM is unavailable or call fails."""
    if get_llm() is None:
        return None
    try:
        return _invoke(system_prompt, user_prompt, max_tokens=CHAT_MAX_TOKENS)
    except Exception as exc:
        logger.error("LLM chat failed: %s", exc)
        return None


def chat_json(system_prompt: str, user_prompt: str) -> Optional[dict[str, Any]]:
    """Chat completion that must return a JSON object. Extracts the first
    JSON object from the response defensively."""
    if get_llm() is None:
        return None
    try:
        raw = _invoke(
            system_prompt + "\nفقط و فقط یک شیء JSON معتبر برگردان، بدون هیچ متن اضافه.",
            user_prompt,
            max_tokens=JSON_MAX_TOKENS,
        )
    except Exception as exc:
        logger.error("LLM chat_json failed: %s", exc)
        return None
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("LLM returned no JSON object: %.200s", raw)
            return None
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed: %s | raw=%.200s", exc, raw)
        return None
