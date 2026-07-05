"""LLM client (Groq or xAI) for Persian sales responses."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_PROXY,
    USE_LLM,
    XAI_API_KEY,
    XAI_BASE_URL,
    XAI_MODEL,
)
from app.graph.prompts import SALES_SYSTEM_PROMPT
from app.utils.errors import LLMError


class LLMPermanentError(LLMError):
    """Auth/credit failure that must not be retried."""
from app.utils.logging import logger
from app.utils.retry import with_retry

_llm: Any | None = None
_llm_disabled = False


def _disable_llm(reason: str) -> None:
    """Stop using the LLM for this session after a permanent auth/credit error."""
    global _llm, _llm_disabled
    _llm = None
    _llm_disabled = True
    logger.warning("LLM disabled for this session: %s", reason)


class XAIChatClient:
    """Minimal OpenAI-compatible chat client for xAI (https://api.x.ai).

    Uses `requests` directly so no extra SDK is needed. API key is read from
    config and never logged.
    """

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self._url = f"{XAI_BASE_URL.rstrip('/')}/chat/completions"
        self._proxies = {"http": LLM_PROXY, "https": LLM_PROXY} if LLM_PROXY else None

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        payload = {
            "model": XAI_MODEL,
            "temperature": 0.4,
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "system" if isinstance(m, SystemMessage) else "user",
                    "content": m.content,
                }
                for m in messages
            ],
        }
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        response = self._requests.post(
            self._url,
            json=payload,
            headers=headers,
            proxies=self._proxies,
            timeout=60,
        )
        if response.status_code in (401, 403):
            # Permanent: bad key or no credits. Don't retry, don't log the key.
            detail = ""
            try:
                detail = response.json().get("error", "")
            except Exception:
                detail = response.text[:200]
            _disable_llm(f"xAI {response.status_code}: {detail}")
            raise LLMPermanentError(detail)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return AIMessage(content=content or "")


def get_llm() -> Any | None:
    """Return the configured chat model (xAI or Groq), or None if unavailable."""
    global _llm
    if not USE_LLM or _llm_disabled:
        return None
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "xai" and XAI_API_KEY:
        try:
            _llm = XAIChatClient()
            logger.info("xAI LLM initialized: %s", XAI_MODEL)
            return _llm
        except Exception as exc:
            logger.warning("xAI init failed: %s", exc)
            return None

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            _llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0.4,
                max_tokens=1500,
            )
            logger.info("Groq LLM initialized: %s", GROQ_MODEL)
            return _llm
        except ImportError as exc:
            logger.warning("langchain-groq not installed: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)
            return None

    return None


@with_retry(max_retries=2, delay=1.0, exceptions=(Exception,))
def polish_response(draft: str, user_message: str = "", stage: str = "") -> str:
    """
    Enhance draft Persian response with the configured LLM.
    Falls back to draft if LLM unavailable or fails.
    """
    llm = get_llm()
    if not llm or not draft.strip():
        return draft

    prompt = (
        "متن پیش‌نویس زیر را به عنوان فروشنده حرفه‌ای فروشگاه بازنویسی کن.\n"
        "قوانین:\n"
        "- فقط فارسی\n"
        "- اطلاعات محصول (نام، قیمت، ویژگی) را حفظ کن و invent نکن\n"
        "- لحن دوستانه و حرفه‌ای\n"
        "- خروجی فقط متن نهایی برای ارسال به مشتری باشد\n"
    )
    if user_message:
        prompt += f"\nپیام کاربر: {user_message}\n"
    if stage:
        prompt += f"\nمرحله مکالمه: {stage}\n"
    prompt += f"\n--- پیش‌نویس ---\n{draft}"

    try:
        result = llm.invoke(
            [
                SystemMessage(content=SALES_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        content = result.content if hasattr(result, "content") else str(result)
        if content and len(content.strip()) > 20:
            return content.strip()
    except LLMPermanentError:
        return draft
    except Exception as exc:
        logger.warning("LLM polish failed: %s", exc)
        raise LLMError(str(exc)) from exc

    return draft


def reset_llm() -> None:
    global _llm, _llm_disabled
    _llm = None
    _llm_disabled = False
