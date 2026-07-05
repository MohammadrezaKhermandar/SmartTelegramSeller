"""Groq LLM client for Persian sales responses."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER, USE_LLM
from app.graph.prompts import SALES_SYSTEM_PROMPT
from app.utils.errors import LLMError
from app.utils.logging import logger
from app.utils.retry import with_retry

_llm: Any | None = None


def get_llm() -> Any | None:
    """Return Groq chat model if configured."""
    global _llm
    if not USE_LLM or LLM_PROVIDER != "groq" or not GROQ_API_KEY:
        return None
    if _llm is None:
        try:
            from langchain_groq import ChatGroq

            _llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0.4,
                max_tokens=1500,
            )
            logger.info("Groq LLM initialized: %s", GROQ_MODEL)
        except ImportError as exc:
            logger.warning("langchain-groq not installed: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)
            return None
    return _llm


@with_retry(max_retries=2, delay=1.0, exceptions=(Exception,))
def polish_response(draft: str, user_message: str = "", stage: str = "") -> str:
    """
    Enhance draft Persian response with Groq.
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
    except Exception as exc:
        logger.warning("Groq polish failed: %s", exc)
        raise LLMError(str(exc)) from exc

    return draft


def reset_llm() -> None:
    global _llm
    _llm = None
