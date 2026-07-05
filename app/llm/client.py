"""LLM client (OpenRouter, xAI, or Groq) for Persian sales responses."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_PROXY,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    USE_LLM,
    XAI_API_KEY,
    XAI_BASE_URL,
    XAI_MODEL,
)
from app.graph.prompts import SALES_SYSTEM_PROMPT
from app.utils.errors import LLMError
from app.utils.logging import logger
from app.utils.retry import with_retry

_llm: Any | None = None
_llm_disabled = False


class LLMPermanentError(LLMError):
    """Auth/credit failure that must not be retried."""


def _disable_llm(reason: str) -> None:
    """Stop using the LLM for this session after a permanent auth/credit error."""
    global _llm, _llm_disabled
    _llm = None
    _llm_disabled = True
    logger.warning("LLM disabled for this session: %s", reason)


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    return "assistant"


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat client (OpenRouter / xAI)."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        import requests

        self._provider = provider
        self._model = model
        self._requests = requests
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._proxies = {"http": LLM_PROXY, "https": LLM_PROXY} if LLM_PROXY else None
        self._extra_headers = extra_headers or {}

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        payload = {
            "model": self._model,
            "temperature": 0.4,
            "max_tokens": 1500,
            "messages": [
                {"role": _message_role(m), "content": m.content}
                for m in messages
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        response = self._requests.post(
            self._url,
            json=payload,
            headers=headers,
            proxies=self._proxies,
            timeout=90,
        )
        if response.status_code in (401, 403, 402):
            detail = ""
            try:
                body = response.json()
                detail = body.get("error", body)
                if isinstance(detail, dict):
                    detail = detail.get("message", str(detail))
            except Exception:
                detail = response.text[:200]
            _disable_llm(f"{self._provider} {response.status_code}: {detail}")
            raise LLMPermanentError(str(detail))
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return AIMessage(content=content or "")


def get_llm() -> Any | None:
    """Return the configured chat model, or None if unavailable."""
    global _llm
    if not USE_LLM or _llm_disabled:
        return None
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        try:
            _llm = OpenAICompatibleChatClient(
                provider="OpenRouter",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_MODEL,
                base_url=OPENROUTER_BASE_URL,
                extra_headers={
                    "HTTP-Referer": "https://github.com/MohammadrezaKhermandar/SmartTelegramSeller",
                    "X-Title": OPENROUTER_APP_NAME,
                },
            )
            logger.info("OpenRouter LLM initialized: %s", OPENROUTER_MODEL)
            return _llm
        except Exception as exc:
            logger.warning("OpenRouter init failed: %s", exc)
            return None

    if LLM_PROVIDER == "xai" and XAI_API_KEY:
        try:
            _llm = OpenAICompatibleChatClient(
                provider="xAI",
                api_key=XAI_API_KEY,
                model=XAI_MODEL,
                base_url=XAI_BASE_URL,
            )
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
        "- سلام یا خوش‌آمد نگو؛ مکالمه در جریان است\n"
        "- سوال جدیدی که در پیش‌نویس نیست اضافه نکن\n"
        "- ساختار و ترتیب موارد پیش‌نویس (مثل لیست محصولات) را حفظ کن\n"
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
        logger.warning("LLM polish failed (using draft): %s", exc)
        return draft

    return draft


def reset_llm() -> None:
    global _llm, _llm_disabled
    _llm = None
    _llm_disabled = False
