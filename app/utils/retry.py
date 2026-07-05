"""Retry decorator for transient failures."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

from app.config import MAX_RETRIES, RETRY_DELAY
from app.utils.logging import logger

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_retries: int = MAX_RETRIES,
    delay: float = RETRY_DELAY,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Retry wrapper – used for CSV download, vector index, LLM, URL fetch."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "%s attempt %d/%d failed: %s",
                        func.__name__,
                        attempt,
                        max_retries,
                        exc,
                    )
                    if attempt < max_retries:
                        time.sleep(delay * attempt)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
