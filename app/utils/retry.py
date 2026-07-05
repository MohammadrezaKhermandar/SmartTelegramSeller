"""Generic retry decorator with exponential backoff.

Used around every fragile boundary in the project:
LLM calls, vector DB queries, CSV loading, HTTP fetches and Telegram API calls.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Callable, Iterable, Type, TypeVar

from app.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: Iterable[Type[BaseException]] = (Exception,),
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff + jitter.

    Works for both sync and async functions.
    """
    exc_tuple = tuple(exceptions)

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: BaseException | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exc_tuple as exc:  # noqa: PERF203
                        last_error = exc
                        if attempt == max_attempts:
                            break
                        delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                        delay += random.uniform(0, delay / 2)
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt, max_attempts, exc, delay,
                        )
                        await asyncio.sleep(delay)
                assert last_error is not None
                logger.error("%s exhausted retries: %s", func.__name__, last_error)
                raise last_error

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exc_tuple as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay += random.uniform(0, delay / 2)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            assert last_error is not None
            logger.error("%s exhausted retries: %s", func.__name__, last_error)
            raise last_error

        return sync_wrapper  # type: ignore[return-value]

    return decorator
