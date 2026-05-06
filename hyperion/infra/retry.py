"""
Retry utilities with exponential backoff for network resilience.
"""
from __future__ import annotations

import time
import random
import functools
import logging
from typing import TypeVar, Callable, Optional

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable)


def retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    jitter: bool = True,
):
    """Decorator: retry function with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay in seconds
        backoff: Multiplicative factor for delay
        exceptions: Tuple of exceptions to catch
        jitter: Add random jitter to delay
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"Retry exhausted: {func.__name__} failed after {max_attempts} attempts")
                        raise
                    sleep_time = current_delay
                    if jitter:
                        sleep_time *= (0.5 + random.random())
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}. "
                        f"Waiting {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                    current_delay *= backoff
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator


def safe_call(func: Callable, default=None, timeout: int = 30, **kwargs):
    """Call function with timeout and default fallback."""
    import threading

    result = [default]
    exception = [None]

    def target():
        try:
            result[0] = func(**kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.warning(f"Timeout ({timeout}s) calling {func.__name__}")
        return default
    if exception[0]:
        raise exception[0]
    return result[0]