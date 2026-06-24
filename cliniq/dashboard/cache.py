"""
ClinIQ Dashboard — In-memory KPI cache
5-minute TTL per cache key. Thread-safe for Dash's multi-threaded server.
Cache is intentionally simple: key → (value, expiry_timestamp).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

_CACHE: dict[str, tuple[Any, float]] = {}
_LOCK = threading.Lock()
TTL_SECONDS = 300  # 5 minutes


def cache_get(key: str) -> Optional[Any]:
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            del _CACHE[key]
            return None
        return value


def cache_set(key: str, value: Any, ttl: int = TTL_SECONDS) -> None:
    with _LOCK:
        _CACHE[key] = (value, time.monotonic() + ttl)


def cache_invalidate(key: str) -> None:
    with _LOCK:
        _CACHE.pop(key, None)


def cache_clear() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_size() -> int:
    with _LOCK:
        return len(_CACHE)


def cached(key_fn):
    """
    Decorator: cache the result of a function using key_fn(*args, **kwargs) as key.
    Usage:
        @cached(lambda ts_id: f"risk:{ts_id}")
        def get_risk(ts_id): ...
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            result = cache_get(key)
            if result is not None:
                return result
            result = fn(*args, **kwargs)
            cache_set(key, result)
            return result
        return wrapper
    return decorator
