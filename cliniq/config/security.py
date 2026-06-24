"""
ClinIQ — Security hardening utilities
Rate limiting for AI endpoint, input sanitisation, security headers.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, status


# ---------------------------------------------------------------------------
# Simple in-process rate limiter (replace with Redis for multi-instance)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token bucket rate limiter.
    Tracks requests per (client_ip, endpoint) pair.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        """Raise 429 if the key has exceeded the rate limit."""
        now = time.monotonic()
        window_start = now - self.window_seconds
        bucket = self._buckets[key]

        # Prune old entries
        self._buckets[key] = [t for t in bucket if t > window_start]

        if len(self._buckets[key]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.max_requests} requests "
                       f"per {self.window_seconds}s",
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._buckets[key].append(now)

    def reset(self, key: str) -> None:
        """Reset the bucket for a key (useful in tests)."""
        self._buckets.pop(key, None)

    def request_count(self, key: str) -> int:
        """Return current request count in the window for a key."""
        now = time.monotonic()
        window_start = now - self.window_seconds
        return sum(1 for t in self._buckets.get(key, []) if t > window_start)


# Module-level limiters
ai_limiter   = RateLimiter(max_requests=20, window_seconds=60)   # AI endpoint
api_limiter  = RateLimiter(max_requests=200, window_seconds=60)  # General API


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SQL_PATTERNS  = re.compile(
    r"(--|;|/\*|\*/|xp_|UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO"
    r"|DELETE\s+FROM|UPDATE\s+SET)",
    re.IGNORECASE,
)


def sanitise_string(value: str, max_length: int = 1000) -> str:
    """
    Strip control characters, truncate, and flag obvious SQL injection.
    Raises ValueError on obvious injection attempts.
    """
    if not isinstance(value, str):
        return str(value)[:max_length]

    # Strip control chars
    cleaned = _CONTROL_CHARS.sub("", value).strip()

    # Truncate
    cleaned = cleaned[:max_length]

    # Reject obvious SQL injection
    if _SQL_PATTERNS.search(cleaned):
        raise ValueError(f"Input contains disallowed pattern: {cleaned[:50]}")

    return cleaned


def sanitise_int(value, min_val: int = 1, max_val: int = 10_000) -> int:
    """Coerce to int and clamp to [min_val, max_val]."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected integer, got: {value!r}")
    if not (min_val <= v <= max_val):
        raise ValueError(f"Value {v} out of range [{min_val}, {max_val}]")
    return v


# ---------------------------------------------------------------------------
# Security response headers middleware helper
# ---------------------------------------------------------------------------

SECURITY_HEADERS = {
    "X-Content-Type-Options":    "nosniff",
    "X-Frame-Options":           "DENY",
    "X-XSS-Protection":          "1; mode=block",
    "Referrer-Policy":           "strict-origin-when-cross-origin",
    "Permissions-Policy":        "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}
