"""Authentication and per-client rate limiting for request handlers.

API keys are checked against SQLite (``app.auth.store``); keyless demo traffic
is limited by an in-memory sliding window per client IP.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import date

from fastapi import HTTPException, Request

from app import config
from app.auth.store import key_daily_usage, key_rate_limit, validate_key

# Keyless-demo sliding window: ip -> deque[timestamps].
_DEMO_HITS: dict[str, deque[float]] = defaultdict(deque)


def reset_demo_state() -> None:
    """Clear demo rate-limit state (tests only)."""
    _DEMO_HITS.clear()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _demo_limit_per_day() -> int:
    try:
        return int(os.environ.get("DEMO_LIMIT_PER_DAY", "20"))
    except ValueError:
        return 20


def check_demo_limit(request: Request) -> None:
    """Sliding window over the last 24h for keyless demo hits; 429 + Retry-After."""
    limit = _demo_limit_per_day()
    ip = client_ip(request)
    now = time.time()
    dq = _DEMO_HITS[ip]
    while dq and dq[0] <= now - config.DEMO_WINDOW_S:
        dq.popleft()
    if len(dq) >= limit:
        retry_after = max(1, int(dq[0] + config.DEMO_WINDOW_S - now) + 1)
        raise HTTPException(
            status_code=429,
            detail="demo rate limit exceeded (keyless); retry later or use an API key",
            headers={"Retry-After": str(retry_after)},
        )
    dq.append(now)


def check_key_limit(key_id: int) -> None:
    limit = key_rate_limit(key_id)
    used = key_daily_usage(key_id, date.today().isoformat())
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail="API key daily rate limit exceeded",
            headers={"Retry-After": "3600"},
        )


def authenticate(request: Request, x_api_key: str | None) -> int | None:
    """Return key_id for valid keys, None for keyless demo (rate-limited).

    Raises 401 for a present-but-invalid/revoked key, 429 when limits hit.
    """
    if x_api_key:
        key_id = validate_key(x_api_key)
        if key_id is None:
            raise HTTPException(status_code=401, detail="invalid or revoked API key")
        check_key_limit(key_id)
        return key_id
    check_demo_limit(request)
    return None
