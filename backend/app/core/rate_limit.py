"""Per-user rate limiting on /ask (design doc Section 14): both an abuse control and a
real cost control once generation is swapped from local Ollama to a hosted, billed model
at scale. Fixed-window counter, in-memory — a single-process prototype has no need for a
shared limiter backend like Redis.

Deliberately NOT a FastAPI dependency (Depends() dependencies all run before the route
body, unconditionally) — the emergency check in routes_ask.py must run before this is
ever consulted, the same way it runs before the query cache. A rate-limited user must
still get an emergency card, not a 429, for a chest-pain query. So this is called
explicitly, after that check, as a plain function.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, status

from backend.app.config import get_settings


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


_settings = get_settings()
ask_rate_limiter = RateLimiter(max_requests=_settings.rate_limit_per_minute, window_seconds=60)


def enforce_ask_rate_limit(user_id: str) -> None:
    if not ask_rate_limiter.allow(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {ask_rate_limiter.max_requests} requests per minute.",
        )
