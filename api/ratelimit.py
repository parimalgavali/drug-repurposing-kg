"""In-process sliding-window limits for paid endpoints (one API worker; see docs/API.md)."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allows `limit` hits per `window` seconds per key; `limit <= 0` disables it."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def hit(self, key: str = "") -> float:
        """Record a hit and return 0, or return the seconds to wait if the key is over the limit."""
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return q[0] + self.window - now
            q.append(now)
            if len(self.hits) > 10_000:
                for k in [k for k, v in self.hits.items() if not v or v[-1] <= now - self.window]:
                    del self.hits[k]
            return 0.0


def retry_after(seconds: float) -> str:
    return str(max(1, math.ceil(seconds)))
