"""Semantic Scholar rate limiter — backoff with Retry-After honor + jitter.

Semantic Scholar is more lenient than arXiv (esp. with API key) but enforces
429s. We start at 1s/request, exponentially back off on 429, and parse the
`Retry-After` header when present.
"""

from __future__ import annotations

import logging
import random
import threading
import time

logger = logging.getLogger(__name__)


class SemanticScholarRateLimiter:
    def __init__(
        self,
        *,
        base_interval_seconds: float = 1.0,
        backoff_factor: float = 1.5,
        max_retries: int = 3,
        jitter: float = 0.2,
    ) -> None:
        self.base_interval = float(base_interval_seconds)
        self.backoff_factor = float(backoff_factor)
        self.max_retries = int(max_retries)
        self.jitter = float(jitter)
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call_at + self.base_interval - now
            if wait > 0:
                time.sleep(wait + self._jitter())
            self._last_call_at = time.monotonic()

    def backoff_for_attempt(self, attempt: int, retry_after_seconds: float | None = None) -> float:
        """Return the number of seconds to sleep before retry attempt N (0-indexed)."""
        if retry_after_seconds is not None and retry_after_seconds > 0:
            return float(retry_after_seconds) + self._jitter()
        return self.base_interval * (self.backoff_factor**attempt) + self._jitter()

    def _jitter(self) -> float:
        if self.jitter <= 0:
            return 0.0
        return random.uniform(-self.jitter, self.jitter) * self.base_interval


def parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    try:
        return float(header_value.strip())
    except ValueError:
        return None
