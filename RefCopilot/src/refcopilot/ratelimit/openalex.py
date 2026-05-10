"""OpenAlex rate limiter — base interval + exponential backoff on 429.

OpenAlex with an authenticated key publishes a generous quota (10 req/s peak),
so a 0.2s base interval (5 req/s) is comfortably polite while keeping batch
runs fast. ``Retry-After`` is honored when the server sends it.
"""

from __future__ import annotations

import logging
import random
import threading
import time

logger = logging.getLogger(__name__)


class OpenAlexRateLimiter:
    def __init__(
        self,
        *,
        base_interval_seconds: float | None = None,
        min_interval_seconds: float | None = None,  # alias kept for older callers
        backoff_factor: float = 2.0,
        max_retries: int = 3,
        jitter: float = 0.2,
    ) -> None:
        if base_interval_seconds is None and min_interval_seconds is None:
            base_interval_seconds = 0.2
        elif base_interval_seconds is None:
            base_interval_seconds = min_interval_seconds
        self.base_interval = float(base_interval_seconds)
        self.backoff_factor = float(backoff_factor)
        self.max_retries = int(max_retries)
        self.jitter = float(jitter)
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0

    @property
    def min_interval(self) -> float:
        return self.base_interval

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call_at + self.base_interval - now
            if wait > 0:
                logger.debug("openalex rate limiter sleeping %.2fs", wait)
                time.sleep(wait + self._jitter())
            self._last_call_at = time.monotonic()

    def backoff_for_attempt(
        self, attempt: int, retry_after_seconds: float | None = None
    ) -> float:
        if retry_after_seconds is not None and retry_after_seconds > 0:
            return float(retry_after_seconds) + self._jitter()
        return self.base_interval * (self.backoff_factor**attempt) + self._jitter()

    def _jitter(self) -> float:
        if self.jitter <= 0:
            return 0.0
        return random.uniform(-self.jitter, self.jitter) * self.base_interval
