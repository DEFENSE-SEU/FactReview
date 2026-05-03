"""arXiv rate limiter — at least 3s between requests (per arXiv's robots policy)."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class ArxivRateLimiter:
    """Serializes requests with a minimum interval. Thread-safe."""

    def __init__(self, *, min_interval_seconds: float = 3.0) -> None:
        self.min_interval = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call_at + self.min_interval - now
            if wait > 0:
                logger.debug("arxiv rate limiter sleeping %.2fs", wait)
                time.sleep(wait)
                now = time.monotonic()
            self._last_call_at = now
