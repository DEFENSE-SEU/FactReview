"""One in-process arXiv connection shared across adapter instances and loops.

Deployments with several processes or machines still need a shared external
scheduler. This gate does not coordinate independent third-party clients.
"""

import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Lock
from time import monotonic

import anyio


class ArxivCooldownError(RuntimeError):
    """The server's cooling period prevents a new HTTP request."""


class ArxivRequestGate:
    def __init__(self, interval: float = 3.2):
        self.interval = interval
        self._lock = Lock()
        self._next_start = 0.0
        self._cooldown_until = 0.0

    def observe_retry_after(self, status: int, header: str | None):
        """Called while holding the request slot; never retry the failed request."""
        if status not in (429, 503) or not header:
            return
        value = header.strip()
        try:
            if re.fullmatch(r"[0-9]+", value):
                seconds = float(value)
            else:
                deadline = parsedate_to_datetime(value)
                if deadline.tzinfo is None:
                    return
                seconds = (deadline - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return
        self._cooldown_until = max(self._cooldown_until, monotonic() + max(0, seconds))

    def _check_cooldown(self):
        remaining = self._cooldown_until - monotonic()
        if remaining > 0:
            raise ArxivCooldownError(
                f"arXiv requested a cooling period; HTTP request withheld ({remaining:.0f}s remaining)"
            )

    @asynccontextmanager
    async def slot(self):
        # A nonblocking thread lock works across independent async event loops.
        # Waiting remains cancellable; no abandoned worker can acquire the lock.
        self._check_cooldown()
        while not self._lock.acquire(blocking=False):
            await anyio.sleep(0.05)
            self._check_cooldown()
        admitted = False
        try:
            self._check_cooldown()
            delay = self._next_start - monotonic()
            if delay > 0:
                await anyio.sleep(delay)
            admitted = True
            yield
        finally:
            if admitted:
                # Spacing after completion is conservative and also covers slow
                # connection setup, failures and cancelled requests.
                self._next_start = monotonic() + self.interval
            self._lock.release()


ARXIV_REQUESTS = ArxivRequestGate()
