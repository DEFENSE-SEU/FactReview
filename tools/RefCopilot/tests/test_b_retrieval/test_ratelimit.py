"""Test B.2 — rate limiters."""

from __future__ import annotations

import time

import pytest

from refcopilot.ratelimit.arxiv import ArxivRateLimiter
from refcopilot.ratelimit.semantic_scholar import (
    SemanticScholarRateLimiter,
    parse_retry_after,
)


def test_arxiv_serializes_with_min_interval():
    limiter = ArxivRateLimiter(min_interval_seconds=0.05)
    times: list[float] = []
    for _ in range(3):
        limiter.acquire()
        times.append(time.monotonic())
    diffs = [b - a for a, b in zip(times, times[1:])]
    for d in diffs:
        # Allow a small slack for timer noise.
        assert d >= 0.045, f"interval {d:.4f}s too short"


def test_s2_backoff_for_attempt():
    limiter = SemanticScholarRateLimiter(base_interval_seconds=1.0, backoff_factor=2.0, jitter=0.0)
    # Without retry-after, exponential backoff
    assert limiter.backoff_for_attempt(0, retry_after_seconds=None) == pytest.approx(1.0)
    assert limiter.backoff_for_attempt(1, retry_after_seconds=None) == pytest.approx(2.0)
    assert limiter.backoff_for_attempt(2, retry_after_seconds=None) == pytest.approx(4.0)


def test_s2_honors_retry_after():
    limiter = SemanticScholarRateLimiter(base_interval_seconds=1.0, jitter=0.0)
    # Retry-After overrides backoff
    assert limiter.backoff_for_attempt(0, retry_after_seconds=7.5) == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("3", 3.0),
        ("0.5", 0.5),
        (None, None),
        ("not-a-number", None),
        (" 5 ", 5.0),
    ],
)
def test_parse_retry_after(header, expected):
    assert parse_retry_after(header) == expected
