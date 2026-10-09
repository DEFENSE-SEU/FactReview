"""Mocked HTTP admission across adapter instances, event loops and cancellation."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise

import httpx
import pytest

from fact_generation.positioning import paper_search
from tests.test_paper_search_adapter import _adapter
from util import arxiv_requests


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Gate tests must not contact external services")

    # conftest blocks sockets while preserving Windows asyncio's wakeup pair.
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)


class Transport:
    def __init__(self, *, failure=None, responses=None):
        self.active = 0
        self.maximum = 0
        self.starts = []
        self.ends = []
        self.options = []
        self.failure = failure
        self.responses = list(responses or [])

    def client(self, **options):
        self.options.append(options)
        transport = self

        class Client:
            async def __aenter__(self):
                transport.active += 1
                transport.maximum = max(transport.maximum, transport.active)
                return self

            async def __aexit__(self, *args):
                transport.ends.append(time.monotonic())
                transport.active -= 1

            async def get(self, url, **kwargs):
                transport.starts.append(time.monotonic())
                await asyncio.sleep(0.005)
                if transport.failure is not None:
                    failure, transport.failure = transport.failure, None
                    raise failure
                status, headers = transport.responses.pop(0) if transport.responses else (200, {})
                return httpx.Response(
                    status,
                    headers=headers,
                    content=b"%PDF-mocked" if "/pdf/" in url else b"<feed/>",
                    request=httpx.Request("GET", url, **kwargs),
                )

        return Client()


@pytest.mark.asyncio
async def test_three_adapter_routes_share_connection_and_completion_spacing(monkeypatch):
    gate = arxiv_requests.ArxivRequestGate(0.02)
    transport = Transport()
    monkeypatch.setattr(paper_search, "ARXIV_REQUESTS", gate)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", transport.client)
    results = await asyncio.gather(
        _adapter()._arxiv_fetch_single("1706.03762"),
        _adapter()._arxiv_query("attention", max_results=1),
        _adapter()._download_pdf("https://arxiv.org/pdf/1706.03762"),
    )
    assert results == [None, [], b"%PDF-mocked"]
    assert transport.maximum == 1 and transport.active == 0
    assert all(start - end >= 0.02 for start, end in zip(transport.starts[1:], transport.ends))
    assert sorted(option["timeout"] for option in transport.options) == [45, 45, 60]
    assert sum(option.get("follow_redirects", False) for option in transport.options) == 1


@pytest.mark.asyncio
async def test_transport_failure_releases_shared_connection_without_retry(monkeypatch):
    transport = Transport(failure=httpx.ReadTimeout("injected timeout"))
    monkeypatch.setattr(paper_search, "ARXIV_REQUESTS", arxiv_requests.ArxivRequestGate(0.01))
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", transport.client)
    with pytest.raises(httpx.ReadTimeout, match="injected timeout"):
        await _adapter()._arxiv_fetch_single("1706.03762")
    assert await _adapter()._arxiv_fetch_single("2001.07685") is None
    assert len(transport.starts) == 2 and transport.maximum == 1 and transport.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["waiting", "spacing", "request"])
async def test_cancelled_task_never_leaves_shared_gate_locked(phase):
    gate = arxiv_requests.ArxivRequestGate(0.02)
    entered = asyncio.Event()

    async def operation():
        async with gate.slot():
            entered.set()
            await asyncio.sleep(10)

    if phase == "waiting":
        assert gate._lock.acquire(blocking=False)
    elif phase == "spacing":
        gate._next_start = time.monotonic() + 10
    task = asyncio.create_task(operation())
    if phase == "request":
        await asyncio.wait_for(entered.wait(), 1)
    else:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if phase == "waiting":
        gate._lock.release()
    assert not gate._lock.locked()
    gate._next_start = 0
    async with asyncio.timeout(1), gate.slot():
        assert gate._lock.locked()
    assert not gate._lock.locked()


def test_same_gate_works_across_threads_and_independent_event_loops():
    gate = arxiv_requests.ArxivRequestGate(0.01)
    barrier = threading.Barrier(3)
    spans = []

    def worker():
        barrier.wait()

        async def operation():
            async with gate.slot():
                start = time.monotonic()
                await asyncio.sleep(0.005)
                spans.append((start, time.monotonic()))

        asyncio.run(operation())

    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(worker) for _ in range(3)]
        for future in futures:
            future.result(timeout=3)
    spans.sort()
    assert len(spans) == 3
    assert all(right[0] - left[1] >= 0.01 for left, right in pairwise(spans))
    assert not gate._lock.locked()


def test_default_interval_keeps_existing_three_second_margin():
    assert arxiv_requests.ARXIV_REQUESTS.interval == 3.2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("route", ["metadata", "search", "pdf"])
async def test_retry_after_is_shared_fail_fast_until_expiry(monkeypatch, status, route):
    now = [100.0]
    monkeypatch.setattr(arxiv_requests, "monotonic", lambda: now[0])
    gate = arxiv_requests.ArxivRequestGate(0.01)
    transport = Transport(responses=[(status, {"Retry-After": "120"})])
    monkeypatch.setattr(paper_search, "ARXIV_REQUESTS", gate)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", transport.client)

    async def request():
        adapter = _adapter()
        if route == "metadata":
            return await adapter._arxiv_fetch_single("1706.03762")
        if route == "search":
            return await adapter._arxiv_query("attention", max_results=1)
        return await adapter._download_pdf("https://arxiv.org/pdf/1706.03762")

    with pytest.raises(httpx.HTTPStatusError) as original:
        await request()
    assert original.value.response.status_code == status
    assert not gate._lock.locked()
    with pytest.raises(arxiv_requests.ArxivCooldownError, match="HTTP request withheld"):
        await request()
    assert len(transport.starts) == 1
    now[0] = 221.0
    await request()
    assert len(transport.starts) == 2 and transport.active == 0


@pytest.mark.asyncio
async def test_http_date_cooldown_does_not_shorten_an_existing_server_deadline(monkeypatch):
    monkeypatch.setattr(arxiv_requests, "monotonic", lambda: 100.0)
    gate = arxiv_requests.ArxivRequestGate()
    async with gate.slot():
        deadline = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
        gate.observe_retry_after(429, deadline)
        assert 218 < gate._cooldown_until <= 220
        preserved = gate._cooldown_until
        gate.observe_retry_after(503, "10")
        assert gate._cooldown_until == preserved
    with pytest.raises(arxiv_requests.ArxivCooldownError):
        async with gate.slot():
            pytest.fail("Server cooling period was ignored")


@pytest.mark.parametrize(
    "status,header", [(200, "120"), (429, None), (503, "nonsense"), (429, "-1"), (429, "1.5")]
)
def test_non_rate_status_and_invalid_retry_after_keep_default_policy(monkeypatch, status, header):
    monkeypatch.setattr(arxiv_requests, "monotonic", lambda: 100.0)
    gate = arxiv_requests.ArxivRequestGate()
    gate.observe_retry_after(status, header)
    assert gate._cooldown_until == 0
    assert gate.interval == 3.2
