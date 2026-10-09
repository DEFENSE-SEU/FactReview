import asyncio
from unittest.mock import AsyncMock

import pytest

from fact_generation.positioning import paper_search
from fact_generation.positioning.paper_search import PaperReadConfig, PaperSearchAdapter, PaperSearchConfig
from util.arxiv_requests import ArxivRequestGate


def adapter(*, remote=False):
    return PaperSearchAdapter(
        search_cfg=PaperSearchConfig(
            enabled=False,
            provider="arxiv",
            base_url=None,
            api_key=None,
            endpoint="/search",
            timeout_seconds=1,
            health_endpoint="/health",
            health_timeout_seconds=1,
        ),
        read_cfg=PaperReadConfig(
            base_url="https://reader.invalid" if remote else None,
            api_key=None,
            endpoint="/read",
            timeout_seconds=1,
        ),
    )


async def require_own_timeout(awaitable):
    task = asyncio.create_task(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=2.5)
        assert task in done, "The reader did not enforce its own deadline"
        with pytest.raises(TimeoutError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_reader_deadline_spans_metadata_and_pdf_download(monkeypatch):
    reader = adapter()
    events = []

    async def metadata(_):
        await asyncio.sleep(0.6)
        events.append("metadata_returned")
        return {"arxiv_id": "2401.00001", "pdf_url": "https://arxiv.org/pdf/2401.00001"}

    async def download(_):
        try:
            await asyncio.sleep(0.6)
            return b"%PDF invalid fixture"
        finally:
            events.append("download_released")

    monkeypatch.setattr(reader, "_arxiv_fetch_single", metadata)
    monkeypatch.setattr(reader, "_download_pdf", download)
    await require_own_timeout(reader.read_papers(items=[{"id": "2401.00001"}]))
    assert events == ["metadata_returned", "download_released"]


@pytest.mark.asyncio
async def test_reader_deadline_includes_shared_gate_queue(monkeypatch):
    reader = adapter()
    gate = ArxivRequestGate(interval=0)
    monkeypatch.setattr(paper_search, "ARXIV_REQUESTS", gate)
    http = AsyncMock(side_effect=AssertionError("Queued operation must not call HTTP"))
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", http)
    async with gate.slot():
        await require_own_timeout(reader.read_papers(items=[{"id": "2401.00001"}]))
        assert gate._lock.locked(), "A timed-out waiter must not release another task's slot"
    http.assert_not_called()
    async with gate.slot():
        assert gate._lock.locked()


@pytest.mark.asyncio
async def test_remote_reader_total_deadline_preserves_external_cancellation(monkeypatch):
    reader = adapter(remote=True)
    entered, released = asyncio.Event(), asyncio.Event()

    async def remote(items):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    monkeypatch.setattr(reader, "_read_remote", remote)
    task = asyncio.create_task(reader.read_papers(items=[{"id": "external-cancellation"}]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()
    released.clear()
    await require_own_timeout(reader.read_papers(items=[{"id": "own-deadline"}]))
    assert released.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_reader_healthy_response_and_single_delegate_are_unchanged(monkeypatch, remote):
    reader = adapter(remote=remote)
    response = {"success": True, "items": [{"id": "one", "evidence": [{"page": 4, "text": "source"}]}]}
    delegate = AsyncMock(return_value=response)
    unused = AsyncMock(side_effect=AssertionError("No fallback or second request was authorized"))
    monkeypatch.setattr(reader, "_read_remote", delegate if remote else unused)
    monkeypatch.setattr(reader, "_read_arxiv_fallback", unused if remote else delegate)
    items = [{"id": "one"}]
    assert await reader.read_papers(items=items) is response
    delegate.assert_awaited_once_with(items)
    unused.assert_not_awaited()
