"""Only comparison scheduling changes; retrieval and scientific responses stay fixed."""

import asyncio
import copy
import threading

import pytest

from common import run_stats
from llm.client import LLMConfig
from tests.test_literature_service_responsibility_v2 import boundaries, comparison, inputs
from verification.literature import verify_literature


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No real model, retrieval or process is allowed")

    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("verification.literature.llm_json", forbidden)
    monkeypatch.setattr("verification.literature._default_adapter", forbidden)
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )


async def verify(tmp_path, claim, material, callback):
    searcher, reader, _ = boundaries()
    return await verify_literature(
        claim,
        material,
        submission_deadline="2021-01-01",
        searcher=searcher,
        reader=reader,
        call=callback,
        output_dir=tmp_path / "audit",
    )


@pytest.mark.asyncio
async def test_blocked_sync_comparison_keeps_loop_timer_and_reader_progressing(tmp_path):
    claim, material = inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    progressed_before_return = []
    response = {"status": "ok", "comparisons": [comparison()]}

    def blocked(**kwargs):
        entered.set()
        progressed_before_return.append(release.wait(0.5))
        return copy.deepcopy(response)

    async def other_reader_and_timer():
        while not entered.is_set():
            await asyncio.sleep(0)
        # This represents a pending same-loop reader deadline/completion callback.
        await asyncio.sleep(0.01)
        release.set()

    result, _ = await asyncio.gather(verify(tmp_path, claim, material, blocked), other_reader_and_timer())
    assert progressed_before_return == [True]
    assert len(result.evidence) == 1 and result.evidence[0].sufficient


@pytest.mark.asyncio
async def test_sync_async_and_returned_awaitable_keep_response_and_stats_context(tmp_path):
    claim, material = inputs(tmp_path)
    original = claim.model_dump(mode="json"), material.model_dump(mode="json")
    response = {"status": "ok", "comparisons": [comparison()]}
    frozen_response = copy.deepcopy(response)
    requests, results, contexts = [], [], []
    stats_file = tmp_path / "stats.json"

    def produce(kwargs):
        requests.append(kwargs)
        contexts.append((run_stats.stats_path(), run_stats.current_module()))
        run_stats.record_llm_call(provider="mock", model="fixture", usage={"total_tokens": 2})
        return response

    def sync(**kwargs):
        return produce(kwargs)

    async def async_call(**kwargs):
        await asyncio.sleep(0)
        return produce(kwargs)

    def returned_awaitable(**kwargs):
        return async_call(**kwargs)

    with run_stats.run_scope(stats_file), run_stats.module_scope("analysis"):
        for callback in (sync, async_call, returned_awaitable):
            results.append((await verify(tmp_path, claim, material, callback)).model_dump(mode="json"))
        stats = run_stats.read()["modules"]["analysis"]
    assert results[0] == results[1] == results[2]
    assert requests[0] == requests[1] == requests[2]
    assert all(path == stats_file.resolve() and module == "analysis" for path, module in contexts)
    assert stats["token_usage"]["requests"] == 3 and stats["token_usage"]["total_tokens"] == 6
    assert stats["failed_requests"] == stats["unavailable_usage_requests"] == 0
    assert response == frozen_response
    assert original == (claim.model_dump(mode="json"), material.model_dump(mode="json"))
