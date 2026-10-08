import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from common import run_stats


def test_parallel_v2_model_calls_preserve_all_recorded_usage(tmp_path, monkeypatch):
    path = tmp_path / "stats.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(path))
    run_stats.initialize(path)

    def record(index):
        run_stats.record_llm_call(
            module="verification.theory" if index % 2 else "screening_claims",
            provider="fixture",
            model="mock",
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(80)))
    usage = run_stats.with_totals(run_stats.read(path))["total"]["token_usage"]
    assert usage["requests"] == 80
    assert usage["input_tokens"] == 160
    assert usage["output_tokens"] == 240
    assert usage["total_tokens"] == 400


def test_scoped_runs_isolate_usage_and_inherit_context_in_async_workers(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(parent))
    monkeypatch.setenv("FACTREVIEW_ACTIVE_STATS_MODULE", "parse")
    run_stats.initialize(parent)
    parent_bytes = parent.read_bytes()
    ready = Barrier(2)

    def run(index):
        path = tmp_path / f"run-{index}.json"
        with run_stats.run_scope(path), run_stats.module_scope("analysis"):
            ready.wait(timeout=10)

            async def record():
                await asyncio.to_thread(
                    run_stats.record_llm_call,
                    provider="fixture",
                    model=f"model-{index}",
                    usage={"input_tokens": index},
                )

            asyncio.run(record())
        return run_stats.read(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (11, 22)))
    for index, result in zip((11, 22), results, strict=True):
        assert result["modules"]["analysis"]["token_usage"]["input_tokens"] == index
        assert result["modules"]["analysis"]["models"] == {f"model-{index}": 1}
    assert parent.read_bytes() == parent_bytes
    assert os.environ["FACTREVIEW_RUN_STATS_PATH"] == str(parent)
    assert os.environ["FACTREVIEW_ACTIVE_STATS_MODULE"] == "parse"


def test_nested_run_scope_restores_path_and_module_after_failure(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(parent))
    monkeypatch.setenv("FACTREVIEW_ACTIVE_STATS_MODULE", "parse")
    outer, inner = tmp_path / "outer.json", tmp_path / "inner.json"
    with run_stats.run_scope(outer), run_stats.module_scope("analysis"):
        with pytest.raises(RuntimeError, match="fixture failure"):
            with run_stats.run_scope(inner), run_stats.module_scope("execution"):
                assert run_stats.stats_path() == inner.resolve()
                raise RuntimeError("fixture failure")
        assert run_stats.stats_path() == outer.resolve()
        assert run_stats.current_module() == "analysis"
    assert run_stats.stats_path() == parent.resolve()
    assert run_stats.current_module() == "parse"


def test_run_scope_restores_context_when_statistics_initialization_fails(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(parent))
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(OSError):
        with run_stats.run_scope(directory):
            pytest.fail("Cannot initialize a statistics file at a directory")
    assert run_stats.stats_path() == parent.resolve()
