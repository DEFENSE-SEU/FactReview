from concurrent.futures import ThreadPoolExecutor

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
