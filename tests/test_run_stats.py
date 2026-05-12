from __future__ import annotations

from common import run_stats
from llm.codex_client import _extract_usage


def test_run_stats_records_fixed_modules_and_totals(tmp_path, monkeypatch) -> None:
    stats_file = tmp_path / "run_stats.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(stats_file))
    run_stats.initialize(stats_file)

    run_stats.record_duration("parse", 2.5)
    run_stats.record_module_status("parse", "ok")
    run_stats.record_llm_call(
        module="analysis",
        provider="openai-codex",
        model="gpt-5.5",
        usage={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
        prompt="ignored when usage is exact",
    )
    run_stats.record_llm_call(
        module="report_generation",
        provider="provider-without-usage",
        model="model-x",
        usage={},
        prompt="A" * 40,
        response_text="B" * 12,
    )

    payload = run_stats.with_totals(run_stats.read(stats_file))

    assert tuple(payload["modules"].keys()) == run_stats.MODULE_ORDER
    assert payload["modules"]["analysis"]["token_usage"]["total_tokens"] == 125
    assert payload["modules"]["report_generation"]["estimated"] is True
    assert payload["modules"]["report_generation"]["token_usage"]["estimated_requests"] == 1
    assert payload["total"]["token_usage"]["requests"] == 2
    assert payload["total"]["estimated"] is True
    assert payload["modules"]["parse"]["token_usage"]["total_tokens"] == 0


def test_module_scope_supplies_module_for_thread_fallback_env(tmp_path, monkeypatch) -> None:
    stats_file = tmp_path / "run_stats.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(stats_file))
    run_stats.initialize(stats_file)

    with run_stats.module_scope("reference_check"):
        run_stats.record_llm_call(
            provider="openai-codex",
            model="gpt-5.5",
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )

    payload = run_stats.with_totals(run_stats.read(stats_file))
    assert payload["modules"]["reference_check"]["token_usage"]["total_tokens"] == 10


def test_codex_sse_usage_extraction_handles_response_completed_payload() -> None:
    payload = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
            }
        },
    }

    assert _extract_usage(payload) == {
        "input_tokens": 123,
        "output_tokens": 45,
        "total_tokens": 168,
    }


def test_codex_sse_usage_extraction_handles_direct_usage_payload() -> None:
    payload = {
        "type": "response.in_progress",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
        },
    }

    assert _extract_usage(payload) == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


def test_stats_payload_round_trips_with_totals(tmp_path, monkeypatch) -> None:
    stats_file = tmp_path / "run_stats.json"
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(stats_file))
    run_stats.initialize(stats_file)
    run_stats.record_llm_call(
        module="execution",
        provider="openai-codex",
        model="gpt-5.5",
        usage={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
    )

    payload = run_stats.with_totals(run_stats.read(stats_file))
    stats_file.write_text(__import__("json").dumps(payload), encoding="utf-8")
    reloaded = run_stats.with_totals(run_stats.read(stats_file))

    assert reloaded["total"]["token_usage"]["total_tokens"] == 11
    assert reloaded["modules"]["execution"]["token_usage"]["total_tokens"] == 11
