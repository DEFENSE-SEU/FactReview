"""A failed host-ledger recovery still delivers the original partial review."""

import json
from pathlib import Path

import pytest

import pipeline_v2
from tests.test_pipeline_v2 import (
    offline_boundaries as offline_boundaries,
)
from tests.test_pipeline_v2 import (
    run_tiny,
)
from tests.test_pipeline_v2 import (
    tiny_inputs as tiny_inputs,
)


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_execution_recovery_failure_preserves_primary_error_and_partial_report(
    tiny_inputs, monkeypatch, recovery_fails,
):
    marker = "PRIVATE_RECOVERY_INPUT_DO_NOT_PUBLISH_612"
    original_verify = pipeline_v2.verify_claims
    retained = {}
    recovery_calls = []

    async def verification(*args, **kwargs):
        result = await original_verify(*args, **kwargs)
        retained.update({claim.id: claim.model_dump(mode="json") for claim in result.claims})
        return result

    def fail_execution(*args, **kwargs):
        raise ValueError("primary execution failure")

    def recovery(output_dir, plans):
        recovery_calls.append((output_dir, [plan.id for plan in plans]))
        if recovery_fails:
            raise RuntimeError(marker)
        return [], ["Original recovery audit: no complete host records."]

    monkeypatch.setattr(pipeline_v2, "verify_claims", verification)
    monkeypatch.setattr(pipeline_v2, "execute_plans", fail_execution)
    monkeypatch.setattr(pipeline_v2, "recover_execution_records", recovery)
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)

    assert len(recovery_calls) == 1 and recovery_calls[0][1]
    assert runner.call_count == 0
    assert summary["stages"]["execution"] == "failed"
    assert all(summary["stages"][stage] == "ok" for stage in ("assessment", "report", "teaser"))
    assert summary["stage_errors"] == {"execution": "ValueError: primary execution failure"}
    review_text = Path(summary["outputs"]["report_json"]).read_text(encoding="utf-8")
    review = json.loads(review_text)
    # This existing model fixture has incomplete claim coverage; retain it in delivery.
    assert summary["claim_coverage"]["status"] == "partial"
    assert review["run_status"] == "partial" and review["incomplete_stages"] == ["screening", "execution"]
    assert review["ledger"] == [] and len(review["claims"]) == len(retained) == 4
    for claim in review["claims"]:
        before = retained[claim["id"]]
        assert claim["evidence"] == before["evidence"]
        assert not any(row["source"] == "execution" for row in claim["evidence"])
        assert claim["conditions"] == before["conditions"]
        assert claim["source_quote"] == before["source_quote"]
        assert claim["verification_limitations"][-1]["kind"] == "stage_failed"
        assert claim["verification_limitations"][-1]["stage"] == "execution"
        assert claim["verification_limitations"][-1]["reason"] == "ValueError: primary execution failure"
    checks = [row for row in review["delivery_checks"] if row["component"] == "execution.recovery"]
    markdown = Path(summary["outputs"]["report_markdown"]).read_text(encoding="utf-8")
    if recovery_fails:
        assert len(checks) == 1 and checks[0]["state"] == "unavailable"
        assert checks[0]["responsibility"] == "system"
        assert "Host execution record recovery unavailable (RuntimeError)" in checks[0]["reason"]
        assert checks[0]["reason"] in summary["issues"]
        assert "execution.recovery" in markdown
    else:
        assert checks == []
        assert "Original recovery audit: no complete host records." in summary["issues"]
    assert marker not in json.dumps(summary) + review_text + markdown
    saved = json.loads((Path(summary["run_dir"]) / "full_pipeline_summary.json").read_text(encoding="utf-8"))
    assert saved["stage_errors"] == summary["stage_errors"]
    assert saved["delivery_checks"] == summary["delivery_checks"]
