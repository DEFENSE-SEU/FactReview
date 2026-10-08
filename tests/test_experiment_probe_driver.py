"""A live probe must preserve failed calls and validate recovered observations."""

import importlib.util
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidencePointer
from schemas.materials import SharedMaterials
from verification.contracts import BranchResult, RejectedPlan

spec = importlib.util.spec_from_file_location(
    "experiment_probe", Path(__file__).resolve().parents[1] / "scripts/check_v2_experiments.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Process"))
    monkeypatch.setattr(probe, "llm_json", lambda **kw: pytest.fail("Unmocked model"))
    source = tmp_path / "source"
    (source / "materials").mkdir(parents=True)
    (source / "screening").mkdir()
    material = SharedMaterials(
        paper_key="fixture",
        source_pdf="fixture.pdf",
        markdown="original",
        markdown_path="fixture.md",
        content_list_path="content.json",
        provider="mock",
        blocks=[],
    )
    (source / "materials/materials.json").write_text(material.model_dump_json(), encoding="utf-8")
    claim = Claim(
        id="c",
        text="Original claim",
        loc=ClaimLocation(page=1),
        conditions=[Condition(id="c1", description="Original condition")],
        needs=["Experiments"],
    )
    (source / "screening/screening.json").write_text(
        json.dumps({"claims": [claim.model_dump(mode="json")]}), encoding="utf-8"
    )
    return source, tmp_path / "output"


def support(covered=None):
    return BranchResult(
        evidence=[
            Evidence(
                source="paper_internal",
                pointer=EvidencePointer(locator="fixture.pdf", page=1, quote="Original"),
                covered=covered or ["c1"],
                direction="support",
                sufficient=True,
            )
        ]
    )


def test_rejected_plan_keeps_valid_observations_and_independent_assessment(saved, monkeypatch):
    def verify(*args, **kwargs):
        raise RejectedPlan("Unbound metric", support())

    monkeypatch.setattr(probe, "verify_experiments", verify)
    directory, summary = probe.run_probe(
        *saved[:1], ["c"], saved[1], expectations={"c": {"supported_conditions": ["c1"]}}
    )
    row = summary["cases"][0]
    assert summary["ok"] and row["expectations_passed"]
    assert row["assessed_status"] == "supported"
    assert row["issues"] == ["Execution plan rejected: Unbound metric"]
    assert (directory / "case-001/assessed.json").is_file()


def test_foreign_coverage_in_rejected_plan_is_failure(saved, monkeypatch):
    def verify(*args, **kwargs):
        raise RejectedPlan("Unbound metric", support(["foreign"]))

    monkeypatch.setattr(probe, "verify_experiments", verify)
    _, summary = probe.run_probe(saved[0], ["c"], saved[1])
    assert not summary["ok"] and summary["cases"][0]["status"] == "failed"


def test_completed_empty_result_cannot_pass_positive_expectation(saved, monkeypatch):
    monkeypatch.setattr(probe, "verify_experiments", lambda *a, **kw: BranchResult())
    _, summary = probe.run_probe(
        saved[0], ["c"], saved[1], expectations={"c": {"supported_conditions": ["c1"]}}
    )
    assert summary["status"] == "completed" and not summary["ok"]
    assert summary["expectations_passed"] is False


def test_no_expectations_never_claims_oracle_pass(saved, monkeypatch):
    monkeypatch.setattr(probe, "verify_experiments", lambda *a, **kw: BranchResult())
    _, summary = probe.run_probe(saved[0], ["c"], saved[1])
    assert summary["ok"] and summary["expectations_passed"] is None
    assert summary["cases"][0]["assessed_status"] == "unverified"


@pytest.mark.parametrize(
    "claim_ids,expected",
    [
        (["missing"], {}),
        (["c", "c"], {}),
        (["c"], {"c": {"supported_conditions": ["foreign"]}}),
        (["c"], {"c": {"supported_conditions": ["c1"], "unsupported_conditions": ["c1"]}}),
    ],
)
def test_bad_selections_fail_before_any_external_call(saved, monkeypatch, claim_ids, expected):
    monkeypatch.setattr(probe, "verify_experiments", lambda *a, **kw: pytest.fail("Verification called"))
    with pytest.raises(ValueError):
        probe.run_probe(saved[0], claim_ids, saved[1], expectations=expected)
    assert not saved[1].exists()


def test_failed_model_is_saved_with_provider_identity_and_without_credential(saved, monkeypatch):
    cfg = LLMConfig("mock", "fixture", "https://example.test", "secret-probe-key")

    def verify(claim, materials, *, call):
        call(prompt="original", system="review", module="verification.experiments", cfg=cfg)

    def model(**kwargs):
        raise RuntimeError("request failed secret-probe-key")

    monkeypatch.setattr(probe, "verify_experiments", verify)
    directory, summary = probe.run_probe(saved[0], ["c"], saved[1], call=model)
    assert not summary["ok"]
    record = json.loads(next((directory / "case-001").glob("model-*.json")).read_text(encoding="utf-8"))
    assert record["provider"] == "mock" and record["model"] == "fixture"
    assert "[redacted]" in record["error"]
    assert all(
        "secret-probe-key" not in path.read_text(encoding="utf-8") for path in directory.rglob("*.json")
    )


def test_source_change_invalidates_run(saved, monkeypatch):
    def verify(*args, **kwargs):
        path = saved[0] / "screening/screening.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return BranchResult()

    monkeypatch.setattr(probe, "verify_experiments", verify)
    _, summary = probe.run_probe(saved[0], ["c"], saved[1])
    assert summary["status"] == "completed" and not summary["source_unchanged"] and not summary["ok"]


def test_scope_service_failure_remains_failure_after_observation_recovery(saved, monkeypatch):
    def verify(claim, materials, *, call):
        try:
            call(
                prompt="source",
                system="review",
                module="verification.experiments.scope",
                cfg=LLMConfig("mock", "fixture", None, None),
            )
        except RuntimeError:
            return BranchResult(issues=["Scope service unavailable"])

    def model(**kwargs):
        raise RuntimeError("Connection lost")

    monkeypatch.setattr(probe, "verify_experiments", verify)
    _, summary = probe.run_probe(saved[0], ["c"], saved[1], call=model)
    assert summary["status"] == "completed" and not summary["ok"]
    assert summary["cases"][0]["failed_model_calls"] == 1


@pytest.mark.parametrize("response", [{"status": "error", "error": "unavailable"}, {"error": "unknown"}, []])
def test_unsuccessful_model_response_is_failure_after_scope_recovery(saved, monkeypatch, response):
    from screening.checks import ask

    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )

    def verify(claim, materials, *, call):
        try:
            ask("review", {}, module="verification.experiments.scope", call=call)
        except RuntimeError:
            return BranchResult(issues=["Scope request failed"])

    monkeypatch.setattr(probe, "verify_experiments", verify)
    directory, summary = probe.run_probe(saved[0], ["c"], saved[1], call=lambda **kw: response)
    assert summary["status"] == "completed" and not summary["ok"]
    assert summary["model_boundary"] == "injected"
    assert summary["cases"][0]["failed_model_calls"] == 1
    record = json.loads(next((directory / "case-001").glob("model-*.json")).read_text(encoding="utf-8"))
    assert record["response"] == response and record["error"]


def test_failed_case_cannot_claim_expectation_pass(saved, monkeypatch):
    def verify(*args, **kwargs):
        raise ValueError("invalid observation")

    monkeypatch.setattr(probe, "verify_experiments", verify)
    _, summary = probe.run_probe(
        saved[0], ["c"], saved[1], expectations={"c": {"supported_conditions": ["c1"]}}
    )
    assert not summary["ok"] and summary["expectations_passed"] is False


def test_interrupted_request_retains_running_case_and_request_without_claiming_pass(saved, monkeypatch):
    def verify(claim, materials, *, call):
        call(
            prompt="original",
            system="review",
            module="verification.experiments",
            cfg=LLMConfig("mock", "fixture", None, None),
        )

    def model(**kwargs):
        summary_path = next(saved[1].glob("*/summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["status"] == "running" and summary["cases"][0]["claim"] == "c"
        assert next(summary_path.parent.glob("case-001/model-*.json")).is_file()
        raise KeyboardInterrupt

    monkeypatch.setattr(probe, "verify_experiments", verify)
    with pytest.raises(KeyboardInterrupt):
        probe.run_probe(saved[0], ["c"], saved[1], call=model)
    summary = json.loads(next(saved[1].glob("*/summary.json")).read_text(encoding="utf-8"))
    assert summary["status"] == "running" and "ok" not in summary
