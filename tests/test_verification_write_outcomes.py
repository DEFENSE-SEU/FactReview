"""One offline differential control for bounded program-only failure observations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidenceNeed, EvidencePointer
from schemas.materials import SharedMaterials
from verification import dispatch
from verification import rejection_diagnostics as diagnostics
from verification.contracts import BranchResult, RejectedPlan


def _before(name):
    path = Path(__file__).resolve().parent / "fixtures/original_sources"
    module = ModuleType(f"before_{name}")
    exec(compile((path / f"before-{name}.py").read_bytes(), f"before_{name}.py", "exec", dont_inherit=True), module.__dict__)
    return module


def _materials(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir(exist_ok=True)
    return SharedMaterials(
        paper_key="fixture", source_pdf=str(source / "paper.pdf"), markdown="Claim",
        markdown_path=str(source / "paper.md"), content_list_path=str(source / "content.json"), provider="mock",
    )


def _claim(needs):
    return Claim(id="c", text="Claim", loc=ClaimLocation(page=1),
                 conditions=[Condition(id="d", description="original condition")], needs=needs)


async def test_program_write_outcomes_preserve_original_contract(tmp_path, monkeypatch):
    def external(*args, **kwargs):
        pytest.fail("unmocked external boundary")

    monkeypatch.setattr("llm.client.llm_json", external)
    monkeypatch.setattr("httpx.Client.send", external)
    monkeypatch.setattr("httpx.AsyncClient.send", external)
    monkeypatch.setattr("requests.sessions.Session.request", external)
    monkeypatch.setattr("subprocess.run", external)
    monkeypatch.setattr("subprocess.Popen", external)
    original_writer, original_dispatch = _before("rejection_diagnostics"), _before("dispatch")
    materials = _materials(tmp_path)
    fixed_code = "experiments_joint_final_validation_failed"
    expected = {
        "written": "written", "disabled": "disabled_or_invalid_arguments",
        "invalid_branch": "disabled_or_invalid_arguments", "invalid_code": "disabled_or_invalid_arguments",
        "invalid_destination": "invalid_destination", "overlap": "input_overlap",
        "io_failed": "io_failed", "path_error": "unknown", "identity_changed": "path_identity_changed",
        "capacity": "payload_capacity",
    }

    def writer_run(module, mode, root):
        directory = root / "diagnostics"
        code, branch = fixed_code, "Experiments"
        if mode == "disabled":
            directory = None
        elif mode == "invalid_destination":
            directory = root / "wrong"
        elif mode == "overlap":
            directory = Path(materials.markdown_path).parent / "diagnostics"
        elif mode == "invalid_code":
            code = "SECRET_BODY_NEVER_CAPTURED"
        elif mode == "invalid_branch":
            branch = "SECRET_BODY_NEVER_CAPTURED"
        original_plain, original_open = module.plain_path, Path.open

        def plain(path):
            if mode == "path_error":
                raise ValueError("SECRET_BODY_NEVER_CAPTURED")
            if mode == "identity_changed" and path == directory and path.exists():
                return SimpleNamespace(resolve=lambda **kwargs: root / "different")
            return original_plain(path)

        def open_file(path, mode="r", *args, **kwargs):
            if mode == "x" and writer_mode == "io_failed":
                raise OSError("SECRET_BODY_NEVER_CAPTURED")
            return original_open(path, mode, *args, **kwargs)

        writer_mode = mode
        with monkeypatch.context() as control:
            control.setattr(module, "plain_path", plain)
            control.setattr(Path, "open", open_file)
            if mode == "capacity":
                control.setattr(module.json, "dumps", lambda *args, **kwargs: "x" * 512)
            writer = module.FirstRejectionDiagnostic(directory, materials, branch)
            first = writer.record(code, first_subguard="unknown")
            second = writer.record(code, first_subguard="unknown")
        raw = [path.read_bytes() for path in directory.glob("*.json")] if directory is not None else []
        return first, second, writer.attempted, raw

    for mode, outcome in expected.items():
        before = writer_run(original_writer, mode, tmp_path / "old" / mode)
        disabled = writer_run(diagnostics, mode, tmp_path / "disabled" / mode)
        with diagnostics.write_outcome_scope(enabled=True) as collector:
            enabled = writer_run(diagnostics, mode, tmp_path / "enabled" / mode)
            snapshot = collector.close_snapshot()
        assert before == disabled == enabled
        assert before[:3] == (None, None, True)
        assert len(before[3]) == (1 if mode == "written" else 0)
        assert snapshot["closed"] is True and snapshot["adoption_authorized"] is False
        assert snapshot["coverage"] == "captured_scope_events_only"
        rows = snapshot["events"]
        subguard_outcomes = set() if mode in {"invalid_branch", "invalid_code"} else {"joint_subguard_unknown"}
        assert {row["outcome"] for row in rows} == {
            "constructed", "record_called", "attempted", "already_attempted", outcome,
        } | subguard_outcomes
        assert [row for row in rows if row["outcome"] == "joint_subguard_unknown"] == (
            [{"branch": "Experiments", "code": fixed_code, "stage": "writer",
              "outcome": "joint_subguard_unknown", "count": 1}] if subguard_outcomes else []
        )
        assert sum(row["count"] for row in rows if row["outcome"] == "record_called") == 2
        assert "SECRET_BODY_NEVER_CAPTURED" not in json.dumps(snapshot)
        assert str(tmp_path) not in json.dumps(snapshot)
        if mode == "written":
            assert len(before[3][0]) <= 512
            assert json.loads(before[3][0]) == {
                "schema": "verification-rejection-diagnostics-v2", "branch": "Experiments",
                "scope": "first_evidence_validation_rejection", "code": fixed_code, "first_subguard": "unknown",
            }

    class Uncaught:
        @property
        def source_pdf(self):
            raise RuntimeError("same uncaught exception")

    for module in (original_writer, diagnostics):
        with diagnostics.write_outcome_scope(enabled=True), pytest.raises(RuntimeError) as caught:
            module.FirstRejectionDiagnostic(tmp_path / "uncaught" / "diagnostics", Uncaught(), "Experiments").record(fixed_code)
        assert caught.value.args == ("same uncaught exception",)

    stages = {
        "missing": "branch_lookup", "raise": "invoke_or_branch_contract",
        "bad_contract": "invoke_or_branch_contract", "bad_recovery": "rejected_plan_recovery",
        "foreign_evidence": "result_validation", "rejected": None, "sync": None, "async": None,
    }
    for mode, expected_stage in stages.items():
        calls = []

        def branch(c, m):
            calls.append((c.model_dump(mode="json"), m.model_dump(mode="json")))
            if mode == "raise":
                raise ValueError("schema words are not a stage classifier")
            if mode == "bad_contract":
                return {"extra": "invalid"}
            if mode in {"rejected", "bad_recovery"}:
                exc = RejectedPlan("invalid plan", BranchResult())
                if mode == "bad_recovery":
                    exc.observations.plans.append(object())
                raise exc
            if mode == "foreign_evidence":
                return BranchResult(evidence=[Evidence(
                    source="paper_internal", pointer=EvidencePointer(locator="paper", quote="Claim", page=1),
                    covered=["foreign"], direction="support",
                )])
            diagnostics.FirstRejectionDiagnostic(None, m, "Experiments").record(fixed_code)
            return BranchResult()

        async def async_branch(c, m):
            await asyncio.sleep(0)
            return branch(c, m)

        branches = {} if mode == "missing" else {EvidenceNeed.EXPERIMENTS: async_branch if mode == "async" else branch}
        before = await original_dispatch.verify_claims(
            [_claim([EvidenceNeed.EXPERIMENTS])], materials, tmp_path / "dispatch_old" / mode, branches=branches,
        )
        old_calls = list(calls)
        calls.clear()
        with diagnostics.write_outcome_scope(enabled=True) as collector:
            after = await dispatch.verify_claims(
                [_claim([EvidenceNeed.EXPERIMENTS])], materials, tmp_path / "dispatch_new" / mode, branches=branches,
            )
            snapshot = collector.close_snapshot()
        assert before.model_dump(mode="json") == after.model_dump(mode="json")
        assert old_calls == calls
        assert before.plans == [] and before.claims[0].evidence == []
        failed = [row for row in snapshot["events"] if row["code"] == "experiments_dispatch_failed"]
        assert failed == ([] if expected_stage is None else [{
            "branch": "Experiments", "code": "experiments_dispatch_failed", "stage": expected_stage,
            "outcome": "failed", "count": 1,
        }])
        if mode in {"sync", "async"}:
            assert {row["outcome"] for row in snapshot["events"]} == {
                "constructed", "record_called", "attempted", "disabled_or_invalid_arguments",
            }

    async def parallel_branch(c, m):
        await asyncio.to_thread(lambda: diagnostics.FirstRejectionDiagnostic(None, m, "Experiments").record(fixed_code))
        return BranchResult()

    with diagnostics.write_outcome_scope(enabled=True) as outer:
        await asyncio.gather(*(parallel_branch(None, materials) for _ in range(8)))
        with diagnostics.write_outcome_scope(enabled=True) as inner:
            diagnostics.record_experiments_dispatch_failure("result_validation")
            assert len(inner.close_snapshot()["events"]) == 1
        with diagnostics.write_outcome_scope():
            diagnostics.record_experiments_dispatch_failure("branch_lookup")
        snapshot = outer.close_snapshot()
        assert len(snapshot["events"]) == 4 and all(row["count"] == 8 for row in snapshot["events"])
        await asyncio.to_thread(diagnostics.record_experiments_dispatch_failure, "result_validation")
        assert outer.close_snapshot() == snapshot
    assert diagnostics._OUTCOME_CONTEXT.get() is None
    for disabled in (False, None, 1, "true"):
        with diagnostics.write_outcome_scope(enabled=disabled) as empty:
            assert empty is None

    with diagnostics.write_outcome_scope(enabled=True) as collector:
        with monkeypatch.context() as control:
            control.setattr(diagnostics, "_OUTCOME_MAX_COUNT", 1)
            diagnostics.record_experiments_dispatch_failure("branch_lookup")
            diagnostics.record_experiments_dispatch_failure("branch_lookup")
        snapshot = collector.close_snapshot()
        assert snapshot["overflow"] is True and snapshot["events"][0]["count"] == 1
    with diagnostics.write_outcome_scope(enabled=True) as collector:
        with monkeypatch.context() as control:
            control.setattr(diagnostics, "_OUTCOME_MAX_ROWS", 0)
            diagnostics.record_experiments_dispatch_failure("branch_lookup")
        assert collector.close_snapshot()["overflow"] is True
    with diagnostics.write_outcome_scope(enabled=True) as collector:
        with monkeypatch.context() as control:
            control.setattr(diagnostics._WriteOutcomes, "_record", lambda *args: (_ for _ in ()).throw(ValueError("diagnostic failed")))
            assert writer_run(diagnostics, "written", tmp_path / "observer_failure")[3]
        snapshot = collector.close_snapshot()
        assert snapshot["unavailable"] is True and snapshot["events"] == []
    with monkeypatch.context() as control:
        control.setattr(diagnostics, "RLock", lambda: (_ for _ in ()).throw(RuntimeError("lock failed")))
        with diagnostics.write_outcome_scope(enabled=True) as collector:
            assert collector is None
            assert writer_run(diagnostics, "written", tmp_path / "scope_failure")[3]
