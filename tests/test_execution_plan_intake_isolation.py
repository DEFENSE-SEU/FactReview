"""One public intake control; every external execution boundary is forbidden."""

import copy
import hashlib
import json

import httpx
import pytest
import requests
from pydantic_core import PydanticSerializationError

from fact_generation.execution import runner as v2
from fact_generation.execution.recovery import recover_execution_records
from review.report.renderer import render_markdown
from schemas.claim import ExecutionPlan
from schemas.review import FinalReview
from tests.test_execution import inputs as original_inputs


@pytest.mark.parametrize("failure", ["dump", "origin", "shared"])
def test_plan_intake_isolation(tmp_path, monkeypatch, failure):
    external_calls = []

    def forbidden(*args, **kwargs):
        external_calls.append(True)
        raise AssertionError("External operation was not mocked")

    for owner, name in ((v2, "docker_runner"), (v2, "docker_ensure_paper_image"),
                        (v2, "run_command"), (requests.sessions.Session, "request"),
                        (httpx.Client, "send"), (httpx.AsyncClient, "send")):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)

    first, claim, materials = original_inputs.__wrapped__(tmp_path)
    bad = first.model_copy(update={"id": "bad"}, deep=True)
    later = first.model_copy(update={"id": "later"}, deep=True)
    plans = [bad, first, later]
    dump = ExecutionPlan.model_dump
    bad_dumps = []
    config = {"approval_mode": "interactive", "max_attempts": 0, "refine_with_llm": False}
    approvals = []
    runs = []

    def approve(plan, cost):
        approvals.append(plan.id)
        return True

    def runner(request):
        runs.append(request.plan.id)
        return v2.RunOutcome(returncode=0)

    if failure == "shared":
        repository_type = type(materials.repository)
        repository_dump = repository_type.model_dump

        def shared_dump(self, *args, **kwargs):
            value = repository_dump(self, *args, **kwargs)
            value["opaque_program_fixture"] = float("nan")
            return value

        with monkeypatch.context() as patch:
            patch.setattr(repository_type, "model_dump", shared_dump)
            with pytest.raises(ValueError):
                v2.execute_plans(plans, [claim], materials, tmp_path / "shared", config=config,
                                 approver=approve, runner=runner)
        assert approvals == runs == []
        assert not (tmp_path / "shared/intake_unavailable.json").exists()

        claim_type = type(claim)
        claim_dump = claim_type.model_dump

        def shared_claim_dump(self, *args, **kwargs):
            value = claim_dump(self, *args, **kwargs)
            value["conditions"][0]["settings"]["opaque_program_fixture"] = float("nan")
            return value

        with monkeypatch.context() as patch:
            patch.setattr(claim_type, "model_dump", shared_claim_dump)
            with pytest.raises(ValueError):
                v2.execute_plans(plans, [claim], materials, tmp_path / "shared-claim", config=config,
                                 approver=approve, runner=runner)
        assert approvals == runs == [] and not external_calls
        return

    def tracked_dump(self, *args, **kwargs):
        if self.id == bad.id:
            if failure == "dump":
                bad_dumps.append(True)
                raise PydanticSerializationError("private fixture must never enter diagnostic text")
            value = dump(self, *args, **kwargs)
            value["target_conditions"][0]["settings"]["opaque_program_fixture"] = float("nan")
            return value
        return dump(self, *args, **kwargs)
    monkeypatch.setattr(ExecutionPlan, "model_dump", tracked_dump)

    output = tmp_path / "isolated"
    repository = materials.repository.model_dump(mode="json")
    expected = {plan.id: hashlib.sha256(json.dumps(
        {"plan": dump(plan, mode="json"), "repository": repository},
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest() for plan in (first, later)}
    frozen = []
    origin = getattr(v2, "_resource_digest", None)
    if origin is not None:
        def tracked_origin(snapshot, repository_snapshot):
            value = origin(snapshot, repository_snapshot)
            if snapshot["id"] in expected:
                frozen.append((snapshot["id"], value))
            return value
        monkeypatch.setattr(v2, "_resource_digest", tracked_origin)

    def approved_after_freeze(plan, cost):
        if not approvals:
            assert set(expected.items()).issubset(set(frozen))
            assert json.loads((output / "intake_unavailable.json").read_text(encoding="utf-8"))
        return approve(plan, cost)

    result = v2.execute_plans(plans, [claim], materials, output, config=config,
                              approver=approved_after_freeze, runner=runner)
    assert approvals == runs == [first.id, later.id]
    assert [row["plan"]["id"] for row in result.ledger] == [first.id, later.id]
    assert [row["resource_origin_sha256"] for row in result.ledger] == list(expected.values())
    assert not result.claims[0].evidence
    assert [question.reason for question in result.claims[0].questions] == [
        "runtime conditions do not uniquely align with every requested claim condition"
    ] * 2
    assert claim.evidence == [] and claim.verification_limitations == []
    record = json.loads((output / "intake_unavailable.json").read_text(encoding="utf-8"))
    code = "plan_dump_unavailable" if failure == "dump" else "plan_origin_unavailable"
    assert record == [{"schema_id": "execution-plan-intake-unavailable-v1", "input_ordinal": 0,
                       "execution_ordinal": 0, "plan_id": "bad", "claim_id": claim.id,
                       "condition_ids": ["c1"], "state": "intake_unavailable", "error_code": code,
                       "approved": False, "attempts_started_by_this_intake": 0}]
    check = result.delivery_checks[0]
    limitation = result.claims[0].verification_limitations[0]
    assert check.component == "execution.intake" and check.state == "unavailable"
    assert check.claim_id == limitation.claim_id == claim.id and limitation.kind == "stage_failed"
    assert limitation.condition_ids == ["c1"] and "intake_unavailable.json#/0" in limitation.reason
    assert "private fixture" not in json.dumps(result.model_dump(mode="json"))
    review = FinalReview(paper_key="fixture", run_id="mock", claims=result.claims, ledger=result.ledger,
                         run_status="partial", incomplete_stages=["execution"], execution_requested=True,
                         delivery_checks=result.delivery_checks)
    dump_count = len(bad_dumps)
    restored = FinalReview.model_validate_json(review.model_dump_json())
    assert v2.ExecutionResult.model_validate_json(result.model_dump_json()) == result
    text = render_markdown(restored, issues=result.issues)
    assert "intake_unavailable.json#/0" in text and "execution.intake" in text
    assert len(bad_dumps) == dump_count

    forged = copy.deepcopy(result.ledger[0])
    forged["plan"]["id"] = bad.id
    bad_record = output / "run_0000/ledger.json"
    bad_record.parent.mkdir()
    bad_record.write_text(json.dumps(forged), encoding="utf-8")
    before = {row.parent.name: row.read_bytes() for row in output.glob("run_*/ledger.json")}
    recovered, issues = recover_execution_records(output, plans)
    assert recovered == result.ledger and any(code in issue for issue in issues)
    assert "without their evidence being adopted" in issues[-1] and not claim.evidence
    assert {row.parent.name: row.read_bytes() for row in output.glob("run_*/ledger.json")} == before

    approvals.clear()
    runs.clear()

    def mutate_later(plan, cost):
        if plan.id == first.id:
            later.task.command.append("--changed-after-original-freeze")
        return approve(plan, cost)

    late = v2.execute_plans([first, later], [claim], materials, tmp_path / "late", config=config,
                            approver=mutate_later, runner=runner)
    assert runs == [first.id] and not late.ledger[1]["approved"] and late.ledger[1]["attempts"] == []
    assert late.ledger[1]["plan"]["task"]["command"] == ["python", "eval.py"]
    assert late.ledger[1]["resource_validation"][-1]["state"] == "invalid"
    assert not late.claims[0].evidence and not external_calls
