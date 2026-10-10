"""Necessary offline controls for unresolved execution operation responsibility."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import pipeline_v2
from fact_generation.execution import v2
from fact_generation.execution.recovery import recover_execution_records
from schemas.review import DeliveryCheck
from tests import test_execution_v2 as execution_fixtures
from tests import test_pipeline_v2 as pipeline_fixtures
from tests.test_execution_v2 import observation
from tests.test_pipeline_v2 import run_tiny
from util.subprocess_runner import CommandResult
from verification.dispatch import VerificationResult

execution_inputs = execution_fixtures.inputs
pipeline_offline_boundaries = pipeline_fixtures.offline_boundaries
tiny_inputs = pipeline_fixtures.tiny_inputs
_DOCKER_PRODUCER = v2.docker_runner


@pytest.fixture(autouse=True)
def offline(monkeypatch, pipeline_offline_boundaries):
    def forbidden(*args, **kwargs):
        raise AssertionError("External operation was not mocked")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: object())
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    monkeypatch.setattr(v2, "run_command", forbidden)
    monkeypatch.setattr(v2, "docker_runner", _DOCKER_PRODUCER)


def _execute(values, output, runner, **kwargs):
    plan, claim, materials = values
    return v2.execute_plans([plan], [claim], materials, output, runner=runner, **kwargs)


@pytest.mark.parametrize("failure", ["service", "protocol"])
def test_refinement_failure_is_partial_and_next_plan_still_runs(
    execution_inputs, tmp_path, monkeypatch, failure
):
    plan, claim, materials = execution_inputs
    first = plan.model_copy(deep=True)
    first.task.command = []
    second = plan.model_copy(update={"id": "p2"}, deep=True)

    def completion(*args, **kwargs):
        if failure == "service":
            raise ConnectionError("fixture service unavailable")
        return {"command": "python eval.py", "metric_output": None}

    monkeypatch.setattr("llm.client.llm_json", completion)
    runner = Mock(return_value=v2.RunOutcome(returncode=0, observations=[observation()]))
    result = v2.execute_plans(
        [first, second], [claim], materials, tmp_path / "out", runner=runner
    )
    assert runner.call_count == 1 and runner.call_args.args[0].plan.id == "p2"
    assert len(result.ledger) == 2 and result.ledger[0]["reason"]
    assert result.ledger[0]["operation_failures"][0]["component"] == "execution.refinement"
    assert len(result.delivery_checks) == 1
    check = result.delivery_checks[0]
    assert (check.stage, check.component, check.state, check.responsibility, check.claim_id) == (
        "execution", "execution.refinement", "failed", "system", claim.id
    )
    assert "ledger.json" in check.reason and not result.claims[0].questions
    assert len(result.claims[0].evidence) == 1 and result.claims[0].evidence[0].aligned
    assert Path(tmp_path / "out" / "run_0000" / "ledger.json").is_file()
    assert claim.evidence == [] and claim.questions == []


@pytest.mark.parametrize("failure", ["exception", "protocol", "repair", "declared"])
def test_runner_and_repair_protocol_failures_do_not_become_author_questions(
    execution_inputs, tmp_path, failure
):
    def runner(request):
        if failure == "exception":
            raise OSError("fixture transport failure")
        if failure == "protocol":
            return {"returncode": "invalid"}
        if failure == "declared":
            return v2.RunOutcome(
                returncode=0, observations=[observation()],
                operation_failures=[{
                    "component": "execution.environment", "reason": "fixture explicit environment failure",
                }],
            )
        return v2.RunOutcome(returncode=1, stderr="fixture author process failed")

    def repair(*args):
        if failure == "repair":
            raise ConnectionError("fixture repair service failed")
        return None

    result = _execute(execution_inputs, tmp_path / "out", runner, repairer=repair)
    component = {
        "repair": "execution.repair", "declared": "execution.environment",
    }.get(failure, "execution.runner")
    assert len(result.delivery_checks) == 1 and result.delivery_checks[0].component == component
    assert result.ledger[0]["operation_failures"][0]["component"] == component
    assert not result.claims[0].questions and not result.claims[0].evidence
    assert result.claims[0].verification_limitations[-1].kind == "stage_failed"
    assert result.claims[0].verification_limitations[-1].stage == "execution"


@pytest.mark.parametrize("failure", ["build", "transport", "cleanup", "author127"])
def test_docker_producer_classifies_explicit_operation_failures_only(
    execution_inputs, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(
        v2, "docker_ensure_paper_image",
        lambda *args, **kwargs: (failure != "build", "fixture-image"),
    )
    commands = []

    def transport(command, cwd, timeout_sec):
        commands.append(command)
        if command[1] == "run":
            if failure == "transport":
                raise OSError("fixture Docker transport exception")
            return CommandResult(command, cwd, 127, "", "fixture author process failed", .01)
        assert command[1:3] == ["rm", "--force"]
        return CommandResult(command, cwd, 1 if failure == "cleanup" else 0, "", "", .01)

    monkeypatch.setattr(v2, "run_command", transport)
    result = _execute(
        execution_inputs, tmp_path / "out", v2.docker_runner,
        config={"max_attempts": 0},
    )
    if failure == "author127":
        assert result.delivery_checks == [] and result.ledger[0]["operation_failures"] == []
        assert result.claims[0].questions and not result.claims[0].verification_limitations
    else:
        component = {
            "build": "execution.environment", "transport": "execution.runner",
            "cleanup": "execution.cleanup",
        }[failure]
        assert [check.component for check in result.delivery_checks] == [component]
        assert not result.claims[0].questions
        assert result.ledger[0]["attempts"][0]["operation_failures"][0]["component"] == component
    assert not result.claims[0].evidence
    assert len(commands) == (0 if failure == "build" else 2)


def test_repaired_operation_failure_is_history_and_policy_block_is_not_system_failure(
    execution_inputs, tmp_path, monkeypatch
):
    builds = []

    def builder(*args, **kwargs):
        builds.append(kwargs)
        return len(builds) > 1, "fixture-image"

    monkeypatch.setattr(v2, "docker_ensure_paper_image", builder)
    monkeypatch.setattr(v2, "run_command", lambda command, cwd, timeout_sec: CommandResult(
        command, cwd, 0, json.dumps({"observations": [observation().model_dump()]}), "", .01
    ))
    result = _execute(
        execution_inputs, tmp_path / "repaired", v2.docker_runner,
        repairer=lambda *args: v2.Repair(dependencies=["numpy"], reason="fixture environment repair"),
    )
    assert len(result.ledger[0]["attempts"]) == 2
    assert result.ledger[0]["attempts"][0]["operation_failures"][0]["component"] == "execution.environment"
    assert result.ledger[0]["operation_failures"] == [] and result.delivery_checks == []
    assert result.claims[0].evidence[0].aligned and not result.claims[0].questions
    execution_inputs[0].run_mode = "training"
    blocked = _execute(
        execution_inputs, tmp_path / "blocked", Mock(side_effect=AssertionError("must not launch")),
        config={"training_budget": 0},
    )
    assert blocked.delivery_checks == [] and blocked.ledger[0]["attempts"] == []
    assert "training budget exhausted" in blocked.ledger[0]["reason"]


def test_execution_producer_failure_reaches_all_canonical_outputs(tiny_inputs, monkeypatch):
    check = DeliveryCheck(
        stage="execution", component="execution.runner", state="failed", claim_id="claim_001",
        reason="Transport unavailable. Audit: execution/run_0000/ledger.json#/operation_failures/0.",
    )

    async def verification(claims, *args, **kwargs):
        return VerificationResult(claims=claims)

    def execution(plans, claims, *args, **kwargs):
        return v2.ExecutionResult(claims=claims, delivery_checks=[check])

    monkeypatch.setattr(pipeline_v2, "verify_claims", verification)
    monkeypatch.setattr(pipeline_v2, "execute_plans", execution)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    report = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    expected = check.model_dump(mode="json")
    assert summary["stages"]["execution"] == summary["stages"]["report"] == "ok"
    assert summary["run_status"] == report["run_status"] == teaser["run_status"] == "partial"
    assert expected in summary["delivery_checks"]
    assert report["delivery_checks"] == teaser["delivery_checks"] == summary["delivery_checks"]
    assert "execution" in summary["incomplete_stages"]


def test_recovery_preserves_historical_outcomes_and_rejects_foreign_failure_fields(
    execution_inputs, tmp_path
):
    plan, claim, materials = execution_inputs
    plans = [plan.model_copy(update={"id": identifier}, deep=True) for identifier in ("old", "new", "bad")]
    result = v2.execute_plans(
        plans, [claim], materials, tmp_path / "out",
        runner=lambda request: v2.RunOutcome(returncode=0, observations=[observation()]),
    )
    rows = result.ledger
    del rows[0]["operation_failures"]
    del rows[0]["attempts"][0]["operation_failures"]
    rows[2]["operation_failures"] = [{"component": "foreign", "reason": "must be rejected"}]
    paths = [tmp_path / "out" / f"run_{number:04d}" / "ledger.json" for number in range(3)]
    for path, row in zip(paths, rows):
        path.write_text(json.dumps(row), encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    recovered, issues = recover_execution_records(tmp_path / "out", plans)
    assert recovered == rows[:2]
    assert any("bad" in issue and "unknown" in issue for issue in issues)
    assert [path.read_bytes() for path in paths] == before
    assert claim.evidence == []
