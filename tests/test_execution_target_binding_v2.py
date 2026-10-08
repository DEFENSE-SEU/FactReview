"""Execution independently verifies targets; no producer or transport is trusted implicitly."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from fact_generation.execution.v2 import Repair, RunOutcome, execute_plans
from tests.test_execution_v2 import bind_fixture_sources, observation
from tests.test_execution_v2 import inputs as inputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Target-binding tests must mock external boundaries")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def test_legacy_readable_plan_is_blocked_before_approval_or_runner(inputs, tmp_path):
    plan, claim, materials = inputs
    plan.target_bindings = {}
    plan = type(plan).model_validate_json(plan.model_dump_json())
    approver = Mock(return_value=True)
    runner = Mock(return_value=RunOutcome(returncode=0, observations=[observation()]))
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "legacy",
        config={"approval_mode": "interactive"},
        approver=approver,
        runner=runner,
    )
    approver.assert_not_called()
    runner.assert_not_called()
    assert not result.ledger[0]["approved"]
    assert not result.ledger[0]["attempts"]
    assert not result.claims[0].evidence
    assert "binding" in result.ledger[0]["reason"].lower()
    assert result.claims[0].questions


def test_changed_paper_value_cannot_be_rehabilitated_by_matching_runtime(inputs, tmp_path):
    plan, claim, materials = inputs
    plan.y_paper = {"c1": 0.2}
    runner = Mock(return_value=RunOutcome(returncode=0, observations=[observation(value=0.2)]))
    result = execute_plans([plan], [claim], materials, tmp_path / "tampered", runner=runner)
    runner.assert_not_called()
    assert not result.claims[0].evidence
    assert not result.ledger[0]["approved"]
    assert result.ledger[0]["reason"]


@pytest.mark.parametrize(
    "change",
    [
        "binding_value",
        "unknown_version",
        "selector",
        "hash",
        "claim",
        "missing_file",
        "file",
        "block",
        "raw_dict",
    ],
)
def test_forged_or_stale_target_is_a_local_blocker(inputs, tmp_path, change):
    plan, claim, materials = inputs
    binding = plan.target_bindings["c1"]
    if change == "binding_value":
        plan.y_paper["c1"] = 0.2
        plan.target_bindings["c1"] = binding.model_copy(update={"value": 0.2})
    elif change == "unknown_version":
        plan.target_bindings["c1"] = binding.model_copy(update={"version": 99})
    elif change == "selector":
        binding.selector.number_id = "number_foreign"
    elif change == "hash":
        binding.artifact_sha256 = "0" * 64
    elif change == "claim":
        claim.text = "d1 test seed 1 accuracy is 0.8."
    elif change == "missing_file":
        materials.markdown_path = str(tmp_path / "missing.md")
    elif change == "file":
        Path(materials.markdown_path).write_text("changed paper", encoding="utf-8")
    elif change == "block":
        materials.blocks[0].text = "d1 test seed 1 accuracy is 0.8."
    else:
        plan.target_bindings["c1"] = {}
    runner = Mock(return_value=RunOutcome(returncode=0, observations=[observation()]))
    repairer = Mock()
    result = execute_plans([plan], [claim], materials, tmp_path / "invalid", runner=runner, repairer=repairer)
    runner.assert_not_called()
    repairer.assert_not_called()
    assert not result.claims[0].evidence
    assert not result.ledger[0]["approved"]
    assert result.ledger[0]["paper_target_validation"][0]["verified"] is False


def test_invalid_cached_plan_does_not_hide_another_healthy_plan(inputs, tmp_path):
    plan, claim, materials = inputs
    legacy = plan.model_copy(update={"id": "legacy", "target_bindings": {}}, deep=True)
    runner = Mock(return_value=RunOutcome(returncode=0, observations=[observation()]))
    result = execute_plans([legacy, plan], [claim], materials, tmp_path / "multiple", runner=runner)
    assert runner.call_count == 1
    assert runner.call_args.args[0].plan.id == plan.id
    assert not result.ledger[0]["approved"] and result.ledger[1]["approved"]
    assert len(result.claims[0].evidence) == 1 and result.claims[0].evidence[0].sufficient


def test_mismatched_plan_condition_does_not_hide_healthy_neighbor(inputs, tmp_path):
    plan, claim, materials = inputs
    bad = plan.model_copy(update={"id": "bad-condition"}, deep=True)
    bad.target_conditions[0].dataset = "other"
    runner = Mock(return_value=RunOutcome(returncode=0, observations=[observation()]))
    result = execute_plans([bad, plan], [claim], materials, tmp_path / "conditions", runner=runner)
    assert runner.call_count == 1 and runner.call_args.args[0].plan.id == plan.id
    assert not result.ledger[0]["approved"] and "conditions differ" in result.ledger[0]["reason"]
    assert result.ledger[1]["approved"]
    assert len(result.claims[0].evidence) == 1 and result.claims[0].evidence[0].sufficient


def test_bound_plan_roundtrip_preserves_valid_binding_audit(inputs, tmp_path):
    plan, claim, materials = inputs
    plan = type(plan).model_validate_json(plan.model_dump_json())
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "roundtrip",
        runner=lambda _: RunOutcome(returncode=0, observations=[observation()]),
    )
    assert result.claims[0].evidence[0].sufficient
    checks = result.ledger[0]["paper_target_validation"]
    assert [row["phase"] for row in checks] == ["before_approval", "before_run", "before_judge"]
    assert all(
        row["verified"] and row["bindings"]["c1"] == plan.target_bindings["c1"].model_dump(mode="json")
        for row in checks
    )
    assert result.ledger[0]["alignment"][0]["paper_target_binding"] == plan.target_bindings["c1"].model_dump(
        mode="json"
    )


@pytest.mark.parametrize("returncode", [0, 1])
def test_paper_change_during_runner_retains_logs_without_evidence_or_repair(inputs, tmp_path, returncode):
    plan, claim, materials = inputs

    def runner(request):
        Path(materials.markdown_path).write_text("changed after launch", encoding="utf-8")
        return RunOutcome(returncode=returncode, stdout="actual output", observations=[observation()])

    repairer = Mock()
    result = execute_plans([plan], [claim], materials, tmp_path / "during", runner=runner, repairer=repairer)
    repairer.assert_not_called()
    assert not result.claims[0].evidence
    row = result.ledger[0]
    assert len(row["attempts"]) == 1 and not row["repairs"]
    assert row["attempts"][0]["observations"][0]["value"] == 0.9
    assert Path(row["attempts"][0]["logs"]["stdout"]).read_text(encoding="utf-8") == "actual output"
    assert row["paper_target_validation"][-1]["phase"] == "before_judge"
    assert not row["paper_target_validation"][-1]["verified"]


def test_runtime_unit_fields_must_agree_without_borrowing_paper_units(inputs, tmp_path):
    plan, claim, materials = inputs
    plan.y_paper["c1"] = 90
    claim.conditions[0].settings["unit"] = "percent"
    bind_fixture_sources(inputs, [("c1", "d1 test seed 1 accuracy is 90%.", "90%")])
    actual = observation(value=90, unit="seconds").model_copy(
        update={"settings": dict(claim.conditions[0].settings)}
    )
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "unit_conflict",
        runner=lambda _: RunOutcome(returncode=0, observations=[actual]),
    )
    assert not result.claims[0].evidence
    assert "Runtime unit fields conflict" in result.ledger[0]["reason"]


def test_approval_callback_cannot_make_a_stale_target_reach_preparation(inputs, tmp_path):
    plan, claim, materials = inputs

    def approver(*args):
        Path(materials.markdown_path).write_text("changed by callback", encoding="utf-8")
        return True

    runner = Mock()
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "approval",
        config={"approval_mode": "interactive"},
        approver=approver,
        runner=runner,
    )
    runner.assert_not_called()
    assert not result.ledger[0]["approved"]
    assert not (tmp_path / "approval/run_0000/workspace").exists()


def test_source_change_in_repair_cannot_launch_a_second_attempt(inputs, tmp_path):
    plan, claim, materials = inputs
    runner = Mock(
        return_value=RunOutcome(returncode=1, stderr="ModuleNotFoundError: No module named 'numpy'")
    )

    def repairer(*args):
        Path(materials.markdown_path).write_text("changed during repair", encoding="utf-8")
        return Repair(dependencies=["numpy"], reason="dependency")

    result = execute_plans([plan], [claim], materials, tmp_path / "repair", runner=runner, repairer=repairer)
    assert runner.call_count == 1
    assert not result.claims[0].evidence
    assert len(result.ledger[0]["attempts"]) == 1
    assert result.ledger[0]["paper_target_validation"][-1]["phase"] == "before_run"
    assert not result.ledger[0]["paper_target_validation"][-1]["verified"]


@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("percent", 90, True),
        ("%", 90, True),
        (None, 90, False),
        ("fraction", 0.9, False),
        ("seconds", 90, False),
    ],
)
def test_explicit_paper_unit_requires_independent_runtime_scale(inputs, tmp_path, unit, value, expected):
    plan, claim, materials = inputs
    plan.y_paper["c1"] = 90
    bind_fixture_sources(inputs, [("c1", "d1 test seed 1 accuracy is 90%.", "90%")])
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "unit",
        runner=lambda _: RunOutcome(returncode=0, observations=[observation(value=value, unit=unit)]),
    )
    assert any(item.sufficient for item in result.claims[0].evidence) is expected
    if not expected:
        assert not result.claims[0].evidence
        assert "comparison unavailable" in result.ledger[0]["reason"]
