"""Execution resource intake identity; all execution boundaries stay mocked."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from fact_generation.execution import v2
from fact_generation.execution.recovery import recover_execution_records
from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from tests import test_execution_v2 as original_fixtures

original_inputs = original_fixtures.inputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External operation was not mocked")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    monkeypatch.setattr(v2, "run_command", forbidden)


def _contract(plan, claim, materials):
    return build_resource_contract(
        claim, materials, condition_ids=plan.condition_ids, entry_script=plan.task.entry_script,
        config=plan.task.config, data_paths=plan.task.data_paths, weight_paths=plan.task.weight_paths,
    )


@pytest.fixture
def selected(original_inputs):
    plan, claim, materials = original_inputs
    root = Path(materials.repository.root)
    for name, value in {"data.json": "[1,2]", "other.json": "[3,4]", "weights.bin": "fixture weights"}.items():
        (root / name).write_text(value, encoding="utf-8")
    materials.repository = index_repository(root)
    plan.task.data_paths, plan.task.weight_paths = ["data.json"], ["weights.bin"]
    plan.task.resource_contract = _contract(plan, claim, materials)
    return plan, claim, materials


def test_resealed_task_replacement_cannot_replace_approved_origin(selected, tmp_path):
    plan, claim, materials = selected
    runner = Mock(side_effect=AssertionError("Altered resources must not run"))

    def approve(copy, cost):
        plan.task.data_paths = ["other.json"]
        plan.task.resource_contract = _contract(plan, claim, materials)
        return True

    result = v2.execute_plans(
        [plan], [claim], materials, tmp_path / "out", config={"approval_mode": "interactive"},
        approver=approve, runner=runner,
    )
    assert not runner.called and result.ledger[0]["attempts"] == []
    row = result.ledger[0]
    assert not row["approved"] and row["plan"]["task"]["data_paths"] == ["data.json"]
    assert row["resource_validation"][-1]["point"] == "after_approval"
    assert row["resource_validation"][-1]["state"] == "invalid"
    assert "original" in row["reason"].lower() and not result.claims[0].evidence
    assert result.delivery_checks == []


def test_selected_identity_is_checked_at_run_and_judge_without_consumption_claim(selected, tmp_path):
    plan, claim, materials = selected
    result = v2.execute_plans(
        [plan], [claim], materials, tmp_path / "healthy", runner=lambda req: v2.RunOutcome(returncode=0),
    )
    row = result.ledger[0]
    assert [(item["point"], item["state"]) for item in row["resource_validation"]] == [
        ("before_approval", "bound"), ("before_run", "bound"), ("before_judge", "bound"),
    ]
    assert len(row["resource_origin_sha256"]) == 64 and not result.claims[0].evidence
    assert row["resource_validation"][0]["reason"] == "Candidate identity only; runtime consumption remains unverified"

    def mutate_index(req):
        materials.repository.entry_scripts.append("other.json")
        return v2.RunOutcome(returncode=0, observations=[original_fixtures.observation()])

    changed = v2.execute_plans([plan], [claim], materials, tmp_path / "changed", runner=mutate_index)
    assert changed.ledger[0]["resource_validation"][-1]["point"] == "before_judge"
    assert changed.ledger[0]["resource_validation"][-1]["state"] == "invalid"
    assert not changed.claims[0].evidence and changed.delivery_checks == []
    assert claim.evidence == []


def test_approval_cannot_replace_a_later_plans_original_resources(selected, tmp_path):
    first, claim, materials = selected
    later = first.model_copy(deep=True)
    later.id = "p2"

    def approve(copy, cost):
        if copy.id == first.id:
            later.task.data_paths = ["other.json"]
            later.task.resource_contract = _contract(later, claim, materials)
        return True

    runner = Mock(side_effect=lambda request: v2.RunOutcome(returncode=0))
    result = v2.execute_plans(
        [first, later], [claim], materials, tmp_path / "out",
        config={"approval_mode": "interactive", "max_attempts": 0}, approver=approve, runner=runner,
    )
    assert [call.args[0].plan.id for call in runner.call_args_list] == [first.id]
    assert result.ledger[1]["plan"]["task"]["data_paths"] == ["data.json"]
    assert not result.ledger[1]["approved"] and result.ledger[1]["attempts"] == []
    assert result.ledger[1]["resource_validation"][-1]["state"] == "invalid"
    assert not result.claims[0].evidence


def test_history_migrates_only_additive_empty_fields_and_keeps_original_record(original_inputs, tmp_path):
    plan, claim, materials = original_inputs
    output = tmp_path / "history"
    result = v2.execute_plans([plan], [claim], materials, output, runner=lambda req: v2.RunOutcome(returncode=1))
    original = json.loads(json.dumps(result.ledger[0]))
    for serialized in [original["plan"], original["attempts"][0]["request"]["plan"]]:
        for key in ("data_paths", "weight_paths", "resource_contract"):
            serialized["task"].pop(key)
    original.pop("resource_validation", None)
    original.pop("resource_origin_sha256", None)
    ledger = output / "run_0000" / "ledger.json"
    ledger.write_text(json.dumps(original), "utf-8")
    recovered, issues = recover_execution_records(output, [plan])
    assert recovered == [original] and "audit only" in issues[-1]
    assert "resource_contract" not in recovered[0]["plan"]["task"]
    bad = json.loads(json.dumps(original))
    bad["plan"]["task"]["foreign_resource_field"] = True
    ledger.write_text(json.dumps(bad), "utf-8")
    assert recover_execution_records(output, [plan])[0] == []
    bad = json.loads(json.dumps(original))
    bad["resource_origin_sha256"] = "not_a_digest"
    ledger.write_text(json.dumps(bad), "utf-8")
    assert recover_execution_records(output, [plan])[0] == []
    assert claim.evidence == []
