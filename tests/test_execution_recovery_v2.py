"""Partial-run history is recovered without accepting arbitrary or unfinished ledgers."""

import json

import pytest

from fact_generation.execution import v2
from fact_generation.execution.recovery import recover_execution_records
from fact_generation.execution.v2_config import ExecutionConfig
from schemas.claim import Condition, ExecutionPlan
from tests.test_execution_v2 import inputs as execution_inputs
from tests.test_execution_v2 import observation


def plan(identifier="p1"):
    condition = Condition(id="c1", dataset="d1", metric="accuracy", settings={"split": "test"})
    return ExecutionPlan(
        id=identifier,
        claim_id="claim1",
        condition_ids=["c1"],
        target_conditions=[condition],
        run_mode="evaluation",
        y_paper={"c1": 0.9},
        feasibility="blocked",
        blocker="No released weights",
        priority="high",
    )


def row(source):
    return {
        "plan": source.model_dump(mode="json"),
        "approval_mode": "auto",
        "training_budget": 0,
        "training_used_before": 0,
        "approved": False,
        "reason": source.blocker,
        "attempts": [],
        "repairs": [],
        "alignment": [],
        "paper_target_validation": [],
        "tolerance_profile": "alignment",
        "config": ExecutionConfig().model_dump(mode="json"),
    }


def save(root, number, record):
    path = root / f"run_{number:04d}" / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_second_plan_top_level_failure_preserves_first_history_without_adopting_evidence(
    tmp_path, monkeypatch
):
    original, claim, materials = execution_inputs.__wrapped__(tmp_path)
    second = original.model_copy(update={"id": "p2"}, deep=True)
    before = claim.model_dump(mode="json")
    graph = v2._execute_graph

    def fail_second(request, *args):
        if request.plan.id == second.id:
            started = tmp_path / "out" / "run_0001" / "attempt_0"
            started.mkdir()
            (started / "started.log").write_text("Attempt started before the stage failed", encoding="utf-8")
            raise LookupError("synthetic top-level stage failure")
        return graph(request, *args)

    monkeypatch.setattr(v2, "_execute_graph", fail_second)
    with pytest.raises(LookupError, match="top-level"):
        v2.execute_plans(
            [original, second],
            [claim],
            materials,
            tmp_path / "out",
            config=ExecutionConfig(refine_with_llm=False, max_attempts=0),
            runner=lambda _: v2.RunOutcome(returncode=0, observations=[observation()]),
        )
    path = tmp_path / "out" / "run_0000" / "ledger.json"
    bytes_before = path.read_bytes()
    ledger, issues = recover_execution_records(tmp_path / "out", [original, second])
    assert ledger == [json.loads(bytes_before)]
    assert ledger[0]["attempts"][0]["observations"][0]["value"] == 0.9
    assert claim.model_dump(mode="json") == before and claim.evidence == []
    assert path.read_bytes() == bytes_before
    assert any("p2" in issue and "unknown" in issue for issue in issues)
    assert any("without their evidence being adopted" in issue for issue in issues)


@pytest.mark.parametrize(
    "mutation", ["unknown", "changed_plan", "missing_field", "bool_count", "bad_attempt", "bad_config"]
)
def test_bad_row_excluded_while_healthy_neighbor_survives(tmp_path, mutation):
    bad, good = plan("bad"), plan("good")
    record = row(bad)
    if mutation == "unknown":
        record["plan"]["id"] = "other"
    elif mutation == "changed_plan":
        record["plan"]["y_paper"]["c1"] = 0.1
    elif mutation == "missing_field":
        del record["attempts"]
    elif mutation == "bool_count":
        record["training_budget"] = False
    elif mutation == "bad_attempt":
        record["attempts"] = [{"returncode": 0}]
    elif mutation == "bad_config":
        record["config"]["training_budget"] = "0"
    save(tmp_path, 0, record)
    save(tmp_path, 1, row(good))
    ledger, issues = recover_execution_records(tmp_path, [bad, good])
    assert ledger == [row(good)] and any("bad" in issue for issue in issues)


@pytest.mark.parametrize("malformed_duplicate", [False, True])
def test_duplicate_ids_permanently_revoke_first_healthy_record(tmp_path, malformed_duplicate):
    p, neighbor = plan(), plan("p2")
    save(tmp_path, 0, row(p))
    duplicate = row(p)
    if malformed_duplicate:
        duplicate["plan"]["id"] = " p1 "
        del duplicate["approved"]
    save(tmp_path, 1, duplicate)
    save(tmp_path, 2, row(p))
    save(tmp_path, 3, row(neighbor))
    ledger, issues = recover_execution_records(tmp_path, [p, neighbor])
    assert ledger == [row(neighbor)]
    assert any("duplicate records" in issue for issue in issues)


@pytest.mark.parametrize("content", ['{"plan":', '{"plan":null}', '{"plan":{},"plan":{}}', "NaN"])
def test_invalid_json_never_aborts_healthy_recovery(tmp_path, content):
    p = plan()
    path = save(tmp_path, 0, row(p))
    path.write_text(content, encoding="utf-8")
    save(tmp_path, 1, row(p))
    ledger, issues = recover_execution_records(tmp_path, [p])
    assert ledger == [row(p)] and any("run_0000" in issue for issue in issues)


def test_linked_and_nested_files_and_aggregate_are_never_adopted(tmp_path):
    p = plan()
    outside = tmp_path / "other"
    save(outside, 0, row(p))
    output = tmp_path / "out"
    output.mkdir()
    (output / "run_0000").symlink_to(outside / "run_0000", target_is_directory=True)
    save(output / "runtime_scratch", 1, row(p))
    (output / "ledger.json").write_text(json.dumps([row(p)]), encoding="utf-8")
    ledger, issues = recover_execution_records(output, [p])
    assert ledger == [] and any("unknown" in issue for issue in issues)


def test_duplicate_expected_plans_are_not_bound_and_absent_root_is_explicit(tmp_path):
    p = plan()
    save(tmp_path, 0, row(p))
    ledger, issues = recover_execution_records(tmp_path, [p, p.model_copy(deep=True)])
    assert ledger == [] and any("duplicate input" in issue for issue in issues)
    ledger, issues = recover_execution_records(tmp_path / "missing", [p])
    assert ledger == [] and any("whether attempts executed is unknown" in issue for issue in issues)


@pytest.mark.parametrize(
    "mutation", ["foreign_run", "request_plan", "missing_outcome", "bool_returncode", "request_config"]
)
def test_changed_attempt_is_excluded_without_losing_healthy_neighbor(tmp_path, mutation):
    p, claim, materials = execution_inputs.__wrapped__(tmp_path)
    output = tmp_path / "out"
    v2.execute_plans(
        [p],
        [claim],
        materials,
        output,
        config=ExecutionConfig(refine_with_llm=False, max_attempts=0),
        runner=lambda _: v2.RunOutcome(returncode=0, observations=[observation()]),
    )
    path = output / "run_0000" / "ledger.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    attempt = record["attempts"][0]
    if mutation == "foreign_run":
        attempt["request"]["run_dir"] = str(tmp_path / "different_run")
    elif mutation == "request_plan":
        attempt["request"]["plan"]["y_paper"]["c1"] = 0.1
    elif mutation == "missing_outcome":
        del attempt["observations"]
    elif mutation == "bool_returncode":
        attempt["returncode"] = False
    elif mutation == "request_config":
        attempt["request"]["config"]["training_budget"] = 1
    save(output, 0, record)
    good = plan("good")
    save(output, 1, row(good))
    original = path.read_bytes()
    ledger, issues = recover_execution_records(output, [p, good])
    assert ledger == [row(good)] and any("p1" in issue and "unknown" in issue for issue in issues)
    assert path.read_bytes() == original


def test_linked_ledger_file_is_excluded(tmp_path):
    p = plan()
    external = save(tmp_path / "other", 0, row(p))
    output = tmp_path / "out"
    (output / "run_0000").mkdir(parents=True)
    (output / "run_0000" / "ledger.json").symlink_to(external)
    ledger, issues = recover_execution_records(output, [p])
    assert ledger == [] and any("run_0000" in issue for issue in issues)
