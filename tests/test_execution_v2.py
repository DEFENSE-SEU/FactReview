"""Offline L3 behavior tests: all Docker/LLM transports are replaced at boundaries."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fact_generation.execution import v2
from fact_generation.execution.nodes.plan import _default_tolerance
from fact_generation.execution.v2 import ExecutionConfig, Observation, Repair, RunOutcome, execute_plans
from fact_generation.execution.v2_config import metric_tolerance
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from util.subprocess_runner import CommandResult


@pytest.fixture
def inputs(tmp_path):
    repo = tmp_path / "released"
    repo.mkdir()
    (repo / "eval.py").write_text("print('released evaluator')\n", encoding="utf-8")
    (repo / "model.py").write_text("model = 'author model'\n", encoding="utf-8")
    condition = Condition(id="c1", dataset="d1", metric="accuracy", settings={"split": "test", "seed": 1})
    claim = Claim(
        id="claim1",
        text="Accuracy is .9",
        loc=ClaimLocation(page=2),
        conditions=[condition],
        needs=["Experiments"],
    )
    plan = ExecutionPlan(
        id="p1",
        claim_id=claim.id,
        condition_ids=["c1"],
        target_conditions=[condition],
        task=ExecutionTask(entry_script="eval.py", command=["python", "eval.py"]),
        run_mode="evaluation",
        y_paper={"c1": 0.9},
        feasibility="ready",
        priority="high",
    )
    materials = SharedMaterials(
        paper_key="paper",
        source_pdf="paper.pdf",
        markdown="Accuracy .9",
        markdown_path="paper.md",
        content_list_path="content.json",
        provider="fixture",
        repository=index_repository(repo),
    )
    return plan, claim, materials


def observation(**changes):
    return Observation(
        dataset="d1", metric="accuracy", settings={"split": "test", "seed": 1}, **{"value": 0.9, **changes}
    )


def run(inputs, tmp_path, runner, **kwargs):
    plan, claim, materials = inputs
    return execute_plans([plan], [claim], materials, tmp_path / "out", runner=runner, **kwargs)


def approved_artifact(inputs, obs):
    contract = obs.released_recomputation.model_dump()
    indexed = next(item for item in inputs[2].repository.files if item.path == contract["artifact_path"])
    contract["artifact_sha256"] = indexed.sha256
    return {"author_artifacts": {inputs[0].id: [contract]}}


def test_aligned_support_retains_traceable_logs_and_preserves_inputs(inputs, tmp_path):
    original = inputs[1].model_dump()

    def runner(request):
        assert Path(request.workspace) != Path(inputs[2].repository.root)
        return RunOutcome(
            returncode=0, observations=[observation()], stdout="actual runtime output", tokens=7
        )

    result = run(inputs, tmp_path, runner)
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.direction == "support" and evidence.aligned
    assert Path(evidence.pointer.locator).is_file()
    row = result.ledger[0]
    assert row["approved"] and row["approval_mode"] == "auto" and row["tolerance_profile"] == "alignment"
    assert row["attempts"][0]["tokens"] == 7
    assert row["attempts"][0]["request"]["command"] == ["python", "eval.py"]
    assert inputs[1].model_dump() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "other"),
        ("metric", "f1"),
        ("settings", {"split": "test"}),
        ("settings", {"split": "test", "seed": True}),
    ],
)
def test_no_evidence_from_misaligned_actual_conditions(inputs, tmp_path, field, value):
    obs = observation().model_copy(update={field: value})
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]))
    assert not result.claims[0].evidence
    assert result.claims[0].questions
    assert result.ledger[0]["alignment"][0]["aligned"] is False


def test_missing_observed_conditions_never_copy_targets(inputs, tmp_path):
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0))
    assert not result.claims[0].evidence
    assert "align" in result.ledger[0]["reason"]


def test_aligned_discrepancy_is_resolvable_concern(inputs, tmp_path):
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[observation(value=0.2)]))
    evidence = result.claims[0].evidence[0]
    assert evidence.direction == "flaw" and evidence.concern and evidence.overturnable
    assert not evidence.sufficient
    assert evidence.provenance.environment_explanation_possible


def test_released_recomputation_requires_hash_and_real_metadata(inputs, tmp_path):
    _, _, materials = inputs
    inputs[1].conditions[0].settings["aggregation"] = "mean"
    inputs[0].target_conditions[0].settings["aggregation"] = "mean"
    path = Path(materials.repository.root) / "results.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "d1",
                "metric": "accuracy",
                "settings": {"split": "test", "seed": 1, "aggregation": "mean"},
                "scores": [0.1, 0.3],
            }
        ),
        encoding="utf-8",
    )
    materials.repository = index_repository(Path(materials.repository.root))
    obs = observation(
        value=0.2,
        released_recomputation={
            "artifact_path": "results.json",
            "artifact_kind": "logs",
            "values_key": "scores",
            "dataset_key": "dataset",
            "metric_key": "metric",
            "settings_key": "settings",
            "operation": "mean",
        },
    )
    obs.settings["aggregation"] = "mean"
    config = approved_artifact(inputs, obs)
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]), config=config)
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and not evidence.overturnable
    assert evidence.provenance.artifact_sha256
    assert Path(evidence.provenance.recomputation_pointer).is_file()
    assert not evidence.provenance.environment_explanation_possible


def test_missing_released_artifact_does_not_create_decisive_flaw(inputs, tmp_path):
    obs = observation(
        value=0.2,
        released_recomputation={
            "artifact_path": "missing.json",
            "artifact_kind": "logs",
            "values_key": "scores",
            "dataset_key": "dataset",
            "metric_key": "metric",
            "settings_key": "settings",
            "operation": "identity",
        },
    )
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]))
    assert not result.claims[0].evidence[0].sufficient
    assert result.issues and "proof rejected" in result.issues[0]


def test_three_repair_rounds_mean_four_runs_and_no_fifth(inputs, tmp_path):
    attempts = []

    def runner(request):
        attempts.append(request.repair_round)
        return RunOutcome(returncode=1, stderr="missing runtime dependency")

    def repairer(request, outcome):
        return Repair(dependencies=[f"dependency{request.repair_round}"], reason="dependency repair")

    result = run(inputs, tmp_path, runner, repairer=repairer)
    assert attempts == [0, 1, 2, 3]
    assert len(result.ledger[0]["repairs"]) == 3
    assert all(item["accepted"] and item["diff"] for item in result.ledger[0]["repairs"])
    assert not result.claims[0].evidence
    with pytest.raises(ValidationError):
        ExecutionConfig(max_attempts=4)


@pytest.mark.parametrize("flag", ["--model", "--loss", "--dataset", "--eval-split", "--baseline", "--seed"])
def test_repair_rejects_protected_method_arguments(inputs, tmp_path, flag):
    result = run(
        inputs,
        tmp_path,
        lambda _: RunOutcome(returncode=1),
        repairer=lambda *_: Repair(launch_arguments={flag: "changed"}, reason="try different method"),
    )
    assert len(result.ledger[0]["attempts"]) == 1
    assert not result.ledger[0]["repairs"][0]["accepted"]
    assert "protected" in result.ledger[0]["repairs"][0]["reason"]


def test_repair_direct_source_write_rejected_with_diff(inputs, tmp_path):
    def malicious(request, _):
        (Path(request.workspace) / "model.py").write_text("model = 'different'\n", encoding="utf-8")
        return Repair(dependencies=["numpy"], reason="dependency")

    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=1), repairer=malicious)
    record = result.ledger[0]["repairs"][0]
    assert not record["accepted"]
    assert "model.py" in record["unauthorized_file_diffs"]
    assert "author model" in (Path(inputs[2].repository.root) / "model.py").read_text()


def test_runtime_mutation_of_evaluator_cannot_produce_support(inputs, tmp_path):
    def mutate(request):
        (Path(request.workspace) / "eval.py").write_text("print(.9)", encoding="utf-8")
        return RunOutcome(returncode=0, observations=[observation()])

    result = run(inputs, tmp_path, mutate)
    assert not result.claims[0].evidence
    assert "protected released files" in result.ledger[0]["reason"]


@pytest.mark.parametrize(
    "mode,config",
    [
        ("training", {}),
        ("training", {"training_budget": 1}),
        ("evaluation", {"approval_mode": "interactive"}),
    ],
)
def test_denied_or_missing_operator_never_creates_workspace(inputs, tmp_path, mode, config):
    inputs[0].run_mode = mode
    inputs[0].priority = "medium"

    def forbidden(_):
        raise AssertionError("runner must not be called")

    result = run(inputs, tmp_path, forbidden, config=config)
    assert not result.ledger[0]["approved"]
    assert not (tmp_path / "out" / "run_0000" / "workspace").exists()


def test_training_budget_counts_retries(inputs, tmp_path):
    inputs[0].run_mode = "training"
    result = run(
        inputs,
        tmp_path,
        lambda _: RunOutcome(returncode=1),
        config={"training_budget": 1},
        repairer=lambda *_: Repair(dependencies=["numpy"], reason="missing dependency"),
    )
    assert len(result.ledger[0]["attempts"]) == 1
    assert not result.ledger[0]["repairs"]


def test_interactive_approval_records_mode_and_estimate(inputs, tmp_path):
    seen = []
    result = run(
        inputs,
        tmp_path,
        lambda _: RunOutcome(returncode=0, observations=[observation()]),
        config={"approval_mode": "interactive"},
        approver=lambda plan, cost: seen.append((plan.id, cost)) or True,
    )
    assert seen == [("p1", "unknown")]
    assert result.ledger[0]["approved"] and result.ledger[0]["approval_mode"] == "interactive"


def test_blocked_kept_and_order_is_priority_then_ready_then_mode(inputs, tmp_path):
    plan, claim, materials = inputs
    plans = [
        plan.model_copy(update={"id": "low", "priority": "low"}),
        plan.model_copy(update={"id": "blocked", "feasibility": "blocked", "blocker": "no weights"}),
        plan.model_copy(update={"id": "train", "run_mode": "training"}),
        plan.model_copy(update={"id": "eval"}),
    ]
    result = execute_plans(
        plans,
        [claim],
        materials,
        tmp_path / "out",
        config={"training_budget": 1},
        runner=lambda _: RunOutcome(returncode=0, observations=[observation()]),
    )
    assert [item["plan"]["id"] for item in result.ledger] == ["eval", "train", "blocked", "low"]
    assert result.ledger[2]["reason"] == "no weights"


def test_default_runner_uses_mocked_docker_transport_and_actual_stdout(inputs, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "test-image:1"))

    def docker_command(**kwargs):
        seen.append(kwargs)
        return ["docker", "run", "test-image:1", *kwargs["cmd"]]

    monkeypatch.setattr(v2, "docker_run_paper_image", docker_command)

    def transport(command, cwd, timeout_sec):
        assert command[:2] == ["docker", "run"]
        payload = {"observations": [observation().model_dump(mode="json")]}
        return CommandResult(command, cwd, 0, "FACTREVIEW_OBSERVATIONS=" + json.dumps(payload), "", 0.2)

    monkeypatch.setattr(v2, "run_command", transport)
    result = run(inputs, tmp_path, None)
    assert result.claims[0].evidence[0].sufficient
    assert seen[0]["env_passthrough"] == []
    assert result.ledger[0]["attempts"][0]["environment"]["image"] == "test-image:1"


def test_preserved_named_tolerance_profiles():
    assert _default_tolerance("mrr", 0.5) == 0.02
    assert metric_tolerance("mrr", 0.5) == 0.01
    assert _default_tolerance("accuracy", 90) == 2
    assert _default_tolerance("other", 10) == 0.5
    assert metric_tolerance("loss", 10) == 0.05
    assert metric_tolerance("mr", 100) == 30


def test_plan_cannot_change_claim_conditions(inputs, tmp_path):
    inputs[0].target_conditions[0] = inputs[0].target_conditions[0].model_copy(update={"dataset": "other"})
    with pytest.raises(ValueError, match="conditions differ"):
        run(inputs, tmp_path, lambda _: RunOutcome(returncode=0))


def test_subdirectory_entry_keeps_index_path_and_runtime_workdir_distinct(inputs, tmp_path):
    plan, _, materials = inputs
    root = Path(materials.repository.root)
    (root / "scripts").mkdir()
    (root / "scripts" / "eval.py").write_text("print('eval')\n", encoding="utf-8")
    materials.repository = index_repository(root)
    plan.task.entry_script = "scripts/eval.py"
    plan.task.workdir = "scripts"
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[observation()]))
    assert result.claims[0].evidence[0].sufficient


def test_repair_wrapper_forwards_original_command_and_records_file_diff(inputs, tmp_path):
    def runner(request):
        if request.repair_round == 0:
            return RunOutcome(returncode=1)
        wrapper = Path(request.workspace) / request.command[1]
        assert "['python', 'eval.py']" in wrapper.read_text()
        assert "subprocess.call" in wrapper.read_text()
        return RunOutcome(returncode=0, observations=[observation()])

    result = run(inputs, tmp_path, runner, repairer=lambda *_: Repair(wrapper=True, reason="launch wrapper"))
    assert result.claims[0].evidence[0].sufficient
    repair = result.ledger[0]["repairs"][0]
    assert repair["accepted"] and repair["file_diffs"][".factreview/wrapper_1.py"]


@pytest.mark.parametrize("override", [-0.1, float("nan"), float("inf")])
def test_invalid_tolerance_config_rejected(override):
    with pytest.raises(ValidationError):
        ExecutionConfig(tolerance_overrides={"accuracy": override})


def test_llm_refinement_uses_mock_and_existing_entry(inputs, tmp_path, monkeypatch):
    plan, _, materials = inputs
    root = Path(materials.repository.root)
    (root / "config.json").write_text('{"seed": 1}', encoding="utf-8")
    materials.repository = index_repository(root)
    plan.task.command = []
    plan.task.config = "config.json"
    seen = []

    def llm(prompt, system, cfg, module):
        seen.append(json.loads(prompt))
        return {"command": ["python", "eval.py", "--config", "config.json"], "metric_output": None}

    monkeypatch.setattr("llm.client.llm_json", llm)
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[observation()]))
    assert result.claims[0].evidence[0].sufficient
    assert "config.json" in seen[0]["files"]
    assert result.ledger[0]["refinement"]["mode"] == "llm"


def test_llm_cannot_replace_script_with_inline_evaluation(inputs, tmp_path, monkeypatch):
    inputs[0].task.command = []
    inputs[0].task.config = "config.json"
    monkeypatch.setattr(
        "llm.client.llm_json", lambda *args, **kwargs: {"command": ["python", "-c", "print(.9)"]}
    )
    called = []
    result = run(inputs, tmp_path, lambda request: called.append(request) or RunOutcome(returncode=0))
    assert not called and not result.claims[0].evidence
    assert "released script" in result.ledger[0]["reason"]


def test_metadata_claimed_by_runtime_does_not_widen_paper_tolerance(inputs, tmp_path):
    result = run(
        inputs,
        tmp_path,
        lambda _: RunOutcome(returncode=0, observations=[observation(value=0.2, reported_variance=1)]),
    )
    assert not result.claims[0].evidence[0].sufficient
    assert result.ledger[0]["alignment"][0]["tolerance"] == 0.02


def test_released_metadata_mismatch_never_becomes_decisive(inputs, tmp_path):
    materials = inputs[2]
    root = Path(materials.repository.root)
    (root / "results.json").write_text(
        json.dumps(
            {
                "dataset": "another-dataset",
                "metric": "accuracy",
                "settings": {"split": "test", "seed": 1},
                "score": 0.2,
            }
        ),
        encoding="utf-8",
    )
    materials.repository = index_repository(root)
    obs = observation(
        value=0.2,
        released_recomputation={
            "artifact_path": "results.json",
            "artifact_kind": "logs",
            "values_key": "score",
            "dataset_key": "dataset",
            "metric_key": "metric",
            "settings_key": "settings",
            "operation": "identity",
        },
    )
    config = approved_artifact(inputs, obs)
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]), config=config)
    assert not result.claims[0].evidence[0].sufficient
    assert "metadata differs" in result.issues[0]


def test_partial_matching_conditions_support_only_covered_subset(inputs, tmp_path):
    plan, claim, _ = inputs
    second = claim.conditions[0].model_copy(update={"id": "c2", "dataset": "d2"})
    claim.conditions.append(second)
    plan.target_conditions.append(second)
    plan.condition_ids.append("c2")
    plan.y_paper["c2"] = 0.8
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[observation()]))
    assert result.claims[0].evidence[0].covered == ["c1"]
    assert len(result.claims[0].evidence) == 1
    assert "every requested" in result.ledger[0]["reason"]


def test_default_dependency_repair_uses_known_missing_module(inputs, tmp_path):
    def runner(request):
        if "numpy" in request.dependencies:
            return RunOutcome(returncode=0, observations=[observation()])
        return RunOutcome(returncode=1, stderr="ModuleNotFoundError: No module named 'numpy'")

    result = run(inputs, tmp_path, runner)
    assert result.claims[0].evidence[0].sufficient
    assert result.ledger[0]["repairs"][0]["proposal"]["dependencies"] == ["numpy"]


def test_stale_metric_file_cannot_support_new_execution(inputs, tmp_path, monkeypatch):
    plan, _, materials = inputs
    root = Path(materials.repository.root)
    (root / "metrics.json").write_text(
        json.dumps({"observations": [observation().model_dump()]}), encoding="utf-8"
    )
    materials.repository = index_repository(root)
    plan.task.metric_output = "metrics.json"
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    monkeypatch.setattr(v2, "docker_run_paper_image", lambda **kwargs: ["docker", "run", "fixture-image"])
    monkeypatch.setattr(
        v2, "run_command", lambda cmd, cwd, timeout_sec: CommandResult(cmd, cwd, 0, "", "", 0)
    )
    result = run(inputs, tmp_path, None)
    assert not result.claims[0].evidence
    assert "not refreshed" in result.ledger[0]["attempts"][0]["issue"]


def test_compgcn_author_json_decoded_by_default_docker_path(inputs, tmp_path, monkeypatch):
    plan, claim, _ = inputs
    condition = Condition(
        id="c1",
        dataset="FB15k-237",
        metric="MRR",
        settings={"split": "test", "score_func": "transe", "opn": "sub"},
    )
    plan.target_conditions = [condition]
    claim.conditions = [condition]
    plan.y_paper = {"c1": 0.335}
    plan.task.command += ["--out", "metrics/result.json"]
    author_json = {
        "ok": True,
        "split": "test",
        "dataset": "FB15k-237",
        "score_func": "transe",
        "opn": "sub",
        "ckpt": "/app/checkpoints/author",
        "mrr": 0.335,
        "mr": 194,
        "hits@1": 0.245,
        "hits@3": 0.365,
        "hits@10": 0.514,
    }
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    monkeypatch.setattr(v2, "docker_run_paper_image", lambda **kwargs: ["docker", "run", "fixture-image"])

    def transport(cmd, cwd, timeout_sec):
        target = Path(cwd) / "workspace" / "metrics" / "result.json"
        target.parent.mkdir()
        target.write_text(json.dumps(author_json), encoding="utf-8")
        return CommandResult(cmd, cwd, 0, '{"ok": true, "out": "metrics/result.json"}', "", 0.2)

    monkeypatch.setattr(v2, "run_command", transport)
    result = run(inputs, tmp_path, None)
    assert result.claims[0].evidence[0].sufficient
    assert result.claims[0].evidence[0].provenance.runtime_conditions[0].settings == condition.settings
    attempt = result.ledger[0]["attempts"][0]
    raw = json.loads(Path(attempt["logs"]["raw_output"]).read_text())
    audit = json.loads(Path(attempt["logs"]["output_mapping"]).read_text())
    assert raw == author_json and any(item["metric_path"] == ["mrr"] for item in audit["selectors"])


def test_explicit_nested_output_mapping_reads_only_runtime_values():
    from fact_generation.execution.v2_config import OutputMapping
    from fact_generation.execution.v2_outputs import decode_output

    payload = {
        "evaluation": {
            "data": "actual-dataset",
            "configuration": {"seed": 7, "split": "valid"},
            "scores": {"accuracy": 0.7},
        }
    }
    mapping = OutputMapping(
        root_path=["evaluation"],
        dataset_path=["data"],
        settings_path=["configuration"],
        metric_paths={"accuracy": ["scores", "accuracy"]},
    )
    observed, audit = decode_output(payload, mapping)
    assert observed == [
        {
            "dataset": "actual-dataset",
            "metric": "accuracy",
            "settings": {"seed": 7, "split": "valid"},
            "value": 0.7,
        }
    ]
    assert audit[0]["root_path"] == ["evaluation"]
    del payload["evaluation"]["data"]
    with pytest.raises(ValueError, match="dataset is missing"):
        decode_output(payload, mapping)


def test_normal_nested_metrics_retain_observed_seed_without_plan_values():
    from fact_generation.execution.v2_outputs import decode_output

    observed, _ = decode_output(
        {"dataset": "actual", "split": "test", "seed": 7, "metrics": {"accuracy": 0.8, "f1": 0.6}}
    )
    assert [item["metric"] for item in observed] == ["accuracy", "f1"]
    assert observed[0]["settings"] == {"split": "test", "seed": 7}
    with pytest.raises(ValueError, match="settings"):
        decode_output({"dataset": "actual", "accuracy": 0.8})


@pytest.mark.parametrize("variance", [float("inf"), float("nan"), -0.1, "large"])
def test_invalid_paper_variance_never_expands_tolerance(inputs, tmp_path, variance):
    inputs[0].target_conditions[0].settings["reported_variance"] = variance
    inputs[1].conditions[0].settings["reported_variance"] = variance
    obs = observation(value=0.2)
    obs.settings["reported_variance"] = variance
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]))
    assert not result.claims[0].evidence
    assert "comparison unavailable" in result.ledger[0]["reason"]
    assert result.ledger[0]["alignment"][0]["comparable"] is False


def test_finite_variance_without_paper_pointer_does_not_expand_tolerance(inputs, tmp_path):
    inputs[0].target_conditions[0].settings["reported_variance"] = 1.0
    inputs[1].conditions[0].settings["reported_variance"] = 1.0
    obs = observation(value=0.2)
    obs.settings["reported_variance"] = 1.0
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]))
    assert not result.claims[0].evidence[0].sufficient
    assert result.ledger[0]["alignment"][0]["tolerance"] == 0.02
    assert "no verified paper pointer" in result.ledger[0]["alignment"][0]["variance_source"]["reason"]


def test_grounded_paper_statistic_can_expand_tolerance(inputs, tmp_path):
    inputs[2].blocks.append(
        MaterialBlock(id="variance", text="Accuracy standard deviation is 0.04.", loc=ClaimLocation(page=2))
    )
    config = {
        "paper_variances": {
            "p1": {
                "c1": {"value": 0.04, "block_id": "variance", "quote": "Accuracy standard deviation is 0.04."}
            }
        }
    }
    result = run(
        inputs,
        tmp_path,
        lambda _: RunOutcome(returncode=0, observations=[observation(value=0.87)]),
        config=config,
    )
    assert result.claims[0].evidence[0].sufficient
    assert result.ledger[0]["alignment"][0]["variance_source"]["verified"]
    assert result.ledger[0]["alignment"][0]["variance_source"]["binding_mode"] == "operator_confirmed"
    assert result.ledger[0]["alignment"][0]["tolerance"] == 0.04


def test_runtime_cannot_choose_another_released_numeric_column(inputs, tmp_path):
    materials = inputs[2]
    root = Path(materials.repository.root)
    (root / "results.json").write_text(
        json.dumps(
            {
                "dataset": "d1",
                "metric": "accuracy",
                "settings": {"seed": 1, "split": "test"},
                "score": 0.9,
                "threshold": 0.2,
            }
        ),
        encoding="utf-8",
    )
    materials.repository = index_repository(root)
    obs = observation(
        value=0.2,
        released_recomputation={
            "artifact_path": "results.json",
            "artifact_kind": "logs",
            "values_key": "score",
            "dataset_key": "dataset",
            "metric_key": "metric",
            "settings_key": "settings",
            "operation": "identity",
        },
    )
    config = approved_artifact(inputs, obs)
    obs.released_recomputation.values_key = "threshold"
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=0, observations=[obs]), config=config)
    assert not result.claims[0].evidence[0].sufficient
    assert result.claims[0].evidence[0].overturnable
    assert "pre-run approved" in result.issues[0]


def test_blocked_plan_has_explanation_evidence_without_status_effect(inputs, tmp_path):
    inputs[0].feasibility = "blocked"
    inputs[0].blocker = "released weights missing"
    result = run(inputs, tmp_path, lambda _: pytest.fail("blocked plan executed"))
    evidence = result.claims[0].evidence[0]
    assert not evidence.sufficient and not evidence.affects_claim and not evidence.aligned
    assert not evidence.concern and evidence.note == "released weights missing"
    assert json.loads(Path(evidence.pointer.locator).read_text())[evidence.pointer.key] == evidence.note


@pytest.mark.parametrize(
    "name", ["expected_metrics", "target", "paper", "y_paper", "ExpectedMetrics", "target_metrics"]
)
def test_output_mapping_cannot_select_paper_targets(name):
    from fact_generation.execution.v2_config import OutputMapping
    from fact_generation.execution.v2_outputs import decode_output

    payload = {
        "dataset": "d1",
        "settings": {"split": "test"},
        "metrics": {"accuracy": 0.2},
        name: {"accuracy": 0.9},
    }
    mapping = OutputMapping(metric_paths={"accuracy": [name, "accuracy"]})
    with pytest.raises(ValueError, match="expected/target/paper"):
        decode_output(payload, mapping)


def test_output_mapping_cannot_select_paper_root_or_paper_settings():
    from fact_generation.execution.v2_config import OutputMapping
    from fact_generation.execution.v2_outputs import decode_output

    actual = {"dataset": "d1", "settings": {"split": "test"}, "metrics": {"accuracy": 0.2}}
    with pytest.raises(ValueError, match="expected/target/paper"):
        decode_output(
            {"paper": actual},
            OutputMapping(root_path=["paper"], metric_paths={"accuracy": ["metrics", "accuracy"]}),
        )
    with pytest.raises(ValueError, match="expected/target/paper"):
        decode_output(
            {**actual, "expected_settings": {"split": "test"}},
            OutputMapping(
                settings_path=["expected_settings"], metric_paths={"accuracy": ["metrics", "accuracy"]}
            ),
        )


def test_output_mapping_conflict_with_actual_metric_is_rejected():
    from fact_generation.execution.v2_config import OutputMapping
    from fact_generation.execution.v2_outputs import decode_output

    payload = {
        "dataset": "d1",
        "settings": {"split": "test"},
        "metrics": {"accuracy": 0.2},
        "other": {"accuracy": 0.9},
    }
    with pytest.raises(ValueError, match="conflicts with an identified actual metric"):
        decode_output(payload, OutputMapping(metric_paths={"accuracy": ["other", "accuracy"]}))
    decoded, _ = decode_output({**payload, "expected_metrics": {"accuracy": 0.9}})
    assert decoded[0]["value"] == 0.2


@pytest.mark.parametrize("flag", ["ok", "success"])
def test_failure_markers_cover_canonical_root_nested_root_and_observations(flag):
    from fact_generation.execution.v2_config import OutputMapping
    from fact_generation.execution.v2_outputs import decode_output

    canonical = {"observations": [observation().model_dump()]}
    with pytest.raises(ValueError, match="reports failure"):
        decode_output({flag: False, **canonical})
    canonical["observations"][0][flag] = False
    with pytest.raises(ValueError, match="reports failure"):
        decode_output(canonical)
    mapping = OutputMapping(root_path=["result"], metric_paths={"accuracy": ["metrics", "accuracy"]})
    with pytest.raises(ValueError, match="reports failure"):
        decode_output(
            {"result": {flag: False, "dataset": "d1", "settings": {}, "metrics": {"accuracy": 0.9}}}, mapping
        )
    with pytest.raises(ValueError, match="reports failure"):
        decode_output(
            {flag: False, "result": {"dataset": "d1", "settings": {}, "metrics": {"accuracy": 0.9}}}, mapping
        )


def test_canonical_observation_conflict_with_actual_metric_is_rejected():
    from fact_generation.execution.v2_outputs import decode_output

    with pytest.raises(ValueError, match="conflicts with an identified actual metric"):
        decode_output({"metrics": {"accuracy": 0.2}, "observations": [observation(value=0.9).model_dump()]})
