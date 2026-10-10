"""Complete released partitions are measured without granting scientific evidence."""
import copy
import json
import shutil

import pytest

from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_targets import bind_execution_target


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Partition analysis controls forbid services and processes")
    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)


def inputs(tmp_path, metric="mse", *, boolean_label=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "eval.py").write_text("print('released analysis entry')\n", encoding="utf-8")
    rows = [{"label": 1, "prediction": 2}, {"label": 1, "prediction": 2}, {"label": 2, "prediction": 2}]
    if boolean_label:
        rows[0]["label"] = True
    artifact = {"dataset": "Benchmark", "model": "ReleasedModel", "metric": metric,
                "partitions": {"test": rows, "train": copy.deepcopy(rows)}}
    (repo / "data.json").write_text(json.dumps(artifact), encoding="utf-8")
    definition = ("Benchmark test contains three released records, including two repeated records. "
                  "ReleasedModel predictions and labels are provided for all test records. "
                  "mse is the mean squared prediction error; accuracy is exact-match fraction. "
                  "This assertion concerns released statistics and makes no new inference claim.")
    (repo / "README.md").write_text(definition, encoding="utf-8")
    value = "0.6666666666666666" if metric == "mse" else "0.3333333333333333"
    quote = f"On Benchmark test sample_count 3, model ReleasedModel reports {metric} {value}."
    paper = quote + "\n" + definition
    md = tmp_path / "paper.md"
    md.write_text(paper, encoding="utf-8")
    condition = Condition(id="c1", dataset="Benchmark", metric=metric,
                          settings={"model": "ReleasedModel", "split": "test", "sample_count": 3})
    claim = Claim(id="c", text=quote, source_block_id="b", source_quote=quote,
                  loc=ClaimLocation(page=1, char_start=0, char_end=len(quote)), conditions=[condition],
                  needs=["Experiments"], source_refs=[ClaimSourceRef(source_block_id="b", source_quote=definition,
                  loc=ClaimLocation(page=1, char_start=len(quote)+1, char_end=len(paper)), covered=["c1"])])
    materials = SharedMaterials(paper_key="p", source_pdf="paper.pdf", markdown=paper, markdown_path=str(md),
        content_list_path="content.json", provider="fixture", blocks=[MaterialBlock(id="b", text=paper,
        loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))], repository=index_repository(repo))
    task = ExecutionTask(entry_script="eval.py", command=["python", "eval.py"], data_paths=["data.json"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"],
        entry_script=task.entry_script, config=None, data_paths=task.data_paths, weight_paths=[])
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        run_mode="analysis", y_paper={"c1": float(value)}, feasibility="ready", priority="high",
        target_bindings={"c1": bind_execution_target(claim, condition,
            {"block_id": "b", "quote": quote, "token": value}, materials)})
    roles = ("dataset", "split", "model", "metric", "population", "qualifiers")
    proposal = {"version": "released-partition-analysis-v1", "artifact_path": "data.json",
        "partition_selector": ["partitions", "test"], "label_selector": ["label"], "prediction_selector": ["prediction"],
        "dataset_selector": ["dataset"], "model_selector": ["model"], "metric_selector": ["metric"],
        "metric_definition": "mean_squared_error" if metric == "mse" else "exact_match_fraction",
        "paper": {role: {"block_id": "b", "quote": definition, "start": len(quote)+1, "end": len(paper)} for role in roles},
        "sources": {role: {"path": "README.md", "quote": definition, "start": 0, "end": len(definition)} for role in roles}}
    workspace = tmp_path / "workspace"
    shutil.copytree(repo, workspace)
    return plan, claim, materials, workspace, proposal


def test_complete_multiline_partition_and_immutable_scope(tmp_path):
    from fact_generation.execution.partition_analysis import analyze_partition
    plan, claim, materials, workspace, proposal = inputs(tmp_path)
    before = claim.model_dump(mode="json"), plan.model_dump(mode="json")
    result = analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)
    assert result["host_measurement"]["value"] == pytest.approx(2/3)
    assert result["host_measurement"]["sample_count"] == 3
    assert result["context"]["rows"] == [{"index": 0, "label": 1, "prediction": 2},
        {"index": 1, "label": 1, "prediction": 2}, {"index": 2, "label": 2, "prediction": 2}]
    assert result["status"] == "unresolved"
    assert not any(result[key] for key in ("scientific_qualification", "alignment", "support", "runtime_output_authenticated"))
    assert result["derived_observations"] == []
    assert before == (claim.model_dump(mode="json"), plan.model_dump(mode="json"))
    other = tmp_path / "accuracy"
    other.mkdir()
    plan, claim, materials, workspace, proposal = inputs(other, "accuracy")
    result = analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)
    assert result["host_measurement"]["value"] == pytest.approx(1/3)
    assert result["host_measurement"]["unit"] == "fraction"
    (workspace / "data.json").write_text('{}', encoding="utf-8")
    assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None


def test_equal_wrong_split_and_missing_tail_are_rejected(tmp_path):
    from fact_generation.execution.partition_analysis import analyze_partition
    plan, claim, materials, workspace, proposal = inputs(tmp_path)
    wrong = copy.deepcopy(proposal)
    wrong["partition_selector"] = ["partitions", "train"]
    result = analyze_partition(wrong, plan=plan, claim=claim, materials=materials, workspace=workspace)
    assert result["host_measurement"] is None and "partition" in result["reason"]
    result = analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace, max_rows=2)
    assert result["host_measurement"] is None and "capacity" in result["reason"]
    artifact = json.loads((workspace / "data.json").read_text(encoding="utf-8"))
    artifact["partitions"]["test"].pop()
    (workspace / "data.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None


def test_unreviewed_inference_settings_and_unbound_modes_never_gain_qualification(tmp_path):
    from fact_generation.execution.partition_analysis import analyze_partition
    plan, claim, materials, workspace, proposal = inputs(tmp_path)
    result = analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)
    assert result["host_measurement"] is not None and not result["scientific_qualification"]
    assert "model_inference_unproven" in result["unresolved"]
    invalid = copy.deepcopy(proposal)
    invalid["label_selector"] = [True]
    assert analyze_partition(invalid, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
    for mode in ("evaluation", "training"):
        changed = plan.model_copy(deep=True)
        changed.run_mode = mode
        assert analyze_partition(proposal, plan=changed, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
    legacy = plan.model_copy(deep=True)
    legacy.task.resource_contract = None
    assert analyze_partition(proposal, plan=legacy, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
    modified = claim.model_copy(deep=True)
    modified.conditions[0].settings["preprocessing"] = "unproven"
    changed = plan.model_copy(deep=True)
    changed.target_conditions = modified.conditions
    changed.task.resource_contract = build_resource_contract(modified, materials, condition_ids=["c1"],
        entry_script=changed.task.entry_script, config=None, data_paths=changed.task.data_paths, weight_paths=[])
    assert analyze_partition(proposal, plan=changed, claim=modified, materials=materials, workspace=workspace)["host_measurement"] is None
    other = tmp_path / "boolean"
    other.mkdir()
    plan, claim, materials, workspace, proposal = inputs(other, boolean_label=True)
    assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
