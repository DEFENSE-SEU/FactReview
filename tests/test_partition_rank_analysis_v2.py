"""Complete rank statistics and authenticated canonical output, external calls forbidden."""
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from fact_generation.execution import v2
from fact_generation.execution.partition_analysis import analyze_partition
from fact_generation.execution.partition_analysis_runtime import build_partition_runtime_context
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_launch import bind_source_sites
from fact_generation.execution.runtime_observer import _snapshot
from preprocessing.materials import index_repository
from tests.test_partition_analysis_v2 import inputs
from tests.test_runtime_science_v2 import captured_runner
from verification.experiment_targets import bind_execution_target


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Rank controls prohibit external and process calls")
    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)


def fixture(directory, ranks, value):
    plan, claim, materials, workspace, proposal = inputs(directory)
    root = Path(materials.repository.root)
    source = '''import json
def driver():
    with open("data.json", encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data["partitions"]["test"]
    value = sum(1 / row["rank"] for row in rows) / len(rows)
    return {"metrics": {"MRR": value}, "observations": [{"dataset": "Benchmark", "metric": "MRR", "settings": {"split": "test", "score_func": "transe", "opn": "sub"}, "value": value}]}
print(json.dumps(driver()))
'''
    (root / "eval.py").write_text(source, encoding="utf-8")
    artifact = {"dataset": "Benchmark", "model": "ReleasedModel", "metric": "MRR",
                "partitions": {"test": [{"rank": rank} for rank in ranks]}}
    (root / "data.json").write_text(json.dumps(artifact), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"score_func": "transe", "opn": "sub"}), encoding="utf-8")
    definition = "ReleasedModel released ranks cover every Benchmark test record. MRR is the mean of reciprocal positive integer ranks. The scoring function is transe with sub operation. This assertion checks released statistics without new inference."
    (root / "README.md").write_text(definition, encoding="utf-8")
    quote = f"On Benchmark test score_func transe opn sub, model ReleasedModel reports MRR {value}."
    paper = quote + "\n" + definition
    claim.text = claim.source_quote = quote
    claim.loc.char_end = len(quote)
    claim.conditions[0].metric = "MRR"
    claim.conditions[0].settings = {"model": "ReleasedModel", "split": "test", "score_func": "transe", "opn": "sub"}
    claim.source_refs[0].source_quote = definition
    claim.source_refs[0].loc.char_start, claim.source_refs[0].loc.char_end = len(quote)+1, len(paper)
    materials.markdown = materials.blocks[0].text = paper
    materials.blocks[0].loc.char_end = len(paper)
    Path(materials.markdown_path).write_text(paper, encoding="utf-8")
    materials.repository = index_repository(root)
    plan.target_conditions = copy.deepcopy(claim.conditions)
    plan.y_paper = {"c1": value}
    plan.task.config = "config.json"
    plan.task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"],
        entry_script="eval.py", config="config.json", data_paths=["data.json"], weight_paths=[])
    plan.target_bindings = {"c1": bind_execution_target(claim, claim.conditions[0],
        {"block_id": "b", "quote": quote, "token": str(value)}, materials)}
    proposal.update(metric_definition="mean_reciprocal_rank", label_selector=None, prediction_selector=None,
                    rank_selector=["rank"], config_settings_selector={"score_func": ["score_func"], "opn": ["opn"]})
    for row in proposal["paper"].values():
        row.update(quote=definition, start=len(quote)+1, end=len(paper))
    for row in proposal["sources"].values():
        row.update(quote=definition, start=0, end=len(definition))
    shutil.rmtree(workspace)
    shutil.copytree(root, workspace)
    return plan, claim, materials, workspace, proposal, source


def test_complete_rank_measurement_and_exact_selected_config(tmp_path):
    for name, ranks, value in (("five", [2,2,2,4,4], .4), ("hundred", [2]*34+[4]*66, .335)):
        directory = tmp_path / name
        directory.mkdir()
        plan, claim, materials, workspace, proposal, _ = fixture(directory, ranks, value)
        before = plan.model_dump(mode="json"), claim.model_dump(mode="json")
        result = analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)
        assert result["host_measurement"] is not None, result["reason"]
        assert result["host_measurement"]["value"] == pytest.approx(value)
        assert result["host_measurement"]["sample_count"] == len(ranks)
        assert result["context"]["rows"] == [{"index": i, "rank": rank} for i, rank in enumerate(ranks)]
        assert result["context"]["actual_settings"] == claim.conditions[0].settings
        assert result["context"]["selected_config"]["sha256"] == hashlib.sha256((workspace/"config.json").read_bytes()).hexdigest()
        assert not any(result[k] for k in ("scientific_qualification", "alignment", "support"))
        assert before == (plan.model_dump(mode="json"), claim.model_dump(mode="json"))
        for key in ("opn", "score_func"):
            bad = copy.deepcopy(proposal)
            bad["config_settings_selector"].pop(key)
            assert analyze_partition(bad, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
        (workspace/"config.json").write_text('{}', encoding="utf-8")
        assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None
    directory = tmp_path / "different_paper_value"
    directory.mkdir()
    plan, claim, materials, workspace, proposal, _ = fixture(directory, [2,2,2,4,4], .9)
    assert plan.y_paper == {"c1": .9}
    assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"]["value"] == pytest.approx(.4)
    for i, bad_rank in enumerate((True, 0, -1, 2.0, float("inf"))):
        directory = tmp_path / f"invalid{i}"
        directory.mkdir()
        plan, claim, materials, workspace, proposal, _ = fixture(directory, [bad_rank], .4)
        assert analyze_partition(proposal, plan=plan, claim=claim, materials=materials, workspace=workspace)["host_measurement"] is None


def test_canonical_rank_raw_preserved_and_conflicts_rejected(tmp_path):
    plan, claim, materials, workspace, proposal, source = fixture(tmp_path, [2,2,2,4,4], .4)
    run = tmp_path / "run"
    run.mkdir()
    shutil.move(str(workspace), run/"workspace")
    workspace = run/"workspace"
    (run/"source_manifest.json").write_text(json.dumps({f.path: f.sha256 for f in materials.repository.files}), encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    sites = bind_source_sites([{"path": "eval.py", "qualname": "driver", "firstlineno": node.lineno}],
        workspace=workspace, supplied_files={"eval.py": source})
    request = v2.RunRequest(plan=plan, workspace=str(workspace), run_dir=str(run), command=plan.task.command,
        workdir=".", metric_output=None, repair_round=0, config=v2.ExecutionConfig(), source_sites=sites)
    outcome = captured_runner({}, {})(request)
    payload = {"metrics": {"MRR": .4}, "observations": [{"dataset": "Benchmark", "metric": "MRR",
        "settings": {"split": "test", "score_func": "transe", "opn": "sub"}, "value": .4}]}
    attempt = run/"attempt_0"
    report = json.loads((attempt/"runtime_observer.json").read_text(encoding="utf-8"))
    report["events"] = [{"event": "call", "site": 0, "invocation": 1, "order": 1, "parent_invocation": None, "arguments": {}},
                        {"event": "return", "site": 0, "invocation": 1, "order": 2, "value": _snapshot(payload, 65536)}]
    raw = json.dumps(report).encode()
    (attempt/"runtime_observer.json").write_bytes(raw)
    launch = outcome.environment["runtime_observer"]
    Path(launch["audit_output"]).write_bytes(raw)
    launch["report_sha256"] = hashlib.sha256(raw).hexdigest()
    (attempt/"runtime_observer_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    outcome.stdout = json.dumps(payload)+"\n"
    (attempt/"run_stdout.log").write_text(outcome.stdout, encoding="utf-8")
    (attempt/"raw_output.json").write_text(json.dumps(payload), encoding="utf-8")
    decoded, mapping = v2.decode_output(payload)
    outcome.observations = [v2.Observation.model_validate(row) for row in decoded]
    (attempt/"output_mapping.json").write_text(json.dumps({"source": str(attempt/"raw_output.json"), "selectors": mapping}), encoding="utf-8")
    outcome.logs.update(raw_output=str(attempt/"raw_output.json"), output_mapping=str(attempt/"output_mapping.json"))
    wire = {"version": "partition-runtime-v1", "analysis": proposal, "raw_value_selector": ["observations",0,"value"]}
    files = {name: (workspace/name).read_text(encoding="utf-8") for name in ("eval.py", "README.md", "config.json", "data.json")}
    before = outcome.model_dump(mode="json")
    context = build_partition_runtime_context(wire, plan=plan, claim=claim, materials=materials, request=request, outcome=outcome, source_files=files)
    assert context["actual_observation"]["value"] == pytest.approx(.4)
    assert context["actual_observation"]["settings"] == claim.conditions[0].settings
    assert context["catalog"]["actual/analysis"]["runtime_decoded_observations"] == decoded
    assert outcome.model_dump(mode="json") == before
    assert context["catalog"]["actual/analysis"]["model_inference_performed"] is False
    for change in ("raw", "source", "config"):
        changed = outcome.model_copy(deep=True)
        changed_files = copy.deepcopy(files)
        if change == "raw":
            changed.observations[0].value = .9
        else:
            changed_files["eval.py" if change == "source" else "config.json"] += " altered"
        with pytest.raises(ValueError):
            build_partition_runtime_context(wire, plan=plan, claim=claim, materials=materials,
                request=request, outcome=changed, source_files=changed_files)
