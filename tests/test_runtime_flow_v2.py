"""Three finite-flow boundaries; local controlled Python only, external calls forbidden."""

import ast
import copy
import hashlib

import pytest

from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_observer import prepare_site, run_observed
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials

SOURCE = '''import json
from pathlib import Path

def reader(path):
    return json.loads(Path(path).read_text())

def inference(data, weights):
    return data["x"] * weights["w"]

def metric(predictions, labels):
    return predictions - labels

def driver():
    data = reader("data.json")
    weights = reader("weights.json")
    predictions = inference(data, weights)
    value = metric(predictions, data["label"])
    return value

print(driver())
'''


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Finite flow controls forbid external boundaries")
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
                 "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(name, forbidden)


def fixture(tmp_path, source=SOURCE):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "eval.py").write_text(source, encoding="utf-8")
    (root / "data.json").write_text('{"x": 3, "label": 5}', encoding="utf-8")
    (root / "weights.json").write_text('{"w": 2}', encoding="utf-8")
    quote = "Linear model test residual on Toy data is 1."
    condition = Condition(id="c1", dataset="Toy", metric="residual", settings={"model": "Linear", "split": "test"})
    claim = Claim(id="c", text=quote, loc=ClaimLocation(page=1), source_block_id="b",
                  source_quote=quote, conditions=[condition], needs=["Experiments"])
    materials = SharedMaterials(paper_key="p", source_pdf="paper.pdf", markdown=quote,
        markdown_path=str(tmp_path / "paper.md"), content_list_path="content.json", provider="fixture",
        blocks=[MaterialBlock(id="b", text=quote, loc=claim.loc)], repository=index_repository(root))
    task = ExecutionTask(entry_script="eval.py", command=["python", "eval.py"],
                         data_paths=["data.json"], weight_paths=["weights.json"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"],
        entry_script="eval.py", config=None, data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition],
        task=task, run_mode="evaluation", y_paper={"c1": 1}, feasibility="ready", priority="high")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    sites = {"status": "bound", "sites": [
        {"path": "eval.py", "sha256": hashlib.sha256((root / "eval.py").read_bytes()).hexdigest(),
         "qualname": name, "firstlineno": functions[name].lineno}
        for name in ("driver", "reader", "inference", "metric")]}
    proposal = {"version": "builtin-json-flow-v1", "roles": {
        name: {"site": index, "quote": ast.get_source_segment(source, functions[name])}
        for index, name in enumerate(("driver", "reader", "inference", "metric"))},
        "conditions": [{"condition": condition.model_dump(mode="json"), "paper_quote": quote,
                        "model_rationale": "The located inference is proposed as the paper model.",
                        "metric_rationale": "The located metric is proposed as the reported protocol."}]}
    args = dict(plan=plan, claim=claim, materials=materials, supplied_files={"eval.py": source}, source_sites=sites)
    return root, proposal, args


def observed(root, tmp_path, args):
    sites = [prepare_site(str(root / row["path"]), row["qualname"], row["firstlineno"])
             for row in args["source_sites"]["sites"]]
    report = run_observed("eval.py", [], str(root), sites, str(tmp_path / "events.json"))
    request = {"command": ["python", "eval.py"], "cwd": str(root), "runtime_root": str(root),
               "repair_round": 0, "launch_sha256": "a" * 64, "config_sha256": "b" * 64}
    return report, request


def test_actual_finite_flow_is_rebuilt_and_never_scientific_grant(tmp_path):
    from fact_generation.execution.runtime_flow import bind_builtin_flow, validate_builtin_flow
    root, proposal, args = fixture(tmp_path)
    before = args["claim"].model_dump(mode="json")
    binding = bind_builtin_flow(proposal, **args)
    assert binding["status"] == "bound"
    report, request = observed(root, tmp_path, args)
    result = validate_builtin_flow(binding, **args, observer_report=report, run_request=request, raw_value=1)
    assert result["status"] == "witnessed"
    assert result["data_participation"] and result["weights_participation"] and result["labels_participation"]
    assert result["value"] == 1 and result["scientific_qualification"] is False
    assert result["raw_output_binding"] == "unresolved" and not result["alignment"] and not result["support"]
    assert result["evidence_refs"]["runtime_events"]["weights_reader"] == {"call": "/events/3", "return": "/events/4"}
    assert result["evidence_refs"]["source_roles"]["inference"]["firstlineno"] == 7
    assert args["claim"].model_dump(mode="json") == before
    # Exact parent/order/argument edges are mandatory even when values equal.
    for alteration in ("parent", "order", "arguments", "raw", "return_type", "bool", "container", "duplicate"):
        bad = copy.deepcopy(report)
        if alteration == "parent":
            bad["events"][1]["parent_invocation"] = None
        elif alteration == "order":
            bad["events"][2]["order"] = bad["events"][1]["order"]
        elif alteration == "arguments":
            bad["events"][5]["arguments"]["weights"] = {"type": "dict", "items": [["w", 3]]}
        elif alteration == "return_type":
            bad["events"][-1]["value"] = 1.0
        elif alteration == "bool":
            bad["events"][-1]["value"] = True
        elif alteration == "container":
            bad["events"][5]["arguments"]["weights"] = {"type": "tuple", "items": [2]}
        elif alteration == "duplicate":
            bad["events"][3] = copy.deepcopy(bad["events"][1])
        result = validate_builtin_flow(binding, **args, observer_report=bad, run_request=request,
                                       raw_value=2 if alteration == "raw" else 1)
        assert result["status"] == "unresolved", alteration


def test_constants_unused_weights_and_metadata_cannot_create_edges(tmp_path):
    from fact_generation.execution.runtime_flow import bind_builtin_flow, validate_builtin_flow
    variants = [SOURCE.replace('inference(data, weights)', 'inference(data, weights)', 1).replace(
        'predictions = inference(data, weights)', 'predictions = inference({"x": 3}, weights)'),
        SOURCE.replace('data["x"] * weights["w"]', 'data["x"] * 2'),
        SOURCE.replace('data["x"] * weights["w"]', 'data["x"] * (weights["w"] - weights["w"])'),
        SOURCE.replace('return predictions - labels', 'return 1'),
        SOURCE.replace('return data["x"] * weights["w"]', 'if data["x"]:\n        return 6\n    return 6')]
    for index, source in enumerate(variants):
        directory = tmp_path / str(index)
        directory.mkdir()
        root, proposal, args = fixture(directory, source)
        binding = bind_builtin_flow(proposal, **args)
        if binding["status"] == "bound":
            report, request = observed(root, directory, args)
            result = validate_builtin_flow(binding, **args, observer_report=report, run_request=request, raw_value=1)
            assert result["status"] == "unresolved"
        else:
            assert binding["status"] == "unresolved"
    directory = tmp_path / "metadata"
    directory.mkdir()
    _, proposal, args = fixture(directory)
    proposal["roles"]["inference"]["verified"] = True
    assert bind_builtin_flow(proposal, **args)["status"] == "unresolved"


def test_identity_unknown_and_changed_inputs_fail_closed(tmp_path):
    from fact_generation.execution.runtime_flow import bind_builtin_flow, validate_builtin_flow
    root, proposal, args = fixture(tmp_path)
    binding = bind_builtin_flow(proposal, **args)
    report, request = observed(root, tmp_path, args)
    for alteration in ("unknown", "code", "version", "site", "command", "cwd", "claim", "binding", "output", "optimize"):
        bad, changed, frozen = copy.deepcopy(report), copy.deepcopy(request), copy.deepcopy(binding)
        local = dict(args)
        if alteration == "unknown":
            bad["unresolved"] = ["thread_observed"]
        elif alteration == "code":
            bad["sites"][0]["code_sha256"] = "f" * 64
        elif alteration == "version":
            bad["interpreter"]["version"] = "3.11.0 unknown runtime"
        elif alteration == "site":
            bad["sites"][0]["path"] = "/outside/eval.py"
        elif alteration == "command":
            changed["command"] = ["python", "other.py"]
        elif alteration == "cwd":
            changed["cwd"] = str(root.parent)
        elif alteration == "output":
            bad["entry"]["argv"] = ["eval.py", "foreign-output"]
        elif alteration == "optimize":
            bad["interpreter"]["optimize"] = False
        elif alteration == "claim":
            local["claim"] = args["claim"].model_copy(deep=True)
            local["claim"].conditions[0].settings["split"] = "train"
        else:
            frozen["proposal"]["conditions"][0]["condition"]["settings"]["split"] = "train"
        result = validate_builtin_flow(frozen, **local, observer_report=bad, run_request=changed, raw_value=1)
        assert result["status"] == "unresolved", alteration
    args["plan"].task.resource_contract = None
    assert bind_builtin_flow(proposal, **args)["status"] == "unresolved"
    args["plan"].task.resource_contract = binding["resource_contract"]
    (root / "weights.json").write_text('{"w": 99}', encoding="utf-8")
    assert validate_builtin_flow(binding, **args, observer_report=report, run_request=request, raw_value=1)["status"] == "unresolved"
