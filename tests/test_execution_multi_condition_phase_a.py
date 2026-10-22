"""One public two-setting claim control; all external boundaries are mocked."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from assessment.rules import assess_claim
from common import run_stats
from fact_generation.execution import runner as v2
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_observer import _snapshot
from preprocessing.materials import index_repository
from schemas.claim import (
    Claim,
    ClaimLocation,
    ClaimSourceRef,
    Condition,
    ExecutionPlan,
    ExecutionTask,
)
from schemas.materials import MaterialBlock, SharedMaterials
from tests.execution_science_fixtures import record_observer
from tests.test_runtime_flow import SOURCE
from util.subprocess_runner import CommandResult
from verification.experiment_targets import bind_execution_target

ROLES = ("dataset", "split", "model", "metric", "population", "qualifiers")


def _inputs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = SOURCE.replace('"data.json"', '"evaluation.json"')
    source = source.replace('data["x"]', 'data["partitions"]["test"][0]["x"]')
    source = source.replace('data["label"]', 'data["partitions"]["test"][0]["label"]')
    source = source.replace("return predictions - labels", "return (predictions - labels) * (predictions - labels)")
    analysis = (
        "import json\nfrom pathlib import Path\n"
        "def driver():\n"
        "    data = json.loads((Path(__file__).resolve().parent / 'released.json').read_text(encoding='utf-8'))\n"
        "    rows = data['partitions']['validation']\n"
        "    value = sum((row['prediction'] - row['label']) ** 2 for row in rows) / len(rows)\n"
        "    return {'observations': [{'dataset': data['dataset'], 'metric': data['metric'], "
        "'settings': data['settings'], 'value': value}]}\n"
        "print(json.dumps(driver()))\n"
    )
    evaluation = {"dataset": "Benchmark", "metric": "mse", "partitions": {"test": [{"x": 3, "label": 5}]}}
    released = {"dataset": "Benchmark", "metric": "mse", "model": "Affine",
        "settings": {"model": "Affine", "split": "validation", "sample_count": 3},
        "partitions": {"validation": [{"label": 1, "prediction": 2},
            {"label": 2, "prediction": 3}, {"label": 3, "prediction": 4.01}]}}
    weights = {"model": "Affine", "w": 2}
    conditions = [Condition(id="test-setting", dataset="Benchmark", metric="mse",
        settings={"model": "Affine", "split": "test", "sample_count": 1}),
        Condition(id="validation-setting", dataset="Benchmark", metric="mse", settings=released["settings"])]
    sentences = ["On Benchmark test sample_count 1, model Affine reports mse 1.",
                 "On Benchmark validation sample_count 3, model Affine reports mse 1."]
    quote = " ".join(sentences)
    # One property (MSE reported as 1) across two settings, with explicit common limits.
    shared = "Both fixed Benchmark partitions use Affine, no preprocessing, and complete-partition mean squared error. No training-history or population-performance conclusion is asserted."
    definitions = [
        "Benchmark test is the complete held-out row x=3,label=5. Affine predicts x*w with released coefficient w=2. "
        "mse is the mean squared prediction error across all 1 test rows. " + shared,
        "Benchmark validation consists of all 3 released rows (label,prediction)=(1,2),(2,3),(3,4.01). "
        "These predictions were released for Affine. mse is the mean squared prediction error over the complete validation partition. "
        "This setting permits checking the released predictions without performing new inference. " + shared,
    ]
    readme = "\n\n".join(definitions)
    for name, value in {"eval.py": source, "eval_released.py": analysis, "README.md": readme,
        "evaluation.json": json.dumps(evaluation), "released.json": json.dumps(released),
        "weights.json": json.dumps(weights)}.items():
        (root / name).write_bytes(value.encode())
    paper = quote + "\n\n" + readme
    md = tmp_path / "paper.md"
    md.write_bytes(paper.encode())
    refs = [ClaimSourceRef(source_block_id="paper", source_quote=definition,
        loc=ClaimLocation(page=1, char_start=paper.index(definition), char_end=paper.index(definition)+len(definition)),
        covered=[condition.id]) for condition, definition in zip(conditions, definitions)]
    claim = Claim(id="one-property", text=quote, source_quote=quote, source_block_id="paper",
        loc=ClaimLocation(page=1, char_start=0, char_end=len(quote)), source_refs=refs,
        conditions=conditions, needs=["Experiments"])
    materials = SharedMaterials(paper_key="two-settings", source_pdf=str(md), markdown=paper,
        markdown_path=str(md), content_list_path="content.json", provider="fixture",
        blocks=[MaterialBlock(id="paper", text=paper, loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))],
        repository=index_repository(root))
    plans = []
    for i, condition in enumerate(conditions):
        entry = "eval.py" if i == 0 else "eval_released.py"
        data = "evaluation.json" if i == 0 else "released.json"
        task = ExecutionTask(entry_script=entry, command=["python", entry], data_paths=[data],
            weight_paths=["weights.json"] if i == 0 else [])
        task.resource_contract = build_resource_contract(claim, materials, condition_ids=[condition.id],
            entry_script=entry, config=None, data_paths=task.data_paths, weight_paths=task.weight_paths)
        plans.append(ExecutionPlan(id="setting-plan-"+str(i), claim_id=claim.id, condition_ids=[condition.id],
            target_conditions=[condition], task=task, run_mode="evaluation" if i == 0 else "analysis",
            y_paper={condition.id: 1}, feasibility="ready", priority="high",
            target_bindings={condition.id: bind_execution_target(claim, condition,
                {"block_id": "paper", "quote": sentences[i], "token": "1", "value_context": "mse 1"}, materials)}))
    return plans, claim, materials, definitions, evaluation, weights, released


def _proposal(payload, definitions, *, wrong_source=False):
    plan = payload["plan"]
    condition = plan["target_conditions"][0]
    entry = plan["task"]["entry_script"]
    files = payload["science_files"]
    source = files[entry]
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    names = ("driver", "reader", "inference", "metric") if plan["run_mode"] == "evaluation" else ("driver",)
    index = 0 if plan["run_mode"] == "evaluation" else 1
    definition = definitions[index]
    block = payload["paper_sources"][0]
    paper = {role: {"block_id": block["id"], "quote": definition, "start": block["text"].index(definition),
        "end": block["text"].index(definition)+len(definition)} for role in ROLES}
    if wrong_source and index == 0:
        other = definitions[1]
        paper["qualifiers"] = {"block_id": block["id"], "quote": other,
            "start": block["text"].index(other), "end": block["text"].index(other)+len(other)}
    sources = {}
    for role in ROLES:
        path = entry if index == 0 and role in ("model", "metric") else "README.md"
        text = ast.get_source_segment(source, functions["inference" if role == "model" else "metric"]) if path == entry else definition
        sources[role] = {"path": path, "quote": text, "start": files[path].index(text),
                         "end": files[path].index(text)+len(text)}
    proposal = {"version": "builtin-source-science-v1", "dataset_selector": ["dataset"],
        "metric_selector": ["metric"], "model_selector": ["model"],
        "partition_selector": ["partitions", condition["settings"]["split"]], "paper": paper, "sources": sources}
    output = {"command": plan["task"]["command"], "metric_output": None,
        "source_sites": [{"path": entry, "qualname": name, "firstlineno": functions[name].lineno} for name in names],
        "science_proposal": proposal}
    if index == 0:
        output["flow_proposal"] = {"version": "builtin-json-flow-v1",
            "roles": {name: {"site": i, "quote": ast.get_source_segment(source, functions[name])} for i, name in enumerate(names)},
            "conditions": [{"condition": condition, "paper_quote": payload["scientific_claim"]["source_quote"],
                "model_rationale": "Affines use the actual selected coefficient.", "metric_rationale": "Complete squared error."}]}
    else:
        proposal.update(version="released-partition-analysis-v1", artifact_path=plan["task"]["data_paths"][0],
            label_selector=["label"], prediction_selector=["prediction"], metric_definition="mean_squared_error")
        output["science_proposal"] = {"version": "partition-runtime-v1", "analysis": proposal,
                                      "raw_value_selector": ["observations", 0, "value"]}
    return output


def _evaluation_events(command, cwd, data, weights):
    attempt = Path(cwd) / "attempt_0"
    launch = json.loads((attempt / "runtime_observer_launch.json").read_text())
    config = json.loads((Path(launch["trusted_dir"]) / "config.json").read_text())
    prediction = data["partitions"]["test"][0]["x"] * weights["w"]
    label = data["partitions"]["test"][0]["label"]
    value = (prediction - label) ** 2
    arguments = [{}, {"path": "evaluation.json"}, {"path": "weights.json"},
        {"data": data, "weights": weights}, {"predictions": prediction, "labels": label}]
    returns = [value, data, weights, prediction, value]
    indices = [0, 1, 1, 2, 3]
    pattern = [("call", 0), ("call", 1), ("return", 1), ("call", 2), ("return", 2),
        ("call", 3), ("return", 3), ("call", 4), ("return", 4), ("return", 0)]
    events = []
    for order, (kind, i) in enumerate(pattern, 1):
        row = {"event": kind, "site": indices[i], "invocation": i+1, "order": order}
        if kind == "call":
            row.update(parent_invocation=None if i == 0 else 1,
                arguments={key: _snapshot(item, 65536) for key, item in arguments[i].items()})
        else:
            row["value"] = _snapshot(returns[i], 65536)
        events.append(row)
    report = {"version": "python-source-events-v1", "unresolved": [], "execution": {"status": "completed"},
        "interpreter": {"version": "3.11.14 (mock producer)", "executable": "/usr/local/bin/python", "optimize": 0},
        "entry": {"argv": [config["entry"], *config["args"]], "cwd": config["cwd"], "path": "/app/eval.py",
                  "sha256": launch["source_sha256"]["eval.py"]},
        "sites": [dict(row, code_sha256=hashlib.sha256((row["qualname"]+str(i)).encode()).hexdigest()) for i, row in enumerate(config["sites"])],
        "events": events, "source_hashes_after": {"/app/"+key: sha for key, sha in launch["source_sha256"].items()},
        "scope_limits": ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles", "native_and_child_execution_not_covered"]}
    Path(launch["audit_output"]).write_bytes(json.dumps(report).encode())
    return value


def test_singleton_plans_qualify_only_their_condition_in_a_complete_multi_condition_claim(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No real LLM, retrieval, HTTP, Docker or subprocess is permitted")
    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_proxy_env", lambda _: {})
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "mock-image"))
    plans, claim, materials, definitions, data, weights, released = _inputs(tmp_path)
    original = claim.model_dump(mode="json")
    original_materials = materials.model_dump(mode="json")
    original_plans = [plan.model_dump(mode="json") for plan in plans]
    root = Path(materials.repository.root)
    original_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    rows = released["partitions"]["validation"]
    measured = sum((row["prediction"]-row["label"])**2 for row in rows)/len(rows)
    actual_payload = {"observations": [{"dataset": released["dataset"], "metric": released["metric"],
        "settings": released["settings"], "value": measured}]}
    audit = []
    current = {"mode": "healthy"}

    def transport(command, cwd, timeout_sec):
        assert command[:2] == ["docker", "run"]
        launch = json.loads((Path(cwd) / "attempt_0" / "runtime_observer_launch.json").read_text())
        entry = json.loads((Path(launch["trusted_dir"]) / "config.json").read_text())["entry"]
        if entry == "eval.py":
            value = _evaluation_events(command, cwd, data, weights)
            stdout = str(value)+"\n"
        else:
            assert entry == "eval_released.py"
            record_observer(command, cwd, actual_payload)
            stdout = json.dumps(actual_payload)
        return CommandResult(command, cwd, 0, stdout, "", .01, termination="completed")

    def model(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        run_stats.record_llm_call(module=module, provider="mock", model="multi-condition",
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        if "plan" in payload:
            assert payload["scientific_claim"]["conditions"] == original["conditions"]
            assert payload["scientific_claim"]["source_refs"] == original["source_refs"]
            assert payload["scientific_claim"]["text"] == original["text"]
            return _proposal(payload, definitions, wrong_source=current["mode"] == "wrong-source")
        context = payload["context"]
        assert context["scientific_claim"]["conditions"] == original["conditions"]
        assert context["scientific_claim"]["source_refs"] == original["source_refs"]
        assert context["scientific_claim"]["text"] == original["text"]
        assert context["plan"]["condition_ids"] == [context["condition"]["id"]]
        assert context["resource_contract"]["condition_sha256"].keys() == {c["id"] for c in original["conditions"]}
        selected = context["condition"]["id"]
        audit.append({"mode": current["mode"], "context": context})
        response = {"version": context["version"], "context_digest": context["context_digest"],
            "condition_id": selected, "obligations": [{"id": key, "decision": "confirmed", "source_ids": sources,
                "rationale": "The complete selected sources establish this setting and the shared qualifiers."}
                for key, sources in context["obligations"].items()], "unresolved": []}
        if context["version"] == "analysis-v1":
            response["measurement_scope"] = "released_statistics"
        if selected == "validation-setting" and current["mode"] == "missing-qualification":
            response["obligations"][-1]["decision"] = "unresolved"
            response["unresolved"] = [response["obligations"][-1]["id"]]
        if selected == "validation-setting" and current["mode"] == "wrong-condition":
            response["condition_id"] = "test-setting"
        return response

    monkeypatch.setattr(v2, "run_command", transport)
    monkeypatch.setattr("llm.client.llm_json", model)
    results = {}
    for mode in ("healthy", "missing-qualification", "wrong-condition", "wrong-source"):
        current["mode"] = mode
        results[mode] = v2.execute_plans(plans, [claim], materials, tmp_path / mode,
            config={"max_attempts": 0, "approval_mode": "auto", "training_budget": 0})
        assert claim.model_dump(mode="json") == original
        assert materials.model_dump(mode="json") == original_materials
        assert [plan.model_dump(mode="json") for plan in plans] == original_plans
        assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()} == original_files
    healthy = results["healthy"]
    assert len(healthy.claims[0].evidence) == 2
    assert [item.covered for item in healthy.claims[0].evidence] == [["test-setting"], ["validation-setting"]]
    assert all(item.source == "execution" and item.aligned and item.sufficient for item in healthy.claims[0].evidence)
    assert assess_claim(healthy.claims[0]).status.value == "supported"
    assert healthy.ledger[0]["alignment"][0]["observed"]["value"] == 1
    assert healthy.ledger[1]["alignment"][0]["observed"]["value"] == measured != 1
    assert all(row["alignment"][0]["consistent"] for row in healthy.ledger)
    assert all(row["alignment"][0]["tolerance"] == .05 for row in healthy.ledger)
    for mode in ("missing-qualification", "wrong-condition"):
        result = results[mode]
        assert [item.covered for item in result.claims[0].evidence] == [["test-setting"]]
        assert assess_claim(result.claims[0]).status.value == "unverified"
        assert not result.ledger[1]["alignment"][0]["aligned"]
    wrong = results["wrong-source"]
    assert [item.covered for item in wrong.claims[0].evidence] == [["validation-setting"]]
    assert assess_claim(wrong.claims[0]).status.value == "unverified"
    assert not wrong.ledger[0]["alignment"][0]["aligned"]
    bad = json.loads(Path(wrong.ledger[0]["attempts"][0]["logs"]["scientific_consumption"]).read_text())
    assert "paper_definition_outside_original_condition_sources" in bad["reason"]
    assert bad["derived_observations"] == [] and bad["model_calls"] == 0
    assert len(audit) == 7 and [row["context"]["version"] for row in audit[:2]] == ["builtin-source-science-v1", "analysis-v1"]
    for result in results.values():
        assert len(result.ledger) == 2 and all(len(row["attempts"]) == 1 for row in result.ledger)
        for row in result.ledger:
            attempt = row["attempts"][0]
            assert attempt["returncode"] == 0
            science = json.loads(Path(attempt["logs"]["scientific_consumption"]).read_text())
            if science["scientific_qualification"]:
                assert science["context"]["condition"]["id"] == row["plan"]["condition_ids"][0]
                assert science["usage"]["state"] == "measured" and science["usage"]["requests"] == 1
    (tmp_path / "multi-condition-public-audit.json").write_text(json.dumps({"semantic_calls": len(audit),
        "external_calls": 0, "measurements": [1, measured], "contexts": audit,
        "claims": {mode: result.claims[0].model_dump(mode="json") for mode, result in results.items()}}, indent=2))
