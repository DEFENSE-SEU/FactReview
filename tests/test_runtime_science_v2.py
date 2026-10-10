"""Two actual-source scientific scope controls; every external boundary is mocked."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_flow import bind_builtin_flow
from fact_generation.execution.runtime_launch import (
    bind_source_sites,
    prepare_observer_launch,
    protect_observer_docker_argv,
)
from fact_generation.execution.runtime_observer import _snapshot
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from tests.test_runtime_flow_v2 import SOURCE
from verification.experiment_targets import bind_execution_target


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scientific consumption controls forbid external/process calls")
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.Popen", "subprocess.run"):
        monkeypatch.setattr(name, forbidden)
    for name in ("docker_runner", "run_command", "docker_ensure_paper_image"):
        monkeypatch.setattr(v2, name, forbidden)


def inputs(tmp_path, split="test", complete=True):
    root = tmp_path / "repo"
    root.mkdir()
    source = SOURCE.replace('data["x"]', f'data["partitions"]["{split}"][0]["x"]')
    source = source.replace('data["label"]', f'data["partitions"]["{split}"][0]["label"]')
    source = source.replace("return predictions - labels", "return (predictions - labels) * (predictions - labels)")
    (root / "eval.py").write_text(source, encoding="utf-8")
    data = {"dataset": "Benchmark", "metric": "mse", "partitions": {
        "test": [{"x": 3, "label": 5}], "train": [{"x": 3, "label": 5}]}}
    weights = {"model": "Affine", "w": 2}
    (root / "data.json").write_text(json.dumps(data), encoding="utf-8")
    (root / "weights.json").write_text(json.dumps(weights), encoding="utf-8")
    definition = ("Benchmark is the explicitly constructed two-partition benchmark: test is the complete "
        "held-out row (x=3,label=5); train is a separate equal-valued row. Affine predicts x*w with "
        "the released coefficient w=2, without preprocessing. mse is the mean squared prediction "
        "error over the complete test partition, containing exactly one row. No training-history claim is made.")
    (root / "README.md").write_text(definition, encoding="utf-8")
    quote = "On Benchmark test, model Affine reports mse 1."
    paper = quote + "\n" + (definition if complete else "")
    md = tmp_path / "paper.md"
    md.write_text(paper, encoding="utf-8")
    condition = Condition(id="c1", dataset="Benchmark", metric="mse", settings={"model": "Affine", "split": "test"})
    loc = ClaimLocation(page=1, char_start=0, char_end=len(quote))
    refs = [ClaimSourceRef(source_block_id="b", source_quote=definition,
        loc=ClaimLocation(page=1, char_start=len(quote)+1, char_end=len(paper)), covered=["c1"])] if complete else []
    claim = Claim(id="c", text=quote, loc=loc, source_block_id="b", source_quote=quote,
        source_refs=refs, conditions=[condition], needs=["Experiments"])
    materials = SharedMaterials(paper_key="p", source_pdf="paper.pdf", markdown=paper, markdown_path=str(md),
        content_list_path="content.json", provider="fixture", blocks=[MaterialBlock(id="b", text=paper,
        loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))], repository=index_repository(root))
    task = ExecutionTask(entry_script="eval.py", command=["python", "eval.py"], data_paths=["data.json"], weight_paths=["weights.json"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"], entry_script="eval.py",
        config=None, data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        run_mode="evaluation", y_paper={"c1": 1}, feasibility="ready", priority="high",
        target_bindings={"c1": bind_execution_target(claim, condition, {"block_id": "b", "quote": quote, "token": "1"}, materials)})
    return plan, claim, materials, source, definition, data, weights


def refinement(plan, workspace, config, *, claim, materials):
    source = (workspace / "eval.py").read_text(encoding="utf-8")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    names = ("driver", "reader", "inference", "metric")
    sites = bind_source_sites([{"path": "eval.py", "qualname": name, "firstlineno": functions[name].lineno}
        for name in names], workspace=workspace, supplied_files={"eval.py": source})
    flow = {"version": "builtin-json-flow-v1", "roles": {name: {"site": i,
        "quote": ast.get_source_segment(source, functions[name])} for i, name in enumerate(names)},
        "conditions": [{"condition": claim.conditions[0].model_dump(mode="json"), "paper_quote": claim.source_quote,
                        "model_rationale": "Located Affine formula.", "metric_rationale": "Located squared error."}]}
    files = {path: (workspace / path).read_text(encoding="utf-8") for path in ("eval.py", "README.md", "data.json", "weights.json")}
    # A proposal selects source locations and JSON keys; it cannot supply trusted facts.
    roles = ("dataset", "split", "model", "metric", "population", "qualifiers")
    definition = files["README.md"]
    paper = {role: {"block_id": "b", "quote": definition, "start": len(claim.source_quote)+1,
        "end": len(claim.source_quote)+1+len(definition)} for role in roles}
    sources = {}
    for role in roles:
        path = "eval.py" if role in ("model", "metric") else "README.md"
        quote = ast.get_source_segment(source, functions["inference" if role == "model" else "metric"]) if path == "eval.py" else definition
        sources[role] = {"path": path, "quote": quote, "start": files[path].index(quote), "end": files[path].index(quote)+len(quote)}
    proposal = {"version": "builtin-source-science-v1", "dataset_selector": ["dataset"],
        "partition_selector": ["partitions", "test"], "model_selector": ["model"], "metric_selector": ["metric"],
        "paper": paper, "sources": sources}
    return list(plan.task.command), None, {"source_sites": sites, "flow_binding": bind_builtin_flow(flow,
        plan=plan, claim=claim, materials=materials, supplied_files={"eval.py": source}, source_sites=sites),
        "flow_files": {"eval.py": source}, "science_proposal": proposal, "science_files": files}


def captured_runner(data, weights):
    def run(request):
        attempt = Path(request.run_dir) / "attempt_0"
        attempt.mkdir()
        scratch = Path(request.run_dir) / "runtime_scratch"
        scratch.mkdir()
        launch = prepare_observer_launch(request.command, entry_script="eval.py", workdir=".", workspace=request.workspace,
            metric_output=None, source_sites=request.source_sites, trusted_dir=attempt / "observer_trusted",
            runtime_dir=scratch, repair_round=0, runtime_python="3.11")
        assert launch["status"] == "ready"
        env = {"EXECUTION_RUN_DIR": "/workspace/run_dir", "EXECUTION_ARTIFACT_DIR": "/workspace/run_dir/artifacts",
            "EXECUTION_PAPER_DIR": "/app", "EXECUTION_PAPER_ROOT": "/app", "PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPYCACHEPREFIX": "/workspace/run_dir/.pycache",
            "XDG_CACHE_HOME": "/workspace/run_dir/.cache", "HF_HOME": "/workspace/run_dir/.cache/huggingface",
            "MPLCONFIGDIR": "/workspace/run_dir/.cache/matplotlib", "FACTREVIEW_REPAIR_ROUND": "0"}
        argv = ["docker", "run", "--rm", "--name", "factreview-mock", "-v", request.workspace+":/app",
            "-v", str(scratch)+":/workspace/run_dir", "-w", "/app/."]
        for key, value in env.items():
            argv += ["-e", key+"="+value]
        argv += ["mock-image", *request.command]
        argv = protect_observer_docker_argv(argv, launch=launch)["argv"]
        calls = [{}, {"path": "data.json"}, {"path": "weights.json"}, {"data": data, "weights": weights}, {"predictions": 6, "labels": 5}]
        returns = [1, data, weights, 6, 1]
        indices = [0, 1, 1, 2, 3]
        pattern = [("call", 0), ("call", 1), ("return", 1), ("call", 2), ("return", 2),
            ("call", 3), ("return", 3), ("call", 4), ("return", 4), ("return", 0)]
        events = []
        for order, (kind, i) in enumerate(pattern, 1):
            row = {"event": kind, "site": indices[i], "invocation": i+1, "order": order}
            if kind == "call":
                row.update(parent_invocation=None if i == 0 else 1, arguments={key: _snapshot(value, 65536) for key, value in calls[i].items()})
            else:
                row["value"] = _snapshot(returns[i], 65536)
            events.append(row)
        report = {"version": "python-source-events-v1", "unresolved": [], "execution": {"status": "completed"},
            "interpreter": {"version": "3.11.14 (mock producer)", "executable": "/usr/local/bin/python", "optimize": 0},
            "entry": {"argv": ["eval.py"], "cwd": "/app", "path": "/app/eval.py", "sha256": launch["source_sha256"]["eval.py"]},
            "sites": [dict(row, path="/app/"+row["path"], code_sha256=str(i+1)*64) for i, row in enumerate(request.source_sites["sites"])],
            "events": events, "source_hashes_after": {"/app/"+key: value for key, value in launch["source_sha256"].items()},
            "scope_limits": ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles", "native_and_child_execution_not_covered"]}
        raw = json.dumps(report).encode()
        Path(launch["audit_output"]).write_bytes(raw)
        (attempt / "runtime_observer.json").write_bytes(raw)
        launch.update(status="observed", report_status="completed", report_unresolved=[], report_sha256=hashlib.sha256(raw).hexdigest())
        (attempt / "runtime_observer_launch.json").write_text(json.dumps(launch), encoding="utf-8")
        (attempt / "run_stdout.log").write_bytes(b"1\n")
        return v2.RunOutcome(returncode=0, stdout="1\n", commands=[argv], environment={"transport": "docker",
            "image": "mock-image", "container_name": "factreview-mock", "runtime_observer": launch,
            "process_termination": {"kind": "completed", "exception_type": None}}, logs={"runtime_observer": str(attempt / "runtime_observer.json"),
            "runtime_observer_launch": str(attempt / "runtime_observer_launch.json"), "stdout": str(attempt / "run_stdout.log")})
    return run


def run_case(tmp_path, monkeypatch, split="test", complete=True):
    plan, claim, materials, _source, _definition, data, weights = inputs(tmp_path, split, complete)
    original = claim.model_dump(mode="json")
    calls = []
    def scope(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        if "plan" in payload:
            assert payload["science_files"]["data.json"] == (Path(materials.repository.root) / "data.json").read_text(encoding="utf-8")
            assert payload["science_read_scope"]["unavailable"] == []
            _, _, proposed = refinement(plan, Path(materials.repository.root), None, claim=claim, materials=materials)
            return {"command": list(plan.task.command), "metric_output": None,
                "source_sites": [{key: row[key] for key in ("path", "qualname", "firstlineno")}
                                 for row in proposed["source_sites"]["sites"]],
                "flow_proposal": proposed["flow_binding"]["proposal"], "science_proposal": proposed["science_proposal"]}
        calls.append(payload)
        run_stats.record_llm_call(module=module, provider="mock", model="mock", usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        context = payload["context"]
        return {"version": "builtin-source-science-v1", "context_digest": context["context_digest"], "condition_id": "c1",
            "obligations": [{"id": key, "decision": "confirmed", "source_ids": value, "rationale": "Actual full source and paper definition agree."}
                for key, value in context["obligations"].items()], "unresolved": []}
    monkeypatch.setattr("llm.client.llm_json", scope)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    result = v2.execute_plans([plan], [claim], materials, tmp_path / "out", config={"max_attempts": 0}, runner=captured_runner(data, weights))
    assert claim.model_dump(mode="json") == original
    return result, calls


def test_complete_actual_scope_creates_separate_observation_before_original_judge(tmp_path, monkeypatch):
    result, calls = run_case(tmp_path, monkeypatch)
    assert len(result.claims[0].evidence) == 1
    assert len(calls) == 1
    row = result.ledger[0]
    attempt = row["attempts"][0]
    assert attempt["returncode"] == 0 and attempt["stdout"] == "1\n" and attempt["observations"] == []
    audit = json.loads(Path(attempt["logs"]["scientific_consumption"]).read_text(encoding="utf-8"))
    assert audit["scientific_qualification"] is True and not audit["alignment"] and not audit["support"]
    assert audit["usage"]["state"] == "measured" and audit["usage"]["logical_calls"] == audit["usage"]["requests"] == 1
    assert audit["tokens"] == 14 and Path(audit["token_source"]).is_file()
    assert audit["derived_observations"][0] == {"dataset": "Benchmark", "metric": "mse", "settings": {"model": "Affine", "split": "test"}, "value": 1}
    assert row["alignment"][0]["aligned"] and row["alignment"][0]["consistent"]
    assert result.claims[0].evidence[0].pointer.locator == attempt["logs"]["scientific_consumption"]
    assert result.delivery_checks == []
    # An independently indexed result sentence alone has no method/provenance definition.
    directory = tmp_path / "missing"
    directory.mkdir()
    missing, missing_calls = run_case(directory, monkeypatch, complete=False)
    assert not missing.claims[0].evidence and missing_calls == []


def test_equal_actual_value_wrong_partition_cannot_alias_to_paper_split(tmp_path, monkeypatch):
    result, calls = run_case(tmp_path, monkeypatch, split="train")
    row = result.ledger[0]
    assert row["attempts"][0]["returncode"] == 0 and row["attempts"][0]["stdout"] == "1\n"
    assert row["attempts"][0]["observations"] == [] and not result.claims[0].evidence
    assert calls == [] and all(not item.get("aligned") for item in row["alignment"])
    audit = json.loads(Path(row["attempts"][0]["logs"]["scientific_consumption"]).read_text(encoding="utf-8"))
    assert not audit["scientific_qualification"] and audit["derived_observations"] == []
    assert "partition" in audit["reason"] and result.delivery_checks == []
