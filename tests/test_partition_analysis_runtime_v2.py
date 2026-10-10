"""Released-data authority through real refinement/judge; transport and LLM mocked."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_observer import _snapshot
from preprocessing.materials import index_repository
from tests.test_partition_analysis_v2 import inputs as partition_inputs
from tests.test_runtime_science_v2 import captured_runner
from verification.experiment_targets import bind_execution_target

SOURCE = '''import json
def driver():
    with open("data.json", encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data["partitions"]["test"]
    value = sum((r["prediction"] - r["label"]) ** 2 for r in rows) / len(rows)
    return {"metrics": {"mse": value}}
print(json.dumps(driver()))
'''


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Partition runtime controls prohibit external/process calls")
    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run", "llm.client.llm_json"):
        monkeypatch.setattr(name, forbidden)
    for name in ("docker_runner", "run_command", "docker_ensure_paper_image"):
        monkeypatch.setattr(v2, name, forbidden)


def fixture(tmp_path, reported=None):
    plan, claim, materials, _workspace, analysis = partition_inputs(tmp_path)
    root = Path(materials.repository.root)
    (root / "eval.py").write_text(SOURCE, encoding="utf-8")
    if reported is not None:
        old = claim.source_quote
        new = old.replace("0.6666666666666666", reported)
        definition = (root / "README.md").read_text(encoding="utf-8")
        paper = new + "\n" + definition
        claim.text = claim.source_quote = new
        claim.loc.char_end = len(new)
        claim.source_refs[0].loc.char_start = len(new)+1
        claim.source_refs[0].loc.char_end = len(paper)
        materials.markdown = materials.blocks[0].text = paper
        materials.blocks[0].loc.char_end = len(paper)
        Path(materials.markdown_path).write_text(paper, encoding="utf-8")
        for row in analysis["paper"].values():
            row.update(start=len(new)+1, end=len(paper))
        plan.y_paper["c1"] = float(reported)
        plan.target_bindings["c1"] = bind_execution_target(claim, claim.conditions[0],
            {"block_id": "b", "quote": new, "token": reported}, materials)
    materials.repository = index_repository(root)
    plan.task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"],
        entry_script="eval.py", config=None, data_paths=["data.json"], weight_paths=[])
    return plan, claim, materials, {"version": "partition-runtime-v1", "analysis": analysis,
                                    "raw_value_selector": ["metrics", "mse"]}


def transport(request, *, fault=None):
    # Reuse the existing mock protected-package/argv producer; replace its event
    # payload with this single observed analysis driver, without executing authors.
    outcome = captured_runner({}, {})(request)
    attempt = Path(request.run_dir) / "attempt_0"
    report = json.loads((attempt / "runtime_observer.json").read_text(encoding="utf-8"))
    payload = {"metrics": {"mse": 2/3}}
    if fault == "conflicting_metric":
        payload["mse"] = 9
    elif fault == "canonical_conflict":
        condition = request.plan.target_conditions[0]
        payload["observations"] = [{"dataset": condition.dataset, "metric": condition.metric,
            "settings": dict(condition.settings), "value": 9.0}]
    report["events"] = [
        {"event": "call", "site": 0, "invocation": 1, "order": 1, "parent_invocation": None, "arguments": {}},
        {"event": "return", "site": 0, "invocation": 1, "order": 2, "value": _snapshot(payload, 65536)}]
    raw = json.dumps(report).encode()
    (attempt / "runtime_observer.json").write_bytes(raw)
    launch = outcome.environment["runtime_observer"]
    Path(launch["audit_output"]).write_bytes(raw)
    launch["report_sha256"] = hashlib.sha256(raw).hexdigest()
    (attempt / "runtime_observer_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    outcome.stdout = json.dumps(payload)+"\n"
    (attempt / "run_stdout.log").write_text(outcome.stdout, encoding="utf-8")
    (attempt / "raw_output.json").write_text(json.dumps(payload), encoding="utf-8")
    outcome.logs["raw_output"] = str(attempt / "raw_output.json")
    # The real decoder rejects missing dataset/settings and leaves raw observations
    # empty. Its failure path preserves raw_output and does not create mapping audit.
    with pytest.raises(ValueError):
        v2.decode_output(payload, request.output_mapping)
    if fault == "source":
        path = Path(request.workspace) / "data.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["partitions"]["test"].pop()
        path.write_text(json.dumps(data), encoding="utf-8")
    elif fault == "producer":
        outcome.commands[0].append("unexpected")
    return outcome


def run_case(tmp_path, monkeypatch, *, reported=None, modify=None, fault=None, scope="released_statistics", bad_usage=False):
    plan, claim, materials, proposal = fixture(tmp_path, reported)
    if modify:
        modify(proposal)
    before = claim.model_dump(mode="json"), plan.model_dump(mode="json")
    calls = []
    def llm(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        usage = None if bad_usage and "context" in payload else {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        run_stats.record_llm_call(module=module, provider="mock", model="mock", usage=usage)
        if "plan" in payload:
            assert payload["science_proposal_schema"]["properties"]["version"]["const"] == "partition-runtime-v1"
            node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef))
            return {"command": list(plan.task.command), "metric_output": None,
                "source_sites": [{"path": "eval.py", "qualname": "driver", "firstlineno": node.lineno}],
                "science_proposal": proposal}
        calls.append(payload)
        context = payload["context"]
        assert context["version"] == "analysis-v1"
        assert context["catalog"]["actual/analysis"]["model_inference_performed"] is False
        return {"version": "analysis-v1", "context_digest": context["context_digest"], "condition_id": "c1",
            "measurement_scope": scope, "obligations": [{"id": key, "decision": "confirmed",
                "source_ids": values, "rationale": "Released complete partition and located original statistical assertion agree."}
                for key, values in context["obligations"].items()], "unresolved": []}
    monkeypatch.setattr("llm.client.llm_json", llm)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    result = v2.execute_plans([plan], [claim], materials, tmp_path / "out", config={"max_attempts": 0},
        runner=lambda request: transport(request, fault=fault))
    assert before == (claim.model_dump(mode="json"), plan.model_dump(mode="json"))
    return result, calls


def audit(result):
    attempt = result.ledger[0]["attempts"][0]
    return attempt, json.loads(Path(attempt["logs"]["scientific_consumption"]).read_text(encoding="utf-8"))


def test_full_released_measurement_and_artifact_discrepancy_use_original_judge(tmp_path, monkeypatch):
    from assessment.rules import assess_claim
    for name, reported, status in (("support", None, "supported"), ("mismatch", "0.9", "flawed")):
        directory = tmp_path / name
        directory.mkdir()
        result, calls = run_case(directory, monkeypatch, reported=reported)
        attempt, record = audit(result)
        assert len(calls) == 1 and attempt["returncode"] == 0
        assert attempt["observations"] == [] and json.loads(attempt["stdout"])["metrics"]["mse"] == 2/3
        assert record["scientific_qualification"] and record["usage"]["state"] == "measured"
        assert record["usage"]["logical_calls"] == record["usage"]["requests"] == 1 and record["tokens"] == 14
        assert not record["model_inference_performed"] and not record["alignment"] and not record["support"]
        assert len(record["context"]["catalog"]["actual/analysis"]["measurement"]["rows"]) == 3
        assert result.ledger[0]["alignment"][0]["aligned"] and len(result.claims[0].evidence) == 1
        assert assess_claim(result.claims[0]).status == status
        if reported:
            evidence = result.claims[0].evidence[0]
            assert evidence.provenance.released_artifact and not evidence.provenance.environment_explanation_possible
            assert evidence.sufficient and not evidence.overturnable


def test_equal_wrong_split_and_missing_tail_have_no_semantic_delegate(tmp_path, monkeypatch):
    for name, modify, fault in (("wrong_split", lambda p: p["analysis"].update(partition_selector=["partitions", "train"]), None),
                               ("missing_tail", None, "source")):
        directory = tmp_path / name
        directory.mkdir()
        result, calls = run_case(directory, monkeypatch, modify=modify, fault=fault)
        attempt = result.ledger[0]["attempts"][0]
        assert calls == [] and attempt["returncode"] == (1 if fault else 0)
        assert not result.claims[0].evidence and attempt["observations"] == []
        assert json.loads(attempt["stdout"])["metrics"]["mse"] == 2/3
        if fault:
            assert any(row["component"] == "execution.integrity" for row in attempt["operation_failures"])


def test_unmeasured_unknown_inference_and_forged_producer_do_not_qualify(tmp_path, monkeypatch):
    cases = [("bad_usage", {"bad_usage": True}), ("inference", {"scope": "requires_inference"}),
             ("unknown_source", {"modify": lambda p: p["analysis"]["sources"]["model"].update(path="missing.py")}),
             ("forged", {"fault": "producer"}), ("conflicting_metric", {"fault": "conflicting_metric"}),
             ("canonical_conflict", {"fault": "canonical_conflict"})]
    for name, kwargs in cases:
        directory = tmp_path / name
        directory.mkdir()
        result, calls = run_case(directory, monkeypatch, **kwargs)
        attempt, record = audit(result)
        assert not record["scientific_qualification"] and record["derived_observations"] == []
        assert attempt["observations"] == [] and not result.claims[0].evidence
        assert len(calls) == (1 if name in {"bad_usage", "inference"} else 0)
        if name == "bad_usage":
            assert record["status"] == "failed" and record["tokens"] is None and record["usage"]["state"] == "incomplete"
        if name == "inference":
            assert record["status"] == "unresolved" and not result.delivery_checks
    from fact_generation.execution.runtime_science import build_consumption_context
    directory = tmp_path / "evaluation"
    directory.mkdir()
    plan, claim, materials, proposal = fixture(directory)
    plan.run_mode = "evaluation"
    with pytest.raises(ValueError):
        build_consumption_context(proposal, plan=plan, claim=claim, materials=materials,
                                  request=None, outcome=None, source_files={})
