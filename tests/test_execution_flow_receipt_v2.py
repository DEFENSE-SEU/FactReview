"""Consumer wiring preserves runtime output and separate scientific guards."""

import json
from pathlib import Path

import pytest

from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from tests import test_execution_consumption_v2 as fixtures

original_inputs = fixtures.original_inputs
observation = fixtures.observation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Flow wiring forbids external services, processes and author execution")
    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "fact_generation.execution.v2.docker_runner", "fact_generation.execution.v2.run_command",
                 "fact_generation.execution.v2.docker_ensure_paper_image"):
        monkeypatch.setattr(name, forbidden)


def test_flow_receipt_wiring_preserves_raw_outcome_and_never_grants_science(original_inputs, tmp_path, monkeypatch):
    plan, claim, materials = original_inputs
    root = Path(materials.repository.root)
    (root / "data.json").write_text('[{"feature": 1, "label": 0}]', encoding="utf-8")
    (root / "weights.json").write_text('{"bias": 0}', encoding="utf-8")
    materials.repository = index_repository(root)
    plan.task.data_paths, plan.task.weight_paths = ["data.json"], ["weights.json"]
    plan.task.resource_contract = build_resource_contract(
        claim, materials, condition_ids=plan.condition_ids, entry_script=plan.task.entry_script,
        config=None, data_paths=plan.task.data_paths, weight_paths=plan.task.weight_paths,
    )
    original_claim, original_plan = claim.model_dump(mode="json"), plan.model_dump(mode="json")
    original_files = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    flow = {"status": "bound", "proposal": {"mock": "original refinement response"}}
    sites = {"status": "bound", "sites": [{"path": "eval.py"}]}
    files = {"eval.py": (root / "eval.py").read_text(encoding="utf-8")}
    derived = {"status": "witnessed", "scientific_qualification": False, "alignment": False,
               "support": False, "evidence_refs": {"mock": "captured receipt"}}
    consumer_calls, outcomes, runner_requests = [], [], []

    def refined(*args, **kwargs):
        assert kwargs["claim"].model_dump(mode="json") == original_claim and kwargs["materials"] is materials
        return list(plan.task.command), None, {"source_sites": sites, "flow_binding": flow, "flow_files": files}

    def runner(request):
        runner_requests.append(request)
        outcome = v2.RunOutcome(returncode=0, stdout="raw author output\n", stderr="original stderr\n",
            observations=[observation()], issue="original author issue", environment={"transport": "docker"})
        outcomes.append(outcome)
        return outcome

    def consumer(binding, **kwargs):
        consumer_calls.append((binding, kwargs))
        assert binding == flow and kwargs["plan"].model_dump(mode="json") == original_plan
        assert kwargs["claim"].model_dump(mode="json") == original_claim and kwargs["materials"] is materials
        assert kwargs["supplied_files"] == files and kwargs["source_sites"] == sites
        assert kwargs["runtime_request"].model_dump(mode="json") == runner_requests[0].model_dump(mode="json")
        assert kwargs["runtime_outcome"] is outcomes[0]
        assert kwargs["runtime_outcome"].returncode == 0 and kwargs["runtime_outcome"].observations == [observation()]
        return derived

    monkeypatch.setattr(v2, "_refine", refined)
    monkeypatch.setattr("fact_generation.execution.runtime_flow.validate_builtin_flow", consumer)
    result = v2.execute_plans([plan], [claim], materials, tmp_path / "out", runner=runner,
                             config={"max_attempts": 0, "refine_with_llm": True})
    assert len(consumer_calls) == 1
    attempt = result.ledger[0]["attempts"][0]
    validation_path = Path(attempt["logs"]["source_flow_validation"])
    assert validation_path == Path(attempt["request"]["run_dir"]) / "attempt_0/source_flow_validation.json"
    assert json.loads(validation_path.read_text(encoding="utf-8")) == derived
    assert attempt["environment"]["source_flow_validation"] == derived
    assert attempt["returncode"] == 0 and attempt["stdout"] == "raw author output\n"
    assert attempt["stderr"] == "original stderr\n" and attempt["issue"] == "original author issue"
    assert attempt["observations"] == [observation().model_dump(mode="json")]
    assert result.ledger[0]["alignment"][0]["aligned"] is False and not result.claims[0].evidence
    assert result.ledger[0]["alignment"][0]["consumption"]["status"] == "unresolved"
    assert claim.model_dump(mode="json") == original_claim and plan.model_dump(mode="json") == original_plan
    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == original_files
    monkeypatch.setattr(v2, "_refine", lambda *args, **kwargs: (list(plan.task.command), None, {}))
    legacy = v2.execute_plans([plan], [claim], materials, tmp_path / "without-flow", runner=runner,
                             config={"max_attempts": 0})
    assert len(consumer_calls) == 1 and len(runner_requests) == 2
    assert "source_flow_validation" not in legacy.ledger[0]["attempts"][0]["environment"]
    assert "source_flow_validation" not in legacy.ledger[0]["attempts"][0]["logs"]
