"""Ordinary-v4 public paths; external model and Docker boundaries are mocked."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from assessment.rules import assess_claim
from common import run_stats
from fact_generation.execution import v2
from schemas.claim import ExecutionPlan
from tests.execution_science_fixtures import record_observer
from tests.experiment_authority_fixtures import authority_scope_fixture
from tests.test_execution_multi_condition_phase_a_v2 import _evaluation_events, _inputs, _proposal
from util.subprocess_runner import CommandResult
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.experiment_targets import TargetBindingError, validate_plan_targets
from verification.experiments import verify_experiments

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


def _offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real model, HTTP, retrieval, Docker and subprocess are forbidden")
    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: None)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_proxy_env", lambda _: {})
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "mock-image"))


def _wire(payload, plans):
    required = payload["required_execution_conditions"]
    rows = []
    for spec, plan in zip(required, plans, strict=True):
        condition = plan.target_conditions[0]
        binding = plan.target_bindings[condition.id]
        rows.append({"condition_id": condition.id, "condition_digest": spec["condition_digest"],
            "kind": "ordinary", "plan": {"targets": [{"condition_id": condition.id,
                "reported": binding.reported.model_dump(mode="json"), "projection": None}],
                "entry_script": plan.task.entry_script, "config": plan.task.config,
                "run_mode": plan.run_mode, "feasibility": plan.feasibility, "blocker": plan.blocker,
                "priority": plan.priority, "data_paths": plan.task.data_paths,
                "weight_paths": plan.task.weight_paths}})
    return {"schema_version": "experiments-execution-v4", "execution_domain": "ordinary-conditions-v1",
        "execution_context_digest": payload["execution_context_digest"], "checked_aspects": ASPECTS,
        "items": [], "issues": [], "execution": rows}


def test_default_dispatch_v4_reaches_real_two_condition_execution_and_missing_is_unverified(tmp_path, monkeypatch):
    _offline(monkeypatch)
    plans, claim, materials, definitions, data, weights, released = _inputs(tmp_path)
    original = claim.model_dump(mode="json")
    original_materials = materials.model_dump(mode="json")
    raw_calls, semantic = [], []
    mode = {"value": "healthy"}

    def model(prompt, system, cfg, *, module):
        payload = json.loads(prompt)
        run_stats.record_llm_call(module=module, provider="mock", model="ordinary-v4",
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        if module == "verification.experiments":
            assert payload["claim"] == original
            assert payload["paper_blocks"] == [b.model_dump(mode="json") for b in materials.blocks]
            schema = payload["output_schema"]["properties"]["execution"]
            assert schema["minItems"] == schema["maxItems"] == 2
            assert [r["properties"]["condition_id"]["const"] for r in schema["prefixItems"]] == [c.id for c in claim.conditions]
            assert "execution_choice_context" not in payload
            response = _wire(payload, plans)
            raw_calls.append(copy.deepcopy(response))
            return response
        assert module == "execution"
        if "plan" in payload:
            response = _proposal(payload, definitions)
            response["command"] = ["python", payload["plan"]["task"]["entry_script"]]
            return response
        context = payload["context"]
        assert context["scientific_claim"]["conditions"] == original["conditions"]
        semantic.append(context)
        response = {"version": context["version"], "context_digest": context["context_digest"],
            "condition_id": context["condition"]["id"], "obligations": [{"id": key, "decision": "confirmed",
                "source_ids": sources, "rationale": "Complete selected sources and common qualifiers reviewed."}
                for key, sources in context["obligations"].items()], "unresolved": []}
        if context["version"] == "analysis-v1":
            response["measurement_scope"] = "released_statistics"
        if mode["value"] == "missing" and context["condition"]["id"] == "validation-setting":
            response["obligations"][-1]["decision"] = "unresolved"
            response["unresolved"] = [response["obligations"][-1]["id"]]
        return response

    def transport(command, cwd, timeout_sec):
        assert command[:2] == ["docker", "run"]
        launch = json.loads((Path(cwd) / "attempt_0/runtime_observer_launch.json").read_text())
        config = json.loads((Path(launch["trusted_dir"]) / "config.json").read_text())
        if config["entry"] == "eval.py":
            stdout = str(_evaluation_events(command, cwd, data, weights)) + "\n"
        else:
            rows = released["partitions"]["validation"]
            value = sum((r["prediction"]-r["label"])**2 for r in rows)/len(rows)
            actual = {"observations": [{"dataset": released["dataset"], "metric": released["metric"],
                "settings": released["settings"], "value": value}]}
            record_observer(command, cwd, actual)
            stdout = json.dumps(actual)
        return CommandResult(command, cwd, 0, stdout, "", .01, termination="completed")

    monkeypatch.setattr("llm.client.llm_json", model)
    monkeypatch.setattr(v2, "run_command", transport)
    results = {}
    for name in ("healthy", "missing"):
        mode["value"] = name
        with run_stats.run_scope(tmp_path / name / "run_stats.json"):
            verification = asyncio.run(verify_claims([claim], materials, tmp_path / name / "verification",
                call=model, global_literature=lambda c, m: BranchResult()))
            assert len(verification.plans) == 2, verification.issues
            assert [p.id for p in verification.plans] == [claim.id+".plan.v4.0", claim.id+".plan.v4.1"]
            for ordinal, plan in enumerate(verification.plans):
                origin = plan.execution_origin
                assert origin.ordinal == origin.plan_index == ordinal
                audit = json.loads(Path(origin.audit_locator).read_text())
                assert audit["raw_response"] == raw_calls[-1]
                assert origin.raw_row_pointer == f"/execution/{ordinal}"
                assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
                assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
            results[name] = v2.execute_plans(verification.plans, verification.claims, materials,
                tmp_path / name / "execution", config={"max_attempts": 0, "approval_mode": "auto", "training_budget": 0})
    assert [e.covered for e in results["healthy"].claims[0].evidence] == [["test-setting"], ["validation-setting"]]
    assert assess_claim(results["healthy"].claims[0]).status.value == "supported"
    assert [e.covered for e in results["missing"].claims[0].evidence] == [["test-setting"]]
    assert assess_claim(results["missing"].claims[0]).status.value == "unverified"
    assert len(semantic) == 4
    assert claim.model_dump(mode="json") == original and materials.model_dump(mode="json") == original_materials
    assert all(len(result.ledger) == 2 and all(len(r["attempts"]) == 1 for r in result.ledger) for result in results.values())
    (tmp_path / "public-audit.json").write_text(json.dumps({"raw": raw_calls, "semantic": semantic}, indent=2))


def test_v4_partition_origin_and_local_target_fail_closed_without_legacy_upgrade(tmp_path, monkeypatch):
    _offline(monkeypatch)
    plans, claim, materials, *_ = _inputs(tmp_path)
    before = [p.model_dump(mode="json") for p in plans]
    assert all("execution_origin" not in row for row in before)
    assert all("execution_origin" not in p.model_dump() for p in plans)
    captured = []
    mode = {"value": "healthy"}

    def model(prompt, system, cfg, *, module):
        payload = json.loads(prompt)
        if "candidate_items" in payload:
            response = {"conditions": [{"condition_id": c.id, "claim_quote": claim.text,
                "assertion": "descriptive", "matched_controls_required": False,
                "uncertainty_sensitive": False, "relation": "none", "subject": "", "comparator": "",
                "rationale": "Fixed two-setting claim remains intact."} for c in claim.conditions],
                "items": [{"item_index": 0, "condition_id": claim.conditions[0].id,
                    "applicability": "applicable", "grounds": [{"block_id": claim.source_block_id,
                        "quote": claim.source_quote}], "rationale": "Partial paper observation retained.",
                    "full_support": False, "qualifiers_complete": False, "comparison_objects": "not_comparative"}]}
            return authority_scope_fixture(payload, response, requirement="manuscript_reported_result", basis="reported_measurement", rationale="The fixture reports the complete fixed result and retains its original qualifiers.")
        response = _wire(payload, plans)
        if mode["value"] == "missing":
            response["execution"].pop()
        elif mode["value"] == "duplicate":
            response["execution"][1] = copy.deepcopy(response["execution"][0])
        elif mode["value"] == "wrong-context":
            response["execution_context_digest"] = "0" * 64
        elif mode["value"] == "old":
            response = {"checked_aspects": ASPECTS, "items": [], "plans": [], "issues": []}
        elif mode["value"] == "local":
            response["execution"][0]["plan"]["targets"][0]["condition_id"] = "validation-setting"
        elif mode["value"] == "none":
            response["execution"][0] = {k: v for k, v in response["execution"][0].items() if k != "plan"}
            response["execution"][0].update(kind="none", rationale="No supported execution for this setting.")
        if mode["value"] in ("missing", "duplicate", "wrong-context", "old"):
            response["items"] = [{"aspect": "correspondence", "kind": "paper_support",
                "block_id": claim.source_block_id, "quote": claim.source_quote,
                "covered": [claim.conditions[0].id], "fully_supported_conditions": [],
                "detail": "Original partial paper observation survives an invalid execution partition."}]
        captured.append(copy.deepcopy(response))
        return response

    outcomes = {}
    for name in ("healthy", "missing", "duplicate", "wrong-context", "old", "local", "none"):
        mode["value"] = name
        outcomes[name] = verify_experiments(claim, materials, call=model,
            execution_wire_version="experiments-execution-v4", execution_audit_dir=tmp_path / name,
            scope_binding_repair_rounds=0)
    assert all(outcomes[name].plans == [] and outcomes[name].issues for name in ("missing", "duplicate", "wrong-context", "old"))
    assert all(len(outcomes[name].evidence) == 1 and not outcomes[name].evidence[0].sufficient
        for name in ("missing", "duplicate", "wrong-context", "old"))
    assert [p.condition_ids for p in outcomes["local"].plans] == [["validation-setting"]]
    assert outcomes["local"].verification_limitations[0].condition_ids == ["test-setting"]
    assert outcomes["none"].plans[0].execution_origin.ordinal == 1
    assert outcomes["none"].plans[0].execution_origin.plan_index == 0
    plan = outcomes["healthy"].plans[0]
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
    dropped = plan.model_copy(update={"execution_origin": None})
    with pytest.raises(TargetBindingError):
        validate_plan_targets(dropped, claim, materials)
    audit_path = Path(plan.execution_origin.audit_locator)
    raw = audit_path.read_bytes()
    audit = json.loads(raw)
    audit["raw_response"]["execution"][0]["plan"]["entry_script"] = "eval_released.py"
    audit_path.write_text(json.dumps(audit))
    resealed = plan.model_copy(deep=True)
    resealed.execution_origin.audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    with pytest.raises(TargetBindingError):
        validate_plan_targets(resealed, claim, materials)
    audit_path.write_bytes(raw)
    resource = Path(materials.repository.root) / "weights.json"
    original_resource = resource.read_bytes()
    resource.write_text('{"model":"Affine","w":3}')
    with pytest.raises(TargetBindingError):
        validate_plan_targets(plan, claim, materials)
    resource.write_bytes(original_resource)
    assert [p.model_dump(mode="json") for p in plans] == before
    absent = tmp_path / "not-a-directory"
    absent.write_text("preserve")
    calls_before = len(captured)
    unavailable = verify_experiments(claim, materials, call=model,
        execution_wire_version="experiments-execution-v4", execution_audit_dir=absent)
    assert unavailable.plans == [] and unavailable.issues and len(captured) == calls_before
    assert absent.read_text() == "preserve"
