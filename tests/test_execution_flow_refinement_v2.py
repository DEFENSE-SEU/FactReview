"""Necessary capability-refinement controls; all external operations are mocked."""
import copy
import json

import pytest

from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from schemas.claim import ClaimSourceRef
from schemas.materials import MaterialBlock
from tests.test_runtime_flow_v2 import fixture


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Refinement controls forbid external operations")
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
                 "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(name, forbidden)


def completion(proposal, sites, command):
    return {"command": command, "metric_output": None,
            "source_sites": [{key: row[key] for key in ("path", "qualname", "firstlineno")}
                             for row in sites["sites"]], "flow_proposal": proposal}


def test_existing_command_gets_one_scoped_completion_and_keeps_original_launch(tmp_path, monkeypatch):
    root, proposal, args = fixture(tmp_path)
    plan, claim, materials = args["plan"], args["claim"], args["materials"]
    claim.importance = "core"
    qualifier = "Published linear weights are used on the held-out test split."
    claim.source_refs = [ClaimSourceRef(source_block_id="qualifier", source_quote=qualifier,
                                       loc=claim.loc, covered=["c1"])]
    materials.blocks.append(MaterialBlock(id="qualifier", text=qualifier, loc=claim.loc))
    plan.task.resource_contract = build_resource_contract(claim, materials, condition_ids=plan.condition_ids,
        entry_script=plan.task.entry_script, config=plan.task.config,
        data_paths=plan.task.data_paths, weight_paths=plan.task.weight_paths)
    original = plan.model_dump(mode="json")
    calls = []
    def model(payload, prompt, **kwargs):
        calls.append(json.loads(payload))
        # Existing argv is preserved even when the response proposes another one.
        return completion(proposal, args["source_sites"], ["python", "other.py"])
    monkeypatch.setattr("llm.client.llm_json", model)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    command, output, audit = v2._refine(plan, root, v2.ExecutionConfig(), claim=claim, materials=materials)
    assert command == ["python", "eval.py"] and output is None
    assert len(calls) == 1 and calls[0]["scientific_claim"]["source_quote"] == claim.source_quote
    assert calls[0]["paper_sources"][0]["text"] == materials.blocks[0].text
    assert calls[0]["scientific_claim"]["importance"] == "core"
    assert calls[0]["scientific_claim"]["source_refs"][0]["source_quote"] == qualifier
    assert [row["id"] for row in calls[0]["paper_sources"]] == ["b", "qualifier"]
    assert audit["flow_binding"]["status"] == "bound" and not audit["flow_binding"]["scientific_qualification"]
    assert audit["flow_files"] == args["supplied_files"]
    assert audit["mode"] == "llm" and plan.model_dump(mode="json") == original
    # Existing commands with refinement disabled retain the original no-call path.
    _, _, disabled = v2._refine(plan, root, v2.ExecutionConfig(refine_with_llm=False), claim=claim, materials=materials)
    assert len(calls) == 1 and disabled["flow_binding"]["status"] == "unresolved"


def test_missing_command_binds_original_plan_and_records_actual_refinement(tmp_path, monkeypatch):
    root, proposal, args = fixture(tmp_path)
    plan = args["plan"]
    plan.task.command = []
    before = copy.deepcopy(plan.model_dump(mode="json"))
    calls = []
    def model(payload, prompt, **kwargs):
        calls.append(json.loads(payload))
        return completion(proposal, args["source_sites"], ["python", "eval.py"])
    monkeypatch.setattr("llm.client.llm_json", model)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    command, _, audit = v2._refine(plan, root, v2.ExecutionConfig(), claim=args["claim"], materials=args["materials"])
    assert len(calls) == 1 and command == ["python", "eval.py"]
    assert calls[0]["plan"]["task"]["command"] == [] and plan.model_dump(mode="json") == before
    assert audit["flow_binding"]["status"] == "bound" and audit["flow_files"] == args["supplied_files"]
    assert audit["flow_binding"]["alignment"] is False and audit["flow_binding"]["support"] is False
