"""Fresh paper support requires source authority; external boundaries are mocked."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.experiment_targets import validate_plan_targets
from verification.experiments import verify_experiments

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


def _offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unmocked model/network/process boundary")
    for target in ("screening.checks.llm_json", "requests.sessions.Session.request",
                   "httpx.Client.send", "httpx.AsyncClient.send", "subprocess.Popen", "subprocess.run"):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))


def _inputs(path, mixed=False, definition=False):
    path.mkdir(parents=True, exist_ok=True)
    internal = ("Method A defines the preprocessing protocol on D test with seed 0, optional features disabled and no additional transforms."
                if definition else "Method A reports accuracy 0.75 on D test with seed 0, optional features disabled and no additional transforms.")
    external = "The released repository supplies no A training result."
    texts = [internal, external] if mixed else [internal]
    conditions = [Condition(id="c1", dataset="D", metric=None if definition else "accuracy",
        settings={"split": "test", "seed": 0, "optional": False, "empty": {}})]
    if mixed:
        conditions.append(Condition(id="c2", settings={"repository_scope": "released repository"}))
    materials = SharedMaterials(paper_key="authority", source_pdf="paper.pdf", markdown="",
        markdown_path=str(path / "paper.md"), content_list_path="content.json", provider="mock")
    for index, text in enumerate([*texts, "Unrelated model Z reports another result."]):
        start = len(materials.markdown)
        materials.markdown += text + "\n"
        materials.blocks.append(MaterialBlock(id=f"b{index}", text=text,
            loc=ClaimLocation(page=1, char_start=start, char_end=start+len(text))))
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    claim = Claim(id="claim", text=" ".join(texts), loc=ClaimLocation(page=1),
        source_block_id="b0", source_quote=internal,
        source_refs=[ClaimSourceRef(loc=materials.blocks[i].loc, source_block_id=f"b{i}", source_quote=text,
            covered=[conditions[i].id]) for i, text in enumerate(texts)],
        conditions=conditions, needs=["Experiments"])
    items = [{"aspect": "correspondence", "kind": "paper_support", "block_id": f"b{i}",
        "quote": text, "covered": [conditions[i].id], "fully_supported_conditions": [conditions[i].id],
        "detail": "Independent source-authority review requested."} for i, text in enumerate(texts)]
    return claim, materials, items


def _scope(payload, items, *, external=False, definition=False):
    registry = payload["source_authority_context"]
    conditions = [{"condition_id": c["id"], "assertion": "descriptive", "relation": "none",
        "matched_controls_required": False, "uncertainty_sensitive": False,
        "setting_scopes": {key: "shared" for key in c["settings"]},
        "rationale": "Review this condition and shared qualifiers with other conditions visible."}
        for c in payload["claim"]["conditions"]]
    rows = []
    for i, item in enumerate(items):
        source = next(key for key, record in payload["catalog"]["sources"].items()
                      if record[0] == item["block_id"] and record[3] == "paper_block")
        rows.append({"item_index": i, "condition_id": item["covered"][0], "applicability": "applicable",
            "grounds_source_ids": [source], "rationale": "Exact original source is independently classified.",
            "full_support": True, "qualifiers_complete": True, "comparison_objects": "not_comparative",
            "comparisons": [], "source_uses": []})
    authority = {"schema_version": "source-authority-v1", "context_digest": registry["context_digest"],
        "requirements": [{"unit_id": u["unit_id"], "requirement":
            "external_availability" if external and u["condition_id"] == "c2" else
            ("manuscript_definition" if definition else "manuscript_reported_result"),
            "claim_source_ids": u["claim_source_ids"], "rationale": "Explicit fixture semantic classification."}
            for u in registry["units"]], "candidates": []}
    for pair in registry["pairs"]:
        authority["candidates"].append({"pair_id": pair["pair_id"], "units": [{"unit_id": key,
            "basis": "author_assertion" if external and pair["condition_id"] == "c2" else
            ("definition_or_protocol" if definition else "reported_measurement"),
            "evidence_source_ids": [pair["primary_source_id"]],
            "rationale": "Explicit original passage meaning; no metadata grant."} for key in pair["unit_ids"]]})
    return {"schema_version": "catalog-v3", "paper_scope_format": "catalog-v2", "conditions": conditions, "items": rows,
            "source_authority": authority}


def test_fresh_default_dispatch_and_standalone_retain_complete_same_quote_authority(tmp_path, monkeypatch):
    _offline(monkeypatch)
    for definition in (False, True):
        claim, materials, items = _inputs(tmp_path / str(definition), definition=definition)
        original = copy.deepcopy((claim.model_dump(mode="json"), materials.model_dump(mode="json")))
        calls = []
        def model(**kwargs):
            payload = json.loads(kwargs["prompt"])
            calls.append((kwargs["module"], copy.deepcopy(payload)))
            if kwargs["module"] == "verification.experiments":
                return {"checked_aspects": ASPECTS, "items": items, "plans": []}
            assert kwargs["module"] == "verification.experiments.scope"
            assert payload["output_schema"]["properties"]["schema_version"]["const"] == "catalog-v3"
            registry = payload["source_authority_context"]
            assert payload["claim"] == original[0] and payload["paper_blocks"] == original[1]["blocks"]
            assert [u["field_pointer"] for u in registry["units"]] == ["/text", "/conditions/0/dataset", *([] if definition else ["/conditions/0/metric"]), "/conditions/0/settings/split", "/conditions/0/settings/seed", "/conditions/0/settings/optional", "/conditions/0/settings/empty"]
            assert [u["value"] for u in registry["units"]][-3:] == [0, False, {}]
            return _scope(payload, items, definition=definition)
        with run_stats.run_scope(tmp_path / f"stats-{definition}.json"):
            if definition:
                result = verify_experiments(claim, materials, call=model)
                evidence = result.evidence
            else:
                result = asyncio.run(verify_claims([claim], materials, tmp_path / "verification",
                    call=model, global_literature=lambda c, m: BranchResult()))
                evidence = result.claims[0].evidence
        assert len(evidence) == 1 and evidence[0].sufficient and evidence[0].covered == ["c1"], result.issues
        assert [module for module, _ in calls] == ["verification.experiments", "verification.experiments.scope"]
        assert (claim.model_dump(mode="json"), materials.model_dump(mode="json")) == original
        assert evidence[0].pointer.quote == items[0]["quote"] == claim.source_quote
        audits = sorted(tmp_path.glob("experiment_scope/*.json"), key=lambda p: p.stat().st_mtime_ns)
        audit = json.loads(audits[-1].read_text(encoding="utf-8"))
        assert audit["response"]["schema_version"] == "catalog-v3"
        assert audit["input"]["source_authority_context"] == calls[-1][1]["source_authority_context"]


def test_external_self_assertion_closed_sources_and_retained_execution_fail_closed(tmp_path, monkeypatch):
    _offline(monkeypatch)
    for mode in ("external", "missing", "stale", "synthetic", "foreign", "legacy", "empty_list"):
        claim, materials, items = _inputs(tmp_path / mode, mixed=True)
        if mode == "empty_list":
            claim.conditions[0].settings["empty"] = []
        saved = []
        def model(**kwargs):
            payload = json.loads(kwargs["prompt"])
            if kwargs["module"] == "verification.experiments":
                return {"checked_aspects": ASPECTS, "items": items, "plans": []}
            if mode == "empty_list":
                registry = payload["source_authority_context"]
                assert [u["value"] for u in registry["units"] if u["condition_id"] == "c1"][-3:] == [0, False, []]
                assert payload["claim"]["conditions"][0]["settings"]["empty"] == []
            response = _scope(payload, items, external=True)
            authority = response["source_authority"]
            if mode == "missing":
                authority["requirements"].pop()
            elif mode == "stale":
                authority["context_digest"] = "0" * 64
            elif mode == "synthetic":
                authority["requirements"][0]["claim_source_ids"] = [payload["catalog"]["claim_source_id"]]
            elif mode == "foreign":
                foreign = next(k for k, v in payload["catalog"]["sources"].items() if v[0] == "b2")
                authority["candidates"][0]["units"][0]["evidence_source_ids"] = [foreign]
            elif mode == "legacy":
                response.pop("source_authority")
                response["schema_version"] = "catalog-v2"
            saved.append(copy.deepcopy(response))
            return response
        with run_stats.run_scope(tmp_path / f"{mode}.json"):
            result = verify_experiments(claim, materials, call=model)
        assert not any(e.sufficient and "c2" in e.covered for e in result.evidence)
        if mode == "external":
            assert any(e.sufficient and e.covered == ["c1"] for e in result.evidence), result.issues
        else:
            assert not any(e.sufficient for e in result.evidence), result.issues
        assert result.plans == [] and result.issues
        if mode == "empty_list":
            assert any("no finite catalog cases" in issue for issue in result.issues)
        audit = json.loads(sorted((tmp_path / "experiment_scope").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)[-1].read_text(encoding="utf-8"))
        assert audit["response"] == saved[-1]

    # One actual public choice producer still retains its independent 19-obligation contract.
    from tests.test_execution_projection_choices_v2 import review, select
    from tests.test_execution_projection_semantics_v2 import inputs
    (tmp_path / "choice").mkdir()
    claim, materials, _, _ = inputs(tmp_path / "choice")
    calls = []
    def choice_model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(payload)
        candidate = next(iter(payload["execution_choice_context"]["candidates"].values()))
        if kwargs["module"] == "verification.experiments":
            return {"schema_version": "experiments-execution-v3", "checked_aspects": ASPECTS,
                "items": [], "execution": {"kind": "choice", "selection": select(candidate)}}
        response = _scope(payload, [], external=False)
        for row in response["source_authority"]["requirements"]:
            row["requirement"] = "unresolved"
        response["execution_choice_reviews"] = [review(candidate)]
        return response
    with run_stats.run_scope(tmp_path / "choice.json"):
        result = verify_experiments(claim, materials, call=choice_model, scope_binding_repair_rounds=0)
    assert len(result.plans) == 1 and result.evidence == [], result.issues
    assert len(next(iter(calls[0]["execution_choice_context"]["candidates"].values()))["obligations"]) == 19
    plan = result.plans[0]
    validate_plan_targets(plan, claim, materials)
    path = Path(plan.target_bindings["c1"].projection.scope_audit)
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["response"]["schema_version"] == "catalog-v3"
    audit["response"]["source_authority"]["context_digest"] = "0" * 64
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_plan_targets(plan, claim, materials)
