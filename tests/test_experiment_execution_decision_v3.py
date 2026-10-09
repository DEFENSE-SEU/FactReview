"""Explicit execution wire; all model/service boundaries stay injected."""

import copy
import json
from pathlib import Path

import pytest

from tests.test_execution_projection_choices_v2 import offline, public  # noqa: F401


def choice_wire(raw, *_):
    selection = raw.pop("execution_choices")[0]
    assert raw.pop("plans") == []
    raw["schema_version"] = "experiments-execution-v3"
    raw["execution"] = {"kind": "choice", "selection": selection}


def test_explicit_choice_raw_scope_roundtrip_and_consumer(tmp_path):
    from schemas.claim import ExecutionPlan
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, calls, raw = public(tmp_path, first_change=choice_wire)
    assert len(calls) == 2 and len(result.plans) == 1, result.issues
    plan = result.plans[0]
    projection = plan.target_bindings["c1"].projection
    audit = json.loads(Path(projection.scope_audit).read_text("utf-8"))
    assert audit["first_pass_response"] == raw[0]
    assert audit["first_pass_normalized"]["execution_choices"] == [raw[0]["execution"]["selection"]]
    assert audit["input"] == calls[1]["payload"]
    assert audit["input"]["candidate_plans"] == []
    assert audit["execution_wire"]["raw_origin_pointer"] == "/execution/selection"
    assert projection.selection.model_dump(mode="json") == raw[0]["execution"]["selection"]
    clone = ExecutionPlan.model_validate_json(plan.model_dump_json())
    assert clone == plan and validate_plan_targets(clone, claim, materials) == plan.target_bindings


def paper_item(raw, claim):
    raw["items"] = [
        dict(
            aspect="correspondence",
            kind="paper_support",
            block_id=claim.source_block_id,
            quote=claim.source_quote,
            covered=["c1"],
            fully_supported_conditions=[],
            detail="Unchanged partial paper observation.",
        )
    ]


def paper_scope(raw, claim, materials):
    from verification.experiment_catalog import build_catalog

    sid = next(
        sid
        for sid, s in build_catalog(claim, materials)["sources"].items()
        if s.get("block_id") == claim.source_block_id and s["kind"] == "claim_source"
    )
    raw["conditions"] = [
        dict(
            condition_id="c1",
            assertion="descriptive",
            matched_controls_required=False,
            uncertainty_sensitive=False,
            relation="none",
            rationale="Fixed predictions.",
        )
    ]
    raw["items"] = [
        dict(
            item_index=0,
            condition_id="c1",
            applicability="applicable",
            grounds_source_ids=[sid],
            rationale="Partial observation.",
            full_support=False,
            qualifiers_complete=False,
            comparison_objects="not_comparative",
        )
    ]


@pytest.mark.parametrize("bad", ["mixed", "unknown_kind", "missing", "wrong_version", "unknown_candidate"])
def test_bad_execution_is_local_and_never_selected_automatically(tmp_path, bad):
    def first(raw, claim, materials):
        choice_wire(raw)
        paper_item(raw, claim)
        if bad == "mixed":
            raw["plans"] = []
        elif bad == "unknown_kind":
            raw["execution"]["kind"] = "choose_it"
        elif bad == "missing":
            raw.pop("execution")
        elif bad == "wrong_version":
            raw["schema_version"] = "future-version"
        else:
            raw["execution"]["selection"]["candidate_id"] = "choice_unknown"

    _, _, result, calls, _ = public(tmp_path, first_change=first, scope_change=paper_scope)
    assert not result.plans and result.issues
    assert len(calls) == 2 and "execution_choice_context" not in calls[1]["payload"]
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.evidence[0].covered == ["c1"]


def test_none_and_legacy_lower_only_the_explicit_branch(tmp_path):
    from tests.test_execution_projection_semantics_v2 import inputs
    from verification.experiments import lower_execution_response

    def first(raw, *_):
        choice_wire(raw)
        raw["execution"] = {"kind": "none", "rationale": "No execution requested."}

    _, _, result, calls, raw = public(tmp_path / "none", first_change=first)
    assert not result.plans and len(calls) == 1
    claim, materials, proposal, old = inputs(tmp_path / "legacy")
    plan = copy.deepcopy(old[4])
    plan["targets"][0]["projection"] = proposal.model_dump(mode="json")
    versioned = {**raw[0], "execution": {"kind": "legacy", "plan": plan}}
    before = copy.deepcopy(versioned)
    lowered, wire = lower_execution_response(versioned)
    assert lowered["plans"] == [plan] and lowered["execution_choices"] == []
    assert wire["raw_origin_pointer"] == "/execution/plan" and versioned == before
    assert claim.conditions and materials.blocks  # Original fixture and target remain intact.


@pytest.mark.parametrize("change", ["raw", "normalized", "missing_origin", "adapter_hash", "pointer"])
def test_choice_consumer_rejects_tampered_origin_even_with_updated_file_hash(tmp_path, change):
    from verification.execution_projection import file_hash
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, _, _ = public(tmp_path, first_change=choice_wire)
    plan = result.plans[0]
    projection = plan.target_bindings["c1"].projection
    path = Path(projection.scope_audit)
    audit = json.loads(path.read_text("utf-8"))
    if change == "raw":
        audit["first_pass_response"]["execution"]["selection"]["rationale"] += " changed"
    elif change == "normalized":
        audit["first_pass_normalized"]["execution_choices"][0]["rationale"] += " changed"
    elif change == "missing_origin":
        audit.pop("execution_wire")
    elif change == "adapter_hash":
        audit["execution_wire"]["adapter_sha256"] = "0" * 64
    else:
        audit["execution_wire"]["raw_origin_pointer"] = "/plans/0"
    path.write_text(json.dumps(audit), encoding="utf-8")
    projection.scope_audit_sha256 = file_hash(path)
    with pytest.raises(ValueError):
        validate_plan_targets(plan, claim, materials)


@pytest.mark.parametrize("phase", ["selection", "review", "malformed_key"])
def test_private_nested_choice_cannot_escape_or_revive(tmp_path, monkeypatch, phase):
    from llm.client import LLMConfig
    from verification.execution_projection_choices import retained_choice_first_response

    key = "sk-v3-unit-NOT-A-REAL-KEY-37921"
    cfg = LLMConfig(provider="mock", model="mock", base_url="https://fixture.invalid/v1", api_key=key)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    held = []

    def first(raw, claim, materials):
        choice_wire(raw)
        if phase == "selection":
            raw["execution"]["selection"]["rationale"] += key
        elif phase == "malformed_key":
            raw["execution"]["selection"][key] = {"[REDACTED]": 1, key: 2}
        paper_item(raw, claim)
        held.append((raw, copy.deepcopy(raw)))

    def scope(raw, claim, materials):
        paper_scope(raw, claim, materials)
        if phase == "review":
            raw["execution_choice_reviews"][0]["rationale"] += key

    _, _, result, calls, _ = public(tmp_path, first_change=first, scope_change=scope)
    assert not result.plans and len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert key not in result.model_dump_json() and key not in json.dumps(calls)
    assert all(raw == old for raw, old in held)
    audits = [
        p for p in tmp_path.rglob("*.json") if p.parent.name in {"experiment_scope", "execution_decisions"}
    ]
    assert audits and all(key not in p.read_text("utf-8") for p in audits)
    if phase == "selection":
        audit = json.loads(
            next(p for p in audits if p.parent.name == "execution_decisions").read_text("utf-8")
        )
        assert audit["execution_wire"]["wire_representation"] == "redacted_copy"
        audit.pop("choice_privacy", None)
        with pytest.raises(ValueError):
            retained_choice_first_response(audit)


def test_lowering_keeps_raw_whitespace_and_does_not_claim_it_was_normalized():
    from verification.experiments import lower_execution_response

    selection = dict(
        version="released-predictions-choice-v1",
        condition_id="c1",
        decision="select",
        candidate_id="choice-example",
        rationale="  raw rationale  ",
    )
    raw = dict(
        schema_version="experiments-execution-v3",
        checked_aspects=[],
        items=[],
        execution=dict(kind="choice", selection=selection),
    )
    lowered, _ = lower_execution_response(raw)
    assert lowered["execution_choices"] == [selection]
    assert lowered["execution_choices"][0]["rationale"] == "  raw rationale  "


def test_bound_origin_cannot_downgrade_by_replacing_only_audit_and_file_hash(tmp_path):
    from verification.execution_projection import file_hash
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, _, _ = public(tmp_path, first_change=choice_wire)
    plan = result.plans[0]
    assert validate_plan_targets(plan, claim, materials)
    projection = plan.target_bindings["c1"].projection
    path = Path(projection.scope_audit)
    audit = json.loads(path.read_text("utf-8"))
    audit["first_pass_response"] = audit.pop("first_pass_normalized")
    audit.pop("execution_wire")
    path.write_text(json.dumps(audit), encoding="utf-8")
    projection.scope_audit_sha256 = file_hash(path)
    with pytest.raises(ValueError):
        validate_plan_targets(plan, claim, materials)


@pytest.mark.parametrize("versioned", [False, True])
def test_all_generated_choices_bind_origin_and_unbound_json_is_readable_only(tmp_path, versioned):
    from schemas.claim import ExecutionPlan
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, _, _ = public(tmp_path, first_change=choice_wire if versioned else None)
    plan = result.plans[0]
    origin = plan.target_bindings["c1"].projection.execution_origin
    assert origin is not None
    assert origin.wire_version == ("experiments-execution-v3" if versioned else "legacy")
    assert validate_plan_targets(plan, claim, materials)
    raw = plan.model_dump(mode="json")
    raw["target_bindings"]["c1"]["projection"].pop("execution_origin")
    old = ExecutionPlan.model_validate(raw)
    assert old.target_bindings["c1"].projection.execution_origin is None
    with pytest.raises(ValueError):
        validate_plan_targets(old, claim, materials)
