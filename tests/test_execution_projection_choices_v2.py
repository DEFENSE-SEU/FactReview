"""Finite choice contracts; every model/runtime boundary is injected."""

import copy
import json

import pytest

from tests.test_execution_projection_semantics_v2 import inputs
from verification.execution_projection import ProjectionError, field_inventory
from verification.execution_projection_choices import (
    build_choice_registry,
    choice_context,
    decode_choice_reviews,
    decode_choices,
    revalidate_registry,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External/default boundary must be explicitly mocked")

    for name in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "screening.checks.llm_json",
    ):
        monkeypatch.setattr(name, forbidden)


def candidate(tmp_path):
    claim, materials, _, _ = inputs(tmp_path)
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    return claim, materials, registry, next(iter(registry["candidates"].values()))


def select(c):
    return dict(
        version="released-predictions-choice-v1",
        condition_id=c["condition_id"],
        decision="select",
        candidate_id=c["candidate_id"],
        rationale="Explicit injected selection of the full original condition.",
    )


def review(c):
    return dict(
        version="released-predictions-choice-v1",
        condition_id=c["condition_id"],
        candidate_id=c["candidate_id"],
        classification="absolute_fixed_predictions",
        decision="confirmed",
        reviews=[
            dict(
                obligation_id=o["obligation_id"],
                decision="confirmed",
                rationale="Independent injected interpretation of this unchanged obligation.",
            )
            for o in c["obligations"]
        ],
        rationale="All original fields and whole claim reviewed.",
    )


def test_original_006_registry_preserves_full_input(tmp_path):
    claim, materials, registry, c = candidate(tmp_path)
    assert registry["snapshot"]["claim"] == claim.model_dump(mode="json")
    assert len(c["proposal"]["field_bindings"]) == len(field_inventory(claim.conditions[0])) == 10
    assert c["proposal"]["claim_bindings"][0]["start"] == 0
    assert c["proposal"]["claim_bindings"][0]["end"] == len(claim.text)
    assert choice_context(registry)["status"] == "structural_choices_only"
    revalidate_registry(registry, claim, materials)


@pytest.mark.parametrize("key,value", [("training_epochs", 4), ("F1", 0.75), ("foo", 4)])
def test_value_alone_cannot_authorize_numeric_role(tmp_path, key, value):
    claim, materials, _, _ = inputs(tmp_path)
    claim.conditions[0].settings[key] = value
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"]
    assert registry["unavailable"]


def test_explicit_select_and_independent_all_obligations(tmp_path):
    _, _, registry, c = candidate(tmp_path)
    selected, errors = decode_choices([select(c)], registry, [])
    assert not errors and len(selected) == 1
    reviews, errors = decode_choice_reviews([review(c)], selected, registry)
    assert not errors and len(reviews) == 1


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unknown", "padded", "unresolved"])
def test_bad_review_cannot_be_restored(tmp_path, kind):
    _, _, registry, c = candidate(tmp_path)
    selected, _ = decode_choices([select(c)], registry, [])
    bad = review(c)
    if kind == "duplicate":
        bad["reviews"].append(copy.deepcopy(bad["reviews"][0]))
    elif kind == "missing":
        bad["reviews"].pop()
    elif kind == "unknown":
        bad["reviews"][0]["obligation_id"] = "unknown"
    elif kind == "padded":
        bad["condition_id"] = " " + c["condition_id"] + " "
    else:
        bad["reviews"][0]["decision"] = "unresolved"
    decisions, errors = decode_choice_reviews([bad, review(c)], selected, registry)
    assert not decisions and errors


@pytest.mark.parametrize("kind", ["registry", "claim", "block", "markdown"])
def test_frozen_registry_rejects_mutation(tmp_path, kind):
    claim, materials, registry, c = candidate(tmp_path)
    if kind == "registry":
        c["reported"]["token"] = "0.9"
    elif kind == "claim":
        claim.text += " It generalizes."
    elif kind == "block":
        materials.blocks[0] = materials.blocks[0].model_copy(update={"text": "changed"})
    else:
        materials.markdown += "changed"
    with pytest.raises(ProjectionError):
        revalidate_registry(registry, claim, materials)


def public(tmp_path, *, first_change=None, scope_change=None):
    from common.run_stats import run_scope
    from verification.experiments import verify_experiments

    claim, materials, _, _ = inputs(tmp_path)
    calls, originals, cached = [], [], []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        calls.append(dict(module=kw["module"], payload=copy.deepcopy(payload)))
        context = payload.get("execution_choice_context")
        c = next(iter(context["candidates"].values())) if context else cached[0]
        if not cached:
            cached.append(c)
        if kw["module"] == "verification.experiments":
            raw = dict(
                checked_aspects=["correspondence", "fairness", "isolation", "stability", "consistency"],
                items=[],
                plans=[],
                execution_choices=[select(c)],
            )
            if first_change:
                first_change(raw, claim, materials)
        else:
            assert kw["module"] == "verification.experiments.scope"
            raw = dict(
                schema_version="catalog-v2", conditions=[], items=[], execution_choice_reviews=[review(c)]
            )
            if scope_change:
                scope_change(raw, claim, materials)
        originals.append(copy.deepcopy(raw))
        return raw

    with run_scope(tmp_path / "stats.json"):
        result = verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=0)
    return claim, materials, result, calls, originals


def test_public_two_calls_raw_and_instance_roundtrip_and_l3(tmp_path):
    from pathlib import Path

    from schemas.claim import ChoiceProjectionRecord, ExecutionPlan, ProjectedExecutionTargetBinding
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, calls, raw = public(tmp_path)
    assert len(calls) == 2
    assert len(result.plans) == 1, result.issues
    plan = result.plans[0]
    assert plan.feasibility == "ready" and plan.target_conditions == claim.conditions
    assert plan.y_paper == {"c1": 0.75}
    b = plan.target_bindings["c1"]
    assert isinstance(b.projection, ChoiceProjectionRecord)
    for clone in (
        ProjectedExecutionTargetBinding.model_validate(b),
        ProjectedExecutionTargetBinding.model_validate(b.model_dump()),
        ProjectedExecutionTargetBinding.model_validate_json(b.model_dump_json()),
    ):
        assert clone.projection.record_version == "released-predictions-choice-v1"
        assert (
            clone.projection.selection == b.projection.selection
            and clone.projection.choice_review == b.projection.choice_review
        )
        assert clone.model_dump() == b.model_dump()
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
    audit = json.loads(Path(b.projection.scope_audit).read_text("utf-8"))
    assert audit["first_pass_response"] == raw[0] and audit["response"] == raw[1]
    assert audit["input"] == calls[1]["payload"]
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
    claim.notes.append("Downstream annotation")
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
    claim.needs = []
    with pytest.raises(ValueError):
        validate_plan_targets(plan, claim, materials)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "unresolved",
        "unknown",
        "padded",
        "duplicate",
        "wrong_version",
        "malformed",
        "null",
        "false",
        "source_change",
    ],
)
def test_public_bad_first_selection_never_auto_upgrades(tmp_path, kind):
    def change(raw, claim, materials):
        row = raw["execution_choices"][0]
        if kind == "empty":
            raw["execution_choices"] = []
        elif kind == "unresolved":
            row["decision"] = "unresolved"
            row.pop("candidate_id")
        elif kind == "unknown":
            row["candidate_id"] = "choice_unknown"
        elif kind == "padded":
            row["condition_id"] = " c1 "
        elif kind == "duplicate":
            raw["execution_choices"].append(copy.deepcopy(row))
        elif kind == "wrong_version":
            row["version"] = "unknown"
        elif kind == "malformed":
            raw["execution_choices"] = {"bad": True}
        elif kind == "null":
            raw["execution_choices"] = None
        elif kind == "false":
            raw["execution_choices"] = False
        else:
            materials.markdown += " changed during first call"

    _, _, result, calls, _ = public(tmp_path, first_change=change)
    assert not result.plans and len(calls) == 1
    if kind != "empty":
        assert result.issues


@pytest.mark.parametrize(
    "kind",
    ["missing", "duplicate", "unknown_obligation", "wrong_version", "source_change", "resource_change"],
)
def test_public_bad_scope_never_creates_plan(tmp_path, kind):
    def change(raw, claim, materials):
        row = raw["execution_choice_reviews"][0]
        if kind == "missing":
            row["reviews"].pop()
        elif kind == "duplicate":
            raw["execution_choice_reviews"].append(copy.deepcopy(row))
        elif kind == "unknown_obligation":
            row["reviews"][0]["obligation_id"] = "ob_unknown"
        elif kind == "wrong_version":
            row["version"] = "unknown"
        elif kind == "source_change":
            materials.blocks[0].text += " changed during second call"
        else:
            from pathlib import Path

            path = Path(materials.repository.root) / "data/test_predictions.json"
            path.write_text("[]", encoding="utf-8")

    _, _, result, calls, _ = public(tmp_path, scope_change=change)
    assert not result.plans and len(calls) == 2 and result.verification_limitations


def test_prediction_disagreement_does_not_pre_filter_structure(tmp_path):
    from pathlib import Path

    from tests.test_execution_projection_v2 import refresh

    claim, materials, _, _ = inputs(tmp_path)
    before = build_choice_registry(claim, materials)
    file = Path(materials.repository.root) / "data/test_predictions.json"
    rows = json.loads(file.read_text("utf-8"))
    for row in rows:
        row["prediction"] = row["label"]
    file.write_text(json.dumps(rows), encoding="utf-8")
    refresh(materials, "data/test_predictions.json")
    after = build_choice_registry(claim, materials)
    assert len(before["candidates"]) == len(after["candidates"]) == 1
    assert before["registry_sha256"] != after["registry_sha256"]
    assert next(iter(after["candidates"].values()))["value"] == 0.75


def test_choice_runs_existing_native_host_measurement_with_mock_runner(tmp_path):
    from pathlib import Path

    from assessment.rules import assess_claim
    from fact_generation.execution.v2 import ExecutionConfig, Observation, RunOutcome, execute_plans

    claim, materials, verified, calls, _ = public(tmp_path)
    runs = []
    observation = Observation(
        dataset="MiniSet", metric="accuracy", settings={"split": "test", "model": "ExactMatch"}, value=0.75
    )

    def runner(request):
        runs.append(request)
        return RunOutcome(
            returncode=0,
            observations=[observation],
            stdout=json.dumps({"observations": [observation.model_dump(mode="json")]}),
        )

    result = execute_plans(
        verified.plans,
        [claim],
        materials,
        tmp_path / "execution",
        config=ExecutionConfig(refine_with_llm=False, training_budget=0),
        runner=runner,
    )
    assert len(calls) == 2 and len(runs) == 1
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.aligned and evidence.direction == "support"
    measurement = json.loads(Path(evidence.pointer.locator).read_text("utf-8"))
    assert measurement["sample_count"] == 4 and measurement["measurement"]["value"] == 0.75
    assert measurement["measurement"]["unit"] == "fraction"
    assert "no model inference or training performed" in evidence.note
    assert assess_claim(result.claims[0]).status.value == "supported"


@pytest.mark.parametrize("kind", ["selection", "review", "candidate", "registry", "condition", "audit_raw"])
def test_consumer_rejects_saved_record_tampering(tmp_path, kind):
    from pathlib import Path

    from verification.experiment_targets import validate_plan_targets

    claim, materials, verified, _, _ = public(tmp_path)
    plan = verified.plans[0]
    p = plan.target_bindings["c1"].projection
    if kind == "selection":
        p.selection_index = 1
    elif kind == "review":
        p.review_index = 1
    elif kind == "candidate":
        p.candidate_id = "unknown"
    elif kind == "registry":
        p.registry_snapshot["snapshot"]["stable_claim_sha256"] = "changed"
    elif kind == "condition":
        plan.target_conditions[0].settings["examples"] = 5
    else:
        path = Path(p.scope_audit)
        data = json.loads(path.read_text("utf-8"))
        data["first_pass_response"]["execution_choices"] = []
        path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ValueError):
        validate_plan_targets(plan, claim, materials)


def test_legacy_null_is_never_auto_selected(tmp_path):
    from tests.test_execution_projection_v2 import proposal_input
    from verification.contracts import RejectedPlan

    def old_target(raw, claim, materials):
        old = proposal_input(claim, materials, tmp_path)
        plan = copy.deepcopy(old[4])
        plan["targets"][0]["projection"] = None
        raw["plans"] = [plan]
        raw.pop("execution_choices")

    with pytest.raises(RejectedPlan, match="condition's metric") as error:
        public(tmp_path, first_change=old_target)
    assert not error.value.observations.plans


def test_legacy_and_choice_collision_rejects_both(tmp_path):
    from tests.test_execution_projection_v2 import proposal_input

    def collide(raw, claim, materials):
        raw["plans"] = [copy.deepcopy(proposal_input(claim, materials, tmp_path)[4])]

    _, _, result, calls, _ = public(tmp_path, first_change=collide)
    assert not result.plans and any("legacy" in s.lower() for s in result.issues)
    assert len(calls) == 2


def test_cap_does_not_silently_take_first_candidate(tmp_path, monkeypatch):
    claim, materials, _, _ = inputs(tmp_path)
    monkeypatch.setattr("verification.execution_projection_choices.MAX_CHOICES", 0)
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"] and "bound exceeded" in registry["unavailable"][0]["reason"]


def test_total_combination_budget_rejects_partial_registry(tmp_path, monkeypatch):
    claim, materials, _, _ = inputs(tmp_path)
    good = build_choice_registry(claim, materials)
    assert len(good["candidates"]) == 1
    monkeypatch.setattr("verification.execution_projection_choices.MAX_COMBINATION_CHECKS", 1)
    bounded = build_choice_registry(claim, materials)
    assert not bounded["candidates"]
    assert bounded["budget"]["checks_used"] == 1
    assert "budget exhausted" in bounded["unavailable"][0]["reason"]


def test_resource_budget_stops_recipe_reads(tmp_path, monkeypatch):
    claim, materials, _, _ = inputs(tmp_path)
    monkeypatch.setattr("verification.execution_projection_choices.MAX_RESOURCE_BYTES", 1)

    def forbidden(*args):
        pytest.fail("Oversized combination must be rejected before recipe reads")

    monkeypatch.setattr("verification.execution_projection_choices.entry_recipe", forbidden)
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"] and registry["budget"]["resource_bytes"] == 0
    assert "resource-size budget exhausted" in registry["unavailable"][0]["reason"]


@pytest.mark.parametrize("margin,available", [(-1, False), (0, True), (1, True)])
def test_cumulative_resource_budget_exact_boundary(tmp_path, monkeypatch, margin, available):
    claim, materials, _, _ = inputs(tmp_path)
    used = build_choice_registry(claim, materials)["budget"]["resource_bytes"]
    assert used > 1
    monkeypatch.setattr("verification.execution_projection_choices.MAX_RESOURCE_BYTES", used + margin)
    registry = build_choice_registry(claim, materials)
    assert bool(registry["candidates"]) is available
    assert registry["budget"]["resource_bytes"] <= used + margin


def test_repeated_resource_combinations_count_toward_conservative_budget(tmp_path, monkeypatch):
    claim, materials, _, _ = inputs(tmp_path)
    used = build_choice_registry(claim, materials)["budget"]["resource_bytes"]
    # Duplicate index choices must not create unbounded repeated reads. The
    # budget is cumulative work, including repeated immutable input paths.
    materials.repository.entry_scripts.append(materials.repository.entry_scripts[0])
    repeated = build_choice_registry(claim, materials)
    assert repeated["budget"]["resource_bytes"] > used
    monkeypatch.setattr("verification.execution_projection_choices.MAX_RESOURCE_BYTES", used)
    bounded = build_choice_registry(claim, materials)
    assert not bounded["candidates"]
    assert "resource-size budget exhausted" in bounded["unavailable"][0]["reason"]


def test_invalid_choice_retains_partial_paper_observation(tmp_path):
    from verification.experiment_catalog import build_catalog

    def first(raw, claim, materials):
        raw["execution_choices"][0]["candidate_id"] = "choice_unknown"
        raw["items"] = [
            dict(
                aspect="correspondence",
                kind="paper_support",
                block_id=claim.source_block_id,
                quote=claim.source_quote,
                covered=["c1"],
                fully_supported_conditions=[],
                detail="An unchanged partial observation.",
            )
        ]

    def scope(raw, claim, materials):
        source_id = next(
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
                grounds_source_ids=[source_id],
                rationale="Original partial observation.",
                full_support=False,
                qualifiers_complete=False,
                comparison_objects="not_comparative",
            )
        ]

    _, _, result, calls, _ = public(tmp_path, first_change=first, scope_change=scope)
    assert len(calls) == 2 and not result.plans
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.evidence[0].covered == ["c1"]


@pytest.mark.parametrize(
    "kind", ["claim_suffix", "metric", "sample", "description", "bool", "nested_numeric"]
)
def test_no_candidate_for_incomplete_or_changed_finite_meaning(tmp_path, kind):
    claim, materials, _, _ = inputs(tmp_path)
    condition = claim.conditions[0]
    if kind == "claim_suffix":
        claim.text += " The method generalizes to new populations."
    elif kind == "metric":
        condition.metric = "F1"
    elif kind == "sample":
        condition.settings["scope_qualifiers"][0] = "Four fixed validation predictions"
    elif kind == "description":
        condition.description += " with inferred future performance"
    elif kind == "bool":
        condition.settings["examples"] = True
    else:
        condition.settings["nested"] = {"sample_count": 4}
    assert not build_choice_registry(claim, materials)["candidates"]


def test_decoded_config_keys_precede_numeric_roles():
    from schemas.claim import Condition
    from verification.execution_projection_choices import _field_keys

    cfg = {"dataset": "D", "metric": "accuracy", "settings": {"a/b": 4, "a~b": 0.75, "examples": 4}}
    runtime = Condition(id="c", **cfg)
    assert _field_keys("/settings/a~1b", 4, cfg, runtime, 0.75, 4) == {"runtime:a/b"}
    assert _field_keys("/settings/a~0b", 0.75, cfg, runtime, 0.75, 4) == {"runtime:a~b"}
    assert _field_keys("/settings/examples", 4, cfg, runtime, 0.75, 4) == {"runtime:examples"}
    assert _field_keys("/settings/a/b", 4, cfg, runtime, 0.75, 4) is None
    with pytest.raises(ProjectionError):
        _field_keys("/settings/examples", 4.0, cfg, runtime, 0.75, 4)


def test_unknown_identity_tombstones_whole_choice_response(tmp_path):
    _, _, registry, c = candidate(tmp_path)
    bad = select(c)
    bad["condition_id"] = ["c1"]
    accepted, errors = decode_choices([bad, select(c)], registry, [])
    assert not accepted and errors
    selected, _ = decode_choices([select(c)], registry, [])
    malformed = review(c)
    malformed.pop("condition_id")
    accepted, errors = decode_choice_reviews([malformed, review(c)], selected, registry)
    assert not accepted and errors


def test_duplicate_responses_cannot_restore_after_third_row(tmp_path):
    _, _, registry, c = candidate(tmp_path)
    first = select(c)
    first["candidate_id"] = "unknown"
    accepted, errors = decode_choices([first, select(c), select(c)], registry, [])
    assert not accepted and errors
    selected, _ = decode_choices([select(c)], registry, [])
    bad = review(c)
    bad["reviews"].pop()
    accepted, errors = decode_choice_reviews([bad, review(c), review(c)], selected, registry)
    assert not accepted and errors
