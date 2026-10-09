"""Injected semantic decisions test the finite contract, not model accuracy."""

import copy
import json
from pathlib import Path

import pytest

from schemas.claim import Claim, SemanticPredictionProjection, SemanticProjectionScopeDecision
from tests.test_execution_projection_v2 import original_inputs, proposal_input
from verification.execution_projection import ProjectionError, digest, field_inventory
from verification.execution_projection_semantics import validate_semantic_obligations


def inputs(tmp_path):
    _, materials = original_inputs(tmp_path)
    saved = json.loads((Path(__file__).parent / "fixtures/execution_projection_006.json").read_text("utf-8"))
    claim = Claim.model_validate(saved["claim"])
    old = proposal_input(claim, materials, tmp_path)
    source = old[0].field_roles[0].source_ids
    from verification.experiment_catalog import build_catalog

    catalog = build_catalog(claim, materials)
    ranking = [
        next(
            k
            for k, v in catalog["sources"].items()
            if v.get("block_id") == "block_11" and v["kind"] == "paper_block"
        )
    ]
    atoms = [
        dict(id="dataset", kind="dataset_identity", source_ids=source, dataset="MiniSet", split="test"),
        dict(id="model", kind="runtime_setting", source_ids=source, key="model", value="ExactMatch"),
        dict(id="value", kind="reported_value", source_ids=source, number_id=old[3]["number_id"]),
        dict(
            id="sample",
            kind="sample_scope",
            source_ids=source,
            count=4,
            population="fixed_released_predictions",
            split="test",
            all_records=True,
        ),
        dict(
            id="definition",
            kind="measurement_definition",
            source_ids=source,
            measure="exact_match_accuracy",
            predicate="prediction_equals_label",
            aggregation="fraction_of_all_records",
            unit="fraction",
        ),
        dict(
            id="boundaries",
            kind="conclusion_boundary",
            source_ids=source,
            excludes=["repeated_run_uncertainty", "population_performance"],
        ),
        dict(id="ranking", kind="conclusion_boundary", source_ids=ranking, excludes=["cross_model_ranking"]),
    ]
    ids = {
        "/dataset": ["dataset"],
        "/metric": ["dataset", "definition"],
        "/settings/model": ["model"],
        "/settings/accuracy": ["value"],
        "/settings/examples": ["sample"],
        "/settings/accuracy_computation": ["definition"],
        "/settings/scope_qualifiers/0": ["sample", "dataset"],
        "/settings/scope_qualifiers/1": ["boundaries"],
        "/settings/scope_qualifiers/2": ["ranking"],
        "/description": ["dataset", "model", "definition"],
    }
    proposal = SemanticPredictionProjection(
        version="released-predictions-v2",
        recipe="exact_match_accuracy",
        data_path="data/test_predictions.json",
        atoms=atoms,
        field_bindings=[dict(path=k, atom_ids=v) for k, v in ids.items()],
        claim_bindings=[
            dict(start=0, end=len(claim.text), atom_ids=["dataset", "model", "value", "sample", "definition"])
        ],
    )
    return claim, materials, proposal, old


def confirm(proposal):
    return SemanticProjectionScopeDecision(
        version=proposal.version,
        plan_index=0,
        condition_id="c1",
        classification="absolute_fixed_predictions",
        dataset_identity_confirmed=True,
        measurement_definition_confirmed=True,
        all_original_qualifiers_preserved=True,
        confirmed_field_paths=[r.path for r in proposal.field_bindings],
        source_ids=list(dict.fromkeys(s for a in proposal.atoms for s in a.source_ids)),
        unresolved=[],
        rationale="Injected independent semantic review.",
        proposal_sha256=digest(proposal.model_dump(mode="json")),
        atom_reviews=[
            dict(
                atom_id=a.id,
                decision="confirmed",
                source_ids=a.source_ids,
                rationale="Explicit original clause.",
            )
            for a in proposal.atoms
        ],
        field_reviews=[
            dict(**r.model_dump(), decision="confirmed", rationale="Whole original value retained.")
            for r in proposal.field_bindings
        ],
        claim_reviews=[
            dict(**r.model_dump(), decision="confirmed", rationale="Every original clause retained.")
            for r in proposal.claim_bindings
        ],
    )


def validate(claim, materials, proposal, old):
    from verification.execution_projection import entry_recipe
    from verification.experiment_catalog import build_catalog

    recipe = entry_recipe(materials, "scripts/evaluate.py", "configs/evaluation.json", proposal.data_path)
    return validate_semantic_obligations(
        claim,
        claim.conditions[0],
        materials,
        proposal,
        confirm(proposal),
        recipe,
        build_catalog(claim, materials),
        old[3]["number_id"],
        0.75,
    )


def test_original_006_full_semantics_without_rewriting(tmp_path):
    claim, materials, proposal, old = inputs(tmp_path)
    before = copy.deepcopy(claim.model_dump(mode="json"))
    resolved = validate(claim, materials, proposal, old)
    assert resolved["runtime"].metric == "accuracy"
    assert resolved["sample_count"] == 4
    assert len(resolved["field_consumption"]) == len(field_inventory(claim.conditions[0])) == 10
    assert claim.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "change",
    [
        "derived_metric",
        "population",
        "inference",
        "comparison",
        "definition",
        "unknown",
        "omit",
        "duplicate",
        "empty",
        "boolean_count",
        "wrong_split",
    ],
)
def test_wrong_confirmed_scope_cannot_discharge_unknown_or_changed_obligations(tmp_path, change):
    claim, materials, proposal, old = inputs(tmp_path)
    c = claim.conditions[0]
    if change == "derived_metric":
        c.metric = "balanced accuracy"
    elif change == "population":
        c.settings["scope_qualifiers"][1] = "population-performance conclusion is claimed"
    elif change == "inference":
        claim.text += " The model was executed on these examples."
    elif change == "comparison":
        claim.text = claim.text.replace("has accuracy", "beats another model with accuracy")
    elif change == "definition":
        c.settings["accuracy_computation"] += " weighted by confidence"
    elif change == "unknown":
        c.settings["unseen"] = "under distribution shift"
    elif change == "omit":
        proposal.field_bindings.pop()
    elif change == "duplicate":
        proposal.field_bindings.append(proposal.field_bindings[0].model_copy())
    elif change == "empty":
        c.settings["unseen"] = {}
    elif change == "boolean_count":
        c.settings["examples"] = True
    elif change == "wrong_split":
        c.dataset = "MiniSet validation"
    # Repairing coverage positions/confirm flags cannot make altered semantics safe.
    proposal.claim_bindings[0].end = len(claim.text)
    with pytest.raises(ProjectionError):
        validate(claim, materials, proposal, old)


def public(tmp_path, damage=None, *, with_observation=False):
    from common.run_stats import run_scope
    from verification.experiments import verify_experiments

    claim, materials, proposal, old = inputs(tmp_path)
    review = confirm(proposal)
    plan = copy.deepcopy(old[4])
    plan["targets"][0]["projection"] = proposal.model_dump(mode="json")
    item = {
        "aspect": "correspondence",
        "kind": "paper_support",
        "block_id": claim.source_block_id,
        "quote": claim.source_quote,
        "covered": ["c1"],
        "fully_supported_conditions": [],
        "detail": "A partial paper observation must survive an invalid execution projection.",
    }
    first = {
        "checked_aspects": ["correspondence", "fairness", "isolation", "stability", "consistency"],
        "items": [item] if with_observation else [],
        "plans": [plan],
    }
    scope = {
        "schema_version": "catalog-v2",
        "conditions": [
            {
                "condition_id": "c1",
                "assertion": "descriptive",
                "matched_controls_required": False,
                "uncertainty_sensitive": False,
                "relation": "none",
                "rationale": "Fixed released predictions.",
            }
        ],
        "items": [
            {
                "item_index": 0,
                "condition_id": "c1",
                "applicability": "applicable",
                "ground_source_ids": [proposal.atoms[0].source_ids[0]],
                "rationale": "Original partial observation.",
                "full_support": False,
                "qualifiers_complete": False,
                "comparison_objects": "not_comparative",
            }
        ]
        if with_observation
        else [],
        "plan_projection_reviews": [review.model_dump(mode="json")],
    }
    calls = []

    def model(**kw):
        calls.append(kw)
        if kw["module"] == "verification.experiments":
            return copy.deepcopy(first)
        assert kw["module"] == "verification.experiments.scope"
        if damage:
            damage(scope, claim, materials)
        return copy.deepcopy(scope)

    with run_scope(tmp_path / "stats.json"):
        result = verify_experiments(claim, materials, call=model)
    return claim, materials, result, calls


def test_public_plan_only_two_calls_full_audit_roundtrip_and_consumer_revalidation(tmp_path):
    from schemas.claim import ExecutionPlan
    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, calls = public(tmp_path)
    assert len(calls) == 2
    assert result.plans[0].feasibility == "ready", result.plans[0].blocker
    plan = result.plans[0]
    assert plan.y_paper == {"c1": 0.75}
    assert plan.target_conditions == claim.conditions
    assert not result.evidence
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
    bound = plan.target_bindings["c1"]
    assert len(bound.projection.field_consumption) == 10
    assert bound.projection.claim_consumption[0]["text"] == claim.text
    assert bound.projection.runtime_target.metric == "accuracy"


@pytest.mark.parametrize(
    "damage",
    ["duplicate", "missing", "version", "bool", "source", "atom", "hash", "paper", "resource", "audit"],
)
def test_public_invalid_review_is_plan_local_and_never_ready(tmp_path, damage):
    def change(scope, claim, materials):
        row = scope["plan_projection_reviews"][0]
        if damage == "duplicate":
            scope["plan_projection_reviews"].append(copy.deepcopy(row))
        elif damage == "missing":
            scope["plan_projection_reviews"] = []
        elif damage == "version":
            row["version"] = "unknown"
        elif damage == "bool":
            row["plan_index"] = False
        elif damage == "source":
            row["atom_reviews"][0]["source_ids"] = ["unknown"]
        elif damage == "atom":
            row["atom_reviews"].pop()
        elif damage == "hash":
            row["proposal_sha256"] = "changed"
        elif damage == "paper":
            Path(materials.markdown_path).write_text("changed", "utf-8")
        elif damage == "resource":
            (Path(materials.repository.root) / "configs/evaluation.json").write_text("{}", "utf-8")
        elif damage == "audit":
            row["field_reviews"][0]["atom_ids"] = ["sample"]

    claim, _, result, calls = public(tmp_path, change, with_observation=True)
    assert len(calls) == 2
    assert not result.plans or result.plans[0].feasibility == "blocked"
    assert result.verification_limitations
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.evidence[0].pointer.quote == claim.source_quote


@pytest.mark.parametrize("change", ["rename", "repeat_count", "split_separate"])
def test_role_names_are_not_hardcoded_and_repeated_obligations_are_consistent(tmp_path, change):
    claim, materials, proposal, old = inputs(tmp_path)
    c = claim.conditions[0]
    if change == "rename":
        c.settings["precise_formula"] = c.settings.pop("accuracy_computation")
        next(
            r for r in proposal.field_bindings if r.path.endswith("accuracy_computation")
        ).path = "/settings/precise_formula"
    elif change == "repeat_count":
        c.settings["also_N"] = 4
        from schemas.claim import ProjectionFieldBinding

        proposal.field_bindings.append(ProjectionFieldBinding(path="/settings/also_N", atom_ids=["sample"]))
    else:
        c.dataset = "MiniSet"
        c.settings["split"] = "test"
        from schemas.claim import ProjectionFieldBinding, RuntimeProjectionAtom

        proposal.atoms.append(
            RuntimeProjectionAtom(
                id="split",
                kind="runtime_setting",
                source_ids=proposal.atoms[0].source_ids,
                key="split",
                value="test",
            )
        )
        proposal.field_bindings.append(ProjectionFieldBinding(path="/settings/split", atom_ids=["split"]))
    assert validate(claim, materials, proposal, old)["runtime"].dataset == "MiniSet"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/settings/a~1b", ["settings", "a/b"]),
        ("/settings/a/b", ["settings", "a", "b"]),
        ("/settings/a~0b", ["settings", "a~b"]),
    ],
)
def test_json_pointer_tokens_keep_literal_and_nested_keys_distinct(path, expected):
    from verification.execution_projection_semantics import pointer_tokens

    assert pointer_tokens(path) == expected


def test_canonical_tolerance_only_for_v2_with_original_label_override_priority(tmp_path):
    from fact_generation.execution.v2 import ExecutionConfig, _target_tolerance
    from tests.test_execution_projection_v2 import public_verify

    claim, _, result, _ = public(tmp_path / "new")
    condition, bound = claim.conditions[0], result.plans[0].target_bindings["c1"]
    assert _target_tolerance(condition, bound, ExecutionConfig(), 0.75, -0.03) == 0.02
    cfg = ExecutionConfig(tolerance_overrides={"Test accuracy": 0.01, "accuracy": 0.07})
    assert _target_tolerance(condition, bound, cfg, 0.75, -0.03) == 0.01
    old, _, old_result, _ = public_verify(tmp_path / "old")
    old_condition = old.conditions[0].model_copy(update={"metric": "Test accuracy"})
    old_binding = old_result.plans[0].target_bindings["c1"]
    assert old_binding.projection.proposal.version == "released-predictions-v1"
    assert _target_tolerance(old_condition, old_binding, ExecutionConfig(), 0.75, -0.03) == 0.05


def test_original_006_reaches_released_prediction_l3_without_model_inference(tmp_path):
    from assessment.rules import assess_claim
    from fact_generation.execution.v2 import ExecutionConfig, Observation, RunOutcome, execute_plans

    claim, materials, verified, model_calls = public(tmp_path)
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
    assert len(model_calls) == 2 and len(runs) == 1
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.aligned and evidence.direction == "support"
    assert "no model inference or training performed" in evidence.note
    measurement = json.loads(Path(evidence.pointer.locator).read_text("utf-8"))
    assert measurement["sample_count"] == 4 and measurement["measurement"]["value"] == 0.75
    assert measurement["measurement"]["unit"] == "fraction"
    assessed = assess_claim(result.claims[0])
    assert assessed.status.value == "supported"
    assert result.claims[0].conditions == claim.conditions


@pytest.mark.parametrize(
    "damage",
    [
        "source_duplicate",
        "source_other_condition",
        "claim_suffix",
        "count_conflict",
        "source_filter",
        "dataset_ambiguity",
    ],
)
def test_source_and_semantic_consumer_guards_survive_wrong_confirmation(tmp_path, damage):
    claim, materials, proposal, old = inputs(tmp_path)
    if damage == "source_duplicate":
        proposal.atoms[0].source_ids.append(proposal.atoms[0].source_ids[0])
    elif damage == "source_other_condition":
        from schemas.claim import ClaimSourceRef

        # Explicit primary narrowing cannot be borrowed back through paper_block IDs.
        claim.source_refs.append(
            ClaimSourceRef(
                source_block_id=claim.source_block_id,
                source_quote=claim.source_quote,
                loc=claim.loc,
                covered=["c2"],
            )
        )
    elif damage == "claim_suffix":
        claim.text += " With confidence weighting."
    elif damage == "count_conflict":
        claim.conditions[0].settings["examples"] = 5
    elif damage == "source_filter":
        block = next(b for b in materials.blocks if b.id == claim.source_block_id)
        # A semantically different new mock source is intentionally rejected; no old artifact is edited.
        block.text = block.text.replace(
            "Three of the four fixed test predictions are correct.",
            "Only four fixed test predictions were selected by a confidence filter.",
        )
        claim.source_quote = block.text
    else:
        path = "configs/ambiguous.json"
        actual = Path(materials.repository.root) / path
        actual.write_text(
            json.dumps(
                {"dataset": "MiniSet test", "metric": "accuracy", "settings": {"model": "ExactMatch"}}
            ),
            "utf-8",
        )
        materials.repository.configs.append(path)
        import hashlib

        # Existing repository file schema carries the indexed identity/hash.
        template = materials.repository.files[0].model_copy(deep=True)
        template.path, template.kind, template.sha256 = (
            path,
            "config",
            hashlib.sha256(actual.read_bytes()).hexdigest(),
        )
        materials.repository.files.append(template)
    with pytest.raises((ValueError, KeyError)):
        validate(claim, materials, proposal, old)


@pytest.mark.parametrize(
    "quote,accepted",
    [
        ("Three of the four fixed test predictions are correct.", True),
        ("Only four fixed test predictions were selected by confidence.", False),
        ("There are not four fixed test predictions.", False),
        ("Three of the four fixed validation predictions are correct.", False),
    ],
)
def test_complete_sample_source_phrase_cannot_borrow_a_negated_or_filtered_count(tmp_path, quote, accepted):
    from schemas.claim import Condition
    from verification.execution_projection_semantics import _source_supports

    _, _, proposal, _ = inputs(tmp_path)
    sample = next(a for a in proposal.atoms if a.kind == "sample_scope")
    runtime = Condition(
        id="c1", dataset="MiniSet", metric="accuracy", settings={"split": "test", "model": "ExactMatch"}
    )
    assert _source_supports(sample, quote, runtime, 0.75, 4, {}) is accepted


def test_separate_v2_request_menu_retains_original_inventory_and_current_source_ids(tmp_path):
    from verification.execution_projection import projection_context
    from verification.execution_projection_semantics import semantic_request_context

    claim, materials, _, _ = inputs(tmp_path)
    old = projection_context(claim, materials)
    new = semantic_request_context(claim, materials)
    assert new["field_inventory"]["c1"] == field_inventory(claim.conditions[0])
    before = old["request_choices"]["conditions"]["c1"]["by_config"]["configs/evaluation.json"]
    after = new["request_choices"]["conditions"]["c1"]["by_config"]["configs/evaluation.json"]
    assert not before["dataset_metric_identity_matches"] and not before["prose_scalar_candidates"]
    assert after["finite_metric_scope_compatible"] and len(after["prose_scalar_candidates"]) == 1
    assert after["prose_scalar_candidates"][0]["sentence"] in claim.source_quote
    assert {r["path"]: r["original_value"] for r in after["original_fields"]} == field_inventory(
        claim.conditions[0]
    )
    assert projection_context(claim, materials) == old


@pytest.mark.parametrize("changed", [1, "true", False])
def test_new_population_boolean_never_coerces(changed):
    from schemas.claim import SampleProjectionAtom

    with pytest.raises(ValueError):
        SampleProjectionAtom(
            id="n",
            source_ids=["s"],
            kind="sample_scope",
            count=4,
            population="fixed_released_predictions",
            split="test",
            all_records=changed,
        )


def test_original_v1_projection_schema_remains_strictly_separate(tmp_path):
    from schemas.claim import PredictionProjection, ProjectionScopeDecision

    _, _, proposal, _ = inputs(tmp_path)
    with pytest.raises(ValueError):
        PredictionProjection.model_validate(proposal.model_dump())
    with pytest.raises(ValueError):
        ProjectionScopeDecision.model_validate(confirm(proposal).model_dump())


@pytest.mark.parametrize(
    "pointer,config_key", [("/settings/a~1b", "a/b"), ("/settings/a/b", "a/b"), ("/settings/a~0b", "a~b")]
)
def test_request_menu_runtime_precedence_uses_decoded_direct_keys(tmp_path, pointer, config_key):
    from tests.test_execution_projection_v2 import refresh
    from verification.execution_projection_semantics import semantic_request_context

    claim, materials, _, _ = inputs(tmp_path)
    if pointer == "/settings/a/b":
        claim.conditions[0].settings["a"] = {"b": 42}
    else:
        claim.conditions[0].settings[config_key] = 42
    config_path = Path(materials.repository.root) / "configs/evaluation.json"
    cfg = json.loads(config_path.read_text("utf-8"))
    cfg["settings"][config_key] = 42
    config_path.write_text(json.dumps(cfg), "utf-8")
    refresh(materials, "configs/evaluation.json")
    choices = semantic_request_context(claim, materials)["request_choices"]["conditions"]["c1"]["by_config"][
        "configs/evaluation.json"
    ]
    selected = next(r for r in choices["original_fields"] if r["path"] == pointer)
    assert selected["runtime_key_required"] == (None if pointer == "/settings/a/b" else config_key)
