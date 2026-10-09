"""Finite exact-match phrasing controls; actual006 semantics, all services mocked."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from schemas.claim import Claim, Condition
from tests.test_execution_projection_choices_v2 import review, select
from tests.test_execution_projection_v2 import original_inputs
from verification.execution_projection import field_inventory
from verification.execution_projection_choices import _field_keys, build_choice_registry
from verification.execution_projection_semantics import _definition, _interpret_text

CFG = {"dataset": "MiniSet", "metric": "accuracy", "settings": {"split": "test", "model": "ExactMatch"}}
RUNTIME = Condition(id="c1", dataset=CFG["dataset"], metric=CFG["metric"], settings=CFG["settings"])
SCIENTIFIC_FIELDS = (
    "id",
    "text",
    "loc",
    "source_block_id",
    "source_quote",
    "source_refs",
    "conditions",
    "needs",
    "importance",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from llm.client import LLMConfig

    def forbidden(*args, **kwargs):
        pytest.fail("External/model/process boundary must be explicitly mocked")

    for target in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "subprocess.Popen",
        "screening.checks.llm_json",
    ):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config",
        lambda: LLMConfig(provider="mock", model="mock", base_url=None, api_key=None),
    )


def actual006(tmp_path):
    """Exact scientific fields from bef2's claim006, using the same preexisting paper fixture.

    Original screening SHA 6bc91e0497db9a4fd121fc74d001ef5a4c30b734f72742fd4d7d99f66e6317fb.
    Neither the original saved run nor an old fixture is edited by these tests.
    """
    _, materials = original_inputs(tmp_path)
    old = json.loads((Path(__file__).parent / "fixtures/execution_projection_006.json").read_text("utf-8"))[
        "claim"
    ]
    raw = copy.deepcopy(old)
    raw["text"] = (
        "On MiniSet test, model ExactMatch has accuracy 0.75 over four examples, with accuracy computed by exact equality of each prediction and label."
    )
    raw["conditions"] = [
        {
            "id": "c1",
            "dataset": "MiniSet test",
            "metric": "accuracy",
            "settings": {
                "model": "ExactMatch",
                "accuracy": 0.75,
                "examples": 4,
                "accuracy_computation": "exact equality of each prediction and label",
                "scope": "four fixed test predictions",
                "exclusions": [
                    "no repeated-run uncertainty",
                    "no population-performance conclusion",
                    "no ranking against other models",
                ],
            },
            "description": "Reported MiniSet test accuracy for ExactMatch on four examples.",
        }
    ]
    scientific = {k: raw[k] for k in SCIENTIFIC_FIELDS}
    assert (
        hashlib.sha256(json.dumps(scientific, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        == "a6a7a5fe8f6094e99ad56df7fb7d8a9c3e17add21ef5d3d343edf65bb4671ad2"
    )
    claim = Claim.model_validate(raw)
    assert next(b.text for b in materials.blocks if b.id == claim.source_block_id) == claim.source_quote
    assert (
        next(b.text for b in materials.blocks if b.id == claim.source_refs[0].source_block_id)
        == claim.source_refs[0].source_quote
    )
    return claim, materials


def test_actual006_all_eleven_fields_and_whole_claim_retain_obligations(tmp_path):
    claim, _ = actual006(tmp_path)
    inventory = field_inventory(claim.conditions[0])
    assert len(inventory) == 11
    actual = {p: _field_keys(p, v, CFG, RUNTIME, 0.75, 4) for p, v in inventory.items()}
    assert all(actual.values()), actual
    assert actual["/settings/accuracy_computation"] == {"definition"}
    assert actual["/description"] == {"dataset", "runtime:model", "definition", "sample"}
    assert _interpret_text(claim.text, CFG, RUNTIME, 0.75, 4) == {
        "dataset",
        "runtime:model",
        "value",
        "sample",
        "definition",
    }


def test_actual006_registry_preserves_all_original_fields(tmp_path):
    claim, materials = actual006(tmp_path)
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    candidate = next(iter(registry["candidates"].values()))
    assert candidate["condition"] == claim.conditions[0].model_dump(mode="json")
    assert {b["path"] for b in candidate["proposal"]["field_bindings"]} == set(
        field_inventory(claim.conditions[0])
    )
    assert candidate["proposal"]["claim_bindings"][0]["start"] == 0
    assert candidate["proposal"]["claim_bindings"][0]["end"] == len(claim.text)
    assert candidate["data_path"] == "data/test_predictions.json"
    assert before == (claim.model_dump(), materials.model_dump())


def test_current_actual006_fixed_sample_and_computed_as_registry(tmp_path):
    claim, materials = actual006(tmp_path)
    claim.text = (
        "On MiniSet test, model ExactMatch has accuracy 0.75 over four examples, "
        "computed as exact equality of each prediction and label."
    )
    claim.conditions[0].settings = {
        "model": "ExactMatch",
        "accuracy": 0.75,
        "examples": 4,
        "computation": "fraction computed by exact equality of each prediction and label",
        "qualifiers": [
            "four fixed test predictions",
            "no repeated-run uncertainty",
            "no population-performance conclusion",
            "no ranking against other models",
        ],
    }
    claim.conditions[0].description = "Reported MiniSet test accuracy for ExactMatch on four fixed examples."
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    candidate = next(iter(registry["candidates"].values()))
    assert candidate["condition"] == claim.conditions[0].model_dump(mode="json")
    assert {b["path"] for b in candidate["proposal"]["field_bindings"]} == set(
        field_inventory(claim.conditions[0])
    )
    assert candidate["proposal"]["claim_bindings"][0]["end"] == len(claim.text)
    assert before == (claim.model_dump(), materials.model_dump())


def test_fixed_sample_and_computed_as_use_dynamic_bound_identity():
    cfg = {
        "dataset": "D[2].+",
        "metric": "accuracy",
        "settings": {"split": "held-out", "model": "R+G(v2)"},
    }
    runtime = Condition(id="c1", **cfg)
    assert _interpret_text(
        "Reported D[2].+ held-out accuracy for R+G(v2) on seven fixed examples.",
        cfg,
        runtime,
        0.75,
        7,
    ) == {"dataset", "definition", "runtime:model", "sample"}
    assert _interpret_text(
        "On D[2].+ held-out, model R+G(v2) has accuracy 0.75 over seven examples, "
        "computed as exact equality of each prediction and label.",
        cfg,
        runtime,
        0.75,
        7,
    ) == {"dataset", "definition", "runtime:model", "sample", "value"}


@pytest.mark.parametrize(
    "tail",
    [
        "computed as weighted exact equality of each prediction and label.",
        "computed as exact equality of each prediction and label except hard examples.",
    ],
)
def test_computed_as_preserves_non_equivalent_or_unknown_tail_rejection(tail):
    assert (
        _interpret_text(
            "On MiniSet test, model ExactMatch has accuracy 0.75 over four examples, " + tail,
            CFG,
            RUNTIME,
            0.75,
            4,
        )
        is None
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "exact equality of each prediction and label",
        "the exact equality of every prediction and its label",
        "accuracy computed by exact equality of each prediction and label",
    ],
)
def test_definition_fragment_never_supplies_fraction_source(phrase):
    assert _definition(phrase)
    assert not _definition(phrase, require_fraction=True)
    assert _definition(
        "Accuracy is a fraction, computed by exact equality of each prediction and label.",
        require_fraction=True,
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "not exact equality of each prediction and label",
        "weighted exact equality of each prediction and label",
        "top-k exact equality of each prediction and label",
        "approximate equality of each prediction and label",
        "exact equality of each prediction and label with a tolerance of 0.1",
        "exact equality of each prediction and label, weighted by confidence",
        "F1 computed by exact equality of each prediction and label",
        "exact equality of each prediction and another label",
        "exact equality of each prediction and label except hard examples",
        "accuracy computed by exact equality of each prediction and label on all populations",
        "percent computed by exact equality of each prediction and label",
    ],
)
def test_definition_fragment_rejects_changed_or_residual_meaning(phrase):
    assert not _definition(phrase)
    assert _interpret_text(phrase, CFG, RUNTIME, 0.75, 4) is None


@pytest.mark.parametrize(
    "dataset,split,actor,count",
    [
        ("MiniSet", "test", "ExactMatch", 4),
        ("D[2].+", "held-out", "R+G(v2)", 7),
    ],
)
def test_reported_description_uses_dynamic_identity_and_count(dataset, split, actor, count):
    cfg = {"dataset": dataset, "metric": "accuracy", "settings": {"split": split, "model": actor}}
    runtime = Condition(id="c1", **cfg)
    text = f"Reported {dataset} {split} accuracy for {actor} on {count} examples."
    assert _interpret_text(text, cfg, runtime, 0.75, count) == {
        "dataset",
        "runtime:model",
        "definition",
        "sample",
    }


@pytest.mark.parametrize(
    "text",
    [
        "Reported OtherSet test accuracy for ExactMatch on four examples.",
        "Reported MiniSet train accuracy for ExactMatch on four examples.",
        "Reported MiniSet test accuracy for OtherModel on four examples.",
        "Reported MiniSet test accuracy for ExactMatch on five examples.",
        "Reported MiniSet test accuracy for ExactMatch on 4.0 examples.",
        "Reported MiniSet test accuracy for ExactMatch on True examples.",
        "Reported MiniSet test F1 for ExactMatch on four examples.",
        "Reported MiniSet test top-k accuracy for ExactMatch on four examples.",
        "Reported MiniSet test weighted accuracy for ExactMatch on four examples.",
        "Not reported MiniSet test accuracy for ExactMatch on four examples.",
        "Reported MiniSet test accuracy for ExactMatch on four examples, generalizing to all populations.",
        "Reported MiniSet test accuracy for ExactMatch on four examples except errors.",
    ],
)
def test_reported_description_rejects_wrong_identity_or_residue(text):
    assert _interpret_text(text, CFG, RUNTIME, 0.75, 4) is None


def test_literal_identity_cannot_be_satisfied_by_regex_wildcards():
    cfg = {"dataset": "D[2].+", "metric": "accuracy", "settings": {"model": "R+G(v2)", "split": "test"}}
    runtime = Condition(id="c1", **cfg)
    assert (
        _interpret_text("Reported D2xxx test accuracy for RRGv2 on 4 examples.", cfg, runtime, 0.75, 4)
        is None
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_epochs", 4),
        ("F1", 0.75),
        ("extra", ""),
        ("extra", []),
        ("extra", "predicts unseen data"),
        ("examples", 4.0),
        ("examples", True),
    ],
)
def test_actual006_additional_unknown_field_remains_unavailable(tmp_path, key, value):
    claim, materials = actual006(tmp_path)
    claim.conditions[0].settings[key] = value
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"]
    assert any("Original field" in x["reason"] for x in registry["unavailable"])


@pytest.mark.parametrize(
    "suffix", [" It generalizes to unseen populations.", " except incorrect predictions."]
)
def test_entire_actual_claim_retains_unknown_residual_obligation(tmp_path, suffix):
    claim, materials = actual006(tmp_path)
    claim.text += suffix
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"]
    assert any("Whole claim is outside" in x["reason"] for x in registry["unavailable"])


def test_missing_fraction_in_original_measurement_source_cannot_build_choice(tmp_path):
    claim, materials = actual006(tmp_path)
    old = "Accuracy is a fraction, computed by exact equality of each prediction and label."
    new = "Accuracy computed by exact equality of each prediction and label."
    materials.markdown = materials.markdown.replace(old, new)
    Path(materials.markdown_path).write_text(materials.markdown, "utf-8")
    for block in materials.blocks:
        block.text = block.text.replace(old, new)
        if block.loc.char_start is not None and materials.markdown.count(block.text) == 1:
            block.loc.char_start = materials.markdown.index(block.text)
            block.loc.char_end = block.loc.char_start + len(block.text)
    claim.source_quote = claim.source_quote.replace(old, new)
    claim.loc = next(b.loc.model_copy(deep=True) for b in materials.blocks if b.id == claim.source_block_id)
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"]
    assert any(
        "definition lacks one complete authorized source" in x["reason"] for x in registry["unavailable"]
    )


@pytest.mark.parametrize("confirmed", [True, False])
def test_actual006_public_two_calls_still_require_independent_scope(tmp_path, confirmed):
    from common.run_stats import run_scope
    from verification.experiment_targets import validate_plan_targets
    from verification.experiments import verify_experiments

    claim, materials = actual006(tmp_path)
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    calls, selected = [], []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        calls.append({"module": kw["module"], "payload": payload})
        if kw["module"] == "verification.experiments":
            candidate = next(iter(payload["execution_choice_context"]["candidates"].values()))
            selected.append(candidate)
            return dict(
                checked_aspects=["correspondence", "fairness", "isolation", "stability", "consistency"],
                items=[],
                plans=[],
                execution_choices=[select(candidate)],
            )
        assert kw["module"] == "verification.experiments.scope"
        decision = review(selected[0])
        if not confirmed:
            decision["reviews"][0]["decision"] = "unresolved"
        return dict(schema_version="catalog-v2", conditions=[], items=[], execution_choice_reviews=[decision])

    with run_scope(tmp_path / "stats.json"):
        result = verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=0)
    assert len(calls) == 2
    assert before == (claim.model_dump(), materials.model_dump())
    assert bool(result.plans) is confirmed, result.issues
    if confirmed:
        assert len(result.plans) == 1
        plan = result.plans[0]
        assert plan.feasibility == "ready"
        assert plan.target_conditions == claim.conditions
        assert plan.y_paper == {"c1": 0.75}
        validate_plan_targets(plan, claim, materials)
    assert not result.evidence
