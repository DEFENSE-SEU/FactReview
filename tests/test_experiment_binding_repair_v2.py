"""Bounded repair never changes the original evidence or semantic decisions."""

import copy
import json
from pathlib import Path

import pytest

from tests.test_experiment_joint_sources_v2 import ASPECTS, paper, review_for
from verification.experiments import verify_experiments


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from llm.client import LLMConfig

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", lambda **kw: pytest.fail("Unmocked model call"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network call"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("External process"))


def responses(tmp_path, kind="missing"):
    claim, materials, item, _, bounded = paper(tmp_path)
    good = review_for(bounded)
    bad = copy.deepcopy(good)
    patch = dict(item_index=0, condition_id="c1", status="repair", rationale="Explicit binding correction.")
    if kind == "missing":
        bridge = bad["items"][0]["comparisons"][0]["bridges"].pop()
        patch["added_bridges"] = [{"comparison_index": 0, **{k: v for k, v in bridge.items() if k != "kind"}}]
    else:
        use = bad["items"][0]["source_uses"][1]
        use["roles"].append("setup_definition")
        patch["source_roles"] = [{"source_id": use["source_id"], "roles": ["metric_definition"]}]
    first = dict(checked_aspects=ASPECTS, items=[item.model_dump()], plans=[])
    repair = dict(schema_version="scope-binding-repair-v1", patches=[patch])
    return claim, materials, first, bad, repair


def run(fixture, *, rounds=1, repair_effect=None):
    claim, materials, first, scope, repair = fixture
    original = copy.deepcopy([claim.model_dump(), materials.model_dump(), first, scope, repair])
    calls = []

    def model(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            return copy.deepcopy(first)
        if len(calls) == 2:
            return copy.deepcopy(scope)
        assert len(calls) == 3
        assert kwargs["module"] == "verification.experiments.binding_repair"
        if repair_effect:
            return repair_effect(kwargs)
        return copy.deepcopy(repair)

    result = verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=rounds)
    assert [claim.model_dump(), materials.model_dump(), first, scope, repair] == original
    return result, calls


@pytest.mark.parametrize("kind", ["missing", "roles"])
def test_explicit_mechanical_patch_passes_all_original_gates(tmp_path, kind):
    fixture = responses(tmp_path, kind)
    result, calls = run(fixture)
    assert len(calls) == 3
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    payload = json.loads(calls[-1]["prompt"])
    assert "paper_blocks" not in payload and "gold" not in payload
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["original_scope"] == fixture[3]["items"][0]
    assert "binding_repair" in result.evidence[0].note


@pytest.mark.parametrize("kind", ["missing", "roles"])
def test_round_zero_preserves_old_rejection_and_two_calls(tmp_path, kind):
    result, calls = run(responses(tmp_path, kind), rounds=0)
    assert len(calls) == 2 and result.issues
    assert result.evidence and not any(e.sufficient for e in result.evidence)


@pytest.mark.parametrize("field", ["full_support", "qualifiers_complete", "applicability", "first_full"])
def test_incomplete_original_judgments_never_trigger_repair(tmp_path, field):
    fixture = responses(tmp_path)
    if field == "first_full":
        fixture[2]["items"][0]["fully_supported_conditions"] = []
    else:
        fixture[3]["items"][0][field] = "unverified" if field == "applicability" else False
    result, calls = run(fixture)
    assert len(calls) == 2 and not any(e.sufficient for e in result.evidence)


@pytest.mark.parametrize("part", ["conditions", "items"])
@pytest.mark.parametrize("order", ["before", "after", "third"])
def test_whole_response_tombstone_cannot_be_resurrected(tmp_path, part, order):
    fixture = responses(tmp_path)
    rows = fixture[3][part]
    duplicate = copy.deepcopy(rows[0])
    duplicate["condition_id"] = " c1\t"
    duplicate["rationale"] = []
    if order == "before":
        rows.insert(0, duplicate)
    else:
        rows.append(duplicate)
        if order == "third":
            rows.append(copy.deepcopy(rows[0]))
    result, calls = run(fixture)
    assert len(calls) == 2 and not any(e.sufficient for e in result.evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "noop",
        "rationale_only",
        "wrong_field",
        "wrong_table",
        "foreign_source",
        "semantic_role",
        "duplicate_patch",
        "reorder_context",
    ],
)
def test_forbidden_or_ineffective_patch_stays_insufficient(tmp_path, mutation):
    fixture = responses(tmp_path)
    patch = fixture[4]["patches"][0]
    if mutation in {"noop", "rationale_only"}:
        patch["added_bridges"] = []
        patch["rationale"] = "New explanatory prose only." if mutation == "rationale_only" else "No change."
    elif mutation == "wrong_field":
        patch["added_bridges"][0]["condition_field"] = "settings.split"
    elif mutation == "wrong_table":
        patch["added_bridges"][0]["table_id"] = "another-table"
    elif mutation == "foreign_source":
        patch["added_bridges"][0]["source_ids"] = ["another-candidate-source"]
    elif mutation == "semantic_role":
        patch["source_roles"] = [
            {"source_id": fixture[3]["items"][0]["source_uses"][0]["source_id"], "roles": ["protocol"]}
        ]
    elif mutation == "duplicate_patch":
        fixture[4]["patches"].append(copy.deepcopy(patch))
    else:
        patch["context_source_ids"] = []
    result, calls = run(fixture)
    assert len(calls) == 3 and not any(e.sufficient for e in result.evidence)


def test_mixed_semantic_role_is_not_repaired(tmp_path):
    fixture = responses(tmp_path, "roles")
    fixture[3]["items"][0]["source_uses"][1]["roles"].append("protocol")
    result, calls = run(fixture)
    assert len(calls) == 2 and not any(e.sufficient for e in result.evidence)


def test_repaired_binding_does_not_waive_wrong_numeric_relation(tmp_path):
    fixture = responses(tmp_path)
    fixture[3]["conditions"][0]["relation"] = "lt"
    fixture[3]["items"][0]["comparisons"][0]["relation"] = "lt"
    result, calls = run(fixture)
    assert len(calls) <= 3 and not any(e.sufficient for e in result.evidence)
    assert any("false" in issue for issue in result.issues)


def test_repair_transport_failure_is_local_and_bounded(tmp_path):
    def failed(_):
        raise RuntimeError("offline simulated service failure")

    result, calls = run(responses(tmp_path), repair_effect=failed)
    assert len(calls) == 3 and not any(e.sufficient for e in result.evidence)
    assert any("repair" in issue.lower() for issue in result.issues)


@pytest.mark.parametrize("rounds", [-1, 2, True, 1.0, "1"])
def test_explicit_rounds_contract_is_strict(tmp_path, rounds):
    with pytest.raises(ValueError, match="0 or 1"):
        run(responses(tmp_path), rounds=rounds)


@pytest.mark.parametrize("value", ["0", "1"])
def test_existing_settings_env_reaches_verifier_without_new_cli(tmp_path, monkeypatch, value):
    from common.config import Settings

    monkeypatch.setenv("EXPERIMENT_SCOPE_BINDING_REPAIR_ROUNDS", value)
    settings = Settings(_env_file=None)
    assert settings.experiment_scope_binding_repair_rounds == int(value)
    monkeypatch.setattr("common.config.get_settings", lambda: settings)
    result, calls = run(responses(tmp_path), rounds=None)
    assert len(calls) == 2 + int(value)
    assert any(e.sufficient for e in result.evidence) == bool(int(value))


@pytest.mark.parametrize("value", ["-1", "2", "true", "1.0", " 1 "])
def test_invalid_settings_env_is_rejected(monkeypatch, value):
    from common.config import Settings

    monkeypatch.setenv("EXPERIMENT_SCOPE_BINDING_REPAIR_ROUNDS", value)
    with pytest.raises(ValueError, match="0 or 1"):
        Settings(_env_file=None)


def two_conditions(tmp_path):
    from verification.experiment_catalog import build_catalog
    from verification.experiment_sources import prepare_joint_candidate
    from verification.experiments import ExperimentItem

    fixture = responses(tmp_path)
    claim, materials, first, scope, _ = fixture
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2"}))
    item = ExperimentItem.model_validate(first["items"][0]).model_copy(
        update={"covered": ["c2"], "fully_supported_conditions": ["c2"]}
    )
    catalog = build_catalog(claim, materials)
    first_bounded = prepare_joint_candidate(
        claim, materials, catalog, ExperimentItem.model_validate(first["items"][0]), 0
    )
    rebuilt = review_for(first_bounded)
    bridge = rebuilt["items"][0]["comparisons"][0]["bridges"].pop()
    scope.update(rebuilt)
    fixture[4]["patches"][0]["added_bridges"] = [
        {"comparison_index": 0, **{key: value for key, value in bridge.items() if key != "kind"}}
    ]
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 1)
    # review_for is a c1 fixture helper; only map its condition/case identity.
    compatible = copy.deepcopy(bounded)
    compatible["conditions"]["c1"] = compatible["conditions"]["c2"]
    healthy = review_for(compatible, index=1)
    healthy["conditions"][0]["condition_id"] = "c2"
    healthy["items"][0]["condition_id"] = "c2"
    first["items"].append(item.model_dump())
    scope["conditions"].extend(healthy["conditions"])
    scope["items"].extend(healthy["items"])
    return fixture


@pytest.mark.parametrize("failure", ["service", "bad_patch", "hard_tombstone"])
def test_repair_failure_preserves_other_healthy_condition(tmp_path, failure):
    fixture = two_conditions(tmp_path)
    effect = None
    if failure == "service":

        def effect(_):
            raise RuntimeError("Mocked repair failure")
    elif failure == "bad_patch":
        fixture[4]["patches"][0]["added_bridges"][0]["source_ids"] = ["absent"]
    else:
        bad = copy.deepcopy(fixture[3]["items"][0])
        bad["condition_id"] = " c1\t"
        bad["rationale"] = []
        fixture[3]["items"].insert(0, bad)
    result, calls = run(fixture, repair_effect=effect)
    assert len(calls) == (2 if failure == "hard_tombstone" else 3)
    assert any(e.sufficient and e.covered == ["c2"] for e in result.evidence)
    assert not any(e.sufficient and "c1" in e.covered for e in result.evidence)


@pytest.mark.parametrize("changed", ["claim", "materials", "output", "artifact"])
def test_callback_cannot_replace_frozen_semantic_inputs(tmp_path, monkeypatch, changed):
    import verification.experiments as module

    claim, materials, first, scope, repair = responses(tmp_path)
    parsed = []
    original_validate = module.ExperimentsOutput.model_validate

    def capture(value):
        output = original_validate(value)
        parsed.append(output)
        return output

    monkeypatch.setattr(module.ExperimentsOutput, "model_validate", capture)
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return copy.deepcopy(first if len(calls) == 1 else scope)
        if changed == "claim":
            claim.conditions[0].settings["procedure"] = "different"
        elif changed == "materials":
            materials.blocks[1].text += " altered"
        elif changed == "output":
            parsed[0].items[0].detail = "Replaced candidate during repair"
        else:
            Path(materials.markdown_path).write_text(materials.markdown + " altered", encoding="utf-8")
        return copy.deepcopy(repair)

    result = module.verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=1)
    assert len(calls) == 3 and not any(e.sufficient for e in result.evidence)
    assert result.issues


def test_revalidation_uses_pristine_view_and_keeps_original_audit(tmp_path, monkeypatch):
    import verification.experiments as module

    fixture = responses(tmp_path, "roles")
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(tmp_path / "stats.json"))
    original_validate = module.validate_source_uses
    calls = []

    def tracked(bounded, *args, **kwargs):
        calls.append(bounded)
        if len(calls) == 1:
            bounded["joint_view"].setdefault("semantic_only", []).append({"poison": "failed attempt"})
        return original_validate(bounded, *args, **kwargs)

    monkeypatch.setattr(module, "validate_source_uses", tracked)
    result, _ = run(fixture)
    assert result.evidence[0].sufficient
    audit = json.loads(next((tmp_path / "experiment_scope").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["response"] == fixture[3]
    assert audit["validated"] is False and audit["binding_errors"]
    assert audit["binding_repair"]["response"] == fixture[4]
    assert audit["binding_repair"]["effective_response"] != audit["response"]
    assert not any("poison" in row for row in audit["joint_sources"]["0"]["semantic_only"])
    assert audit["item_locations"][0]["response_pointer"].startswith("/binding_repair/effective_response/")


def test_first_two_requests_are_unchanged_when_repair_is_enabled(tmp_path):
    fixture = responses(tmp_path)
    _, disabled = run(fixture, rounds=0)
    _, enabled = run(fixture, rounds=1)
    assert enabled[:2] == disabled


@pytest.mark.parametrize(
    "field",
    [
        "full_support",
        "qualifiers_complete",
        "applicability",
        "grounds_source_ids",
        "comparisons",
        "source_uses",
        "condition_scope",
    ],
)
def test_patch_cannot_rewrite_original_scope_fields(tmp_path, field):
    fixture = responses(tmp_path)
    fixture[4]["patches"][0][field] = []
    result, calls = run(fixture)
    assert len(calls) == 3 and not any(e.sufficient for e in result.evidence)


@pytest.mark.parametrize("index", [True, "0", -1])
def test_malformed_duplicate_candidate_index_never_enables_repair(tmp_path, index):
    fixture = responses(tmp_path)
    duplicate = copy.deepcopy(fixture[3]["items"][0])
    duplicate["item_index"] = index
    fixture[3]["items"].insert(0, duplicate)
    result, calls = run(fixture)
    assert len(calls) == 2 and not any(e.sufficient for e in result.evidence)


def test_missing_bridge_does_not_authorize_wrong_original_operand(tmp_path):
    fixture = responses(tmp_path)
    comparison = fixture[3]["items"][0]["comparisons"][0]
    comparison["right"] = copy.deepcopy(comparison["left"])
    result, calls = run(fixture)
    assert len(calls) <= 3 and not any(e.sufficient for e in result.evidence)


def test_explicit_unresolved_repair_is_not_provider_success_for_evidence(tmp_path, monkeypatch):
    fixture = responses(tmp_path)
    patch = fixture[4]["patches"][0]
    patch["status"], patch["added_bridges"] = "unresolved", []
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(tmp_path / "stats.json"))
    result, calls = run(fixture)
    assert len(calls) == 3 and not any(e.sufficient for e in result.evidence)
    audit = json.loads(next((tmp_path / "experiment_scope").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["binding_repair"]["outcomes"][0]["status"] == "unresolved"


def test_valid_padded_condition_keeps_repair_and_healthy_neighbor(tmp_path):
    fixture = two_conditions(tmp_path)
    fixture[3]["conditions"][0]["condition_id"] = " c1\t"
    result, calls = run(fixture)
    assert len(calls) == 3
    assert {c for e in result.evidence if e.sufficient for c in e.covered} == {"c1", "c2"}
    payload = json.loads(calls[-1]["prompt"])
    assert payload["pairs"][0]["condition_scope"]["condition_id"] == " c1\t"


def test_unexpected_repair_preparation_failure_keeps_healthy_neighbor(tmp_path, monkeypatch):
    import verification.experiments as module

    def failed(*args, **kwargs):
        raise RuntimeError("Mock preparation fault")

    monkeypatch.setattr(module, "_probe_joint_binding", failed)
    result, calls = run(two_conditions(tmp_path))
    assert len(calls) == 2
    assert any(e.sufficient and e.covered == ["c2"] for e in result.evidence)
    assert not any(e.sufficient and "c1" in e.covered for e in result.evidence)


def test_unexpected_second_pair_failure_cannot_commit_without_effective_audit(tmp_path, monkeypatch):
    import verification.experiments as module

    fixture = two_conditions(tmp_path)
    bridge = fixture[3]["items"][1]["comparisons"][0]["bridges"].pop()
    fixture[4]["patches"].append(
        dict(
            item_index=1,
            condition_id="c2",
            status="repair",
            rationale="Same explicit mechanical fix.",
            added_bridges=[{"comparison_index": 0, **{k: v for k, v in bridge.items() if k != "kind"}}],
        )
    )
    original = module._probe_joint_binding
    calls = []

    def fail_fourth(*args, **kwargs):
        calls.append(args)
        if len(calls) == 4:
            raise RuntimeError("Mock second-pair revalidation fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_probe_joint_binding", fail_fourth)
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(tmp_path / "stats.json"))
    result, model_calls = run(fixture)
    assert len(model_calls) == 3 and not any(e.sufficient for e in result.evidence)
    audit = json.loads(next((tmp_path / "experiment_scope").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["binding_repair"]["status"] == "failed"
    assert audit["binding_repair"]["outcomes"][0]["status"] == "rolled_back"
    assert "effective_response" not in audit["binding_repair"]
    assert all(row["response_pointer"].startswith("/response/") for row in audit["item_locations"])


def test_successful_binding_repair_reports_history_without_current_unconfirmed(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(tmp_path / "stats.json"))
    fixture = responses(tmp_path)
    # Isolate the post-decoder missing bridge: this original source-use row
    # declares only its already-consumed table-reference role.
    fixture[3]["items"][0]["source_uses"][2]["roles"] = ["table_reference"]
    result, calls = run(fixture)
    assert len(calls) == 3 and result.evidence[0].sufficient
    assert not any("Experimental scope review unconfirmed:" in issue for issue in result.issues)
    assert any("Original scope diagnostics and binding repair history:" in issue for issue in result.issues)
    assert "binding_repair=accepted" in result.evidence[0].note
    audit = json.loads(next((tmp_path / "experiment_scope").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["binding_errors"] == []
    assert audit["binding_repair"]["outcomes"][0]["status"] == "accepted"
    assert audit["binding_repair"]["pairs"][0]["binding_issues"]
    disabled, _ = run(responses(tmp_path, "roles"), rounds=0)
    assert any("Experimental scope review unconfirmed:" in issue for issue in disabled.issues)
    unneeded_fixture = responses(tmp_path, "roles")
    unneeded_fixture[3]["items"][0]["full_support"] = False
    unneeded, unneeded_calls = run(unneeded_fixture)
    assert len(unneeded_calls) == 2
    assert any("Experimental scope review unconfirmed:" in issue for issue in unneeded.issues)
