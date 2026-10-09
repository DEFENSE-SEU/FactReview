"""Provider diagnostics cannot authorize or escape through choice narration."""

import copy
import json

import pytest

from llm.client import LLMConfig
from tests.test_execution_projection_choices_v2 import public

KEY = "sk-choice-privacy-unit-NOT-A-REAL-KEY-32987"
CFG = LLMConfig(provider="mock", model="mock", base_url="https://fixture.invalid/v1", api_key=KEY)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("No external/default call is authorized")

    for path in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "screening.checks.llm_json",
    ):
        monkeypatch.setattr(path, forbidden)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: CFG)


@pytest.mark.parametrize("phase", ["unresolved", "selection", "review", "obligation", "malformed_key"])
def test_sensitive_choice_is_rejected_without_reflection(tmp_path, phase):
    raw_snapshots = []

    def first(raw, *_):
        row = raw["execution_choices"][0]
        if phase in {"unresolved", "selection"}:
            row["rationale"] += KEY
        if phase == "unresolved":
            row["decision"] = "unresolved"
            row.pop("candidate_id")
        if phase == "malformed_key":
            row[KEY] = {"[REDACTED]": 1, KEY: 2}
        raw_snapshots.append((raw, copy.deepcopy(raw)))

    def scope(raw, *_):
        row = raw["execution_choice_reviews"][0]
        if phase == "review":
            row["rationale"] += KEY
        if phase == "obligation":
            row["reviews"][0]["rationale"] += KEY
        raw_snapshots.append((raw, copy.deepcopy(raw)))

    _, _, result, calls, _ = public(tmp_path, first_change=first, scope_change=scope)
    assert not result.plans
    assert KEY not in result.model_dump_json()
    assert all(KEY not in json.dumps(call) for call in calls)
    audits = [
        p for p in tmp_path.rglob("*.json") if p.parent.name in {"experiment_scope", "execution_choices"}
    ]
    assert audits and all(KEY not in p.read_text("utf8") for p in audits)
    assert all(raw == before for raw, before in raw_snapshots)
    assert len(calls) == (2 if phase in {"review", "obligation"} else 1)


def test_second_actual_provider_credential_blocks_outgoing_scope(tmp_path, monkeypatch):
    second = LLMConfig(
        provider="mock",
        model="other",
        base_url="https://other.invalid/v1",
        api_key="sk-second-provider-choice-NOT-REAL-54321",
    )
    configs = iter([CFG, second])
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: next(configs))

    def first(raw, *_):
        raw["execution_choices"][0]["rationale"] += second.api_key

    _, _, result, calls, _ = public(tmp_path, first_change=first)
    assert len(calls) == 1
    assert not result.plans and result.verification_limitations
    assert second.api_key not in result.model_dump_json()
    assert all(
        second.api_key not in p.read_text("utf8")
        for p in tmp_path.rglob("*.json")
        if p.parent.name in {"experiment_scope", "execution_choices"}
    )


def mixed_choices(tmp_path):
    from common.run_stats import run_scope
    from tests.test_execution_projection_choices_v2 import inputs, review, select
    from verification.experiments import verify_experiments

    claim, materials, _, _ = inputs(tmp_path)
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2"}, deep=True))
    for ref in claim.source_refs:
        ref.covered.append("c2")
    calls, responses = [], []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(copy.deepcopy(payload))
        candidates = payload["execution_choice_context"]["candidates"]
        c = next(c for c in candidates.values() if c["condition_id"] == "c1")
        if kwargs["module"] == "verification.experiments":
            raw = {
                "checked_aspects": ["correspondence", "fairness", "isolation", "stability", "consistency"],
                "items": [],
                "plans": [],
                "execution_choices": [
                    select(c),
                    {
                        "version": "released-predictions-choice-v1",
                        "condition_id": "c2",
                        "decision": "unresolved",
                        "rationale": KEY,
                    },
                ],
            }
        else:
            raw = {
                "schema_version": "catalog-v2",
                "conditions": [],
                "items": [],
                "execution_choice_reviews": [review(c)],
            }
        responses.append(copy.deepcopy(raw))
        return raw

    with run_scope(tmp_path / "stats.json"):
        result = verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=0)
    return claim, materials, result, calls, responses


def test_rejected_neighbor_preserves_clean_choice_and_l3(tmp_path):
    from pathlib import Path

    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, calls, raw = mixed_choices(tmp_path)
    assert len(calls) == 2 and len(result.plans) == 1, result.issues
    plan = result.plans[0]
    record = plan.target_bindings["c1"].projection
    assert record.selection.model_dump(mode="json") == raw[0]["execution_choices"][0]
    assert record.choice_review.model_dump(mode="json") == raw[1]["execution_choice_reviews"][0]
    assert record.selection_index == 0 and record.review_index == 0
    validate_plan_targets(plan, claim, materials)
    audit = json.loads(Path(record.scope_audit).read_text("utf8"))
    assert audit["choice_privacy"]["wire_representation"] == "redacted_copy"
    assert audit["choice_privacy"]["phases"]["selection"]["rejected_indices"] == [1]
    assert audit["input"] == calls[1]
    assert KEY not in json.dumps(calls) and KEY not in json.dumps(audit)
    assert raw[0]["execution_choices"][1]["rationale"] == KEY


@pytest.mark.parametrize(
    "mutation",
    [
        "deleted",
        "phase_deleted",
        "bool",
        "float",
        "string",
        "negative",
        "out_of_bounds",
        "duplicate",
        "marker_deleted",
        "wrong_hash",
    ],
)
def test_consumer_rejects_changed_privacy_metadata(tmp_path, mutation):
    from pathlib import Path

    from verification.execution_projection import ProjectionError
    from verification.execution_projection_choices import bind_choice_target

    claim, materials, result, _, _ = mixed_choices(tmp_path)
    record = result.plans[0].target_bindings["c1"].projection
    path = Path(record.scope_audit)
    audit = json.loads(path.read_text("utf8"))
    if mutation == "deleted":
        del audit["choice_privacy"]
    elif mutation == "phase_deleted":
        audit["choice_privacy"]["phases"] = {"review": audit["choice_privacy"]["phases"]["selection"]}
    elif mutation == "marker_deleted":
        for rows in (audit["input"]["execution_choices"], audit["first_pass_response"]["execution_choices"]):
            del rows[1]["_choice_privacy_rejected"]
    elif mutation == "wrong_hash":
        audit["choice_privacy"]["phases"]["selection"]["safe_rows_sha256"] = "0" * 64
    else:
        value = {
            "bool": [True],
            "float": [1.0],
            "string": ["1"],
            "negative": [-1],
            "out_of_bounds": [2],
            "duplicate": [1, 1],
        }[mutation]
        audit["choice_privacy"]["phases"]["selection"]["rejected_indices"] = value
    path.write_text(json.dumps(audit), encoding="utf8")
    with pytest.raises(ProjectionError, match="privacy"):
        bind_choice_target(
            claim,
            materials,
            record.registry_snapshot,
            {"index": record.selection_index, "selection": record.selection.model_dump(mode="json")},
            {"index": record.review_index, "review": record.choice_review.model_dump(mode="json")},
            path,
        )


def test_privacy_tombstone_cannot_be_restored_by_clean_duplicate(tmp_path):
    from tests.test_execution_projection_choices_v2 import candidate, review, select
    from verification.execution_projection_choices import ChoicePrivacy, decode_choice_reviews, decode_choices

    _, _, registry, c = candidate(tmp_path)
    privacy = ChoicePrivacy()
    privacy.configs = [CFG]
    bad = select(c)
    bad["rationale"] = KEY
    rows, indices = privacy.rows([bad, select(c)], "selection")
    selected, errors = decode_choices(rows, registry, [], privacy_rejected_indices=indices)
    assert not selected and errors
    selected, _ = decode_choices([select(c)], registry, [])
    bad = review(c)
    bad["reviews"][0]["rationale"] = KEY
    rows, indices = privacy.rows([bad, review(c)], "review")
    retained, errors = decode_choice_reviews(rows, selected, registry, privacy_rejected_indices=indices)
    assert not retained and errors


def test_sensitive_choice_keeps_healthy_paper_observation(tmp_path):
    from verification.experiment_catalog import build_catalog

    def first(raw, claim, materials):
        raw["execution_choices"][0]["rationale"] = KEY
        raw["items"] = [
            {
                "aspect": "correspondence",
                "kind": "paper_support",
                "block_id": claim.source_block_id,
                "quote": claim.source_quote,
                "covered": ["c1"],
                "fully_supported_conditions": [],
                "detail": "Original partial observation.",
            }
        ]

    def scope(raw, claim, materials):
        source = next(
            k
            for k, v in build_catalog(claim, materials)["sources"].items()
            if v.get("block_id") == claim.source_block_id and v["kind"] == "claim_source"
        )
        raw["execution_choice_reviews"] = []
        raw["conditions"] = [
            {
                "condition_id": "c1",
                "assertion": "descriptive",
                "matched_controls_required": False,
                "uncertainty_sensitive": False,
                "relation": "none",
                "rationale": "Fixed predictions.",
            }
        ]
        raw["items"] = [
            {
                "item_index": 0,
                "condition_id": "c1",
                "applicability": "applicable",
                "grounds_source_ids": [source],
                "rationale": "Original partial observation.",
                "full_support": False,
                "qualifiers_complete": False,
                "comparison_objects": "not_comparative",
            }
        ]

    _, _, result, calls, _ = public(tmp_path, first_change=first, scope_change=scope)
    assert len(calls) == 2 and not result.plans
    assert (
        len(result.evidence) == 1
        and result.evidence[0].covered == ["c1"]
        and not result.evidence[0].sufficient
    )
    assert "execution_choices" not in calls[1]["payload"]


@pytest.mark.parametrize("field", ["userinfo", "query", "key_collision"])
def test_endpoint_and_dictionary_keys_are_safe(tmp_path, monkeypatch, field):
    cfg = LLMConfig(
        provider="mock",
        model="mock",
        base_url="https://fixture-user:fixture-pass@fixture.invalid:8443/v1?token=fixture-query",
        api_key=KEY,
    )
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)

    def first(raw, *_):
        row = raw["execution_choices"][0]
        if field == "key_collision":
            row[KEY] = 1
            row["[redacted]"] = 2
        else:
            row["rationale"] = cfg.base_url

    _, _, result, calls, _ = public(tmp_path, first_change=first)
    assert not result.plans and len(calls) == 1
    text = result.model_dump_json() + "".join(
        p.read_text("utf8") for p in tmp_path.rglob("*.json") if p.parent.name == "execution_choices"
    )
    assert all(secret not in text for secret in (KEY, "fixture-user", "fixture-pass", "fixture-query"))
    if field == "key_collision":
        assert "_audit_redaction" in text and '"value": 1' in text and '"value": 2' in text
    else:
        assert "fixture.invalid:8443" in text


def test_nonlist_sensitive_response_is_safe_and_explicitly_marked(tmp_path):
    def first(raw, *_):
        raw["execution_choices"] = {KEY: "malformed"}

    _, _, result, calls, _ = public(tmp_path, first_change=first)
    assert not result.plans and len(calls) == 1
    audit = json.loads(next((tmp_path / "execution_choices").glob("*.json")).read_text("utf8"))
    assert audit["choice_audit_representation"]["wire_representation"] == "redacted_copy"
    assert KEY not in json.dumps(audit) and KEY not in result.model_dump_json()


def test_clean_wire_stays_exact_without_privacy_annotations(tmp_path):
    from pathlib import Path

    from verification.experiment_targets import validate_plan_targets

    claim, materials, result, calls, raw = public(tmp_path)
    record = result.plans[0].target_bindings["c1"].projection
    audit = json.loads(Path(record.scope_audit).read_text("utf8"))
    assert audit["first_pass_response"] == raw[0] and audit["response"] == raw[1]
    assert audit["input"] == calls[1]["payload"]
    assert "choice_privacy" not in audit and "choice_audit_representation" not in audit
    validate_plan_targets(result.plans[0], claim, materials)


@pytest.mark.parametrize("phase", ["selection", "review"])
@pytest.mark.parametrize("bad", [[True], [0.0], ["0"], [-1], [1], [0, 0]])
def test_both_phase_rejection_indices_are_strict(phase, bad):
    from verification.execution_projection import ProjectionError
    from verification.execution_projection_choices import ChoicePrivacy, privacy_indices

    privacy = ChoicePrivacy()
    privacy.configs = [CFG]
    rows = [{"condition_id": "c1", "rationale": KEY}]
    privacy.rows(rows, phase)
    audit = privacy.audit(
        {"first_pass_response": {"execution_choices": rows}, "response": {"execution_choice_reviews": rows}}
    )
    safe = (
        audit["first_pass_response"]["execution_choices"]
        if phase == "selection"
        else audit["response"]["execution_choice_reviews"]
    )
    assert privacy_indices(safe, audit["choice_privacy"], phase) == [0]
    audit["choice_privacy"]["phases"][phase]["rejected_indices"] = bad
    with pytest.raises(ProjectionError, match="privacy"):
        privacy_indices(safe, audit["choice_privacy"], phase)


def test_metadata_only_hash_tamper_invalidates_l3_binding(tmp_path):
    from pathlib import Path

    from verification.experiment_targets import TargetBindingError, validate_plan_targets

    claim, materials, result, _, _ = mixed_choices(tmp_path)
    plan = result.plans[0]
    path = Path(plan.target_bindings["c1"].projection.scope_audit)
    audit = json.loads(path.read_text("utf8"))
    audit["choice_privacy"]["phases"]["selection"]["original_rows_sha256"] = "0" * 64
    path.write_text(json.dumps(audit), encoding="utf8")
    with pytest.raises(TargetBindingError):
        validate_plan_targets(plan, claim, materials)
