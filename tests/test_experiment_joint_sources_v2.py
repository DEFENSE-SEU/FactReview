"""Explicit new joint candidates use two mocked passes; cached claims are never upgraded."""

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_catalog import build_catalog
from verification.experiment_sources import prepare_joint_candidate
from verification.experiments import ExperimentItem, verify_experiments

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", lambda **kw: pytest.fail("Unmocked model call"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network call"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("External process"))


def paper(tmp_path):
    texts = {
        "table": "Table 1: D test results.\n<table><tr><th>Method</th><th>D</th></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>",
        "metric": "The metric on D is accuracy.",
        "setup": "Table 1 reports D test results. Augment denotes the augmented procedure applied to A.",
        "claim": "A with the augmented procedure has higher accuracy than B on D test.",
    }
    materials = SharedMaterials(
        paper_key="joint",
        source_pdf="fixture.pdf",
        markdown="",
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="fixture.json",
        provider="mock",
    )
    for key, text in texts.items():
        start = len(materials.markdown)
        materials.markdown += text + "\n"
        materials.blocks.append(
            MaterialBlock(
                id=key,
                text=text,
                kind="table" if key == "table" else "text",
                loc=ClaimLocation(page=1, char_start=start, char_end=start + len(text)),
            )
        )
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    claim = Claim(
        id="claim",
        text=texts["claim"],
        source_block_id="claim",
        source_quote=texts["claim"],
        loc=materials.blocks[-1].loc,
        needs=["Experiments"],
        conditions=[
            Condition(
                id="c1",
                dataset="D",
                metric="accuracy",
                settings={"method": "A", "baseline": "B", "split": "test", "procedure": "augmented"},
            )
        ],
    )
    item = ExperimentItem(
        aspect="correspondence",
        kind="paper_support",
        block_id="table",
        quote=texts["table"],
        covered=["c1"],
        fully_supported_conditions=["c1"],
        detail="Explicit new joint candidate.",
        additional_sources=[{"block_id": key, "quote": texts[key]} for key in ("metric", "setup")],
    )
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
    return claim, materials, item, catalog, bounded


def review_for(bounded, *, index=0):
    members = {member["block_id"]: member["source_id"] for member in bounded["joint_view"]["members"]}
    cells = {(record["row"], record["column"]): identifier for identifier, record in bounded["cells"].items()}
    table_id = bounded["cells"][cells[1, 1]]["table_id"]
    bridges = [
        dict(
            kind="metric",
            condition_field="metric",
            applies_to="shared",
            table_id=table_id,
            source_ids=[members["metric"], members["setup"]],
            paper_label="accuracy",
            explanation="D accuracy is defined explicitly.",
        ),
        dict(
            kind="setting",
            condition_field="settings.procedure",
            applies_to="subject",
            table_id=table_id,
            source_ids=[members["setup"]],
            paper_label="Augment",
            explanation="Augment is the augmented procedure on A.",
        ),
    ]
    condition = dict(
        condition_id="c1",
        assertion="controlled_comparison",
        matched_controls_required=False,
        uncertainty_sensitive=False,
        relation="gt",
        rationale="Comparison of the exact reported results.",
        setting_scopes={"procedure": "subject", "split": "shared"},
    )
    item = dict(
        item_index=index,
        condition_id="c1",
        applicability="applicable",
        full_support=True,
        qualifiers_complete=True,
        comparison_objects="matched",
        rationale="Table plus definitions establish the condition.",
        grounds_source_ids=list(members.values()),
        source_uses=[
            {"source_id": members["table"], "roles": ["result"], "rationale": "Exact values and roles."},
            {
                "source_id": members["metric"],
                "roles": ["metric_definition"],
                "rationale": "Exact measured quantity.",
            },
            {
                "source_id": members["setup"],
                "roles": ["setup_definition", "table_reference"],
                "rationale": "Exact treatment and named table.",
            },
        ],
        comparisons=[
            dict(
                case_id=bounded["conditions"]["c1"]["cases"][0]["id"],
                left={"kind": "cell", "cell_id": cells[1, 1], "label_cell_id": cells[1, 0]},
                right={"kind": "cell", "cell_id": cells[2, 1], "label_cell_id": cells[2, 0]},
                relation="gt",
                bridges=bridges,
            )
        ],
    )
    return {"schema_version": "catalog-v2", "conditions": [condition], "items": [item]}


def run(p, review=None, items=None):
    claim, materials, item, _, bounded = p
    candidate = {"checked_aspects": ASPECTS, "items": items or [item.model_dump()], "plans": [], "issues": []}
    response = review if review is not None else review_for(bounded)
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(candidate if len(calls) == 1 else response)

    result = verify_experiments(claim, materials, call=model)
    assert len(calls) == 2
    return result, calls


def test_explicit_joint_table_definitions_two_pass_support(tmp_path):
    p = paper(tmp_path)
    before = [p[0].model_dump(), p[1].model_dump()]
    result, calls = run(p)
    assert not result.issues, result.issues
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.sufficient and evidence.covered == ["c1"]
    assert [pointer.quote for pointer in evidence.additional_pointers] == [b.text for b in p[1].blocks[1:3]]
    payload = json.loads(calls[1]["prompt"])
    assert payload["joint_candidates"]["0"]["catalog"]["candidate_id"] == p[4]["joint_view"]["candidate_id"]
    assert [p[0].model_dump(), p[1].model_dump()] == before


@pytest.mark.parametrize(
    "flag", ["first", "full_support", "qualifiers_complete", "unresolved", "no_comparisons"]
)
def test_joint_does_not_upgrade_partial_flags(tmp_path, flag):
    p = paper(tmp_path)
    review = review_for(p[4])
    if flag == "first":
        p[2].fully_supported_conditions = []
        p = (*p[:4], prepare_joint_candidate(p[0], p[1], p[3], p[2], 0))
        review = review_for(p[4])
    elif flag == "unresolved":
        review["items"][0]["unresolved_qualifiers"] = ["Seed robustness remains unresolved."]
    elif flag == "no_comparisons":
        review["items"][0]["comparisons"] = []
    else:
        review["items"][0][flag] = False
    result, _ = run(p, review)
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient


@pytest.mark.parametrize(
    "tamper", ["global_source", "missing_use", "duplicate_use", "unused_role", "foreign_cell", "wrong_setup"]
)
def test_joint_source_or_binding_failure_is_insufficient(tmp_path, tamper):
    p = paper(tmp_path)
    review = review_for(p[4])
    row = review["items"][0]
    if tamper == "global_source":
        row["grounds_source_ids"] = [
            next(k for k, v in p[3]["sources"].items() if v.get("block_id") == "table")
        ]
    elif tamper == "missing_use":
        row["source_uses"].pop()
    elif tamper == "duplicate_use":
        row["source_uses"].append(copy.deepcopy(row["source_uses"][0]))
    elif tamper == "unused_role":
        row["source_uses"][1]["roles"] = ["result"]
    elif tamper == "foreign_cell":
        row["comparisons"][0]["left"]["cell_id"] = next(iter(p[3]["cells"]))
    else:
        row["comparisons"][0]["bridges"][1]["paper_label"] = "B"
    result, _ = run(p, review)
    assert result.issues and not any(e.sufficient for e in result.evidence)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "over_limit", "primary"])
def test_invalid_joint_sources_preserve_only_valid_observation(tmp_path, monkeypatch, tamper):
    p = paper(tmp_path)
    review = review_for(p[4])
    if tamper == "missing":
        p[2].additional_sources[0].quote = "Not in the original source."
    elif tamper == "duplicate":
        p[2].additional_sources.append(copy.deepcopy(p[2].additional_sources[0]))
    elif tamper == "over_limit":
        monkeypatch.setenv("EXPERIMENT_JOINT_MAX_SOURCES", "2")
    else:
        p[2].quote = "Missing primary."
    result, _ = run(p, review)
    assert result.issues and not any(e.sufficient for e in result.evidence)
    if tamper == "primary":
        assert not result.evidence
    elif tamper == "missing":
        assert len(result.evidence[0].additional_pointers) == 1


def with_concern(p):
    review = review_for(p[4])
    review["conditions"][0]["matched_controls_required"] = True
    primary = next(b for b in p[1].blocks if b.id == "claim")
    source_id = next(
        k for k, v in p[3]["sources"].items() if v.get("block_id") == "claim" and v["kind"] == "paper_block"
    )
    item = dict(
        aspect="fairness",
        kind="missing_control",
        block_id="claim",
        quote=primary.text,
        covered=["c1"],
        detail="Matched protocol is needed for this controlled comparison.",
    )
    review["items"].append(
        dict(
            item_index=1,
            condition_id="c1",
            applicability="applicable",
            grounds_source_ids=[source_id],
            rationale="Independent control concern.",
            comparison_objects="matched",
        )
    )
    return review, [p[2].model_dump(), item]


@pytest.mark.parametrize("order", ["valid_bad", "bad_valid", "valid_bad_valid", "missing"])
@pytest.mark.parametrize("malformed", [False, True])
def test_joint_pair_failure_preserves_same_condition_healthy_concern(tmp_path, order, malformed):
    p = paper(tmp_path)
    review, items = with_concern(p)
    valid, concern = review["items"]
    bad = copy.deepcopy(valid)
    bad["condition_id"] = " c1 "
    bad["grounds_source_ids"] = ["not-a-source"]
    if malformed:
        bad.pop("applicability")
    sequence = {
        "valid_bad": [valid, bad],
        "bad_valid": [bad, valid],
        "valid_bad_valid": [valid, bad, copy.deepcopy(valid)],
        "missing": [],
    }[order]
    review["items"] = [*sequence, concern]
    result, _ = run(p, review, items)
    assert len(result.evidence) == 2
    support, concern_evidence = result.evidence
    assert not support.sufficient
    assert concern_evidence.sufficient and concern_evidence.concern


@pytest.mark.parametrize("index", [True, False, "0", 0.0, -1, 99])
def test_noncanonical_joint_index_has_no_pair_local_privilege(tmp_path, index):
    p = paper(tmp_path)
    review, items = with_concern(p)
    review["items"][0]["item_index"] = index
    result, _ = run(p, review, items)
    assert result.issues and len(result.evidence) == 2
    assert not any(e.sufficient or e.concern for e in result.evidence)


def test_raw_fake_joint_cannot_change_legacy_invalidation(tmp_path):
    p = paper(tmp_path)
    review, items = with_concern(p)
    review["items"][1]["joint"] = True
    result, _ = run(p, review, items)
    assert not any(e.sufficient for e in result.evidence)


def test_condition_duplicate_still_invalidates_all_joint_and_single_items(tmp_path):
    p = paper(tmp_path)
    review, items = with_concern(p)
    duplicate = copy.deepcopy(review["conditions"][0])
    duplicate["condition_id"] = " c1 "
    review["conditions"].append(duplicate)
    result, _ = run(p, review, items)
    assert not any(e.sufficient for e in result.evidence)


def test_missing_member_source_error_does_not_erase_healthy_concern(tmp_path):
    p = paper(tmp_path)
    review, items = with_concern(p)
    items[0]["additional_sources"][0]["quote"] = "Missing exact quote."
    result, _ = run(p, review, items)
    assert not result.evidence[0].sufficient and result.evidence[1].concern


def test_original_partial_observation_is_not_merged_or_upgraded(tmp_path):
    p = paper(tmp_path)
    bounded = prepare_joint_candidate(p[0], p[1], p[3], p[2], 1)
    joint_review = review_for(bounded, index=1)
    legacy = copy.deepcopy(joint_review["items"][0])
    legacy["item_index"] = 0
    legacy.pop("source_uses")
    mapping = {key: value["parent_cell_id"] for key, value in bounded["cells"].items()}
    mapping.update({key: value["parent_table_id"] for key, value in bounded["tables"].items()})
    for key, value in bounded["sources"].items():
        mapping[key] = next(
            k
            for k, v in p[3]["sources"].items()
            if v.get("block_id") == value["block_id"] and v["kind"] == "paper_block"
        )

    def remap(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    legacy = remap(legacy)
    joint_review["items"].insert(0, legacy)
    partial = p[2].model_dump()
    partial.update(
        additional_sources=[],
        fully_supported_conditions=[],
        detail="Original standalone partial observation.",
    )
    result, _ = run(p, joint_review, [partial, p[2].model_dump()])
    assert not result.issues, result.issues
    assert len(result.evidence) == 2
    assert not result.evidence[0].sufficient and not result.evidence[0].additional_pointers
    assert result.evidence[1].sufficient and len(result.evidence[1].additional_pointers) == 2
    assert "Original standalone partial observation" in result.evidence[0].note


def test_rejected_plan_retains_verified_joint_observation(tmp_path):
    from verification.contracts import RejectedPlan

    p = paper(tmp_path)
    review = review_for(p[4])
    output = dict(
        checked_aspects=ASPECTS,
        items=[p[2].model_dump()],
        plans=[
            dict(
                targets=[
                    dict(
                        condition_id="foreign", reported=dict(block_id="table", quote=p[2].quote, token="90")
                    )
                ],
                run_mode="analysis",
                feasibility="blocked",
                priority="high",
            )
        ],
    )
    responses = iter([output, review])
    with pytest.raises(RejectedPlan) as error:
        verify_experiments(p[0], p[1], call=lambda **kw: next(responses))
    assert error.value.observations.evidence[0].sufficient
    assert len(error.value.observations.evidence[0].additional_pointers) == 2


def test_source_mutation_during_scope_never_creates_decisive_joint(tmp_path):
    p = paper(tmp_path)
    responses = iter([dict(checked_aspects=ASPECTS, items=[p[2].model_dump()], plans=[]), review_for(p[4])])
    calls = []

    def call(**kw):
        calls.append(kw)
        if len(calls) == 2:
            Path(p[1].markdown_path).write_text(p[1].markdown + "Source changed.", encoding="utf-8")
        return next(responses)

    result = verify_experiments(p[0], p[1], call=call)
    assert result.issues and not any(e.sufficient for e in result.evidence)


def test_authoritative_endpoints_need_no_supporting_member_and_have_separate_audit(tmp_path, monkeypatch):
    p = paper(tmp_path)
    claim, materials, item, _, _ = p
    claim.text += " The exact comparison is 90 vs 80."
    claim.source_quote = claim.text
    claim.conditions[0].description = "90 vs 80"
    block = materials.blocks[-1]
    block.text = claim.text
    block.loc.char_end = block.loc.char_start + len(block.text)
    materials.markdown = "\n".join(b.text for b in materials.blocks) + "\n"
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
    assert all(member["block_id"] != "claim" for member in bounded["joint_view"]["members"])
    monkeypatch.setattr("common.run_stats.stats_path", lambda: tmp_path / "run" / "stats.json")
    result, _calls = run((claim, materials, item, catalog, bounded))
    assert result.evidence[0].sufficient, result.issues
    audit_path = next((tmp_path / "run" / "experiment_scope").glob("*.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    view = audit["joint_sources"]["0"]
    assert len(view["members"]) == 3 and len(result.evidence[0].additional_pointers) == 2
    assert {entry["token"] for entry in view["assertion_consumption"]} == {"90", "80"}
    assert all(
        entry["domain"] == "condition_assertion" and not entry["grants_evidence_access"]
        for entry in view["assertion_consumption"]
    )
    assert view["final_checks"] == [{"condition_id": "c1", "errors": [], "first_pass_full": True}]
    assert all(member["artifact"]["sha256"] for member in view["members"])
    assert "scope_audit=" + str(audit_path) in result.evidence[0].note


def test_assertion_number_ids_cannot_supply_joint_observations(tmp_path):
    p = paper(tmp_path)
    claim, materials, item, _, _ = p
    claim.text += " The exact difference is 10 accuracy."
    claim.source_quote = claim.text
    block = materials.blocks[-1]
    block.text = claim.text
    block.loc.char_end = block.loc.char_start + len(block.text)
    materials.markdown = "\n".join(b.text for b in materials.blocks) + "\n"
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
    assert bounded["assertion_numbers"] and set(bounded["assertion_numbers"]).isdisjoint(bounded["numbers"])
    review = review_for(bounded)
    review["items"][0]["comparisons"][0]["left"] = {
        "kind": "prose",
        "number_id": next(iter(bounded["assertion_numbers"])),
    }
    result, _ = run((claim, materials, item, catalog, bounded), review)
    assert result.issues and not result.evidence[0].sufficient


@pytest.mark.parametrize("name", ["beit039", "beit042"])
def test_new_mock_joint_over_unchanged_real_beit_source_objects(tmp_path, name):
    # These are explicitly NEW mocked first/scope candidates. No historical
    # real model response is read, rewritten, merged or upgraded here.
    from tests import test_experiments_bindings_v2 as fixtures

    original, prior_candidate, prior_review = getattr(fixtures, name)(tmp_path)
    claim, materials, catalog = original
    before = [claim.model_dump(), materials.model_dump()]
    candidates, scopes = [], []
    for index, condition in enumerate(claim.conditions):
        keys = (
            ["block_81", "block_82"]
            if name == "beit039"
            else ["block_67" if index == 0 else "block_81", "block_88"]
        )
        candidate = copy.deepcopy(prior_candidate)
        candidate.update(
            covered=[condition.id],
            fully_supported_conditions=[condition.id],
            additional_sources=[
                {"block_id": key, "quote": next(b.text for b in materials.blocks if b.id == key)}
                for key in keys
            ],
        )
        item = ExperimentItem.model_validate(candidate)
        bounded = prepare_joint_candidate(claim, materials, catalog, item, index)
        assert not bounded["joint_view"]["errors"]
        members = {member["block_id"]: member["source_id"] for member in bounded["joint_view"]["members"]}
        mapping = {value["parent_cell_id"]: key for key, value in bounded["cells"].items()}
        mapping.update({value["parent_table_id"]: key for key, value in bounded["tables"].items()})
        for key, value in catalog["sources"].items():
            if value.get("block_id") in members:
                mapping[key] = members[value["block_id"]]
        old = copy.deepcopy(prior_review["items"][index])
        old["item_index"] = index
        comparison = old["comparisons"][0]
        comparison["left"] = {
            "kind": "cell",
            "cell_id": mapping[comparison.pop("left_cell_id")],
            "label_cell_id": mapping[comparison.pop("left_label_cell_id")],
        }
        comparison["right"] = {
            "kind": "cell",
            "cell_id": mapping[comparison.pop("right_cell_id")],
            "label_cell_id": mapping[comparison.pop("right_label_cell_id")],
        }
        for bridge in comparison["bridges"]:
            bridge["table_id"] = mapping[bridge["table_id"]]
            bridge["source_ids"] = [mapping[key] for key in bridge["source_ids"]]
        old["grounds_source_ids"] = list(members.values())
        old["source_uses"] = [
            dict(
                source_id=members[candidate["block_id"]],
                roles=["result"],
                rationale="Original table values and axes.",
            ),
            dict(
                source_id=members[keys[0]],
                roles=["metric_definition"],
                rationale="Original quantity definition.",
            ),
            dict(
                source_id=members[keys[1]],
                roles=["table_reference", *(["setup_definition"] if name == "beit039" else [])],
                rationale="Original table reference and applicable setup.",
            ),
        ]
        candidates.append(candidate)
        scopes.append(old)
    responses = iter(
        [
            dict(checked_aspects=ASPECTS, items=candidates, plans=[]),
            dict(
                schema_version="catalog-v2",
                conditions=copy.deepcopy(prior_review["conditions"]),
                items=scopes,
            ),
        ]
    )
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        return next(responses)

    result = verify_experiments(claim, materials, call=model)
    assert calls == ["verification.experiments", "verification.experiments.scope"]
    assert not result.issues, result.issues
    assert len(result.evidence) == len(claim.conditions) and all(e.sufficient for e in result.evidence)
    assert all(len(e.additional_pointers) == 2 for e in result.evidence)
    assert [claim.model_dump(), materials.model_dump()] == before
