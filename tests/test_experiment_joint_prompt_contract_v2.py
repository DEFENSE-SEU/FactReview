"""Both model boundaries expose complete joint membership and single-source rules."""

import copy
import json
from pathlib import Path

import pytest

from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock
from tests.test_experiment_joint_sources_v2 import offline as offline
from tests.test_experiment_joint_sources_v2 import paper, review_for
from verification.experiment_catalog import build_catalog
from verification.experiment_sources import prepare_joint_candidate
from verification.experiments import ExperimentItem, verify_experiments


def separate_reference(tmp_path, *, include_reference=True, joint_index=0):
    claim, materials, candidate, _, _ = paper(tmp_path)
    setup = next(block for block in materials.blocks if block.id == "setup")
    setup.text = "Augment denotes the augmented procedure applied to A."
    reference = MaterialBlock(
        id="reference",
        text="Our Table 1 reports D test results.",
        kind="text",
        loc=ClaimLocation(page=1),
    )
    materials.blocks.append(reference)
    materials.markdown = ""
    for block in materials.blocks:
        start = len(materials.markdown)
        materials.markdown += block.text + "\n"
        block.loc.char_start, block.loc.char_end = start, start + len(block.text)
    claim.loc = next(block.loc for block in materials.blocks if block.id == "claim")
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    candidate.additional_sources[1].quote = setup.text
    if include_reference:
        candidate = type(candidate).model_validate(
            {
                **candidate.model_dump(),
                "additional_sources": [
                    *[source.model_dump() for source in candidate.additional_sources],
                    {"block_id": reference.id, "quote": reference.text},
                ],
            }
        )
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, candidate, joint_index)
    review = review_for(bounded, index=joint_index)
    joint = review["items"][0]
    for use in joint["source_uses"]:
        if "setup_definition" in use["roles"]:
            use["roles"] = ["setup_definition"]
    if include_reference:
        member = next(m for m in bounded["joint_view"]["members"] if m["block_id"] == "reference")
        for bridge in joint["comparisons"][0]["bridges"]:
            bridge["source_ids"].append(member["source_id"])
        joint["source_uses"].append(
            dict(
                source_id=member["source_id"], roles=["table_reference"], rationale="Named manuscript table."
            )
        )
    global_reference = next(
        key
        for key, source in catalog["sources"].items()
        if source.get("block_id") == "reference" and source["kind"] == "paper_block"
    )
    partial = dict(
        aspect="correspondence",
        kind="paper_support",
        block_id=reference.id,
        quote=reference.text,
        covered=["c1"],
        fully_supported_conditions=[],
        detail="Reference alone lacks measured values.",
        additional_sources=[],
    )
    partial_scope = dict(
        item_index=1 - joint_index,
        condition_id="c1",
        applicability="applicable",
        full_support=False,
        qualifiers_complete=False,
        comparison_objects="matched",
        grounds_source_ids=[global_reference],
        source_uses=[],
        rationale="The reference alone does not establish the scores.",
    )
    candidates = [candidate.model_dump(), partial]
    review["items"].append(partial_scope)
    if joint_index:
        candidates.reverse()
        review["items"].reverse()
    first = dict(
        checked_aspects=["correspondence", "fairness", "isolation", "stability", "consistency"],
        items=candidates,
        plans=[],
        issues=[],
    )
    return claim, materials, first, review


def invoke(data):
    claim, materials, first, scope = data
    original = copy.deepcopy([claim.model_dump(), materials.model_dump(), first, scope])
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2, "Contract clarification must not add a repair call"
        return copy.deepcopy(first if len(calls) == 1 else scope)

    result = verify_experiments(claim, materials, call=model)
    assert len(calls) == 2
    assert [claim.model_dump(), materials.model_dump(), first, scope] == original
    payload = json.loads(calls[1]["prompt"])
    expected_single = [i for i, item in enumerate(first["items"]) if not item["additional_sources"]]
    assert payload["single_source_candidate_indices"] == expected_single
    assert set(payload["joint_candidates"]) == {
        str(i) for i in range(len(first["items"])) if i not in expected_single
    }
    assert [item["additional_sources"] for item in payload["candidate_items"]] == [
        item["additional_sources"] for item in first["items"]
    ]
    return result, calls


@pytest.mark.parametrize("joint_index", [0, 1])
def test_mixed_candidates_receive_both_stage_contracts_and_preserve_partial(tmp_path, joint_index):
    result, calls = invoke(separate_reference(tmp_path, joint_index=joint_index))
    assert not any("Experimental scope review unconfirmed" in issue for issue in result.issues)
    assert any("reference alone" in issue for issue in result.issues)
    assert result.evidence[joint_index].sufficient
    assert len(result.evidence[joint_index].additional_pointers) == 3
    assert not result.evidence[1 - joint_index].sufficient
    assert not result.evidence[1 - joint_index].additional_pointers
    first = json.loads(calls[0]["prompt"])
    first_description = first["output_schema"]["$defs"]["ExperimentItem"]["properties"]["additional_sources"][
        "description"
    ]
    assert "numbered-table-reference" in first_description
    assert "another partial item" in first_description
    assert "exact manuscript passage explicitly referencing that numbered table" in calls[0]["system"]
    assert "a definition need not repeat the table number" in calls[0]["system"]
    assert "same task, dataset and applicable settings" in calls[0]["system"]
    assert "independent scope review cannot append it" in calls[0]["system"]
    scope = json.loads(calls[1]["prompt"])
    scope_description = scope["output_schema"]["$defs"]["CatalogItemScopeV2"]["properties"]["source_uses"][
        "description"
    ]
    assert "single_source_candidate_indices" in scope_description and "MUST be []" in scope_description
    assert "single-source review uses grounds or bridges" in calls[1]["system"]
    bridge_description = scope["output_schema"]["$defs"]["SourceBridge"]["properties"]["source_ids"][
        "description"
    ]
    assert "each ID's resolved quote" in bridge_description
    assert "may use different IDs" in bridge_description
    assert "explanation does not supply a missing source ID" in bridge_description


@pytest.mark.parametrize("tamper", ["missing_definition", "missing_reference", "wrong_definition_id"])
def test_bridge_explanation_cannot_replace_missing_or_wrong_actual_source_ids(tmp_path, tamper):
    data = separate_reference(tmp_path)
    claim, materials, first, scope = data
    candidate = first["items"][0]
    bounded = prepare_joint_candidate(
        claim, materials, build_catalog(claim, materials), ExperimentItem.model_validate(candidate), 0
    )
    members = {member["block_id"]: member["source_id"] for member in bounded["joint_view"]["members"]}
    bridge = scope["items"][0]["comparisons"][0]["bridges"][0]
    bridge["explanation"] = (
        "The metric on D is accuracy. Our Table 1 reports D test results. "
        "These exact definition and manuscript-reference passages establish the metric."
    )
    assert members["metric"] in bridge["source_ids"] and members["reference"] in bridge["source_ids"]
    if tamper == "missing_reference":
        bridge["source_ids"].remove(members["reference"])
    elif tamper == "missing_definition":
        bridge["source_ids"].remove(members["metric"])
    else:
        # Select a valid declared member with real values and dataset, whose
        # actual quote does not define accuracy. No unknown ID or syntax error.
        bridge["source_ids"][bridge["source_ids"].index(members["metric"])] = members["table"]
        table = next(block for block in materials.blocks if block.id == "table")
        assert "accuracy" not in table.text
    assert len(bridge["source_ids"]) == len(set(bridge["source_ids"]))
    assert set(bridge["source_ids"]) <= set(members.values())
    assert candidate["fully_supported_conditions"] == ["c1"]
    assert scope["items"][0]["full_support"] and scope["items"][0]["qualifiers_complete"]
    result, _ = invoke(data)
    assert len(result.evidence) == 2
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert len(result.evidence[0].additional_pointers) == 3
    assert any(
        "metric_definition" in issue or "External definition has no verified self-reference" in issue
        for issue in result.issues
    ), result.issues


def test_reference_in_another_partial_candidate_grants_no_joint_access(tmp_path):
    result, _ = invoke(separate_reference(tmp_path, include_reference=False))
    assert len(result.evidence) == 2
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert len(result.evidence[0].additional_pointers) == 2
    assert all("Our Table 1" not in pointer.quote for pointer in result.evidence[0].additional_pointers)
    assert any("reference" in issue.lower() for issue in result.issues), result.issues


@pytest.mark.parametrize("flag", ["first", "full_support", "qualifiers_complete"])
def test_complete_members_do_not_upgrade_partial_scientific_judgments(tmp_path, flag):
    data = separate_reference(tmp_path)
    if flag == "first":
        data[2]["items"][0]["fully_supported_conditions"] = []
    else:
        data[3]["items"][0][flag] = False
    result, _ = invoke(data)
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert len(result.evidence[0].additional_pointers) == 3


def test_single_source_uses_still_invalidates_condition_without_auto_correction(tmp_path):
    data = separate_reference(tmp_path)
    partial = data[3]["items"][1]
    partial["source_uses"] = [
        dict(
            source_id=partial["grounds_source_ids"][0],
            roles=["table_reference"],
            rationale="Exact reference.",
        )
    ]
    result, _ = invoke(data)
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert any("Single-source candidates cannot claim joint source_uses" in issue for issue in result.issues)


def test_single_only_scope_metadata_does_not_create_a_joint_candidate(tmp_path):
    claim, materials, first, scope = separate_reference(tmp_path)
    first["items"] = [first["items"][1]]
    scope["items"] = [scope["items"][1]]
    scope["items"][0]["item_index"] = 0
    result, calls = invoke((claim, materials, first, scope))
    payload = json.loads(calls[1]["prompt"])
    assert payload["single_source_candidate_indices"] == [0]
    assert payload["joint_candidates"] == {}
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert not result.evidence[0].additional_pointers
