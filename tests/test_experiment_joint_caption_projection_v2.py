"""Only the complete adjacent selected caption can supplement a bounded table."""

import copy
from pathlib import Path

import pytest

from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock
from tests.test_experiment_joint_sources_v2 import offline as offline
from tests.test_experiment_joint_sources_v2 import paper, review_for, run
from verification.experiment_catalog import _stable_id, build_catalog, resolve_cell, resolve_source
from verification.experiment_sources import prepare_joint_candidate
from verification.experiments import ExperimentItem

OLD = "Table 9: D train accuracy (ms)."
CAPTION = "Table 1: D test results."


def prepare(tmp_path, mode="full", *, caption=CAPTION, old=OLD, unit=None, header="D", settings=None):
    claim, materials, item, _, _ = paper(tmp_path)
    body = item.quote[item.quote.index("<table") :].replace("<th>D</th>", f"<th>{header}</th>")
    block = materials.blocks[0]
    block.text = old + "\n" + caption + "\n" + body
    extras = [source.model_dump() for source in item.additional_sources]
    member = caption + "\n" + body
    if mode != "full":
        member = body
        captions = {
            "separate": [caption],
            "title_only": ["Table 1:"],
            "mid_title": [caption[2:]],
            "mid_body": [caption[caption.index(":") + 2 :]],
            "only_old": [old],
            "split": [caption[: caption.index(":") + 1], caption[caption.index(":") + 1 :]],
            "partial_tail": [CAPTION],
            "body_only": [],
            "other_block": [],
        }[mode]
        extras += [dict(block_id="table", quote=value) for value in captions]
    if mode == "other_block":
        materials.blocks.append(
            MaterialBlock(
                id="unrelated_caption",
                text=CAPTION,
                kind="text",
                loc=ClaimLocation(page=9),
            )
        )
        extras.append(dict(block_id="unrelated_caption", quote=CAPTION))
    if unit:
        claim.conditions[0].settings["unit"] = unit
    claim.conditions[0].settings.update(settings or {})
    materials.markdown = ""
    for current in materials.blocks:
        start = len(materials.markdown)
        materials.markdown += current.text + "\n"
        current.loc.char_start, current.loc.char_end = start, start + len(current.text)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    item = ExperimentItem.model_validate({**item.model_dump(), "quote": member, "additional_sources": extras})
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
    assert not bounded["joint_view"]["errors"]
    original_table_id = next(key for key, value in catalog["tables"].items() if value["table"] == 0)
    original_table = catalog["tables"][original_table_id]
    parent_source = original_table["source_id"]
    # The existing immutable body-only identity must remain selectable by old
    # saved responses. A separately associated caption must not widen this range.
    body_record = {
        **copy.deepcopy(catalog["sources"][parent_source]),
        "start": block.text.index(body),
        "end": block.text.index(body) + len(body),
        "candidate_id": bounded["joint_view"]["candidate_id"],
        "parent_source_ids": [parent_source],
    }
    body_id = _stable_id("src", catalog["paper_key"], catalog["source_pdf"], body_record)
    table_id = _stable_id("table", body_id, 0)
    assert bounded["sources"][body_id] == body_record
    assert table_id in bounded["tables"]
    for parent_cell in catalog["cells"].values():
        if parent_cell["table_id"] == original_table_id:
            key = _stable_id(
                "cell", body_id, 0, parent_cell["row"], parent_cell["column"], parent_cell["token"]
            )
            assert key in bounded["cells"]
    primary_members = copy.deepcopy(bounded)
    primary_members["joint_view"]["members"] = primary_members["joint_view"]["members"][:3]
    review = review_for(primary_members)
    review["items"][0]["grounds_source_ids"] = list(bounded["joint_view"]["member_source_ids"])
    uses = review["items"][0]["source_uses"]
    known = {use["source_id"] for use in uses}
    for source in bounded["joint_view"]["members"]:
        if source["source_id"] not in known:
            uses.append(
                dict(
                    source_id=source["source_id"],
                    roles=["other_qualifier"],
                    rationale="Declared caption context.",
                )
            )
    selected = review["items"][0]["comparisons"][0]["left"]["cell_id"]
    return (claim, materials, item, catalog, bounded), review, selected, body, table_id


@pytest.mark.parametrize("mode", ["full", "separate"])
def test_complete_selected_caption_keeps_body_ids_and_projects_parent_span(tmp_path, mode):
    data, _, selected, body, table_id = prepare(tmp_path, mode)
    catalog, bounded, materials = data[3], data[4], data[1]
    resolved = resolve_cell(bounded, selected, materials)
    assert resolved["quote"] == body
    assert resolved["caption"] == CAPTION
    assert not resolved["caption_ambiguous"]
    projected_id = bounded["tables"][table_id]["caption_source_id"]
    projected = resolve_source(bounded, projected_id, materials)
    original_id = catalog["tables"][bounded["tables"][table_id]["parent_table_id"]]["caption_source_id"]
    assert projected["parent_source_ids"] == [original_id]
    assert projected["block_sha256"] == catalog["sources"][original_id]["block_sha256"]
    assert projected["start"] == len(OLD) + 1
    assert projected["end"] == len(OLD) + 1 + len(CAPTION)
    assert bounded["tables"][table_id]["source_id"] == resolved["source_id"]


@pytest.mark.parametrize("mode", ["full", "separate"])
def test_two_pass_joint_uses_selected_caption_for_reference_and_shared_setting(tmp_path, mode):
    data, review, _, body, _ = prepare(tmp_path, mode)
    result, calls = run(data, review)
    assert not result.issues, result.issues
    assert result.evidence[0].sufficient
    assert len(calls) == 2
    assert data[4]["sources"][next(iter(data[4]["tables"].values()))["source_id"]]["end"] - data[4][
        "sources"
    ][next(iter(data[4]["tables"].values()))["source_id"]]["start"] == len(body)


def test_selected_caption_percent_unit_is_consumed_without_previous_caption_milliseconds(tmp_path):
    data, review, selected, body, _ = prepare(
        tmp_path, caption="Table 1: D test accuracy (%).", unit="percent"
    )
    result, _ = run(data, review)
    assert result.evidence[0].sufficient, result.issues
    assert resolve_cell(data[4], selected, data[1])["quote"] == body


@pytest.mark.parametrize(
    "mode", ["title_only", "mid_title", "mid_body", "only_old", "split", "body_only", "other_block"]
)
def test_partial_or_other_caption_does_not_grant_selected_table_context(tmp_path, mode):
    data, review, selected, _, _ = prepare(tmp_path, mode)
    assert resolve_cell(data[4], selected, data[1])["caption"] == ""
    result, _ = run(data, review)
    assert not any(evidence.sufficient for evidence in result.evidence)


def test_unit_and_protocol_suffix_cannot_be_dropped_from_declared_caption(tmp_path):
    data, review, selected, _, _ = prepare(
        tmp_path,
        "partial_tail",
        caption=CAPTION + " Accuracy (%) uses single-crop protocol.",
        unit="percent",
    )
    assert resolve_cell(data[4], selected, data[1])["caption"] == ""
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient


def test_complete_latest_caption_table_number_must_match_original_reference(tmp_path):
    data, review, selected, _, _ = prepare(tmp_path, caption="Table 4: D test results.")
    assert resolve_cell(data[4], selected, data[1])["caption"] == "Table 4: D test results."
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient
    assert any("table_reference" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "caption,old,header",
    [
        (CAPTION, "Table 9: D test accuracy (%).", "D"),
        ("Table 1: E test accuracy (%).", OLD, "D test"),
        ("Table 1: D train accuracy (%).", OLD, "D test"),
        ("Table 1: D test accuracy (%) (ms).", OLD, "D"),
        ("Table 1: D test accuracy (%)/s.", OLD, "D"),
        ("Table 1: D test accuracy (unknownunit).", OLD, "D"),
    ],
)
def test_caption_unit_and_explicit_axis_conflicts_remain_insufficient(tmp_path, caption, old, header):
    data, review, _, _, _ = prepare(tmp_path, caption=caption, old=old, header=header, unit="percent")
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient, result.issues


def test_fallback_caption_does_not_upgrade_first_pass_partial(tmp_path):
    data, review, _, _, _ = prepare(tmp_path)
    data[2].fully_supported_conditions = []
    data = (*data[:4], prepare_joint_candidate(data[0], data[1], data[3], data[2], 0))
    review = review_for(data[4])
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient


@pytest.mark.parametrize(
    "caption,header,settings",
    [
        ("accuracy (%)", "D test", {}),
        ("D accuracy (%)", "D test", {}),
        ("D test accuracy (%)", "D test", {}),
        ("Results: D test accuracy (%)", "D test", {}),
        ("Results for D test accuracy (%)", "D test", {}),
        ("D test accuracy (%) with seed 42", "D test seed 42", {"seed": 42}),
        ("D test seed:42 accuracy (%)", "D test seed 42", {"seed": 42}),
        ("D test single-crop accuracy (%)", "D test single-crop", {"evaluation": "single-crop"}),
        (
            "D test accuracy (%) with evaluation=single-crop",
            "D test single-crop",
            {"evaluation": "single-crop"},
        ),
    ],
)
def test_caption_units_accept_only_complete_matching_condition_scope(tmp_path, caption, header, settings):
    data, review, _, _, _ = prepare(
        tmp_path,
        caption=f"Table 1: {caption}.",
        header=header,
        settings=settings,
        unit="percent",
    )
    result, _ = run(data, review)
    assert result.evidence[0].sufficient, result.issues


@pytest.mark.parametrize(
    "caption,header,settings",
    [
        ("D train accuracy (%)", "D test", {}),
        ("D test seed 43 accuracy (%)", "D test seed 42", {"seed": 42}),
        ("D test 42 accuracy (%)", "D test seed 42", {"seed": 42}),
        ("D test multi-crop accuracy (%)", "D test single-crop", {"evaluation": "single-crop"}),
        ("A accuracy (%)", "D test", {}),
        ("A only D test accuracy (%)", "D test", {}),
        ("D not test accuracy (%)", "D test", {}),
        ("D test or validation accuracy (%)", "D test", {}),
        ("D test accuracy (%) or latency (ms)", "D test", {}),
        ("D test accuracy (%) and F1 (%)", "D test", {}),
        ("D test accuracy (%) without augmentation", "D test", {}),
        ("D test accuracy (%) at epoch 90", "D test", {}),
    ],
)
def test_inapplicable_caption_cannot_donate_units(tmp_path, caption, header, settings):
    data, review, _, _, _ = prepare(
        tmp_path,
        caption=f"Table 1: {caption}.",
        header=header,
        settings=settings,
        unit="percent",
    )
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient, result.issues
    assert any("units" in issue for issue in result.issues), result.issues


@pytest.mark.parametrize(
    "caption",
    [
        "D train accuracy (ms)",
        "A only D test accuracy (ms)",
        "D test seed 43 accuracy (ms)",
    ],
)
def test_inapplicable_caption_does_not_replace_selected_axis_units(tmp_path, caption):
    data, review, _, _, _ = prepare(
        tmp_path,
        caption=f"Table 1: {caption}.",
        header="D test (%)",
        unit="percent",
    )
    result, _ = run(data, review)
    assert result.evidence[0].sufficient, result.issues


def test_applicable_caption_conflict_with_selected_axis_still_rejects(tmp_path):
    data, review, _, _, _ = prepare(
        tmp_path,
        caption="Table 1: D test accuracy (ms).",
        header="D test (%)",
        unit="percent",
    )
    result, _ = run(data, review)
    assert not result.evidence[0].sufficient
    assert any("units/scales do not match" in issue for issue in result.issues)
