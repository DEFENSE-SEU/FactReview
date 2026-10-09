"""Candidate views authorize original ranges, never clipped or merged contexts."""

from pathlib import Path

import pytest

from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_catalog import build_catalog, resolve_cell, resolve_source
from verification.experiment_sources import prepare_joint_candidate, require_passage
from verification.experiments import ExperimentItem, _expected_number, _occurrence_number

TABLE = "<table><tr><th>Method</th><th>D test accuracy</th></tr><tr><td>A</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
SENTENCE = "On the test split of D, A has 90 accuracy and B has 80 accuracy."


def make(tmp_path, quote=None, extra=None):
    texts = {
        "tables": "Table 1: D test accuracy.\n"
        + TABLE
        + "\nTable 2: E test accuracy.\n"
        + TABLE.replace("D test", "E test"),
        "prose": SENTENCE + "\nOn the train split of E, A has 99 accuracy and B has 79 accuracy.",
        "definition": "D accuracy describes the reported measurement.",
        "claim": "A has higher D test accuracy than B.",
    }
    materials = SharedMaterials(
        paper_key="ranges",
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
                kind="text",
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
                settings={"method": "A", "baseline": "B", "split": "test"},
            )
        ],
    )
    item = ExperimentItem(
        aspect="correspondence",
        kind="paper_support",
        block_id="tables",
        quote=TABLE if quote is None else quote,
        covered=["c1"],
        detail="Range candidate",
        additional_sources=extra or [{"block_id": "definition", "quote": texts["definition"]}],
    )
    catalog = build_catalog(claim, materials)
    view = prepare_joint_candidate(claim, materials, catalog, item, 0)
    return claim, materials, item, catalog, view


def test_complete_table_only_retains_its_axes_without_unquoted_caption_or_other_table(tmp_path):
    _, materials, _, original, view = make(tmp_path)
    assert len(view["tables"]) == 1 and not view["joint_view"]["errors"]
    number_id = next(k for k, v in view["cells"].items() if v["token"] == "90")
    resolved = resolve_cell(view, number_id, materials)
    assert resolved["quote"] == TABLE and resolved["caption"] == ""
    assert resolved["table"] == 0 and resolved["origin_table_index"] == 0
    assert resolved["parent_cell_id"] in original["cells"]
    for source_id, source in original["sources"].items():
        if source.get("block_id") == "tables":
            with pytest.raises(ValueError, match="Unknown"):
                resolve_source(view, source_id, materials)


def test_second_table_preserves_parent_index_with_local_zero(tmp_path):
    quote = TABLE.replace("D test", "E test")
    _, materials, _, _, view = make(tmp_path, quote=quote)
    selected = next(k for k, v in view["cells"].items() if v["token"] == "90")
    record = resolve_cell(view, selected, materials)
    assert record["quote"] == quote
    assert record["table"] == 0 and record["origin_table_index"] == 1


@pytest.mark.parametrize("quote", ["<tr><td>A</td><td>90</td></tr>", "90", "D test accuracy"])
def test_numeric_or_row_fragment_cannot_borrow_original_axes(tmp_path, quote):
    _, _, _, _, view = make(tmp_path, quote=quote)
    assert not view["cells"] and not view["tables"]


def test_complete_sentence_retains_original_number_identity(tmp_path):
    _, materials, _, original, view = make(tmp_path, extra=[{"block_id": "prose", "quote": SENTENCE}])
    assert len(view["numbers"]) == 2
    for identifier in view["numbers"]:
        assert view["numbers"][identifier] == original["numbers"][identifier]
        assert _occurrence_number(view, identifier, materials).quote == SENTENCE
    outside = next(k for k, v in original["numbers"].items() if v["token"] == "99")
    with pytest.raises(ValueError):
        _occurrence_number(view, outside, materials)


def test_two_fragments_cannot_manufacture_shared_prefix_sentence(tmp_path):
    _, _, _, _, view = make(
        tmp_path,
        extra=[
            {"block_id": "prose", "quote": "On the test split of D, A has 90 accuracy"},
            {"block_id": "prose", "quote": "and B has 80 accuracy."},
        ],
    )
    assert not view["numbers"]


def test_different_candidate_ids_cannot_be_substituted(tmp_path):
    claim, materials, item, original, view = make(tmp_path)
    other = prepare_joint_candidate(claim, materials, original, item, 1)
    assert set(view["sources"]).isdisjoint(other["sources"])
    assert set(view["cells"]).isdisjoint(other["cells"])
    with pytest.raises(ValueError):
        resolve_cell(other, next(iter(view["cells"])), materials)


def test_block_and_artifact_mutations_are_detected(tmp_path):
    _, materials, _, _, view = make(tmp_path)
    member = view["joint_view"]["members"][0]
    Path(materials.markdown_path).write_text(materials.markdown + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        resolve_source(view, member["source_id"], materials)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    materials.blocks[0].text += " changed"
    with pytest.raises(ValueError, match="changed"):
        resolve_source(view, member["source_id"], materials)


def test_implicit_context_must_fit_one_declared_member(tmp_path):
    _, materials, _, _, view = make(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        require_passage(view, materials, "tables", materials.blocks[0].text, purpose="bridge")


def test_candidate_members_cannot_authorize_other_condition_expected_number(tmp_path):
    claim, materials, _, _, _ = make(tmp_path)
    first = "D has a 5 point difference."
    second = "E has a 10 point difference."
    block = materials.blocks[-1]
    block.text = first + " " + second
    claim.text = block.text
    claim.source_quote = first
    claim.conditions.append(Condition(id="c2", dataset="E", metric="accuracy"))
    from schemas.claim import ClaimSourceRef

    claim.source_refs = [
        ClaimSourceRef(source_block_id="claim", source_quote=first, loc=block.loc, covered=["c1"]),
        ClaimSourceRef(source_block_id="claim", source_quote=second, loc=block.loc, covered=["c2"]),
    ]
    materials.markdown = "\n".join(b.text for b in materials.blocks) + "\n"
    block.loc.char_end = block.loc.char_start + len(block.text)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    original = build_catalog(claim, materials)
    item = ExperimentItem(
        aspect="correspondence",
        kind="paper_support",
        block_id="tables",
        quote=TABLE,
        covered=["c1"],
        detail="No foreign difference",
        additional_sources=[{"block_id": "claim", "quote": second}],
    )
    view = prepare_joint_candidate(claim, materials, original, item, 0)
    assert not view["joint_view"]["errors"]
    with pytest.raises(ValueError, match="absent"):
        _expected_number(claim, "c1", "10", materials, view)


def test_overlapping_distinct_members_do_not_merge_authority(tmp_path):
    _, materials, _, _, view = make(
        tmp_path,
        extra=[{"block_id": "prose", "quote": SENTENCE[:-1]}, {"block_id": "prose", "quote": SENTENCE[3:]}],
    )
    assert not view["numbers"]
    with pytest.raises(ValueError, match="outside"):
        require_passage(view, materials, "prose", SENTENCE, purpose="shared prefix")
