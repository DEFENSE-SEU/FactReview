"""A consumed subrange retains its declared member's original artifact identity."""

import copy
import hashlib
from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas

from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from tests.test_experiment_joint_sources_v2 import ASPECTS
from tests.test_experiment_joint_sources_v2 import offline as offline
from verification.experiment_catalog import build_catalog, resolve_source
from verification.experiment_sources import prepare_joint_candidate, require_passage
from verification.experiments import ExperimentItem, verify_experiments
from verification.theory import _paper_pointer

TABLE = (
    "<table><tr><th>Method</th><th>D test accuracy</th></tr>"
    "<tr><td>A</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
)
CAPTION = "Table 1: D test accuracy."
MEMBER = CAPTION + "\n" + TABLE
PREFIX = "Earlier context excluded from this candidate.\n\n"
PROTOCOL = "Both A and B use the D test split."
ASSERTION = "A has higher accuracy than B on D test."


def fixture(tmp_path, *, markdown_already_contains_member=False):
    pdf = tmp_path / "paper.pdf"
    canvas = Canvas(str(pdf))
    canvas.drawString(60, 760, CAPTION)
    canvas.drawString(60, 730, "Method     D test accuracy")
    canvas.drawString(60, 710, "A              90")
    canvas.drawString(60, 690, "B              80")
    canvas.showPage()
    canvas.drawString(60, 760, PROTOCOL)
    canvas.drawString(60, 730, ASSERTION)
    canvas.save()
    assert len(PdfReader(pdf).pages) == 2
    assert CAPTION in PdfReader(pdf).pages[0].extract_text()
    markdown = TABLE + "\n" + PROTOCOL + "\n" + ASSERTION + "\n"
    actual = (
        (MEMBER if markdown_already_contains_member else TABLE) + "\n" + PROTOCOL + "\n" + ASSERTION + "\n"
    )
    path = tmp_path / "paper.md"
    path.write_text(actual, encoding="utf-8")
    blocks = [MaterialBlock(id="table", text=PREFIX + MEMBER, kind="table", loc=ClaimLocation(page=1))]
    for identifier, text in (("protocol", PROTOCOL), ("claim", ASSERTION)):
        start = actual.index(text)
        blocks.append(
            MaterialBlock(
                id=identifier,
                text=text,
                kind="text",
                loc=ClaimLocation(
                    page=2,
                    char_start=start,
                    char_end=start + len(text),
                ),
            )
        )
    materials = SharedMaterials(
        paper_key="member_identity",
        source_pdf=str(pdf),
        markdown=markdown,
        markdown_path=str(path),
        content_list_path=str(tmp_path / "parser.json"),
        provider="mock",
        blocks=blocks,
    )
    claim = Claim(
        id="claim",
        text=ASSERTION,
        source_block_id="claim",
        source_quote=ASSERTION,
        loc=blocks[-1].loc,
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
        block_id="table",
        quote=MEMBER,
        covered=["c1"],
        fully_supported_conditions=["c1"],
        detail="The exact table and protocol establish the reported comparison.",
        additional_sources=[dict(block_id="protocol", quote=PROTOCOL)],
    )
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
    assert not bounded["joint_view"]["errors"]
    primary = bounded["joint_view"]["members"][0]
    assert primary["pointer"]["locator"] == str(pdf.resolve())
    assert _paper_pointer(materials, "table", TABLE).locator == str(path.resolve())
    cells = {(value["row"], value["column"]): key for key, value in bounded["cells"].items()}
    cell_source = bounded["cells"][cells[1, 1]]["source_id"]
    record = bounded["sources"][cell_source]
    assert blocks[0].text[record["start"] : record["end"]] == TABLE
    members = {member["block_id"]: member["source_id"] for member in bounded["joint_view"]["members"]}
    scope = dict(
        schema_version="catalog-v2",
        conditions=[
            dict(
                condition_id="c1",
                assertion="descriptive",
                matched_controls_required=False,
                uncertainty_sensitive=False,
                relation="gt",
                rationale="Exact reported table comparison.",
                setting_scopes={"split": "shared"},
            )
        ],
        items=[
            dict(
                item_index=0,
                condition_id="c1",
                applicability="applicable",
                full_support=True,
                qualifiers_complete=True,
                comparison_objects="matched",
                rationale="All stated qualifiers are present.",
                grounds_source_ids=list(members.values()),
                source_uses=[
                    dict(
                        source_id=members["table"], roles=["result"], rationale="Exact table values and axes."
                    ),
                    dict(
                        source_id=members["protocol"],
                        roles=["protocol"],
                        rationale="Both methods use the test split.",
                    ),
                ],
                comparisons=[
                    dict(
                        case_id=bounded["conditions"]["c1"]["cases"][0]["id"],
                        relation="gt",
                        left=dict(kind="cell", cell_id=cells[1, 1], label_cell_id=cells[1, 0]),
                        right=dict(kind="cell", cell_id=cells[2, 1], label_cell_id=cells[2, 0]),
                    )
                ],
            )
        ],
    )
    first = dict(checked_aspects=ASPECTS, items=[item.model_dump()], plans=[], issues=[])
    return claim, materials, first, scope, bounded, cell_source


def public_verify(data, mutate=None):
    claim, materials, first, scope, _, _ = data
    snapshots = copy.deepcopy([first, scope])
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2
        if len(calls) == 2 and mutate:
            mutate(materials)
        return copy.deepcopy(first if len(calls) == 1 else scope)

    result = verify_experiments(claim, materials, call=model)
    assert len(calls) == 2
    assert [first, scope] == snapshots
    return result


def test_pdf_member_allows_exact_table_subrange_that_also_occurs_in_markdown(tmp_path):
    _, materials, _, _, bounded, source_id = fixture(tmp_path)
    before = copy.deepcopy(bounded["joint_view"]["members"])
    resolved = resolve_source(bounded, source_id, materials)
    assert resolved["quote"] == TABLE
    assert bounded["joint_view"]["members"] == before
    consumed = bounded["joint_view"]["consumption"][-1]
    assert consumed["start"] == len(PREFIX) + len(CAPTION) + 1
    assert consumed["end"] == len(PREFIX + MEMBER)
    assert consumed["member_source_id"] == before[0]["source_id"]


def test_two_pass_pdf_member_support_preserves_original_pdf_and_markdown_pointers(tmp_path):
    data = fixture(tmp_path)
    before = [data[0].model_dump(), data[1].model_dump()]
    result = public_verify(data)
    assert not result.issues, result.issues
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    evidence = result.evidence[0]
    assert evidence.pointer.locator == str(Path(data[1].source_pdf).resolve())
    assert evidence.pointer.page == 1 and evidence.pointer.quote == MEMBER
    assert evidence.additional_pointers[0].locator == str(Path(data[1].markdown_path).resolve())
    assert evidence.additional_pointers[0].quote == PROTOCOL
    assert [data[0].model_dump(), data[1].model_dump()] == before


def change(materials, kind):
    if kind in {"pdf_bytes", "markdown_bytes"}:
        path = Path(materials.source_pdf if kind == "pdf_bytes" else materials.markdown_path)
        with path.open("ab") as stream:
            stream.write(b"\n% actual byte change\n")
    elif kind in {"pdf_locator", "markdown_locator"}:
        field = "source_pdf" if kind == "pdf_locator" else "markdown_path"
        old = Path(getattr(materials, field))
        replacement = old.with_name("relocated" + old.suffix)
        replacement.write_bytes(old.read_bytes())
        setattr(materials, field, str(replacement))
    elif kind == "page":
        materials.blocks[0].loc.page = 2
    elif kind == "span":
        materials.blocks[0].loc.char_start = 0
        materials.blocks[0].loc.char_end = len(materials.blocks[0].text)
    elif kind == "block_text":
        materials.blocks[0].text += " Changed parser text."
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["pdf_bytes", "markdown_bytes", "pdf_locator", "markdown_locator", "page", "span", "block_text"]
)
def test_actual_artifact_or_member_identity_changes_still_block_two_pass_support(tmp_path, kind):
    result = public_verify(fixture(tmp_path), lambda materials: change(materials, kind))
    assert result.issues
    assert not any(evidence.sufficient for evidence in result.evidence)


@pytest.mark.parametrize("kind", ["pdf_bytes", "pdf_locator", "page", "span", "block_text"])
def test_direct_consumption_revalidates_original_member_identity(tmp_path, kind):
    _, materials, _, _, bounded, _ = fixture(tmp_path)
    change(materials, kind)
    with pytest.raises(ValueError, match=r"changed|another paper"):
        require_passage(bounded, materials, "table", TABLE, purpose="selected_table")


def test_member_pointer_relocation_is_rejected_even_when_both_files_bytes_stay_identical(tmp_path):
    data = fixture(tmp_path, markdown_already_contains_member=True)
    paths = [Path(data[1].source_pdf), Path(data[1].markdown_path)]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]

    def relocate(materials):
        materials.markdown = Path(materials.markdown_path).read_text(encoding="utf-8")
        assert _paper_pointer(materials, "table", MEMBER).locator == str(paths[1].resolve())

    result = public_verify(data, relocate)
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert any("pointer changed" in issue.lower() for issue in result.issues), result.issues
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before


@pytest.mark.parametrize("quote", [PREFIX.strip(), "D test accuracy"])
def test_child_quote_still_requires_unique_in_member_range(tmp_path, quote):
    _, materials, _, _, bounded, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match=r"unique|outside"):
        require_passage(bounded, materials, "table", quote, purpose="context")


def test_pdf_member_fix_does_not_upgrade_first_pass_partial(tmp_path):
    data = fixture(tmp_path)
    data[2]["items"][0]["fully_supported_conditions"] = []
    result = public_verify(data)
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
