"""Omission qualification is separate from source relevance and claim status."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.literature import verify_literature

TEXT = "Our graph model composes relation embeddings for link prediction."
EXTERNAL = "We compose relation embeddings through message passing for link prediction."
PAPER = {
    "id": "2001.00001",
    "arxiv_id": "2001.00001",
    "title": "Relational composition",
    "published": "2020-01-01",
}


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr(
        "verification.literature.llm_json", Mock(side_effect=AssertionError("Unmocked model"))
    )


def case(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text(TEXT, encoding="utf-8")
    loc = ClaimLocation(page=2, char_start=0, char_end=len(TEXT))
    block = MaterialBlock(id="body", text=TEXT, loc=loc)
    materials = SharedMaterials(
        paper_key="omission",
        title="Graph neural networks",
        abstract=TEXT,
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=TEXT,
        markdown_path=str(path),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=[block],
    )
    claim = Claim(
        id="c",
        text=TEXT,
        source_block_id="body",
        source_quote=TEXT,
        loc=loc,
        conditions=[Condition(id="c1", description="relation composition for link prediction")],
        needs=["Literature"],
    )
    target = {
        "source_block_id": "body",
        "source_quote": TEXT,
        "loc": loc.model_dump(mode="json"),
        "covered": ["c1"],
    }
    return materials, claim, target


def assessment(payload, purpose="related_work"):
    target = payload["manuscript_targets"][0]
    return {
        "version": "omission-v1",
        "decision": "important_missing",
        "basis": "evaluation_baseline" if purpose == "baseline" else "method_positioning",
        "target_source_id": target["source_id"],
        "target_quote": target["source_quote"],
        "external_role": "scientific_contribution",
        "reason": "This earlier relation-composition method provides a concrete point of comparison for the described operator.",
    }


async def run(
    tmp_path,
    *,
    mutate=None,
    qualified=True,
    global_review=False,
    purpose="related_work",
    materials=None,
    claim=None,
    target=None,
):
    if materials is None:
        materials, claim, target = case(tmp_path)
    adapter = Mock()
    adapter.lookup_metadata = None
    adapter.search = AsyncMock(
        return_value={"success": True, "provider": "mock", "papers": [PAPER], "complete": False}
    )
    adapter.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {
                    "id": PAPER["id"],
                    "success": True,
                    "paper": PAPER,
                    "evidence": [{"text": EXTERNAL, "page": 3}],
                }
            ],
        }
    )
    requests, responses = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n")[1])
        requests.append(payload)
        row = {
            "paper_id": PAPER["id"],
            "purpose": purpose,
            "relation": "partial",
            "quote": EXTERNAL,
            "covered": [] if global_review else ["c1"],
            "mechanism": "Both compose relation embeddings.",
            "setting": "Both address link prediction.",
            "protocol": "The methods use different evaluation graphs.",
            "note": "Authors must add this citation.",
        }
        if qualified:
            row["omission_assessment"] = assessment(payload, purpose)
        response = {"status": "ok", "comparisons": [row]}
        if mutate:
            mutate(response, materials, payload)
        responses.append(copy.deepcopy(response))
        return response

    result = await verify_literature(
        None if global_review else claim,
        materials,
        submission_deadline="2022-01-01",
        searcher=adapter,
        reader=adapter,
        call=call,
        output_dir=tmp_path / "audit",
        **({"manuscript_targets": [target]} if global_review else {}),
    )
    audit = json.loads(next((tmp_path / "audit").glob("*-search-audit.json")).read_text("utf-8"))
    assert len(requests) == 1
    assert audit["comparison_response"] == responses[0]
    assert not result.questions
    return result, audit, requests


@pytest.mark.asyncio
async def test_legacy_partial_is_audited_without_author_accusation(tmp_path):
    result, audit, _ = await run(tmp_path, qualified=False)
    assert not result.findings
    assert audit["omission_decisions"][0]["decision"] == "unresolved"
    assert audit["comparison_response"]["comparisons"][0]["note"] == "Authors must add this citation."
    assert not any("Authors must" in issue for issue in result.issues)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "global_review,purpose", [(False, "related_work"), (True, "related_work"), (True, "baseline")]
)
async def test_qualified_partial_has_both_exact_sources_and_real_target_location(
    tmp_path, global_review, purpose
):
    result, audit, _ = await run(tmp_path, global_review=global_review, purpose=purpose)
    (finding,) = result.findings
    assert finding.kind == purpose and finding.level == "missing"
    assert finding.loc.page == 2
    assert [e.source for e in finding.evidence] == ["literature", "paper_internal"]
    assert finding.evidence[0].pointer.quote == EXTERNAL
    assert finding.evidence[1].pointer.quote == TEXT
    assert Path(finding.evidence[1].pointer.locator).read_text("utf-8") == TEXT
    assert all(not e.sufficient and not e.concern and not e.affects_claim for e in finding.evidence)
    assert finding.text != "Authors must add this citation."
    assert audit["omission_decisions"][0]["decision"] == "important_missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["candidate_only", "not_applicable", "unresolved"])
async def test_noncritical_decisions_keep_original_note_only_in_audit(tmp_path, decision):
    def mutate(response, *_):
        response["comparisons"][0]["omission_assessment"]["decision"] = decision

    result, audit, _ = await run(tmp_path, mutate=mutate)
    assert not result.findings and not result.questions
    assert audit["omission_decisions"][0]["decision"] == decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "omission-v99"),
        ("decision", True),
        ("basis", "none"),
        ("target_source_id", "foreign"),
        ("target_quote", "Not in the manuscript"),
        ("external_role", "background_or_bibliography"),
        ("reason", ""),
        ("extra", "not allowed"),
    ],
)
async def test_wrong_confirmed_structural_qualification_cannot_create_missing(tmp_path, field, value):
    def mutate(response, *_):
        response["comparisons"][0]["omission_assessment"][field] = value

    result, audit, _ = await run(tmp_path, mutate=mutate)
    assert not result.findings
    assert audit["omission_decisions"][0]["decision"] == "unresolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["block", "file", "location"])
async def test_target_changed_during_model_callback_is_rejected(tmp_path, mutation):
    def mutate(response, materials, _):
        if mutation == "block":
            materials.blocks[0].text += " Changed."
        elif mutation == "location":
            materials.blocks[0].loc.page = 3
        else:
            Path(materials.markdown_path).write_text(TEXT + " changed", encoding="utf-8")

    result, audit, _ = await run(tmp_path, mutate=mutate)
    assert not result.findings
    assert audit["omission_decisions"][0]["decision"] == "unresolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_conflicting_duplicate_omissions_never_take_a_convenient_row(tmp_path, reverse):
    def mutate(response, *_):
        other = copy.deepcopy(response["comparisons"][0])
        other["omission_assessment"]["decision"] = "candidate_only"
        response["comparisons"].append(other)
        if reverse:
            response["comparisons"].reverse()

    result, audit, _ = await run(tmp_path, mutate=mutate)
    assert not result.findings
    assert len(audit["omission_decisions"]) == 2
    assert all(row["decision"] == "unresolved" for row in audit["omission_decisions"])


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["related_work", "baseline"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_padded_purpose_conflict_revokes_qualified_row_without_accepting_padding(
    tmp_path, purpose, reverse
):
    def mutate(response, *_):
        other = copy.deepcopy(response["comparisons"][0])
        other["purpose"] = " " + purpose + "\t"
        other["omission_assessment"]["decision"] = "candidate_only"
        response["comparisons"].append(other)
        if reverse:
            response["comparisons"].reverse()

    result, audit, _ = await run(tmp_path, purpose=purpose, mutate=mutate)
    assert not result.findings
    assert any(row["purpose"] == " " + purpose + "\t" for row in audit["comparison_response"]["comparisons"])
    assert len(audit["omission_decisions"]) == 1
    assert audit["omission_decisions"][0]["decision"] == "unresolved"
    assert "Duplicate" in audit["omission_decisions"][0]["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["dimensions", "different", "purpose", "bibliography"])
async def test_explicit_qualification_preserves_existing_comparison_guards(tmp_path, mutation):
    materials, claim, target = case(tmp_path)
    if mutation == "bibliography":
        materials.bibliography = [
            MaterialBlock(id="ref", text="[1] Earlier method. arXiv:2001.00001.", loc=ClaimLocation(page=4))
        ]

    def mutate(response, *_):
        row = response["comparisons"][0]
        if mutation == "dimensions":
            row["protocol"] = ""
        elif mutation == "different":
            row["relation"] = "different"
        elif mutation == "purpose":
            row["omission_assessment"]["basis"] = "evaluation_baseline"

    result, _, _ = await run(tmp_path, materials=materials, claim=claim, target=target, mutate=mutate)
    assert not result.findings


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["missing", "malformed", "duplicate"])
async def test_bad_omission_cannot_erase_healthy_cited_support(tmp_path, defect):
    materials, claim, target = case(tmp_path)
    text = TEXT + " [1]"
    materials.markdown = materials.blocks[0].text = claim.text = claim.source_quote = text
    loc = ClaimLocation(page=2, char_start=0, char_end=len(text))
    materials.blocks[0].loc = claim.loc = loc
    Path(materials.markdown_path).write_text(text, encoding="utf-8")
    materials.bibliography = [
        MaterialBlock(id="ref", text="[1] Prior work. arXiv:2001.00001.", loc=ClaimLocation(page=4))
    ]

    def mutate(response, *_):
        row = response["comparisons"][0]
        healthy = {
            **row,
            "purpose": "citation_support",
            "relation": "supports",
            "fully_supported_conditions": ["c1"],
        }
        healthy.pop("omission_assessment", None)
        if defect == "missing":
            row.pop("omission_assessment")
        elif defect == "malformed":
            row["omission_assessment"]["decision"] = True
        else:
            response["comparisons"].append(copy.deepcopy(row))
        response["comparisons"].append(healthy)

    result, _, _ = await run(tmp_path, materials=materials, claim=claim, target=target, mutate=mutate)
    assert not result.findings
    assert any(e.sufficient and e.direction == "support" and e.covered == ["c1"] for e in result.evidence)


@pytest.mark.asyncio
async def test_global_metadata_without_source_never_borrows_first_block_location(tmp_path):
    materials, _, _ = case(tmp_path)
    adapter = Mock(
        search=AsyncMock(return_value={"success": True, "papers": [PAPER]}),
        lookup_metadata=None,
        read_papers=AsyncMock(
            return_value={
                "success": True,
                "items": [
                    {
                        "id": PAPER["id"],
                        "success": True,
                        "paper": PAPER,
                        "evidence": [{"text": EXTERNAL, "page": 3}],
                    }
                ],
            }
        ),
    )

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n")[1])
        assert payload["manuscript_targets"] == []
        return {
            "status": "ok",
            "comparisons": [
                {
                    "paper_id": PAPER["id"],
                    "purpose": "related_work",
                    "relation": "partial",
                    "quote": EXTERNAL,
                    "covered": [],
                    "mechanism": "same",
                    "setting": "same",
                    "protocol": "same",
                    "omission_assessment": {
                        "version": "omission-v1",
                        "decision": "important_missing",
                        "basis": "method_positioning",
                        "target_source_id": "manuscript:invented",
                        "target_quote": TEXT,
                        "external_role": "scientific_contribution",
                        "reason": "Discuss it.",
                    },
                }
            ],
        }

    result = await verify_literature(
        None, materials, submission_deadline="2022-01-01", searcher=adapter, reader=adapter, call=call
    )
    assert not result.findings and not result.questions


def test_target_domain_preserves_ranges_origins_and_excludes_bibliography(tmp_path):
    from verification.literature_omissions import OmissionContext

    materials, _, target = case(tmp_path)
    separate = {
        **target,
        "source_quote": TEXT[:15],
        "loc": {**target["loc"], "char_end": 15},
        "covered": ["c2"],
    }
    context = OmissionContext(
        materials, [target, {**target, "claim_id": "other", "covered": ["c3"]}, separate]
    )
    assert len(context.payload()) == 2
    assert len(context.payload()[0]["origins"]) == 2
    with pytest.raises(ValueError, match="condition scope"):
        context.resolve(context.payload()[1]["source_id"], separate["source_quote"], ["c1"])
    with pytest.raises(ValueError, match="declared range"):
        context.resolve(context.payload()[1]["source_id"], TEXT, [])
    materials.bibliography = [materials.blocks[0]]
    excluded = OmissionContext(materials, [target], global_review=True)
    assert excluded.payload() == [] and excluded.unavailable


@pytest.mark.parametrize("change", ["unknown", "location", "range"])
def test_original_target_input_is_validated_before_being_shown(tmp_path, change):
    from verification.literature_omissions import OmissionContext

    materials, _, target = case(tmp_path)
    if change == "unknown":
        target["source_block_id"] = "other"
    elif change == "location":
        target["loc"]["page"] = 10
    else:
        target["source_quote"] = "Invented method"
    context = OmissionContext(materials, [target], global_review=True)
    assert not context.payload() and context.unavailable
