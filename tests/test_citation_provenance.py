"""Citation context survives paraphrasing and page-only parser locations."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from screening.claims import extract_claims
from verification.literature import _cited_papers, verify_literature


@pytest.fixture
def materials(tmp_path):
    quote = "According to Smith et al. (2020), the bound holds."
    return SharedMaterials(
        paper_key="citation-context",
        title="Graph neural networks for link prediction",
        abstract="Our study establishes a generalization bound.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown="Parser Markdown differs from the page-located content list.",
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=quote, loc=ClaimLocation(page=2))],
        bibliography=[
            MaterialBlock(
                id="r1",
                text="[1] Smith et al. An error bound. 2020. arXiv:2001.00001",
                loc=ClaimLocation(page=5),
            ),
            MaterialBlock(
                id="r2",
                text="[2] Other authors. Unrelated evidence. 2020. arXiv:2001.00002",
                loc=ClaimLocation(page=5),
            ),
        ],
    )


@pytest.fixture(autouse=True)
def config(monkeypatch):
    def cfg():
        return LLMConfig("mock", "fixture", None, None)

    monkeypatch.setattr("screening.claims.resolve_llm_config", cfg)
    monkeypatch.setattr("verification.literature.resolve_llm_config", cfg)
    monkeypatch.setattr("screening.claims.llm_json", Mock(side_effect=AssertionError("Unmocked LLM")))
    monkeypatch.setattr("verification.literature.llm_json", Mock(side_effect=AssertionError("Unmocked LLM")))


def extract(materials):
    return extract_claims(
        materials,
        call=Mock(
            return_value={
                "status": "ok",
                "claims": [
                    {
                        "text": "The bound holds.",
                        "source_block_id": "b1",
                        "source_quote": materials.blocks[0].text,
                        "conditions": [{"id": "bound", "description": "Generalization bound"}],
                        "needs": ["Literature"],
                        "importance": "core",
                    }
                ],
            }
        ),
    )[0]


def boundaries(papers=()):
    paper = {
        "id": "2001.00001",
        "arxiv_id": "2001.00001",
        "title": "An error bound for independent samples",
        "abstract": "A theoretical bound is established.",
        "published": "2020-01-01",
    }
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "mock", "complete": True, "papers": list(papers)}
    )
    searcher.lookup_metadata = AsyncMock(return_value={"success": True, "paper": paper})
    reader = Mock()
    reader.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {"id": paper["id"], "success": True, "evidence": [{"text": "The bound holds.", "page": 2}]}
            ],
        }
    )
    return searcher, reader


def test_extraction_preserves_original_quote_and_page_only_citation(materials):
    claim = Claim.model_validate_json(extract(materials).model_dump_json())
    assert claim.loc == ClaimLocation(page=2)
    assert claim.source_block_id == "b1" and claim.source_quote == materials.blocks[0].text
    assert _cited_papers(claim, materials)[0]["arxiv_id"] == "2001.00001"


def test_new_provenance_never_uses_citations_in_generated_claim_or_neighbor_block(materials):
    claim = extract(materials)
    claim.text += " [2]"
    materials.blocks.append(MaterialBlock(id="b2", text="Another result [2].", loc=ClaimLocation(page=2)))
    assert [p["arxiv_id"] for p in _cited_papers(claim, materials)] == ["2001.00001"]


@pytest.mark.asyncio
async def test_page_only_source_reaches_citation_lookup_and_comparison(materials):
    claim = extract(materials)
    searcher, reader = boundaries()
    call = Mock(
        return_value={
            "status": "ok",
            "comparisons": [
                {
                    "paper_id": "2001.00001",
                    "purpose": "citation_support",
                    "relation": "supports",
                    "quote": "The bound holds.",
                    "covered": ["bound"],
                    "fully_supported_conditions": ["bound"],
                    "mechanism": "Same bound",
                    "setting": "Same assumptions",
                    "protocol": "Same derivation",
                    "note": "The quoted result establishes the stated bound.",
                }
            ],
        }
    )
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-01", searcher=searcher, reader=reader, call=call
    )
    searcher.lookup_metadata.assert_awaited_once_with(identifier="2001.00001")
    assert result.evidence[0].sufficient and result.evidence[0].direction == "support"
    payload = json.loads(call.call_args.kwargs["prompt"].split("\nDATA_JSON:\n")[1])
    assert payload["source_excerpt"] == materials.blocks[0].text
    audit = json.loads(next(materials_path(materials).rglob("*-search-audit.json")).read_text())
    assert audit["claim_source_excerpt"] == materials.blocks[0].text


def materials_path(materials):
    return Path(materials.markdown_path).parent


@pytest.mark.parametrize(
    "changes",
    [
        {"source_block_id": "b1"},
        {"source_quote": "Original."},
        {"source_block_id": "b1", "source_quote": " "},
    ],
)
def test_provenance_fields_are_a_nonempty_pair(materials, changes):
    record = extract(materials).model_dump(exclude={"source_block_id", "source_quote"})
    with pytest.raises(ValidationError):
        Claim.model_validate({**record, **changes})


@pytest.mark.parametrize(
    "damage", ["unknown_block", "changed_quote", "wrong_location", "duplicate_quote", "duplicate_block"]
)
@pytest.mark.asyncio
async def test_unbound_provenance_is_visible_and_cannot_resolve_citations(materials, damage):
    claim = extract(materials)
    if damage == "unknown_block":
        claim.source_block_id = "missing"
    elif damage == "changed_quote":
        claim.source_quote = "Different text [2]."
    elif damage == "wrong_location":
        claim.loc = ClaimLocation(page=3)
    elif damage == "duplicate_quote":
        materials.blocks[0].text *= 2
    else:
        materials.blocks.append(materials.blocks[0].model_copy())
    searcher, reader = boundaries()
    call = Mock(return_value={"status": "ok", "comparisons": []})
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-01", searcher=searcher, reader=reader, call=call
    )
    assert not result.evidence
    assert any("Citation source unavailable" in issue for issue in result.issues)
    searcher.lookup_metadata.assert_not_awaited()
    reader.read_papers.assert_not_awaited()
    call.assert_not_called()


@pytest.mark.parametrize("ambiguous", [False, "same_block", "another_block"])
def test_historical_page_only_fallback_requires_unique_exact_text(materials, ambiguous):
    claim = extract(materials)
    claim.source_block_id = claim.source_quote = None
    claim.text = materials.blocks[0].text
    if ambiguous == "same_block":
        materials.blocks[0].text *= 2
    elif ambiguous == "another_block":
        materials.blocks.append(materials.blocks[0].model_copy(update={"id": "b2"}))
    assert bool(_cited_papers(claim, materials)) is (not ambiguous)


def test_historical_page_only_paraphrase_cannot_guess_page_citations(materials):
    claim = extract(materials)
    claim.source_block_id = claim.source_quote = None
    assert _cited_papers(claim, materials) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("abstract", ["", "Too short to establish identity."])
async def test_missing_submission_identity_cannot_make_own_version_decisive_prior(materials, abstract):
    claim = extract(materials)
    claim.text = "Our novel graph neural network improves link prediction."
    claim.conditions = [Condition(id="novel", description="Novel mechanism")]
    materials.title = ""
    materials.abstract = abstract
    own_version = {
        "id": "2001.00002",
        "arxiv_id": "2001.00002",
        "title": "The submission itself",
        "published": "2020-01-01",
    }
    searcher, reader = boundaries([own_version])
    call = Mock()
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-01", searcher=searcher, reader=reader, call=call
    )
    assert not result.evidence
    assert any("self-version exclusion" in issue for issue in result.issues)
    reader.read_papers.assert_not_awaited()
    call.assert_not_called()
    audit = json.loads(next(materials_path(materials).rglob("*-search-audit.json")).read_text())
    assert audit["self_exclusion"] == {"available": False}
    assert {row["reason"] for row in audit["excluded"]} == {"self_exclusion_unavailable"}
    assert not audit["adequate_for_no_close_prior_work"]


@pytest.mark.asyncio
async def test_abstract_only_identity_preserves_self_exclusion(materials):
    materials.title = ""
    materials.abstract = "Graph neural networks for link prediction establish a new bound under independent sampling assumptions."
    paper = {
        "id": "2001.00002",
        "arxiv_id": "2001.00002",
        "title": "Renamed submission",
        "abstract": materials.abstract,
        "published": "2020-01-01",
    }
    searcher, reader = boundaries([paper])
    await verify_literature(
        None, materials, submission_deadline="2021-01-01", searcher=searcher, reader=reader, call=Mock()
    )
    reader.read_papers.assert_not_awaited()
    audit = json.loads(next(materials_path(materials).rglob("*-search-audit.json")).read_text())
    assert audit["self_exclusion"] == {"available": True}
    assert {row["reason"] for row in audit["excluded"]} == {"submission_version"}
