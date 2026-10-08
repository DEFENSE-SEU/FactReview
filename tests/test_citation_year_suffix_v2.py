"""Keep same-author same-year bibliography labels distinct, including real BERT shapes."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.literature import _cited_papers, verify_literature

BIBLIOGRAPHY = [
    "Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. Deep contextualized word representations. In NAACL.",
    "Matthew Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018b. Dissecting contextual word embeddings: Architecture and representation. In EMNLP.",
    "Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.",
]


def inputs(tmp_path, citation):
    text = f"Prior language models ({citation}) differ from this model."
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    materials = SharedMaterials(
        paper_key="same-year",
        title="Language model attention",
        abstract="A representation study.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
        bibliography=[
            MaterialBlock(id=f"ref{i}", text=row, loc=ClaimLocation(page=10))
            for i, row in enumerate(BIBLIOGRAPHY)
        ],
    )
    claim = Claim(
        id="c1",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[Condition(id="comparison", description="Comparison with prior models")],
        needs=["Literature"],
    )
    return materials, claim


@pytest.mark.parametrize(
    "citation,expected",
    [
        ("Peters et al., 2018a; Radford et al., 2018", [0, 2]),
        ("Peters et al. (2018a)", [0]),
        ("Peters et al. (2018b)", [1]),
        ("Peters et al., 2018a; Peters et al., 2018b", [0, 1]),
        ("Peters et al., 2018b; Peters et al., 2018b", [1]),
    ],
)
def test_suffix_selects_only_the_corresponding_bibliography_entry(tmp_path, citation, expected):
    materials, claim = inputs(tmp_path, citation)
    issues = []
    papers = _cited_papers(claim, materials, issues=issues)
    assert [paper["bibliography_text"] for paper in papers] == [BIBLIOGRAPHY[index] for index in expected]
    assert not issues


def test_missing_suffix_with_multiple_entries_is_explicitly_ambiguous(tmp_path):
    materials, claim = inputs(tmp_path, "Peters et al., 2018")
    issues = []
    assert _cited_papers(claim, materials, issues=issues) == []
    assert len(issues) == 1 and "Ambiguous citation Peters 2018" in issues[0]


def test_single_same_year_candidate_preserves_unsuffixed_compatibility(tmp_path):
    materials, claim = inputs(tmp_path, "Peters et al., 2018")
    materials.bibliography = materials.bibliography[:1]
    issues = []
    assert _cited_papers(claim, materials, issues=issues)[0]["bibliography_text"] == BIBLIOGRAPHY[0]
    assert not issues


def test_duplicate_explicit_label_is_ambiguous_instead_of_guessing(tmp_path):
    materials, claim = inputs(tmp_path, "Peters et al., 2018a")
    materials.bibliography.append(
        MaterialBlock(
            id="duplicate", text="Peters et al. 2018a. A different work.", loc=ClaimLocation(page=11)
        )
    )
    issues = []
    assert not _cited_papers(claim, materials, issues=issues)
    assert any("Ambiguous citation Peters 2018a" in issue for issue in issues)


def test_numeric_label_and_doi_resolution_remain_available_despite_ambiguous_author_year(tmp_path):
    materials, claim = inputs(tmp_path, "Peters et al., 2018; [2]")
    materials.bibliography[0].text = "[1] " + BIBLIOGRAPHY[0] + " doi:10.1234/context-a."
    materials.bibliography[1].text = "[2] " + BIBLIOGRAPHY[1] + " doi:10.1234/context-b."
    issues = []
    papers = _cited_papers(claim, materials, issues=issues)
    assert len(papers) == 1 and papers[0]["doi"] == "10.1234/context-b"
    assert any("Ambiguous citation" in issue for issue in issues)


@pytest.mark.asyncio
async def test_ambiguous_citation_is_saved_and_cannot_certify_empty_search(tmp_path, monkeypatch):
    materials, claim = inputs(tmp_path, "Peters et al., 2018")
    claim.text = "Our novel language model is the first method of its kind."
    claim.conditions = [Condition(id="novelty", description="Novel language model")]
    adapter = Mock()
    adapter.search = AsyncMock(
        return_value={"success": True, "provider": "fixture", "complete": True, "papers": []}
    )
    adapter.read_papers = AsyncMock()
    adapter.lookup_metadata = AsyncMock()
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    call = Mock(side_effect=AssertionError("No source should reach model comparison"))
    result = await verify_literature(
        claim, materials, submission_deadline="2020-01-01", searcher=adapter, reader=adapter, call=call
    )
    assert not result.evidence
    assert any("Ambiguous citation Peters 2018" in issue for issue in result.issues)
    adapter.read_papers.assert_not_awaited()
    adapter.lookup_metadata.assert_not_awaited()
    call.assert_not_called()
    audit = json.loads(next(Path(tmp_path).rglob("c1-search-audit.json")).read_text(encoding="utf-8"))
    assert audit["citation_issues"] and not audit["adequate_for_no_close_prior_work"]
    assert all("peters" not in query.kwargs["query"].lower() for query in adapter.search.await_args_list)
