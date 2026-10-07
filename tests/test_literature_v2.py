"""Literature policy tests mock every search, read, and model boundary."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from util.cutoff_date import concurrent_window_start, parse_submission_deadline, publication_relation
from verification.literature import literature_queries, verify_literature


@pytest.fixture
def materials(tmp_path):
    text = "We propose a novel graph neural network for link prediction."
    path = tmp_path / "paper.md"
    path.write_text(text)
    return SharedMaterials(
        paper_key="tiny",
        title="Novel graph neural networks for link prediction",
        abstract="Graph neural networks improve link prediction through relation embedding.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(path),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=[
            MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1, char_start=0, char_end=len(text)))
        ],
    )


@pytest.fixture
def claim(materials):
    return Claim(
        id="c1",
        text=materials.markdown,
        loc=materials.blocks[0].loc,
        conditions=[Condition(id="cond1", description="novel mechanism for link prediction")],
        needs=["Literature"],
    )


@pytest.fixture
def paper():
    return {
        "id": "2001.00001",
        "arxiv_id": "2001.00001",
        "title": "Compositional relational reasoning",
        "url": "https://arxiv.org/abs/2001.00001",
        "published": "2020-01-01",
        "abstract": "This study describes a message passing mechanism for relational data.",
    }


def boundaries(papers, *, complete=True):
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={
            "success": True,
            "provider": "fixture",
            "complete": complete,
            "papers": papers,
            "count": len(papers),
        }
    )
    reader = Mock()
    by_id = {row["id"]: row for row in papers}

    async def read(*, items):
        identifier = items[0]["id"]
        row = by_id[identifier]
        return {
            "success": True,
            "items": [
                {
                    "id": identifier,
                    "success": True,
                    "paper": row,
                    "evidence": [
                        {
                            "page": 2,
                            "text": "We compose relation embeddings through message passing for link prediction.",
                        }
                    ],
                }
            ],
        }

    reader.read_papers = AsyncMock(side_effect=read)
    return searcher, reader


def comparison(paper, *, relation="different", purpose="novelty", covered=None, quote=None):
    return {
        "paper_id": paper["id"],
        "purpose": purpose,
        "relation": relation,
        "covered": ["cond1"] if covered is None else covered,
        "quote": quote or "compose relation embeddings through message passing",
        "mechanism": "Compare the composition operators in the source and submission.",
        "setting": "Compare relation prediction tasks.",
        "protocol": "Compare link-prediction evaluation protocols.",
        "note": "The source describes the relevant relation-composition mechanism.",
    }


def model(*rows):
    return Mock(return_value={"status": "ok", "comparisons": list(rows)})


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [
        ("2021-05-31", date(2021, 2, 28)),
        ("2020-05-31", date(2020, 2, 29)),
        ("2021-01-31", date(2020, 10, 31)),
    ],
)
def test_three_calendar_month_window(deadline, expected):
    assert concurrent_window_start(parse_submission_deadline(deadline)) == expected


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"published": "2020-10-30"}, "prior"),
        ({"published": "2020-10-31"}, "concurrent"),
        ({"published": "2021-01-31"}, "concurrent"),
        ({"published": "2021-02-01"}, "post_cutoff"),
        ({"year": 2020}, "unknown"),
        ({"year": 2019}, "prior"),
        ({"published": "2020-11"}, "concurrent"),
        ({"published": "2020-10"}, "unknown"),
        ({"updated": "2019-01-01"}, "unknown"),
    ],
)
def test_publication_period_uses_conservative_intervals(metadata, expected):
    assert publication_relation(metadata, parse_submission_deadline("2021-01-31")) == expected


@pytest.mark.parametrize("value", [None, "", "2021", "2021-01", "2101.12345", "2021-02-30"])
def test_submission_deadline_requires_explicit_real_day(value):
    with pytest.raises(ValueError):
        parse_submission_deadline(value)


def test_closed_query_vocabulary_excludes_author_names_and_search_operators(claim, materials):
    claim.text = "John Smith author:Jane Doe proposes novel graph neural network link prediction."
    queries = literature_queries(claim, materials)
    assert len(set(queries)) == 3
    assert all(len(query.split()) <= 10 for query in queries)
    assert all(
        not any(token in query.lower() for token in ("john", "smith", "jane", "doe", "author:"))
        for query in queries
    )


@pytest.mark.asyncio
async def test_same_mechanism_prior_work_is_grounded_flaw_evidence(claim, materials, paper):
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same")),
    )
    evidence = result.evidence[0]
    assert evidence.direction == "flaw" and evidence.sufficient and not evidence.overturnable
    assert evidence.pointer.locator == "arxiv:2001.00001" and evidence.pointer.page == 2
    assert evidence.covered == ["cond1"]
    assert result.plans == []


@pytest.mark.asyncio
async def test_partial_overlap_is_a_resolvable_concern(claim, materials, paper):
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="partial")),
    )
    assert result.evidence[0].concern and result.evidence[0].overturnable
    assert not result.evidence[0].sufficient


@pytest.mark.asyncio
async def test_no_close_prior_work_requires_saved_complete_search_scope(claim, materials, paper):
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper)),
    )
    evidence = result.evidence[0]
    assert evidence.direction == "support" and evidence.sufficient
    assert evidence.pointer.key == "search_scope"
    path = Path(evidence.pointer.locator)
    assert path.is_file()
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["search_scope"] == evidence.pointer.quote
    assert len(audit["queries"]) == 3
    assert audit["reads"][0]["response"]["items"][0]["evidence"][0]["text"]


@pytest.mark.asyncio
async def test_incomplete_retrieval_cannot_support_absence(claim, materials, paper):
    searcher, reader = boundaries([paper], complete=False)
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper)),
    )
    assert not result.evidence
    assert any("insufficient" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_concurrent_work_is_reported_without_novelty_criticism(claim, materials, paper):
    paper["published"] = "2020-11-01"
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same")),
    )
    assert not any(item.direction == "flaw" or item.concern for item in result.evidence)
    assert result.findings[0].level == "concurrent"
    assert not result.findings[0].evidence[0].affects_claim


@pytest.mark.asyncio
async def test_post_cutoff_and_review_pages_are_excluded_before_read(claim, materials, paper):
    paper["published"] = "2021-02-01"
    review = {
        **paper,
        "id": "review",
        "published": "2019-01-01",
        "url": "https://openreview.net/forum?id=submission",
    }
    searcher, reader = boundaries([paper, review])
    call = model()
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=call
    )
    reader.read_papers.assert_not_awaited()
    call.assert_not_called()
    assert not any(item.direction == "flaw" for item in result.evidence)
    audit = json.loads(next(Path(materials.markdown_path).parent.rglob("*-search-audit.json")).read_text())
    assert {row["reason"] for row in audit["excluded"]} == {"post_cutoff", "review_page"}


@pytest.mark.asyncio
async def test_self_near_duplicate_title_is_excluded_before_read(claim, materials, paper):
    paper["title"] = "Novel graph neural networks for link prediction: revised version"
    searcher, reader = boundaries([paper])
    await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    reader.read_papers.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_date_cannot_create_flaw_or_absence_support(claim, materials, paper):
    paper.pop("published")
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same")),
    )
    assert not result.evidence
    assert any("Publication date" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_fabricated_passage_is_rejected(claim, materials, paper):
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same", quote="Invented external fact.")),
    )
    assert not result.evidence
    assert any("ungrounded quote" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_attached_citation_support_and_global_uncited_findings(claim, materials, paper):
    claim.text += " [1]"
    materials.bibliography = [
        MaterialBlock(
            id="ref1", text="[1] Authors. Prior result. arXiv:2001.00001.", loc=ClaimLocation(page=5)
        )
    ]
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="citation_support", relation="supports")),
    )
    assert any(item.direction == "support" and item.sufficient for item in result.evidence)
    assert not result.questions
    global_result = await verify_literature(
        None,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="related_work", relation="partial", covered=[])),
    )
    assert not global_result.findings  # The paper is already in the bibliography.
    materials.bibliography = []
    global_result = await verify_literature(
        None,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="baseline", relation="partial", covered=[])),
    )
    assert global_result.findings[0].kind == "baseline"
    assert global_result.findings[0].evidence[0].pointer.quote


@pytest.mark.asyncio
async def test_missing_deadline_stops_all_external_calls(claim, materials):
    searcher, reader = boundaries([])
    result = await verify_literature(
        claim, materials, submission_deadline="", searcher=searcher, reader=reader, call=model()
    )
    assert result.issues and not result.evidence
    searcher.search.assert_not_awaited()
    reader.read_papers.assert_not_awaited()


@pytest.mark.asyncio
async def test_author_year_citation_matches_locally_without_author_search(claim, materials, paper):
    claim.text += " (Smith et al., 2020)."
    materials.bibliography = [
        MaterialBlock(
            id="ref",
            text="Smith et al. 2020. Compositional relational reasoning. arXiv:2001.00001.",
            loc=ClaimLocation(page=5),
        )
    ]
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="citation_support", relation="supports")),
    )
    assert any(evidence.sufficient and evidence.direction == "support" for evidence in result.evidence)
    assert all("smith" not in request.kwargs["query"].lower() for request in searcher.search.await_args_list)


@pytest.mark.asyncio
async def test_identifier_free_citation_resolves_from_domain_search_title(claim, materials, paper):
    claim.text += " [1]"
    materials.bibliography = [
        MaterialBlock(
            id="ref", text="[1] Smith. Compositional relational reasoning. 2020.", loc=ClaimLocation(page=5)
        )
    ]
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="citation_support", relation="supports")),
    )
    assert any(evidence.sufficient and evidence.direction == "support" for evidence in result.evidence)
    assert not any("lacks a resolvable" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_abstract_only_overlap_cannot_be_decisive_flaw(claim, materials, paper):
    searcher, reader = boundaries([paper])
    reader.read_papers.side_effect = RuntimeError("reader unavailable")
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same", quote=paper["abstract"])),
    )
    assert result.evidence[0].concern and result.evidence[0].overturnable
    assert not result.evidence[0].sufficient
    assert any("Full-text reading unavailable" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_unknown_domain_is_insufficient_without_guessing_author_queries(claim, materials):
    claim.text = "We propose a new quux method by Ada Lovelace."
    materials.title, materials.abstract = "The quux framework", "A quux result."
    searcher, reader = boundaries([])
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    assert not result.evidence
    assert any("recognized technical domain terms" in issue for issue in result.issues)
    searcher.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_abstract_duplicate_is_excluded_even_with_different_title(claim, materials, paper):
    materials.abstract = "This manuscript describes a graph neural network mechanism for relation embedding and link prediction in knowledge graphs."
    paper["abstract"] = materials.abstract
    searcher, reader = boundaries([paper])
    await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    reader.read_papers.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_results_support_only_explicitly_complete_successful_scopes(claim, materials):
    searcher, reader = boundaries([])
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    assert result.evidence[0].sufficient and result.evidence[0].pointer.key == "search_scope"
    searcher.search.return_value["question_results"] = [{"success": False, "count": 0}]
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    assert not result.evidence


@pytest.mark.asyncio
async def test_literature_model_uses_shared_keyword_call_contract(claim, materials, paper):
    searcher, reader = boundaries([paper])

    def call(*, prompt, system, cfg, module):
        assert "DATA_JSON" in prompt
        assert "retrieved literature" in system
        assert cfg.provider
        assert module == "verification_literature"
        return {"status": "ok", "comparisons": [comparison(paper)]}

    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=call
    )
    assert result.evidence[0].sufficient


@pytest.mark.parametrize("malformed", [None, "bad rows", {}, ["bad row"], [None]])
@pytest.mark.asyncio
async def test_malformed_search_papers_never_support_absence(claim, materials, malformed):
    searcher, reader = boundaries([])
    searcher.search.return_value["papers"] = malformed
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    assert not result.evidence
    assert any("incomplete or malformed" in issue for issue in result.issues)
    reader.read_papers.assert_not_awaited()


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "ok", "comparisons": [], "error": "remote failure"},
        {"status": "ok", "comparisons": ""},
        {"status": "ok", "comparisons": {}},
        {"status": "ok", "comparisons": None},
        {"status": "ok", "comparisons": [None]},
    ],
)
@pytest.mark.asyncio
async def test_malformed_comparisons_cannot_support_even_when_only_concurrent_work_exists(
    claim, materials, paper, bad
):
    paper["published"] = "2020-11-01"
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=Mock(return_value=bad),
    )
    assert not result.evidence
    assert result.findings[0].level == "concurrent"
    assert any("no valid comparison" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("relation", "direction", "concern"),
    [
        ("supports", "support", False),
        ("contradicts", "flaw", True),
    ],
)
@pytest.mark.asyncio
async def test_concurrent_cited_work_can_still_check_citation_support(
    claim, materials, paper, relation, direction, concern
):
    claim.text += " [1]"
    paper["published"] = "2020-11-01"
    materials.bibliography = [
        MaterialBlock(id="ref", text="[1] Smith. Prior work. arXiv:2001.00001.", loc=ClaimLocation(page=5))
    ]
    searcher, reader = boundaries([paper])
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, purpose="citation_support", relation=relation)),
    )
    assert result.evidence[0].direction == direction
    assert result.evidence[0].concern is concern
    assert result.findings[0].level == "concurrent"


@pytest.mark.asyncio
async def test_unresolved_citation_identity_is_not_read_before_self_filter(claim, materials):
    claim.text += " [1]"
    materials.bibliography = [
        MaterialBlock(id="ref", text="[1] Unknown version. arXiv:2001.00001.", loc=ClaimLocation(page=5))
    ]
    searcher, reader = boundaries([])
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    reader.read_papers.assert_not_awaited()
    assert not result.evidence
    assert any("Cannot exclude submission versions" in issue for issue in result.issues)


@pytest.mark.parametrize("remaining", ["empty", "post_cutoff", "review"])
@pytest.mark.asyncio
async def test_empty_selected_corpus_needs_explicit_provider_completeness(claim, materials, paper, remaining):
    if remaining == "post_cutoff":
        paper["published"] = "2021-02-01"
    if remaining == "review":
        paper["url"] = "https://openreview.net/forum?id=submission"
    searcher, reader = boundaries([] if remaining == "empty" else [paper])
    searcher.search.return_value.pop("complete")
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-31", searcher=searcher, reader=reader, call=model()
    )
    assert not result.evidence
    assert any("provider must declare complete=true" in issue for issue in result.issues)
    reader.read_papers.assert_not_awaited()


@pytest.mark.parametrize("bad_evidence", [[{}], [{"text": ""}], [{"text": 123}], "bad", {"text": "fake"}])
async def test_invalid_read_evidence_cannot_upgrade_abstract_to_full_text(
    claim, materials, paper, bad_evidence
):
    searcher, reader = boundaries([paper])
    reader.read_papers.side_effect = None
    reader.read_papers.return_value = {
        "success": True,
        "items": [
            {"id": paper["id"], "success": True, "paper": paper, "evidence": bad_evidence},
        ],
    }
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same", quote=paper["abstract"])),
    )
    assert result.evidence[0].concern and not result.evidence[0].sufficient
    assert result.evidence[0].overturnable
    assert any("Full-text reading unavailable" in issue for issue in result.issues)


@pytest.mark.parametrize("failure", ["container", "container_error", "item_error"])
async def test_failed_reader_container_or_item_never_supplies_decisive_evidence(
    claim, materials, paper, failure
):
    searcher, reader = boundaries([paper])
    reader.read_papers.side_effect = None
    item = {
        "id": paper["id"],
        "success": True,
        "paper": paper,
        "evidence": [{"text": paper["abstract"], "page": 2}],
    }
    response = {"success": failure != "container", "items": [item]}
    if failure == "container_error":
        response["error"] = "failed transport"
    if failure == "item_error":
        item["error"] = "failed reader"
    reader.read_papers.return_value = response
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model(comparison(paper, relation="same", quote=paper["abstract"])),
    )
    assert not any(evidence.sufficient for evidence in result.evidence)
    assert any("Full-text reading unavailable" in issue for issue in result.issues)
