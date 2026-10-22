"""Multiple exact claim sources keep citation support inside its original conditions."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.literature import _cited_papers, _claim_source_excerpt, verify_literature


@pytest.fixture(autouse=True)
def external_boundaries(monkeypatch):
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr("verification.literature.llm_json", Mock(side_effect=AssertionError("Unmocked LLM")))
    monkeypatch.setattr(
        "verification.literature._default_adapter", Mock(side_effect=AssertionError("Unmocked retrieval"))
    )


@pytest.fixture
def materials(tmp_path):
    texts = [
        "Our model has two properties.",
        "First property follows Smith et al. (2020).",
        "  Second property follows Jones et al. (2020).\n",
        "Unrelated neighboring result [3].",
    ]
    markdown = "\n\n".join(texts)
    blocks = []
    cursor = 0
    for i, text in enumerate(texts):
        blocks.append(
            MaterialBlock(
                id=f"b{i}",
                text=text,
                loc=ClaimLocation(page=i + 1, char_start=cursor, char_end=cursor + len(text)),
            )
        )
        cursor += len(text) + 2
    return SharedMaterials(
        paper_key="source-scopes",
        title="Transformer image classification",
        abstract="A target study of two architectural properties.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=markdown,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=blocks,
        bibliography=[
            MaterialBlock(id="r1", text="[1] Smith. 2020. First architecture. arXiv:2001.00001"),
            MaterialBlock(id="r2", text="[2] Jones. 2020. Second architecture. arXiv:2001.00002"),
            MaterialBlock(id="r3", text="[3] Other. 2020. Unrelated model. arXiv:2001.00003"),
        ],
    )


@pytest.fixture
def claim(materials):
    first = materials.blocks[0]
    return Claim(
        id="claim-multiple",
        text=first.text,
        loc=first.loc,
        source_block_id=first.id,
        source_quote=first.text,
        source_refs=[
            ClaimSourceRef(source_block_id=block.id, source_quote=block.text, loc=block.loc, covered=[cid])
            for block, cid in zip(materials.blocks[1:3], ["c1", "c2"], strict=True)
        ],
        conditions=[
            Condition(id="c1", description="First property"),
            Condition(id="c2", description="Second property"),
        ],
        needs=["Literature"],
    )


def prior(index=2, *, version="", **extra):
    identifier = f"2001.0000{index}{version}"
    return {
        "id": identifier,
        "arxiv_id": identifier,
        "title": ["", "First architecture", "Second architecture", "Unrelated model"][index],
        "published": "2020-01-01",
        "abstract": "The source proves a specific property.",
        **extra,
    }


def comparison(*, covered, paper_id="2001.00002", relation="supports", purpose="citation_support"):
    return {
        "paper_id": paper_id,
        "purpose": purpose,
        "relation": relation,
        "quote": "The property holds.",
        "covered": covered,
        "fully_supported_conditions": covered if relation == "supports" else [],
        "mechanism": "Same property",
        "setting": "Same architecture",
        "protocol": "Same proof",
        "note": "The retrieved statement concerns these conditions.",
    }


def services(*, search_papers=None, version="", poison=False):
    papers = [prior(1), prior(2, version=version)] if search_papers is None else search_papers
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixture", "complete": True, "papers": papers}
    )
    extra = (
        {"cited": False, "citation_condition_ids": ["c1", "c2"], "citation_sources": [{"fabricated": True}]}
        if poison
        else {}
    )

    async def lookup(*, identifier):
        return {"success": True, "paper": prior(int(identifier.split("v")[0][-1]), version=version, **extra)}

    searcher.lookup_metadata = AsyncMock(side_effect=lookup)
    reader = Mock()

    async def read(*, items):
        identifier = items[0]["id"]
        metadata = prior(int(identifier.split("v")[0][-1]), version=version, **extra)
        return {
            "success": True,
            "items": [
                {
                    "id": identifier,
                    "success": True,
                    "paper": metadata,
                    "evidence": [{"text": "The property holds.", "page": 2}],
                }
            ],
        }

    reader.read_papers = AsyncMock(side_effect=read)
    return searcher, reader


async def verify(claim, materials, tmp_path, comparisons, **boundary_options):
    searcher, reader = services(**boundary_options)
    call = Mock(return_value={"status": "ok", "comparisons": comparisons})
    result = await verify_literature(
        claim,
        materials,
        submission_deadline="2021-01-01",
        searcher=searcher,
        reader=reader,
        call=call,
        output_dir=tmp_path / "audit",
    )
    audit = json.loads((tmp_path / "audit" / "claim-multiple-search-audit.json").read_text(encoding="utf8"))
    payload = json.loads(call.call_args.kwargs["prompt"].split("\nDATA_JSON:\n")[1]) if call.called else None
    return result, audit, payload, searcher, reader


def test_each_citation_is_bound_to_its_exact_passage_and_condition(claim, materials):
    papers = _cited_papers(claim, materials)
    assert [(p["id"], p["citation_condition_ids"]) for p in papers] == [
        ("2001.00001", ["c1"]),
        ("2001.00002", ["c2"]),
    ]
    assert papers[1]["citation_sources"][0]["source_quote"] == materials.blocks[2].text
    assert "2001.00003" not in str(papers)
    summary = _claim_source_excerpt(claim, materials)
    assert "[separate source passage]" in summary
    assert materials.blocks[2].text in summary


def test_primary_same_quote_ref_uses_explicit_scope(claim, materials):
    block = materials.blocks[1]
    claim.source_block_id, claim.source_quote, claim.loc = block.id, block.text, block.loc
    papers = _cited_papers(claim, materials)
    assert papers[0]["citation_condition_ids"] == ["c1"]
    assert len(papers[0]["citation_sources"]) == 1


def test_legacy_primary_without_refs_retains_all_condition_scope(claim, materials):
    block = materials.blocks[1]
    claim.source_block_id, claim.source_quote, claim.loc = block.id, block.text, block.loc
    claim.source_refs = []
    assert _cited_papers(claim, materials)[0]["citation_condition_ids"] == ["c1", "c2"]


def test_same_paper_in_multiple_passages_unions_explicit_scopes(claim, materials):
    materials.bibliography[1].text = materials.bibliography[1].text.replace("2001.00002", "2001.00001")
    papers = _cited_papers(claim, materials)
    assert len(papers) == 1
    assert papers[0]["citation_condition_ids"] == ["c1", "c2"]
    assert len(papers[0]["citation_sources"]) == 2


@pytest.mark.parametrize(
    "damage", ["ref_quote", "ref_location", "ref_block", "primary_quote", "primary_location"]
)
@pytest.mark.asyncio
async def test_any_unbound_source_is_visible_and_cannot_supply_citation_support(
    claim, materials, tmp_path, damage
):
    if damage == "ref_quote":
        claim.source_refs[1].source_quote = "An invented claim [2]."
    elif damage == "ref_location":
        claim.source_refs[1].loc = ClaimLocation(page=16)
    elif damage == "ref_block":
        claim.source_refs[1].source_block_id = "absent"
    elif damage == "primary_quote":
        claim.source_quote = "An invented primary."
    else:
        claim.loc = ClaimLocation(page=16)
    result, audit, payload, _, _ = await verify(
        claim,
        materials,
        tmp_path,
        [comparison(covered=["c2"])],
    )
    assert not result.evidence
    assert any("Citation source unavailable" in issue for issue in result.issues)
    assert not audit["source_available"] and not audit["adequate_for_no_close_prior_work"]
    assert audit["source_refs"] == [ref.model_dump(mode="json") for ref in claim.source_refs]
    assert payload["source_excerpts"] == []
    assert all(not row["cited"] for row in payload["sources"])


@pytest.mark.asyncio
async def test_structured_sources_and_audit_keep_exact_whitespace_and_distinct_locations(
    claim, materials, tmp_path
):
    result, audit, payload, _, _ = await verify(claim, materials, tmp_path, [comparison(covered=["c2"])])
    assert result.evidence[0].sufficient and result.evidence[0].covered == ["c2"]
    assert payload["source_excerpts"] == audit["source_excerpts"]
    assert len(payload["source_excerpts"]) == 3
    assert payload["source_excerpts"][2]["source_quote"] == "  Second property follows Jones et al. (2020).\n"
    assert payload["source_excerpts"][2]["loc"] == materials.blocks[2].loc.model_dump(mode="json")
    assert audit["citation_bindings"][1]["citation_condition_ids"] == ["c2"]
    assert payload["source_excerpt"] != "".join(x["source_quote"] for x in payload["source_excerpts"])


@pytest.mark.parametrize("covered", [["c1"], ["c1", "c2"]])
@pytest.mark.parametrize("relation", ["supports", "contradicts", "unclear"])
@pytest.mark.asyncio
async def test_supplemental_citation_cannot_expand_support_or_concern_to_other_conditions(
    claim,
    materials,
    tmp_path,
    covered,
    relation,
):
    result, _, _, _, _ = await verify(
        claim,
        materials,
        tmp_path,
        [comparison(covered=covered, relation=relation)],
    )
    assert not result.evidence
    assert any("outside the original source condition scope" in issue for issue in result.issues)


@pytest.mark.parametrize("mode", ["search", "lookup", "unresolved_title", "version_alias"])
@pytest.mark.asyncio
async def test_scope_survives_search_metadata_resolution_and_reader_merge(claim, materials, tmp_path, mode):
    version = "v2" if mode == "version_alias" else ""
    papers = [prior(1), prior(2, version=version)]
    if mode == "lookup":
        papers = []
    elif mode == "unresolved_title":
        materials.bibliography[1].text = "[2] Jones. 2020. Second architecture."
    # Untrusted search/read metadata attempts both broadening and deleting citation scope.
    for paper in papers:
        paper.update({"cited": False, "citation_condition_ids": ["c1", "c2"], "citation_sources": []})
    result, audit, payload, searcher, reader = await verify(
        claim,
        materials,
        tmp_path,
        [comparison(covered=["c2"], paper_id=f"2001.00002{version}")],
        search_papers=papers,
        version=version,
        poison=True,
    )
    assert result.evidence[0].sufficient and result.evidence[0].covered == ["c2"]
    row = next(row for row in payload["sources"] if row["paper_id"] == f"2001.00002{version}")
    assert row["citation_condition_ids"] == ["c2"] and row["cited"]
    assert row["citation_sources"] == [claim.source_refs[1].model_dump(mode="json")]
    assert row["paper"]["citation_condition_ids"] == ["c2"]
    assert next(x for x in audit["citation_bindings"] if x["paper_id"] == row["paper_id"])[
        "citation_condition_ids"
    ] == ["c2"]
    assert reader.read_papers.await_count == 2
    if mode in {"lookup", "version_alias"}:
        assert searcher.lookup_metadata.await_count >= 1


@pytest.mark.asyncio
async def test_search_cannot_invent_citation_binding(claim, materials, tmp_path):
    fake = prior(3, cited=True, citation_condition_ids=["c1", "c2"], citation_sources=[{"fabricated": True}])
    result, _, payload, _, _ = await verify(
        claim,
        materials,
        tmp_path,
        [comparison(covered=["c2"], paper_id="2001.00003")],
        search_papers=[fake],
    )
    assert not result.evidence
    row = next(x for x in payload["sources"] if x["paper_id"] == "2001.00003")
    assert not row["cited"] and row["citation_condition_ids"] == []


@pytest.mark.asyncio
async def test_two_unresolved_entries_resolving_to_same_paper_keep_both_scopes(claim, materials, tmp_path):
    materials.bibliography[0].text = "[1] Smith. 2020. Shared architecture description."
    materials.bibliography[1].text = "[2] Jones. 2020. Shared architecture description."
    shared = prior(2, title="Shared architecture description")
    result, _, payload, _, reader = await verify(
        claim,
        materials,
        tmp_path,
        [comparison(covered=["c1", "c2"])],
        search_papers=[shared],
    )
    assert result.evidence[0].sufficient
    assert payload["sources"][0]["citation_condition_ids"] == ["c1", "c2"]
    assert len(payload["sources"][0]["citation_sources"]) == 2
    reader.read_papers.assert_awaited_once()


def test_author_year_cannot_form_across_source_boundaries(claim, materials):
    primary = materials.blocks[0]
    primary.text = "The approach follows Smith"
    primary.loc = ClaimLocation(page=1)
    materials.markdown = "Unavailable normalized text"
    claim.source_quote, claim.loc = primary.text, primary.loc
    ref = materials.blocks[1]
    ref.text, ref.loc = "(2020), with a second property.", ClaimLocation(page=2)
    claim.source_refs = [
        ClaimSourceRef(source_block_id=ref.id, source_quote=ref.text, loc=ref.loc, covered=["c2"])
    ]
    assert _cited_papers(claim, materials) == []
