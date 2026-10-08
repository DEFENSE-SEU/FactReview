"""Original bibliography labels bind citations without author searches or guesses."""

from unittest.mock import AsyncMock, Mock

import pytest

from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.literature import _cited_papers, _match_cited_entries, verify_literature


def materials(tmp_path, text="We use the optimizer [LH19].", entries=None):
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    block = MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1, char_start=0, char_end=len(text)))
    return SharedMaterials(
        paper_key="labels",
        title="A graph neural network evaluation",
        abstract="We evaluate graph models.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(path),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixed mock",
        blocks=[block],
        bibliography=[
            MaterialBlock(id=f"r{i}", text=entry, kind="ref_text", loc=ClaimLocation(page=3))
            for i, entry in enumerate(
                entries
                or [
                    "[LH19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv:1711.05101, 2019."
                ]
            )
        ],
    )


def claim_for(material, *, source_refs=None):
    return Claim(
        id="claim",
        text="The optimizer has the cited property.",
        loc=material.blocks[0].loc,
        source_block_id="b1",
        source_quote=material.blocks[0].text,
        source_refs=source_refs or [],
        conditions=[Condition(id="c1", description="The cited optimizer property")],
        needs=["Literature"],
    )


@pytest.mark.parametrize("label", ["LH19", "ZKHB21", "RPG+21", "ABC2020a", "AB+21b", "BDW<sup>+</sup>20"])
def test_exact_alphanumeric_label_selects_existing_original_entry(tmp_path, label):
    material = materials(
        tmp_path,
        f"The source [{label}] establishes this result.",
        [f"[{label}] Original work. arXiv:2001.00001, 2020."],
    )
    issues = []
    selected = _match_cited_entries(material.markdown, material, issues=issues)
    assert [row["arxiv_id"] for row in selected] == ["2001.00001"]
    assert not issues
    assert selected[0]["bibliography_text"] == material.bibliography[0].text


@pytest.mark.parametrize("citation", ["[RPG+21, XY22]", "[RPG+21; XY22]", "[RPG+21, XY22, RPG+21]"])
def test_grouped_labels_preserve_distinct_entries_without_duplicates(tmp_path, citation):
    material = materials(
        tmp_path, citation, ["[RPG+21] First. arXiv:2101.00001.", "[XY22] Second. arXiv:2201.00002."]
    )
    assert [row["arxiv_id"] for row in _match_cited_entries(citation, material)] == [
        "2101.00001",
        "2201.00002",
    ]


def test_ambiguous_label_is_not_arbitrarily_selected(tmp_path):
    material = materials(
        tmp_path, "We use [AB20].", ["[AB20] First. arXiv:2001.00001.", "[AB20] Second. arXiv:2001.00002."]
    )
    issues = []
    assert not _match_cited_entries(material.markdown, material, issues=issues)
    assert any("Ambiguous citation label [AB20]" in issue for issue in issues)


@pytest.mark.parametrize(
    "text",
    [
        "Use [CLS] before [MASK].",
        "An array [x1, x2] has entries.",
        "The interval [0, 1] is closed.",
        "We use [lh19].",
    ],
)
def test_ordinary_bracket_words_arrays_and_different_case_are_not_bound(tmp_path, text):
    material = materials(tmp_path, text)
    assert not _match_cited_entries(text, material)


def test_unresolved_known_label_retains_candidate_and_visible_resolution_issue(tmp_path, monkeypatch):
    # Selection must preserve an exact citation even when no stable ID is supplied.
    material = materials(tmp_path, entries=["[LH19] Original optimizer paper. Conference 2019."])
    selected = _cited_papers(claim_for(material), material)
    assert len(selected) == 1 and not selected[0]["id"]
    assert selected[0]["citation_condition_ids"] == ["c1"]


def test_unknown_year_shaped_label_is_visible_without_selecting_another_entry(tmp_path):
    material = materials(tmp_path, "The method follows [UNKNOWN21].")
    issues = []
    assert not _match_cited_entries(material.markdown, material, issues=issues)
    assert any("Unresolved citation label [UNKNOWN21]" in issue for issue in issues)


def test_numeric_and_author_year_paths_remain_available(tmp_path):
    material = materials(
        tmp_path,
        "[1] and Smith (2020)",
        ["[1] First. arXiv:1901.00001.", "Smith, Alice. 2020. Second. arXiv:2001.00002."],
    )
    assert [row["arxiv_id"] for row in _match_cited_entries(material.markdown, material)] == [
        "1901.00001",
        "2001.00002",
    ]


@pytest.mark.asyncio
async def test_label_citation_support_uses_exact_condition_scope_and_only_technical_queries(
    tmp_path, monkeypatch
):
    material = materials(tmp_path)
    target = claim_for(material)
    target.conditions.append(Condition(id="c2", description="An independent result"))
    target.source_refs = [
        ClaimSourceRef(
            source_block_id="b1", source_quote=material.markdown, loc=material.blocks[0].loc, covered=["c1"]
        )
    ]
    paper = {
        "id": "1711.05101",
        "arxiv_id": "1711.05101",
        "title": "Decoupled weight decay regularization",
        "published": "2019-01-01",
    }
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixture", "complete": True, "papers": []}
    )
    searcher.lookup_metadata = AsyncMock(return_value={"success": True, "paper": paper})
    reader = Mock()
    passage = "The optimizer decouples weight decay from the gradient update."
    reader.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {
                    "id": paper["id"],
                    "success": True,
                    "paper": paper,
                    "evidence": [{"text": passage, "page": 2}],
                }
            ],
        }
    )
    comparison = {
        "paper_id": paper["id"],
        "purpose": "citation_support",
        "relation": "supports",
        "quote": passage,
        "covered": ["c1"],
        "fully_supported_conditions": ["c1"],
        "mechanism": "Decoupled weight decay.",
        "setting": "Optimizer update.",
        "protocol": "Described procedure.",
        "note": "The original cited property is established.",
    }
    model = Mock(return_value={"status": "ok", "comparisons": [comparison]})
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    result = await verify_literature(
        target,
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model,
        output_dir=tmp_path / "audit",
    )
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    assert result.evidence[0].covered == ["c1"]
    searcher.lookup_metadata.assert_awaited_once_with(identifier=paper["id"])
    assert searcher.search.await_count == 3
    assert all(
        "Loshchilov" not in call.kwargs["query"]
        and "Hutter" not in call.kwargs["query"]
        and "LH19" not in call.kwargs["query"]
        for call in searcher.search.await_args_list
    )
    comparison["covered"] = comparison["fully_supported_conditions"] = ["c1", "c2"]
    invalid = await verify_literature(
        target,
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model,
        output_dir=tmp_path / "invalid-audit",
    )
    assert not invalid.evidence
    assert any("outside the original source condition scope" in issue for issue in invalid.issues)


@pytest.mark.asyncio
async def test_known_label_without_identifier_is_explicitly_unresolved(tmp_path, monkeypatch):
    material = materials(tmp_path, entries=["[LH19] Original optimizer paper. Conference 2019."])
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixture", "complete": True, "papers": []}
    )
    reader = Mock()
    reader.read_papers = AsyncMock()
    model = Mock()
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    result = await verify_literature(
        claim_for(material),
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model,
        output_dir=tmp_path / "audit",
    )
    assert not result.evidence
    assert any("unresolved citation" in issue for issue in result.issues)
    assert result.questions
    reader.read_papers.assert_not_awaited()
    model.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested,extra,read_id,metadata,expected",
    [
        ("2001.00001", {}, "2001.00001", {"id": "2002.00002", "arxiv_id": "2002.00002"}, False),
        ("2001.00001", {}, "2001.00001", {"id": "2002.00002", "arxiv_id": "2001.00001"}, False),
        ("2001.00001", {}, "2001.00001", {"arxiv_id": "not-an-id"}, False),
        ("2001.00001", {}, "arXiv:2001.00001", {"arxiv_id": "https://arxiv.org/abs/2001.00001v2"}, True),
        (
            "2001.00001v2",
            {},
            "https://arxiv.org/pdf/2001.00001v2.pdf",
            {"arxiv_id": "arxiv:2001.00001v2"},
            True,
        ),
        ("2001.00001v2", {}, "2001.00001v2", {"arxiv_id": "2001.00001v3"}, False),
        ("2001.00001v2", {}, "2001.00001v2", {"arxiv_id": "2001.00001"}, False),
        ("10.1234/ABC", {}, "https://doi.org/10.1234/abc", {"doi": "DOI:10.1234/AbC"}, True),
        ("10.1234/ABC", {}, "10.1234/ABC", {"doi": "10.1234/OTHER"}, False),
        ("10.1234/ABC", {}, "10.1234/ABC", {"doi": "not-a-doi"}, False),
        (
            "2001.00001",
            {"doi": "10.1234/ABC"},
            "https://doi.org/10.1234/abc",
            {"arxiv_id": "2001.00001", "doi": "10.1234/abc"},
            True,
        ),
        (
            "2001.00001",
            {"doi": "10.1234/ABC"},
            "2001.00001",
            {"arxiv_id": "2001.00001", "doi": "10.1234/OTHER"},
            False,
        ),
        ("2001.00001", {}, "2001.00001", {"doi": "10.1234/NEW-ALIAS"}, True),
        ("2001.00001", {}, "2001.00001", {}, True),
    ],
)
async def test_reader_explicit_identity_conflicts_cannot_rebind_decisive_passages(
    tmp_path, monkeypatch, requested, extra, read_id, metadata, expected
):
    import json

    from assessment import assess_claim

    text = "We propose a novel graph neural network for link prediction."
    material = materials(tmp_path, text)
    target = claim_for(material)
    target.text = text
    target.conditions = [Condition(id="c1", description="novel mechanism for link prediction")]
    primary_key = "doi" if requested.startswith("10.") else "arxiv_id"
    paper = {
        "id": requested,
        primary_key: requested,
        "title": "A previous relational method",
        "published": "2020-01-01",
        **extra,
    }
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixed", "complete": True, "papers": [paper]}
    )
    passage = "We compose relation embeddings through message passing for link prediction."
    reader = Mock()
    reader.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {
                    "id": read_id,
                    "success": True,
                    "paper": metadata,
                    "evidence": [{"text": passage, "page": 2}],
                }
            ],
        }
    )
    row = {
        "paper_id": requested,
        "purpose": "novelty",
        "relation": "same",
        "quote": passage,
        "covered": ["c1"],
        "mechanism": "Same composition mechanism.",
        "setting": "Same target setting.",
        "protocol": "Same evaluation protocol.",
        "note": "Fixed mock comparison.",
    }
    model = Mock(return_value={"status": "ok", "comparisons": [row]})
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    result = await verify_literature(
        target,
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=model,
        output_dir=tmp_path / "audit",
    )
    assert any(item.sufficient for item in result.evidence) is expected
    assessed = assess_claim(target.model_copy(update={"evidence": result.evidence}))
    assert (assessed.status.value == "flawed") is expected
    audit = json.loads((tmp_path / "audit/claim-search-audit.json").read_text(encoding="utf-8"))
    if not expected:
        assert any("conflicting paper identity" in issue for issue in result.issues)
        assert audit["reads"][0]["identity_rejected"] is True
        assert audit["adequate_for_no_close_prior_work"] is False
        model.assert_not_called()
    elif primary_key == "arxiv_id":
        assert result.evidence[0].pointer.locator.startswith("arxiv:2001.00001")


@pytest.mark.asyncio
async def test_identity_conflict_can_only_retain_original_abstract_as_nondecisive_cue(tmp_path, monkeypatch):
    text = "We propose a novel graph neural network for link prediction."
    material = materials(tmp_path, text)
    target = claim_for(material)
    target.text = text
    target.conditions = [Condition(id="c1", description="novel mechanism for link prediction")]
    abstract = "A relation-composition method is described in this original abstract."
    paper = {
        "id": "2001.00001",
        "arxiv_id": "2001.00001",
        "title": "Previous relational method",
        "published": "2020-01-01",
        "abstract": abstract,
    }
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixed", "complete": True, "papers": [paper]}
    )
    reader = Mock()
    reader.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {
                    "id": paper["id"],
                    "success": True,
                    "paper": {"arxiv_id": "2002.00002", "abstract": "Unrelated metadata must be discarded."},
                    "evidence": [{"text": "Unrelated returned full text.", "page": 2}],
                }
            ],
        }
    )
    row = {
        "paper_id": paper["id"],
        "purpose": "novelty",
        "relation": "same",
        "quote": abstract,
        "covered": ["c1"],
        "mechanism": "Same composition.",
        "setting": "Same task.",
        "protocol": "Same protocol.",
        "note": "Abstract-only cue.",
    }
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    result = await verify_literature(
        target,
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=Mock(return_value={"status": "ok", "comparisons": [row]}),
        output_dir=tmp_path / "audit",
    )
    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert not item.sufficient and item.overturnable
    assert item.pointer.locator == "arxiv:2001.00001" and item.pointer.quote == abstract


async def reader_probe(tmp_path, monkeypatch, originals, responses, *, purpose="novelty", abstract=False):
    from assessment import assess_claim

    first = originals[0]
    identifier = first.get("arxiv_id") or first["doi"]
    citation = f"arXiv:{identifier}" if first.get("arxiv_id") else f"doi:{identifier}"
    text = "A novel graph neural network method [AB20]."
    material = materials(tmp_path, text, [f"[AB20] Original work. {citation}."])
    target = claim_for(material)
    target.text = text
    target.conditions = [Condition(id="c1", description="novel graph mechanism")]
    searcher = Mock()
    searcher.search = AsyncMock(
        return_value={"success": True, "provider": "fixture", "complete": True, "papers": originals}
    )
    reader = Mock()
    reader.read_papers = AsyncMock(side_effect=lambda *, items: responses[items[0]["id"]])
    rows = [
        {
            "paper_id": paper.get("arxiv_id") or paper["doi"],
            "purpose": purpose,
            "relation": "same" if purpose == "novelty" else "supports",
            "quote": paper["abstract"] if abstract else "Exact original method.",
            "covered": ["c1"],
            "fully_supported_conditions": ["c1"],
            "mechanism": "Same method.",
            "setting": "Same task.",
            "protocol": "Same protocol.",
            "note": "Fixed review.",
        }
        for paper in originals
    ]
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    result = await verify_literature(
        target,
        material,
        submission_deadline="2021-01-31",
        searcher=searcher,
        reader=reader,
        call=Mock(return_value={"status": "ok", "comparisons": rows}),
        output_dir=tmp_path / "audit",
    )
    return result, assess_claim(target.model_copy(update={"evidence": result.evidence}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested,metadata,expected_locator",
    [
        ("10.1234/known", {"doi": "10.1234/known", "arxiv_id": "2002.00002"}, "10.1234/known"),
        ("2001.00001", {"arxiv_id": "", "doi": "10.1234/other"}, "arxiv:2001.00001"),
    ],
)
async def test_unverified_cross_namespace_alias_cannot_replace_original_locator(
    tmp_path, monkeypatch, requested, metadata, expected_locator
):
    import json

    key = "doi" if requested.startswith("10.") else "arxiv_id"
    original = {
        "id": requested,
        key: requested,
        "title": "Previous graph operators",
        "published": "2020-01-01",
    }
    response = {
        "success": True,
        "items": [
            {
                "id": requested,
                "success": True,
                "paper": metadata,
                "evidence": [{"text": "Exact original method.", "page": 2}],
            }
        ],
    }
    result, _ = await reader_probe(tmp_path, monkeypatch, [original], {requested: response})
    assert result.evidence and all(item.pointer.locator == expected_locator for item in result.evidence)
    audit = json.loads((tmp_path / "audit/claim-search-audit.json").read_text(encoding="utf-8"))
    assert audit["reads"][0]["response"] == response
    assert audit["reads"][0]["identity_fields_not_adopted"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["metadata_conflict", "unmatched_envelope", "failed_reader", "matched_abstract_control"]
)
async def test_citation_abstract_fallback_after_unbound_reader_cannot_be_sufficient(
    tmp_path, monkeypatch, failure
):
    original = {
        "id": "2001.00001",
        "arxiv_id": "2001.00001",
        "title": "Previous graph operators",
        "published": "2020-01-01",
        "abstract": "The original abstract describes this graph method.",
    }
    metadata = {"arxiv_id": "2002.00002"} if failure == "metadata_conflict" else {}
    envelope = "2002.00002" if failure == "unmatched_envelope" else original["id"]
    response = {
        "success": failure != "failed_reader",
        "items": [{"id": envelope, "success": True, "paper": metadata}],
    }
    result, assessed = await reader_probe(
        tmp_path,
        monkeypatch,
        [original],
        {original["id"]: response},
        purpose="citation_support",
        abstract=True,
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].pointer.locator == "arxiv:2001.00001"
    assert result.evidence[0].sufficient is (failure == "matched_abstract_control")
    assert (assessed.status.value == "supported") is (failure == "matched_abstract_control")
    if failure != "matched_abstract_control":
        assert any("identity" in issue or "could not be bound" in issue for issue in result.issues)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["arxiv_id", "url", "id", "pdf_url"])
async def test_malformed_read_identity_isolated_from_other_healthy_papers(tmp_path, monkeypatch, field):
    papers = [
        {"id": value, "arxiv_id": value, "title": f"Previous operator {index}", "published": "2020-01-01"}
        for index, value in enumerate(("2001.00001", "2002.00002"))
    ]
    responses = {
        paper["id"]: {
            "success": True,
            "items": [
                {
                    "id": paper["id"],
                    "success": True,
                    "paper": {field: "https://[bad"} if index == 0 else paper,
                    "evidence": [{"text": "Exact original method.", "page": 2}],
                }
            ],
        }
        for index, paper in enumerate(papers)
    }
    result, _ = await reader_probe(tmp_path, monkeypatch, papers, responses)
    assert len(result.evidence) == 1
    assert result.evidence[0].pointer.locator == "arxiv:2002.00002" and result.evidence[0].sufficient
    assert any("conflicting paper identity" in issue for issue in result.issues)
