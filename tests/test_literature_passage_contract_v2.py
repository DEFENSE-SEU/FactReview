"""Model inputs distinguish identity metadata from the exact passages eligible for quotes."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_literature_citation_labels_v2 import claim_for, materials
from verification.literature import verify_literature

PAPER = {
    "id": "1711.05101",
    "arxiv_id": "1711.05101",
    "title": "Independent optimizer procedure",
    "published": "2019-01-01",
    "abstract": "The optimizer applies weight decay independently of gradient-based updates.",
}
PASSAGE = "The optimizer decouples weight de- cay\nfrom the gradient update."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: None)
    monkeypatch.setattr("verification.literature._default_adapter", lambda: pytest.fail("Live adapter"))
    monkeypatch.setattr("verification.literature.llm_json", lambda **kw: pytest.fail("Live model"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Docker/process"))


async def inspect(tmp_path, *, quote, evidence=True, duplicate=False, purpose="citation_support"):
    material = materials(tmp_path)
    claim = claim_for(material)
    searcher = Mock(
        search=AsyncMock(return_value={"success": True, "provider": "mock", "complete": False, "papers": []}),
        lookup_metadata=AsyncMock(return_value={"success": True, "paper": deepcopy(PAPER)}),
    )
    item = {
        "id": PAPER["id"],
        "success": True,
        "paper": deepcopy(PAPER),
        "evidence": [{"text": PASSAGE, "page": 2}] if evidence else [],
    }
    reader_items = [item]
    if duplicate:
        reader_items.append({**deepcopy(item), "evidence": [{"text": PAPER["abstract"], "page": 3}]})
    reader = Mock(read_papers=AsyncMock(return_value={"success": True, "items": reader_items}))
    reader_before = deepcopy(reader_items)
    comparison = {
        "paper_id": PAPER["id"],
        "purpose": purpose,
        "relation": "supports" if purpose == "citation_support" else "different",
        "quote": quote,
        "covered": ["c1"] if purpose == "citation_support" else [],
        "fully_supported_conditions": ["c1"] if purpose == "citation_support" else [],
        "mechanism": "The cited source separates weight decay from the gradient update.",
        "setting": "The claimed optimizer update rule.",
        "protocol": "A reported algorithmic procedure.",
        "note": "Fixed mock semantic comparison.",
    }
    model = Mock(return_value={"status": "ok", "comparisons": [comparison]})
    result = await verify_literature(
        claim,
        material,
        submission_deadline="2021-01-01",
        searcher=searcher,
        reader=reader,
        call=model,
        output_dir=tmp_path / "audit",
    )
    model.assert_called_once()
    searcher.lookup_metadata.assert_awaited_once()
    reader.read_papers.assert_awaited_once()
    payload = json.loads(model.call_args.kwargs["prompt"].split("\nDATA_JSON:\n", 1)[1])
    assert payload["claim"] == claim.model_dump(mode="json")
    assert len(payload["sources"]) == 1
    source = payload["sources"][0]
    # The model projection drops only this duplicate metadata field; the raw
    # reader/lookup responses and audit retain it for identity and fallback use.
    assert "abstract" not in source["paper"]
    assert searcher.lookup_metadata.return_value["paper"] == PAPER
    assert reader.read_papers.return_value["items"] == reader_before
    audit = json.loads(next((tmp_path / "audit").glob("*-search-audit.json")).read_text())
    assert audit["metadata_lookups"][0]["response"]["paper"] == PAPER
    assert audit["reads"][0]["response"]["items"] == reader_before
    assert source["reader_identity_verified"] is True
    assert source["citation_condition_ids"] == ["c1"]
    for evidence_item in result.evidence:
        assert evidence_item.pointer.locator == "arxiv:1711.05101"
    return result, source, model.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote,expected",
    [
        (PASSAGE, True),
        (PAPER["abstract"], False),
        (PASSAGE.replace("\n", " "), False),
        (PASSAGE.replace("de- cay", "decay"), False),
    ],
)
async def test_full_text_input_keeps_metadata_separate_and_quote_characters_exact(tmp_path, quote, expected):
    result, source, request = await inspect(tmp_path, quote=quote)
    assert source["passages"] == [{"text": PASSAGE, "page": 2, "source": "full_text"}]
    assert "sources[].passages[].text" in request["system"]
    assert "sources[].paper is identity/context metadata" in request["system"]
    assert bool(result.evidence) is expected
    assert any(e.sufficient and e.covered == ["c1"] for e in result.evidence) is expected
    assert any("ungrounded quote" in issue for issue in result.issues) is (not expected)


@pytest.mark.asyncio
async def test_unused_same_identity_reader_item_cannot_make_metadata_quote_eligible(tmp_path):
    result, source, _ = await inspect(tmp_path, quote=PAPER["abstract"], duplicate=True)
    assert source["passages"] == [{"text": PASSAGE, "page": 2, "source": "full_text"}]
    assert not result.evidence
    assert any("ungrounded quote" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_existing_explicit_abstract_fallback_remains_quoteable(tmp_path):
    result, source, _ = await inspect(tmp_path, quote=PAPER["abstract"], evidence=False)
    assert source["passages"] == [{"text": PAPER["abstract"], "page": None, "source": "abstract"}]
    assert source["full_text"] is False
    assert any("only a retrieved abstract" in issue for issue in result.issues)
    assert any(e.sufficient and e.covered == ["c1"] for e in result.evidence)
    assert not any("ungrounded quote" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_grounded_different_related_work_does_not_create_claim_support_or_missing_finding(tmp_path):
    result, source, _ = await inspect(tmp_path, quote=PASSAGE, purpose="related_work")
    assert source["passages"][0]["text"] == PASSAGE
    assert not result.evidence and not result.findings
    assert not any("ungrounded quote" in issue for issue in result.issues)
