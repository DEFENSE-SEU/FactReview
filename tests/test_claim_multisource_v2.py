"""Original-source provenance remains exact across paragraphs and report round trips."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from llm.client import LLMConfig
from review.report.v2 import write_review
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef
from schemas.materials import MaterialBlock, SharedMaterials
from schemas.review import FinalReview
from screening.claims import ClaimExtractionError, extract_claims


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(
        "screening.claims.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr("screening.claims.llm_json", lambda **_: pytest.fail("Unmocked model"))
    texts = ["These tuning ranges work across tasks.", "Batch sizes: 16, 32.", "Learning rates: 1e-5, 2e-5."]
    markdown = "\n\n".join(texts)
    blocks = []
    offset = 0
    for i, text in enumerate(texts):
        blocks.append(
            MaterialBlock(
                id=f"b{i}",
                text=text,
                loc=ClaimLocation(page=i + 1, char_start=offset, char_end=offset + len(text)),
            )
        )
        offset += len(text) + 2
    paper = SharedMaterials(
        paper_key="multisource",
        source_pdf="paper.pdf",
        markdown=markdown,
        markdown_path="paper.md",
        content_list_path="content.json",
        provider="fixture",
        blocks=blocks,
    )
    candidate = {
        "text": "The stated batch and learning-rate ranges work across tasks.",
        "source_block_id": "b0",
        "source_quote": texts[0],
        "source_refs": [
            {"source_block_id": "b1", "source_quote": texts[1], "covered": ["batch"]},
            {"source_block_id": "b2", "source_quote": texts[2], "covered": ["lr"]},
        ],
        "conditions": [
            {"id": "batch", "description": "Batch range works", "settings": {"batch_sizes": [16, 32]}},
            {
                "id": "lr",
                "description": "Learning rate range works",
                "settings": {"learning_rates": [1e-5, 2e-5]},
            },
        ],
        "needs": ["Experiments"],
        "importance": "secondary",
    }
    return paper, candidate


def test_one_conclusion_retains_multiple_exact_sources_without_manufacturing_support(inputs, tmp_path):
    paper, candidate = inputs

    def model(**request):
        assert "source_refs" in request["system"]
        assert "Corpus choice" in request["system"]
        return {"status": "ok", "claims": [candidate]}

    claims = extract_claims(paper, call=model)
    assert len(claims) == 1 and len(claims[0].conditions) == 2
    claim = claims[0]
    assert not claim.evidence and claim.status == "unverified"
    assert [ref.covered for ref in claim.source_refs] == [["batch"], ["lr"]]
    for ref in claim.source_refs:
        assert paper.markdown[ref.loc.char_start : ref.loc.char_end] == ref.source_quote
    report = FinalReview(paper_key="multi", run_id="test", claims=claims)
    outputs = write_review(report, tmp_path)

    restored = FinalReview.model_validate_json(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert restored.claims[0].source_refs == claim.source_refs
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "Original manuscript sources" in markdown
    assert "conditions: lr" in markdown
    pdf_text = "\n".join(page.extract_text() for page in PdfReader(outputs["pdf"]).pages)
    assert "Learning rates: 1e-5, 2e-5." in pdf_text


@pytest.mark.parametrize(
    "change",
    [
        {"source_quote": "Learning rates: 0.1."},
        {"source_block_id": "unknown"},
        {"covered": ["invented"]},
        {"covered": ["lr", "lr"]},
        {"source_quote": " "},
    ],
)
def test_bad_additional_source_cannot_hide_behind_a_valid_primary_quote(inputs, change):
    paper, candidate = inputs
    candidate["source_refs"][1].update(change)
    with pytest.raises(ClaimExtractionError):
        extract_claims(paper, call=lambda **_: {"status": "ok", "claims": [candidate]})


def test_additional_source_quote_can_be_repaired_with_scope_and_conclusion_preserved(inputs):
    paper, candidate = inputs
    original = candidate["source_refs"][1]["source_quote"]
    candidate["source_refs"][1]["source_quote"] = "Rates 1e-5 and 2e-5"

    def model(**request):
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate]}
        assert {b["id"] for b in json.loads(request["prompt"])["blocks"]} == {"b0", "b1", "b2"}
        refs = [dict(ref) for ref in candidate["source_refs"]]
        refs[1]["source_quote"] = original
        return {
            "repairs": [
                {
                    "index": 1,
                    "source_block_id": "b0",
                    "source_quote": candidate["source_quote"],
                    "source_refs": refs,
                }
            ]
        }

    claim = extract_claims(paper, call=model, max_source_repairs=3)[0]
    assert claim.text == candidate["text"] and claim.source_refs[1].source_quote == original
    assert claim.source_refs[1].covered == ["lr"]


@pytest.mark.parametrize(
    "refs",
    [[], [{"source_block_id": "b2", "source_quote": "Learning rates: 1e-5, 2e-5.", "covered": ["batch"]}]],
)
def test_repair_cannot_drop_or_change_additional_source_scope(inputs, refs):
    paper, candidate = inputs
    candidate["source_refs"][1]["source_quote"] = "Absent"

    def model(**request):
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate]}
        return {
            "repairs": [
                {
                    "index": 1,
                    "source_block_id": "b0",
                    "source_quote": candidate["source_quote"],
                    "source_refs": refs,
                }
            ]
        }

    with pytest.raises(ClaimExtractionError, match="reference coverage"):
        extract_claims(paper, call=model, max_source_repairs=3)


def test_old_saved_claim_remains_readable_and_duplicate_sources_are_rejected(inputs):
    paper, candidate = inputs
    claim = extract_claims(paper, call=lambda **_: {"status": "ok", "claims": [candidate]})[0]
    saved = claim.model_dump()
    saved.pop("source_refs")
    assert Claim.model_validate(saved).source_refs == []
    saved["source_refs"] = [claim.source_refs[0].model_dump()] * 2
    with pytest.raises(ValidationError, match="duplicated"):
        Claim.model_validate(saved)
    with pytest.raises(ValidationError, match="coverage ids must be unique"):
        ClaimSourceRef(
            source_block_id="b1",
            source_quote="Batch sizes: 16, 32.",
            loc=ClaimLocation(page=2),
            covered=["batch", "batch"],
        )
