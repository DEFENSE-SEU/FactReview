"""Explicit publication votes cannot become reviewer advice; protocol actions remain usable."""

import json
from pathlib import Path

import pytest

from assessment import assess_claims
from llm.client import LLMConfig
from review.report.advice import generate_advice
from review.report.v2 import write_review
from schemas.claim import Claim, Condition, Evidence, EvidencePointer
from schemas.review import FinalReview


@pytest.fixture
def assessed_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "fixed", None, None)
    )
    source = tmp_path / "source.md"
    source.write_text(
        "The evaluator accepts correctly formatted samples and rejects malformed samples.", encoding="utf-8"
    )
    claim = Claim(
        id="protocol",
        text=source.read_text("utf-8"),
        loc={"page": 1},
        conditions=[Condition(id="c1", description="input validation")],
        needs=["Code"],
        evidence=[
            Evidence(
                source="paper_internal",
                pointer=EvidencePointer(locator=str(source), quote=source.read_text("utf-8"), page=1),
                covered=["c1"],
                direction="support",
                sufficient=True,
            )
        ],
    )
    return FinalReview(paper_key="publication-vote-fixture", run_id="offline", claims=assess_claims([claim]))


@pytest.mark.parametrize(
    "text",
    [
        "I vote to accept this paper.",
        "My vote is to reject the submission.",
        "We voted to accept this manuscript.",
        "I am voting for rejection of this paper.",
    ],
)
def test_publication_vote_is_unavailable_through_advice_and_report(assessed_review, tmp_path, text):
    before = assessed_review.model_dump(mode="json")

    def call(**_):
        return {
            "status": "ok",
            "claim_id": "protocol",
            "items": [{"text": text, "condition_ids": ["c1"], "basis_refs": ["/evidence/0"]}],
        }

    result = generate_advice(assessed_review, tmp_path / "advice", call=call)
    assert result.counts == {"generated": 0, "unavailable": 1}
    assert assessed_review.model_dump(mode="json") == before
    assert result.review.claims[0].status == "supported"
    audit = json.loads(Path(result.review.claims[0].advice.audit_pointer).read_text("utf-8"))
    assert audit["response"]["items"][0]["text"] == text
    output = write_review(result.review, tmp_path / "report", render_pdf=False)
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert saved["claims"][0]["advice"]["state"] == "unavailable"
    assert "Advice unavailable" in Path(output["markdown"]).read_text("utf-8")


@pytest.mark.parametrize(
    "text",
    [
        "The evaluator accepts valid samples and rejects malformed samples.",
        "The protocol uses a majority vote to accept the sample.",
        "The protocol uses a vote to reject the null hypothesis.",
        "The manuscript reports a majority vote to accept a prediction.",
    ],
)
def test_sample_and_protocol_accept_reject_remain_readable(assessed_review, tmp_path, text):
    def call(**_):
        return {
            "status": "ok",
            "claim_id": "protocol",
            "items": [{"text": text, "condition_ids": ["c1"], "basis_refs": ["/evidence/0"]}],
        }

    result = generate_advice(assessed_review, tmp_path / "advice", call=call)
    assert result.counts == {"generated": 1, "unavailable": 0}
    output = write_review(result.review, tmp_path / "report", render_pdf=False)
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert saved["claims"][0]["advice"]["items"][0]["text"] == text
