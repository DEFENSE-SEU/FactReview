import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from pypdf import PdfReader

from review.report.v2 import write_review
from review.teaser.v2 import write_teaser
from schemas.claim import AuthorQuestion, Claim, ClaimLocation, Condition, Evidence, EvidencePointer, Finding
from schemas.review import FinalReview


def review():
    claims = [
        Claim(
            id=f"c{index}",
            text=f"Claim {status}",
            status=status,
            loc=ClaimLocation(page=index),
            conditions=[Condition(id="a", dataset="D", metric="accuracy")],
            needs=["Experiments"],
            evidence=[
                Evidence(
                    source="paper_internal",
                    pointer=EvidencePointer(locator="paper.md", page=1, quote="D accuracy .9"),
                    covered=["a"],
                    direction="support",
                    sufficient=True,
                )
            ],
            questions=[
                AuthorQuestion(text="Which seed was used?", claim_id=f"c{index}", reason="Not recorded")
            ],
        )
        for index, status in enumerate(["supported", "unverified", "flawed", "questioned"], 1)
    ]
    finding = Finding(
        kind="writing",
        loc=ClaimLocation(page=1),
        level="clarity_issue",
        text="Define the symbol",
        evidence=[claims[0].evidence[0]],
    )
    return FinalReview(
        paper_key="fixture",
        run_id="test",
        claims=claims,
        findings=[finding],
        ledger=[
            {
                "plan_id": "p1",
                "approval_mode": "rule_based",
                "command": ["python", "eval.py"],
                "returncode": 0,
                "aligned": True,
                "logs": "eval.log",
            }
        ],
    )


def test_four_sections_and_status_order_preserve_all_records(tmp_path):
    original = review()
    outputs = write_review(original, tmp_path, issues=["Literature unavailable"], render_pdf=False)
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    headers = ["## 1. Overview", "## 2. Claim list", "## 3. Other findings", "## 4. Execution ledger"]
    assert [markdown.index(header) for header in headers] == sorted(
        markdown.index(header) for header in headers
    )
    assert (
        markdown.index("### c3 — flawed")
        < markdown.index("### c4 — questioned")
        < markdown.index("### c2 — unverified")
        < markdown.index("### c1 — supported")
    )
    assert markdown.count("Which seed was used?") == 4
    assert "paper-internal / support" in markdown and "Literature unavailable" in markdown
    assert '"approval_mode": "rule_based"' in markdown
    assert "accept" not in markdown.lower() and "reject" not in markdown.lower()
    assert original.review_markdown == ""
    restored = FinalReview.model_validate_json(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert len(restored.claims) == 4 and restored.claims[0].status == "flawed"


def test_existing_pdf_renderer_can_render_v2_layout(tmp_path):
    outputs = write_review(review(), tmp_path)
    assert "pdf_error" not in outputs
    text = "\n".join(page.extract_text() for page in PdfReader(outputs["pdf"]).pages)
    for heading in ("Overview", "Claim list", "Other findings", "Execution ledger"):
        assert heading in text


def test_teaser_preserves_four_status_counts_and_source_types(tmp_path):
    result = write_teaser(review(), tmp_path)
    payload = json.loads(Path(result["json"]).read_text())
    assert payload["counts"] == {"flawed": 1, "questioned": 1, "unverified": 1, "supported": 1}
    assert all(claim["source_types"] == ["paper_internal"] for claim in payload["claims"])
    svg = ElementTree.parse(result["image"])
    text = " ".join(svg.getroot().itertext())
    assert all(label in text for label in payload["counts"])
    assert "accept" not in text.lower() and "reject" not in text.lower()
    assert Path(result["prompt"]).exists()


@pytest.mark.parametrize(
    "wording",
    [
        "I recommend acceptance.",
        "I recommend accepting this paper.",
        "This paper should be accepted.",
        "Decision: reject",
        "I recommend\naccepting this paper.",
    ],
)
def test_report_does_not_publish_model_recommendations(tmp_path, wording):
    record = review()
    record.claims[0].notes = [wording]
    with pytest.raises(ValueError, match="publication recommendation"):
        write_review(record, tmp_path, render_pdf=False)
    with pytest.raises(ValueError, match="publication recommendation"):
        write_teaser(record, tmp_path / "teaser")


def test_source_markdown_cannot_add_headings_or_images(tmp_path):
    from markdown_it import MarkdownIt

    record = review()
    record.claims[
        0
    ].text = "## Injected section ![image](https://example.invalid/x) [link](https://example.invalid)"
    record.findings[0].text = "Reject malformed inputs. Model uses x_y and a*b."
    output = write_review(record, tmp_path, render_pdf=False)
    tokens = MarkdownIt().parse(Path(output["markdown"]).read_text(encoding="utf-8"))
    assert sum(token.type == "heading_open" and token.tag == "h2" for token in tokens) == 4
    assert not any(
        child.type in {"image", "link_open"} for token in tokens for child in (token.children or [])
    )


@pytest.mark.parametrize(
    "usage",
    [None, {"requests": 2, "input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "estimated": True}],
)
def test_pdf_token_usage_distinguishes_measured_estimated_and_unknown(tmp_path, usage):
    output = write_review(review(), tmp_path, token_usage=usage)
    text = "\n".join(page.extract_text() for page in PdfReader(output["pdf"]).pages)
    if usage is None:
        assert "Unavailable" in text
        assert "Input 0 | Output 0 | Total 0" not in text
    else:
        assert "Input 10 | Output 20 | Total 30 (estimated)" in " ".join(text.split())
