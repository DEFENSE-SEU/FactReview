import json
import re
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


def test_v2_pdf_preserves_windows_pointers_and_snake_case_identifiers(tmp_path):
    record = review()
    claim = record.claims[0]
    claim.id = "claim_003"
    claim.questions[0].claim_id = claim.id
    claim.text = "The source_repo uses relation_update."
    claim.loc.section = "method_details"
    claim.conditions[0].id = "relation_update"
    claim.evidence[0].covered = ["relation_update"]
    pointer = claim.evidence[0].pointer
    pointer.locator = r"C:\source_repo\model\compgcn_conv.py"
    pointer.quote = "return self.act(out), self.w_rel"
    output = write_review(record, tmp_path)
    assert "pdf_error" not in output
    text = "\n".join(page.extract_text() for page in PdfReader(output["pdf"]).pages)
    for literal in (
        pointer.locator,
        pointer.quote,
        claim.text,
        claim.id,
        claim.loc.section,
        "relation_update",
    ):
        assert literal in text


def test_literal_pdf_mode_preserves_code_and_keeps_explicit_math_and_legacy_default():
    from review.report.pdf_renderer import _markdown_parser, _render_markdown_inline_children

    source = r"ordinary_name and `C:\source_repo\model_file.py` with $x_y$"
    children = _markdown_parser().parse(source)[1].children
    literal = _render_markdown_inline_children(children, inline_code_font="Courier", implicit_math=False)
    assert "ordinary_name" in literal
    assert r"C:\source_repo\model_file.py" in literal
    assert "x<sub>y</sub>" in literal
    legacy = _render_markdown_inline_children(children, inline_code_font="Courier")
    assert "ordinary<sub>n</sub>ame" in legacy
    assert "x<sub>y</sub>" in legacy


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
    children = [child for token in tokens for child in token.children or []]
    assert not any(child.type == "image" for child in children)
    targets = {
        match.group(1)
        for child in children
        if child.type == "html_inline"
        and (
            match := re.fullmatch(
                r'<a id="(factreview-(?:source-[a-f0-9]{64}|evidence-[0-9]{6,}))">', child.content
            )
        )
    }
    assert all(
        child.attrGet("href") in {"#" + target for target in targets}
        for child in children
        if child.type == "link_open"
    )
    assert all(
        "example.invalid" not in (child.attrGet("href") or "")
        for child in children
        if child.type == "link_open"
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
