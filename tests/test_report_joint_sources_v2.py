"""One judgment retains every source usage across serialized and navigable reports."""

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from pypdf import PdfReader

from assessment import assess_claim
from review.report import v2
from review.report.v2 import render_markdown, write_review
from review.teaser.v2 import write_teaser
from schemas.claim import AuthorQuestion, EvidenceNeed, EvidencePointer
from schemas.materials import SharedMaterials
from schemas.review import FinalReview
from tests.test_report_source_links_v2 import evidence, pdf_links, review
from verification.contracts import BranchResult
from verification.dispatch import VerificationResult, verify_claims


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Report tests must not call external services or processes")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def pointer(quote="Metric definition: accuracy is the fraction of correct predictions.", **changes):
    return EvidencePointer(
        **{"locator": "paper.md", "page": 4, "key": "chars:70-140", "quote": quote, **changes}
    )


def joint(*additional, **changes):
    return evidence(additional_pointers=list(additional) or [pointer()], **changes)


def navigation(markdown):
    targets, links = [], []
    for token in MarkdownIt("gfm-like", {"linkify": False}).parse(markdown):
        assert token.type != "html_block"
        for child in token.children or []:
            assert child.type != "image"
            if child.type == "html_inline":
                match = re.fullmatch(
                    r'<a id="(factreview-(?:source-[a-f0-9]{64}|evidence-[0-9]{6,}(?:-source-[0-9]{2,})?))">',
                    child.content,
                )
                assert match or child.content == "</a>"
                if match:
                    targets.append(match.group(1))
            if child.type == "link_open":
                links.append(child.attrGet("href"))
    assert len(targets) == len(set(targets))
    assert set(links) <= {"#" + target for target in targets}
    return targets, links


def test_old_json_adds_only_canonical_empty_field_and_preserves_literal_md_pdf(tmp_path, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 9, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(v2, "datetime", Frozen)
    item = evidence(note="Legacy detail preserved.")
    original = review([[item, item.model_copy(deep=True)]]).model_dump(mode="json")
    for row in original["claims"][0]["evidence"]:
        row.pop("additional_pointers")
    old = tmp_path / "old.json"
    old.write_text(json.dumps(original), encoding="utf-8")
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    restored = FinalReview.model_validate_json(old.read_text(encoding="utf-8"))
    canonical = restored.model_dump(mode="json")
    for row in canonical["claims"][0]["evidence"]:
        assert row.pop("additional_pointers") == []
    assert canonical == original
    outputs = write_review(restored, tmp_path / "rendered")
    assert "pdf_error" not in outputs
    fixtures = Path(__file__).parent / "fixtures"
    assert Path(outputs["markdown"]).read_text(encoding="utf-8") == (
        fixtures / "report_single_source_v2.md"
    ).read_text(encoding="utf-8")
    pdf_text = " ".join(" ".join(p.extract_text() or "" for p in PdfReader(outputs["pdf"]).pages).split())
    assert pdf_text == (fixtures / "report_single_source_v2_pdf.txt").read_text(encoding="utf-8")
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before


def test_joint_source_usages_share_across_primary_and_additional_without_repeating_judgment(tmp_path):
    first = joint(
        pointer(),
        pointer("Protocol uses the same test split.", page=5, key="protocol"),
        note="Joint detail remains complete.",
    )
    first.sufficient = True
    second = evidence(note="Independent second judgment.")
    second.pointer = first.additional_pointers[0].model_copy(deep=True)
    second.additional_pointers = [first.pointer.model_copy(deep=True)]
    original = review([[first], [second]])
    original.claims[0].questions = [
        AuthorQuestion(text="Clarify the protocol?", reason="Original reason", claim_id="claim_1")
    ]
    snapshot = original.model_dump(mode="json", exclude={"review_markdown"})
    outputs = write_review(original, tmp_path, render_pdf=False)
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    saved = FinalReview.model_validate_json(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert saved.model_dump(mode="json", exclude={"review_markdown"}) == snapshot
    assert original.model_dump(mode="json", exclude={"review_markdown"}) == snapshot
    assert sum(len(c.evidence) for c in saved.claims) == 2
    assert markdown.count("sufficient:") == 2 and markdown.count("covers:") == 2
    assert markdown.count("Evidence E") == 2 and markdown.count("Additional pointer") == 3
    for literal in (
        "Joint detail remains complete",
        "Independent second judgment",
        "Clarify the protocol?",
        "Original reason",
        "Claim note 1 remains complete",
    ):
        assert markdown.count(literal) == 1
    readable = "\n".join(
        "".join(child.content for child in token.children or [])
        for token in MarkdownIt("gfm-like", {"linkify": False}).parse(markdown)
    )
    for item in (first.pointer, *first.additional_pointers):
        assert readable.count(item.quote) == 1
    targets, links = navigation(markdown)
    sources = [t for t in targets if t.startswith("factreview-source-")]
    usages = [t for t in targets if t.startswith("factreview-evidence-")]
    assert len(sources) == 3 and len(usages) == 5
    assert all(links.count("#" + target) == 1 for target in usages)
    assert "#factreview-evidence-000001-source-02" in links
    assert "#factreview-evidence-000002-source-02" in links
    assert sum(link in {"#" + source for source in sources} for link in links) == 2
    assert sum(line.startswith("## ") for line in markdown.splitlines()) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("locator", "Paper.md"),
        ("page", 8),
        ("line", 9),
        ("key", "different"),
        ("quote", "Unique original result: A 90.1; B 80.2. "),
    ],
)
def test_joint_sources_deduplicate_only_full_exact_pointer_tuple(field, value):
    first = evidence()
    other = first.pointer.model_copy(update={field: value}, deep=True)
    first.additional_pointers = [other]
    markdown = render_markdown(review([[first]]))
    targets, _ = navigation(markdown)
    assert len([t for t in targets if t.startswith("factreview-source-")]) == 2
    assert "same exact source" not in markdown


def test_additional_html_table_stays_complete_and_shared_with_later_primary_in_pdf(tmp_path):
    rows = "".join(f"<tr><td>Model{i}</td><td>{80 + i / 100:.2f}</td></tr>" for i in range(110))
    table = pointer(
        "<table><tr><th>Model</th><th>D accuracy (%)</th></tr>" + rows + "</table>", key="table_2"
    )
    first = joint(table, note="One joint interpretation.")
    second = evidence(note="A separate interpretation.")
    second.pointer = table.model_copy(deep=True)
    outputs = write_review(review([[first], [second]]), tmp_path)
    assert "pdf_error" not in outputs
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    navigation(markdown)
    pdf = PdfReader(outputs["pdf"])
    text = [page.extract_text() or "" for page in pdf.pages]
    source_page = next(i for i, value in enumerate(text) if "Source S0002; occurrences:" in value)
    usage_page = next(i for i, value in enumerate(text) if "Additional pointer 1:" in value)
    repeated_page = next(i for i, value in enumerate(text) if "same exact source" in value)
    assert repeated_page > source_page + 1
    assert all(any(f"Model{i}" in page for page in text) for i in range(110))
    assert sum("D accuracy (%)" in page for page in text) >= 2
    assert "<table>" not in " ".join(text)
    destinations = pdf_links(pdf)
    assert len(destinations) == 4
    assert any(a == source_page and b == usage_page for a, b, _ in destinations)
    assert any(a == source_page and b == repeated_page for a, b, _ in destinations)
    assert any(a == repeated_page and b == source_page for a, b, _ in destinations)
    restored = FinalReview.model_validate_json(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert restored.claims[0].evidence[0].additional_pointers[0] == table


def test_original_text_cannot_inject_new_usage_anchors_or_links(tmp_path):
    forged = "factreview-evidence-999999-source-99"
    quote = f'<a id="{forged}"></a>[wrong](#{forged}) ![remote](https://example.invalid/image)'
    item = joint(pointer(quote, key="literal"))
    outputs = write_review(review([[item]]), tmp_path)
    assert "pdf_error" not in outputs
    targets, links = navigation(Path(outputs["markdown"]).read_text(encoding="utf-8"))
    assert forged not in targets and "#" + forged not in links
    assert len(pdf_links(PdfReader(outputs["pdf"]))) == 2


def test_additional_usage_target_survives_split_paragraph_on_real_first_page(tmp_path):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    from review.report.pdf_renderer import _AnchoredParagraph

    target = "factreview-evidence-000001-source-02"
    style = ParagraphStyle("cjk", fontName="Helvetica", fontSize=10, leading=14, wordWrap="CJK")
    output = tmp_path / "split.pdf"
    SimpleDocTemplate(str(output)).build(
        [
            _AnchoredParagraph(f'<a name="{target}"/>' + "Exact source text. " * 1800, style),
            PageBreak(),
            Paragraph(f'<a href="#{target}">Return to additional source usage</a>', style),
        ]
    )
    pdf = PdfReader(output)
    assert len(pdf.pages) > 2
    assert [(a, b) for a, b, _ in pdf_links(pdf)] == [(len(pdf.pages) - 1, 0)]


def test_additional_usage_target_moves_with_whole_paragraph_to_next_page(tmp_path):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    from review.report.pdf_renderer import _AnchoredParagraph

    target = "factreview-evidence-000001-source-02"
    style = ParagraphStyle("cjk", fontName="Helvetica", fontSize=10, leading=14, wordWrap="CJK")
    paragraph = _AnchoredParagraph(f'<a name="{target}"/>' + "Additional original source. " * 8, style)
    paragraph.wrap(216, 216)
    assert paragraph.split(216, 6) == []
    output = tmp_path / "moved.pdf"
    SimpleDocTemplate(
        str(output), pagesize=(300, 300), topMargin=36, bottomMargin=36, leftMargin=36, rightMargin=36
    ).build(
        [
            Spacer(1, 210),
            paragraph,
            PageBreak(),
            Paragraph(f'<a href="#{target}">Return to additional source usage</a>', style),
        ]
    )
    pdf = PdfReader(output)
    assert len(pdf.pages) == 3
    assert "Additional original source" not in (pdf.pages[0].extract_text() or "")
    assert "Additional original source" in (pdf.pages[1].extract_text() or "")
    assert [(a, b) for a, b, _ in pdf_links(pdf)] == [(2, 1)]


@pytest.mark.parametrize("sufficient", [False, True])
def test_dispatch_assessment_report_roundtrip_retains_sources_without_expanding_records(tmp_path, sufficient):
    item = joint(
        pointer(),
        pointer("All evaluations use the same protocol.", key="protocol"),
        note="The complete joint explanation.",
    )
    item.sufficient = sufficient
    claim = review([[]]).claims[0]
    initial_notes = list(claim.notes)
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="fixture.pdf",
        markdown="Fixture paper.",
        markdown_path="fixture.md",
        content_list_path="fixture.json",
        provider="mock",
    )
    question = AuthorQuestion(
        claim_id=claim.id, text="Confirm the original setting?", reason="Retained question"
    )
    calls = []

    def branch(*args):
        calls.append(True)
        return BranchResult(evidence=[item], questions=[question], issues=["Retained limitation."])

    verified = asyncio.run(
        verify_claims(
            [claim], materials, tmp_path / "verification", branches={EvidenceNeed.EXPERIMENTS: branch}
        )
    )
    restored = VerificationResult.model_validate_json(
        (tmp_path / "verification/verification.json").read_text(encoding="utf-8")
    )
    assert restored == verified and calls == [True]
    assessed = assess_claim(restored.claims[0])
    assert assessed.status.value == ("supported" if sufficient else "unverified")
    assert assessed.evidence == [item] and assessed.conditions == claim.conditions
    assert assessed.questions == [question] and assessed.notes == [*initial_notes, "Retained limitation."]
    final = FinalReview(paper_key="joint fixture", run_id="offline", claims=[assessed])
    outputs = write_review(final, tmp_path / "report")
    assert "pdf_error" not in outputs
    saved = FinalReview.model_validate_json(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert saved.claims == final.claims
    text = " ".join(page.extract_text() or "" for page in PdfReader(outputs["pdf"]).pages)
    assert all(p.quote in text for p in [item.pointer, *item.additional_pointers])
    teaser = write_teaser(saved, tmp_path / "teaser")
    compact = json.loads(Path(teaser["json"]).read_text(encoding="utf-8"))["claims"]
    assert compact == [
        {
            "id": claim.id,
            "text": claim.text,
            "status": assessed.status.value,
            "source_types": ["paper_internal"],
        }
    ]
    assert "See the report" in Path(teaser["image"]).read_text(encoding="utf-8")
