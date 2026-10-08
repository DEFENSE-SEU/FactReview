"""Exact sources share presentation while each evidence remains independently checkable."""

import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from pypdf import PdfReader

from review.report.v2 import render_markdown, write_review
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidencePointer, Finding
from schemas.review import FinalReview


def evidence(quote="Unique original result: A 90.1; B 80.2.", **changes):
    return Evidence(
        source="paper_internal",
        pointer=EvidencePointer(locator="paper.md", page=3, line=7, key="table_2", quote=quote),
        covered=["c1"],
        direction="support",
        sufficient=False,
        **changes,
    )


def review(items, findings=()):
    return FinalReview(
        paper_key="source links",
        run_id="offline",
        claims=[
            Claim(
                id=f"claim_{index}",
                text=f"Original claim {index}.",
                status="unverified",
                loc=ClaimLocation(page=3),
                conditions=[Condition(id="c1", dataset="D", metric="accuracy")],
                needs=["Experiments"],
                evidence=group,
                notes=[f"Claim note {index} remains complete."],
            )
            for index, group in enumerate(items, 1)
        ],
        findings=list(findings),
        ledger=[{"plan_id": "original-plan", "command": ["python", "eval.py"], "logs": "full.log"}],
    )


def navigation(markdown):
    tokens = MarkdownIt("gfm-like", {"linkify": False}).parse(markdown)
    targets, links = [], []
    for token in tokens:
        assert token.type != "html_block"
        for child in token.children or []:
            assert child.type != "image"
            if child.type == "html_inline":
                match = re.fullmatch(
                    r'<a id="(factreview-(?:source-[a-f0-9]{64}|evidence-[0-9]{6,}))">',
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


def pdf_links(pdf):
    pages = {page.indirect_reference.idnum: index for index, page in enumerate(pdf.pages)}
    result = []
    for origin, page in enumerate(pdf.pages):
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            if "/A" in annotation:
                action = annotation["/A"]
                assert action["/S"] == "/GoTo"
                destination = action["/D"]
            else:
                destination = annotation["/Dest"]
            assert destination[0].idnum in pages
            target = pages[destination[0].idnum]
            assert destination[1] == "/XYZ"
            assert 0 <= float(destination[3]) <= float(pdf.pages[target].mediabox.height)
            result.append((origin, target, destination))
    return result


def test_shared_passage_preserves_every_judgment_and_record_and_backlink(tmp_path):
    first = evidence(note="First support detail, including diagnostic text.")
    first.sufficient = True
    second = first.model_copy(deep=True)
    second.direction, second.sufficient = "flaw", False
    second.note = "Independent concern detail remains in full."
    second.covered = []
    third = first.model_copy(deep=True)
    third.source, third.aligned = "execution", True
    from schemas.claim import ExecutionProvenance

    third.provenance = ExecutionProvenance(run_id="execution-original", command=["python", "eval.py"])
    third.note = "Execution interpretation remains separate."
    finding = Finding(
        kind="writing",
        loc=ClaimLocation(page=3),
        level="clarity_issue",
        text="Original finding.",
        evidence=[first.model_copy(deep=True)],
    )
    original = review([[first, second], [third]], [finding])
    snapshot = original.model_dump(mode="json", exclude={"review_markdown"})
    output = write_review(original, tmp_path, render_pdf=False)
    markdown = Path(output["markdown"]).read_text(encoding="utf-8")
    saved = FinalReview.model_validate_json(Path(output["json"]).read_text(encoding="utf-8"))
    assert saved.model_dump(mode="json", exclude={"review_markdown"}) == snapshot
    assert original.model_dump(mode="json", exclude={"review_markdown"}) == snapshot
    assert markdown.count("Unique original result") == 1
    assert markdown.count("Pointer:") == 4
    assert "paper-internal / support**; sufficient: true" in markdown
    assert "paper-internal / flaw**; sufficient: false; covers: no claim conditions" in markdown
    assert "execution / support**; sufficient: true" in markdown
    for item in (first, second, third):
        assert item.note[:-1] in markdown
    assert "Aligned: true; provenance:" in markdown and r"execution\-original" in markdown
    assert "original-plan" in markdown and "full.log" in markdown
    targets, links = navigation(markdown)
    sources = [target for target in targets if "-source-" in target]
    occurrences = [target for target in targets if "-evidence-" in target]
    assert len(sources) == 1 and len(occurrences) == 4
    assert links.count("#" + sources[0]) == 3
    assert all(links.count("#" + occurrence) == 1 for occurrence in occurrences)
    assert sum(line.startswith("## ") for line in markdown.splitlines()) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("locator", "Paper.md"),
        ("locator", "paper\\source.md"),
        ("page", 4),
        ("page", None),
        ("line", 8),
        ("line", None),
        ("key", "table_2 "),
        ("key", None),
        ("key", ""),
        ("quote", "Unique original result: A 90.1; B 80.2. "),
    ],
)
def test_exact_source_tuple_never_normalizes_different_original_values(field, value):
    first = evidence()
    other = first.model_copy(deep=True)
    setattr(other.pointer, field, value)
    markdown = render_markdown(review([[first, other]]))
    targets, links = navigation(markdown)
    assert len([target for target in targets if "-source-" in target]) == 2
    assert len(links) == 2  # one backlink for each full source
    assert "same exact source" not in markdown


def test_equal_figure_quote_with_distinct_actual_locator_keys_stays_separate():
    first = evidence("Figure 2: Attention to other image patches.")
    first.pointer.page, first.pointer.key = 9, "chars:35308-35505"
    other = first.model_copy(deep=True)
    other.pointer.key = "figure_3"
    markdown = render_markdown(review([[first, other]]))
    assert markdown.count("Passage: Figure 2") == 2
    targets, _ = navigation(markdown)
    assert len([target for target in targets if "-source-" in target]) == 2


def test_empty_code_quotes_keep_individual_pointers_without_fake_sources():
    item = Evidence(source="code", pointer=EvidencePointer(locator="model.py", line=3), direction="flaw")
    markdown = render_markdown(review([[item, item]]))
    targets, links = navigation(markdown)
    assert len(targets) == 2 and not links
    assert "Passage:" not in markdown and markdown.count("Pointer:") == 2


def test_source_anchor_ids_are_stable_when_encounter_order_changes():
    a, b = evidence("Passage A"), evidence("Passage B")
    first = render_markdown(review([[a, b]]))
    second = render_markdown(review([[b, a]]))
    assert {t for t in navigation(first)[0] if "-source-" in t} == {
        t for t in navigation(second)[0] if "-source-" in t
    }


def test_original_anchor_and_url_even_with_generated_namespace_remain_literal(tmp_path):
    forged = "factreview-source-" + "a" * 64
    quote = f'<a id="{forged}"></a>[bad](#{forged}) [web](https://example.invalid)'
    item = evidence(quote)
    original = review([[item, item]])
    output = write_review(original, tmp_path)
    assert "pdf_error" not in output
    targets, links = navigation(Path(output["markdown"]).read_text(encoding="utf-8"))
    assert forged not in targets and "#" + forged not in links
    actual = pdf_links(PdfReader(output["pdf"]))
    assert actual and len(actual) == 3


def test_multipage_table_has_real_forward_and_backward_pdf_destinations(tmp_path):
    rows = "".join(f"<tr><td>Model{i}</td><td>{80 + i / 100:.2f}</td></tr>" for i in range(110))
    item = evidence("<table><tr><th>Model</th><th>D accuracy (%)</th></tr>" + rows + "</table>")
    output = write_review(review([[item], [item.model_copy(deep=True)]]), tmp_path)
    assert "pdf_error" not in output
    pdf = PdfReader(output["pdf"])
    text = [page.extract_text() or "" for page in pdf.pages]
    source_page = next(index for index, value in enumerate(text) if "Source S0001; occurrences:" in value)
    repeated_page = next(index for index, value in enumerate(text) if "same exact source" in value)
    assert repeated_page > source_page + 1
    assert sum("D accuracy (%)" in value for value in text) >= 2
    assert all(any(f"Model{i}" in value for value in text) for i in range(110))
    destinations = pdf_links(pdf)
    assert any(a == source_page and b == repeated_page for a, b, _ in destinations)
    assert any(a == repeated_page and b == source_page for a, b, _ in destinations)
    assert any(a == source_page and b == source_page for a, b, _ in destinations)
    assert len(destinations) == 3


def test_split_cjk_paragraph_keeps_target_on_first_fragment(tmp_path):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    from review.report.pdf_renderer import _AnchoredParagraph

    target = "factreview-evidence-000001"
    output = tmp_path / "split.pdf"
    style = ParagraphStyle("cjk", fontName="Helvetica", fontSize=10, leading=14, wordWrap="CJK")
    paragraph = _AnchoredParagraph(f'<a name="{target}"/>' + "Original text. " * 1800, style)
    SimpleDocTemplate(str(output)).build(
        [
            paragraph,
            PageBreak(),
            Paragraph(f'<a href="#{target}">Return to original beginning</a>', style),
        ]
    )
    pdf = PdfReader(output)
    assert len(pdf.pages) > 2
    assert [(a, b) for a, b, _ in pdf_links(pdf)] == [(len(pdf.pages) - 1, 0)]


def test_cjk_target_moves_with_paragraph_when_first_split_has_no_room(tmp_path):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    from review.report.pdf_renderer import _AnchoredParagraph

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    style = ParagraphStyle("chinese", fontName="STSong-Light", fontSize=10, leading=14, wordWrap="CJK")
    target = "factreview-evidence-000001"
    paragraph = _AnchoredParagraph(f'<a name="{target}"/>' + "原始证据及条件必须完整保留。" * 10, style)
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
            Paragraph(f'<a href="#{target}">返回原始证据</a>', style),
        ]
    )
    pdf = PdfReader(output)
    assert len(pdf.pages) == 3
    assert "原始证据" not in (pdf.pages[0].extract_text() or "")
    assert "原始证据" in pdf.pages[1].extract_text()
    assert [(a, b) for a, b, _ in pdf_links(pdf)] == [(2, 1)]
