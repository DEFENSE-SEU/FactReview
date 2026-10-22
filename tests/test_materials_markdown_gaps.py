"""One synthetic combined contract for preprocessing Markdown source gaps."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import fitz
import pytest
import requests
from PIL import Image

from preprocessing import materials as module
from preprocessing.parse.mineru_adapter import MineruAdapter, MineruParseResult
from schemas.materials import SharedMaterials
from screening import claims


def test_complete_markdown_gaps_preserve_original_source_contract(tmp_path, monkeypatch):
    calls = {"mineru": 0, "llm": 0, "retrieval": 0, "docker": 0, "pdf_text": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"Unexpected {name} call")
        return fail

    monkeypatch.setattr(MineruAdapter, "parse_pdf", forbidden("mineru"))
    monkeypatch.setattr(claims, "llm_json", forbidden("llm"))
    monkeypatch.setattr(requests.Session, "request", forbidden("retrieval"))
    monkeypatch.setattr(subprocess, "run", forbidden("docker"))
    monkeypatch.setattr(subprocess, "Popen", forbidden("docker"))

    class SyntheticImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def resize(self, *args):
            return self

        def save(self, path, **kwargs):
            Path(path).write_bytes(b"synthetic-render-only")

    class SyntheticPage:
        rect = fitz.Rect(0, 0, 600, 800)

        def get_pixmap(self, **kwargs):
            return SyntheticImage()

        get_text = forbidden("pdf_text")

    class SyntheticPDF(list):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: SyntheticPDF([SyntheticPage()]))
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: SyntheticImage())
    paper_pdf = tmp_path / "synthetic.pdf"
    paper_pdf.write_bytes(b"synthetic input; never decoded as PDF")
    original_blocks = module._blocks
    captures = []

    def capture(markdown, rows, issues):
        blocks = original_blocks(markdown, rows, issues)
        captures.append((blocks, [block.model_dump(mode="json") for block in blocks]))
        return blocks

    monkeypatch.setattr(module, "_blocks", capture)

    def build(markdown, rows, name="out"):
        parsed = MineruParseResult(
            markdown=markdown, content_list=copy.deepcopy(rows), image_files=None,
            batch_id="synthetic", raw_result=None, provider="mock",
        )
        result = module.build_materials(
            parsed, paper_pdf=paper_pdf, output_dir=tmp_path / name, paper_key="synthetic"
        )
        assert parsed.content_list == rows
        assert result.markdown == markdown
        assert Path(result.markdown_path).read_bytes() == markdown.encode("utf-8")
        assert json.loads(Path(result.content_list_path).read_text(encoding="utf-8")) == rows
        restored = SharedMaterials.model_validate_json(
            (tmp_path / name / "materials.json").read_text(encoding="utf-8")
        )
        assert restored == result
        return result

    missing = "Missing body cites Figure 1 and Table 1 [1]."
    appendix = "Missing appendix claim."
    partial = "Partial prefix. Remaining qualifier."
    table = "| Column | Value |\n|---|---|\n| Demo | 1 |"
    markdown = "\n\n".join([
        "# Paper", "Retained body cites Figure 1.", missing, "Figure 1: Caption.",
        f"Table 1: Caption.\n{table}", "# References", "[1] Example reference.",
        "# Appendix", appendix, partial,
    ])
    rows = [
        {"type": "text", "text": "Paper", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Retained body cites Figure 1.", "page_idx": 0},
        {"type": "image", "image_caption": ["Figure 1: Caption."], "page_idx": 0, "bbox": [100, 100, 300, 300]},
        {"type": "table", "table_caption": "Table 1: Caption.", "table_body": table,
         "page_idx": 0, "bbox": [400, 100, 600, 300]},
        {"type": "ref_text", "text": "[1] Example reference.", "page_idx": 0},
        {"type": "text", "text": "Partial prefix.", "page_idx": 0},
    ]
    result = build(markdown, rows)
    added = [block for block in result.blocks if block.id.startswith("markdown_gap:")]
    assert [block.text for block in added] == [missing, "# References", "# Appendix", appendix]
    originals, frozen = captures[-1]
    kept = [block for block in result.blocks if not block.id.startswith("markdown_gap:")]
    assert [block.model_dump(mode="json") for block in kept] == frozen
    assert all(current is before for current, before in zip(kept, originals, strict=True))
    assert [block.text for block in result.blocks] == [
        "Paper", rows[1]["text"], missing, rows[2]["image_caption"][0],
        f"Table 1: Caption.\n{table}", "# References", rows[4]["text"],
        "# Appendix", appendix, "Partial prefix.",
    ]
    assert [block.id for block in result.bibliography] == ["block_5"]
    assert result.figures[0].block_id == "block_3" and result.figures[0].parser_row_index == 3
    assert result.tables[0].block_id == "block_4" and result.tables[0].parser_row_index == 4
    assert [ref.text for ref in result.figures[0].references] == [rows[1]["text"], missing]
    assert [ref.text for ref in result.tables[0].references] == [missing]
    for ref in [*result.figures[0].references, *result.tables[0].references]:
        assert markdown[ref.loc.char_start:ref.loc.char_end] == ref.text
    assert [block.loc.section for block in added] == [
        "section_1: Paper", "section_2: References", "section_3: Appendix", "section_3: Appendix"
    ]
    for block in added:
        assert block.loc.page is None
        assert block.text == markdown[block.loc.char_start:block.loc.char_end]
        expected_id = f"markdown_gap:{block.loc.char_start}:{block.loc.char_end}:{hashlib.sha256(block.text.encode()).hexdigest()}"
        assert block.id == expected_id
        assert any(block.id in issue and "page unavailable" in issue for issue in result.issues)
        assert claims._location(block, block.text, markdown) == block.loc
    assert added[0].citations == ["[1]"]
    assert any("partial_overlap" in issue for issue in result.issues)
    assert all(block.text != "Remaining qualifier." for block in result.blocks)
    assert partial in result.markdown
    fake = added[0].model_copy(deep=True)
    fake.loc.char_start += 1
    with pytest.raises(claims.ClaimExtractionError, match="inconsistent character offsets"):
        claims._location(fake, missing, markdown)
    before_material = result.model_dump(mode="json")
    before_material["blocks"] = frozen
    before_digest = hashlib.sha256(json.dumps(before_material, sort_keys=True).encode()).hexdigest()
    after_digest = hashlib.sha256(json.dumps(result.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
    assert before_digest != after_digest
    assert result.model_dump(mode="json")["blocks"] == [b.model_dump(mode="json") for b in result.blocks]

    merge = module._materialize_markdown_gaps

    def merged(text, source_rows):
        issues = []
        blocks = original_blocks(text, source_rows, issues)
        saved = [block.model_dump(mode="json") for block in blocks]
        output = merge(text, blocks, issues)
        retained = [block for block in output if block.id in {b.id for b in blocks}]
        assert retained == blocks
        assert [block.model_dump(mode="json") for block in blocks] == saved
        assert all(left is right for left, right in zip(retained, blocks, strict=True))
        return output, issues, blocks

    repeated = "# Paper\n\nRepeated claim.\n\nRepeated claim.\n\nTail."
    repeated_rows = [rows[0], {"type": "text", "text": "Tail.", "page_idx": 0}]
    repeated_output, _, _ = merged(repeated, repeated_rows)
    repeats = [b for b in repeated_output if b.text == "Repeated claim."]
    assert len(repeats) == 2 and repeats[0].id != repeats[1].id
    assert repeats[0].loc.char_end < repeats[1].loc.char_start
    assert [b.model_dump() for b in merged(repeated, repeated_rows)[0]] == [b.model_dump() for b in repeated_output]
    ambiguous = "# Paper\n\nRepeat. Repeat.\n\nTail."
    ambiguous_block = next(b for b in merged(ambiguous, repeated_rows)[0] if b.text == "Repeat. Repeat.")
    with pytest.raises(claims.ClaimExtractionError, match="ambiguous"):
        claims._location(ambiguous_block, "Repeat.", ambiguous)

    unknown_rows = [rows[0], {"type": "text", "text": "Parser-only section", "text_level": 1, "page_idx": 0}, repeated_rows[1]]
    unknown_output, unknown_issues, unknown_originals = merged("# Paper\n\nMissing.\n\nTail.", unknown_rows)
    assert unknown_output == unknown_originals
    assert any("unknown_insertion_gap" in issue for issue in unknown_issues)
    metadata_rows = [rows[0], {"type": "header", "text": "Parser-only header", "page_idx": 0}, repeated_rows[1]]
    metadata_output, _, _ = merged("# Paper\n\nMissing.\n\nTail.", metadata_rows)
    assert any(b.text == "Missing." and b.loc.page is None for b in metadata_output)
    out_of_order = list(reversed(original_blocks(repeated, repeated_rows, [])))
    disorder_issues = []
    assert merge(repeated, out_of_order, disorder_issues) is out_of_order
    assert any("unknown_insertion_gap" in issue for issue in disorder_issues)

    leading, _, _ = merged("Missing opening.\n\nRetained.", [{"type": "text", "text": "Retained.", "page_idx": 0}])
    assert leading[0].loc.section is None and leading[0].loc.page is None
    assert claims._location(leading[0], "Missing opening.", "Missing opening.\n\nRetained.") == leading[0].loc
    page_heading, _, _ = merged("# Page 9\n\nMissing.\n\nTail.", [repeated_rows[1]])
    assert all(b.loc.page is None for b in page_heading if b.id.startswith("markdown_gap:"))
    crlf = "# Paper\r\n\r\nEntire first line.\r\nEntire second line.\r\n\r\nTail."
    crlf_block = next(b for b in merged(crlf, repeated_rows)[0] if b.id.startswith("markdown_gap:"))
    assert crlf_block.text == "Entire first line.\r\nEntire second line."
    assert crlf[crlf_block.loc.char_start:crlf_block.loc.char_end] == crlf_block.text

    fenced = "# Paper\n\n```python\n# References\nvalue = 1\n```\n\nMissing after code.\n\nTail."
    fenced_rows = [rows[0], {"type": "code", "code_body": "```python\n# References\nvalue = 1\n```", "page_idx": 0}, repeated_rows[1]]
    fenced_output, _, _ = merged(fenced, fenced_rows)
    assert not any(b.kind == "heading" and "References" in b.text for b in fenced_output)
    fenced_missing = next(b for b in fenced_output if b.text == "Missing after code.")
    assert fenced_missing.loc.section == "section_1: Paper"
    for container in ["<!--\n# References\n\nHidden content.\n-->", "<table>\n# References\n\n<tr><td>Demo</td></tr>\n</table>", "$$\n# References\n\nx = 1\n$$", r"\[" + "\n# References\n\nx = 1\n" + r"\]"]:
        opaque = f"# Paper\n\n{container}\n\nMissing after container.\n\nTail."
        opaque_output, opaque_issues, _ = merged(opaque, repeated_rows)
        assert not any(b.kind == "heading" and "References" in b.text for b in opaque_output)
        assert not any(container in b.text for b in opaque_output)
        assert any("unrecognized_markdown_structure" in issue for issue in opaque_issues)
        assert container in opaque
        assert next(b for b in opaque_output if b.text == "Missing after container.").loc.section == "section_1: Paper"
    nested_html = "# Paper\n\n<div>\n<div>\nInner.\n</div>\n# References\nHidden reference.\n</div>\n\nTail."
    nested_result = build(nested_html, repeated_rows, "nested-html")
    assert [block.id for block in nested_result.blocks] == ["block_1", "block_2"]
    assert nested_result.bibliography == []
    assert nested_result.markdown == nested_html
    assert any("unrecognized_markdown_structure" in issue for issue in nested_result.issues)
    for content in [
        "<div>\n<!-- </div> -->\n# References\nHidden reference.\n</div>",
        '<div data-note="</div>">\n# References\nHidden reference.\n</div>',
        '<div>\nprint("</div>")\n# References\nHidden reference.\n</div>',
        "<div>\n<span>Inner.</div>\n# References\nHidden reference.\n</span>",
        "<div>\n<script>const literal = '</div>';</script>\n# References\nHidden reference.\n</div>",
        "<div/>\n# References\nHidden reference.\n</div>",
    ]:
        text = f"# Paper\n\n{content}\n\nTail."
        boundary_result = build(text, repeated_rows, "html-boundary")
        assert [block.id for block in boundary_result.blocks] == ["block_1", "block_2"]
        assert boundary_result.bibliography == []
        assert boundary_result.markdown == text
        assert any("unrecognized_markdown_structure" in issue for issue in boundary_result.issues)

    closed_wrappers = [
        "<?mock\n# References\nHidden reference.\n?>",
        "<![CDATA[\n# References\nHidden reference.\n]]>",
        '<!DOCTYPE mock [\n<!ENTITY demo ">">\n# References\nHidden reference.\n]>',
        '<!DOCTYPE mock [\n<!-- ]> -->\n<!ENTITY demo ">">\n# References\nHidden reference.\n]>',
        "<!CUSTOM\n# References\nHidden reference.\n>",
    ]
    for wrapper in [
        *closed_wrappers,
        "<?mock\n\n# References\nHidden reference.",
        "<![CDATA[\n\n# References\nHidden reference.",
        '<!DOCTYPE mock [\n<!ENTITY demo ">">\n\n# References\nHidden reference.',
        "<!\n\n# References\nHidden reference.",
        "</unknown\n\n# References\nHidden reference.\n>",
        "<![UNKNOWN[\n\n# References\nHidden reference.\n]]>",
        '<!DOCTYPE mock [\n<!ENTITY demo ">">\n# References\nHidden reference.\n>',
    ]:
        text = f"# Paper\n\n{wrapper}\n\nTail."
        wrapper_result = build(text, repeated_rows, "unknown-html-wrapper")
        originals, saved = captures[-1]
        assert [block.id for block in wrapper_result.blocks] == ["block_1", "block_2"]
        assert wrapper_result.bibliography == []
        assert wrapper_result.markdown == text
        assert any("unrecognized_markdown_structure" in issue for issue in wrapper_result.issues)
        assert [block.model_dump(mode="json") for block in wrapper_result.blocks] == saved
        assert all(current is original for current, original in zip(wrapper_result.blocks, originals, strict=True))

    for wrapper in closed_wrappers:
        body = "Complete body after closed wrapper."
        text = f"# Paper\n\n{wrapper}\n\n{body}\n\nTail."
        available = build(text, repeated_rows, "closed-wrapper-body")
        originals, saved = captures[-1]
        retained = [block for block in available.blocks if not block.id.startswith("markdown_gap:")]
        source = next(block for block in available.blocks if block.text == body)
        assert [block.model_dump(mode="json") for block in retained] == saved
        assert all(current is original for current, original in zip(retained, originals, strict=True))
        assert [block.text for block in available.blocks] == ["Paper", body, "Tail."]
        assert available.bibliography == []
        assert source.loc.page is None and source.loc.section == "section_1: Paper"
        assert text[source.loc.char_start:source.loc.char_end] == body
        assert claims._location(source, body, text) == source.loc
        assert any("unrecognized_markdown_structure" in issue for issue in available.issues)
    comparison, _, _ = merged("# Paper\n\n< 5 is a comparison.\n\nComplete body.\n\nTail.", repeated_rows)
    assert [block.text for block in comparison] == ["Paper", "< 5 is a comparison.", "Complete body.", "Tail."]

    subset_pi_wrappers = [
        "<!DOCTYPE mock [\n<?mock ]>\n# References\nHidden reference.\n?>\n]>",
        '<!DOCTYPE mock [\n<?mock "\' <!-- [ ]>\n# References\nHidden reference.\n?>\n]>',
        "<!DOCTYPE mock [\n<?mock ?\n>\n# References\nHidden reference.\n?>\n]>",
        '<!DOCTYPE mock [\n<!ENTITY demo "<?mock ]> <!--">\n# References\nHidden reference.\n]>',
        "<!DOCTYPE mock [\n<!-- <?mock ]> -->\n# References\nHidden reference.\n]>",
    ]
    for wrapper in [
        *subset_pi_wrappers,
        "<!DOCTYPE mock [\n<?mock ]>\n# References\nHidden reference.\n]>",
        '<!DOCTYPE mock [\n<?mock "\' <!-- [ ]>\n# References\nHidden reference.\n]>',
    ]:
        text = f"# Paper\n\n{wrapper}\n\nTail."
        subset = build(text, repeated_rows, "subset-pi")
        originals, saved = captures[-1]
        assert [block.id for block in subset.blocks] == ["block_1", "block_2"]
        assert subset.bibliography == []
        assert subset.markdown == text
        assert any("unrecognized_markdown_structure" in issue for issue in subset.issues)
        assert [block.model_dump(mode="json") for block in subset.blocks] == saved
        assert all(current is original for current, original in zip(subset.blocks, originals, strict=True))
    for wrapper in subset_pi_wrappers:
        body = "Complete body after subset PI."
        text = f"# Paper\n\n{wrapper}\n\n{body}\n\nTail."
        after_subset = build(text, repeated_rows, "closed-subset-pi-body")
        originals, saved = captures[-1]
        retained = [block for block in after_subset.blocks if not block.id.startswith("markdown_gap:")]
        source = next(block for block in after_subset.blocks if block.text == body)
        assert [block.model_dump(mode="json") for block in retained] == saved
        assert all(current is original for current, original in zip(retained, originals, strict=True))
        assert [block.text for block in after_subset.blocks] == ["Paper", body, "Tail."]
        assert after_subset.bibliography == []
        assert source.loc.page is None and source.loc.section == "section_1: Paper"
        assert text[source.loc.char_start:source.loc.char_end] == body
        assert claims._location(source, body, text) == source.loc
        assert any("unrecognized_markdown_structure" in issue for issue in after_subset.issues)

    healthy = "# Paper\n\n- First item.\n- Second item.\n\nFigure 1: Caption.\n\n![](images/source.png)\n\n# References\n\n[1] Ref.\n\n# Appendix\n\nTail."
    healthy_rows = [rows[0], {"type": "list", "list_items": ["First item.", "Second item."], "page_idx": 0}, rows[2],
                    {"type": "text", "text": "References", "text_level": 1, "page_idx": 0},
                    {"type": "ref_text", "text": "[1] Ref.", "page_idx": 0},
                    {"type": "text", "text": "Appendix", "text_level": 1, "page_idx": 0}, repeated_rows[1]]
    for text, source_rows, name in [(healthy, healthy_rows, "healthy"), ("# Paper\n\nFallback.", [], "empty")]:
        normal = build(text, source_rows, name)
        with monkeypatch.context() as old:
            old.setattr(module, "_materialize_markdown_gaps", lambda markdown, blocks, issues: blocks)
            baseline = build(text, source_rows, name)
        assert normal.model_dump(mode="json") == baseline.model_dump(mode="json")
        assert not any(b.id.startswith("markdown_gap:") for b in normal.blocks)
    assert calls == {name: 0 for name in calls}
