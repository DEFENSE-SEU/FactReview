"""Large delivery diagnostics remain lossless without one giant PDF paragraph."""

import json
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from pypdf import PdfReader

from review.report.compact import _delivery_context_markdown
from review.report.v2 import _text, write_review
from schemas.review import FinalReview


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Presentation must not call external services or execution")

    monkeypatch.setattr("requests.sessions.Session.request", fail)
    monkeypatch.setattr("httpx.Client.send", fail)
    monkeypatch.setattr("subprocess.run", fail)
    monkeypatch.setattr("llm.client.llm_json", fail)


def reconstruct(markdown):
    result, entries = {}, []
    for token in MarkdownIt().parse(markdown):
        assert token.type not in {"html_block", "image"}
        if token.type == "inline":
            if token.content == "Delivery context (complete; ordered JSON entries):":
                continue
            assert all(child.type == "text" for child in token.children)
            row = json.loads("".join(child.content for child in token.children))
            entries.append(row)
            if "index" in row:
                assert row["field"] in result
                target = result[row["field"]]
                if "key" in row:
                    target = target[row["key"]]
                assert row["index"] == len(target)
                target.append(row["value"])
            else:
                assert row["field"] not in result
                result[row["field"]] = row["value"]
    return result, entries


def test_1660_original_ordered_issues_never_become_one_large_paragraph():
    issue = 'Original diagnostic with full quote, units and qualifier: "scope remains unresolved". ' * 12
    context = {
        "issues": [issue + str(i % 17) for i in range(1660)],
        "token_usage": {"requests": 19, "failed_requests": 2, "unavailable_usage_requests": 2},
        "anonymity_policy": None,
    }
    frozen = json.dumps(context, ensure_ascii=False)
    markdown = _delivery_context_markdown(context)
    actual, entries = reconstruct(markdown)
    assert json.dumps(actual, ensure_ascii=False) == frozen
    assert json.dumps(context, ensure_ascii=False) == frozen
    assert len(entries) == 1663
    assert [e["index"] for e in entries if "index" in e] == list(range(1660))
    assert len(frozen) > 1_000_000
    assert max(len(line) for line in markdown.splitlines()) < 2000
    assert max(len(t.content) for t in MarkdownIt().parse(markdown) if t.type == "inline") < 2000
    assert actual["issues"][0] == actual["issues"][17]


@pytest.mark.parametrize(
    "issues",
    [
        None,
        [],
        ["duplicate", "duplicate"],
        [
            "line 1\nline 2\r\t雪\\",
            '"quotes" <a href="bad">x</a> ```json\n{}\n```` author\'s literal &#x27; &quot;',
        ],
    ],
)
def test_entries_preserve_types_empty_containers_keys_newlines_and_markup(issues):
    context = {
        "issues": issues,
        "empty": {},
        "array": [],
        "null": None,
        "boolean": False,
        "count": 1,
        "fraction": 0.75,
        "nested": {"quotes": ['a"b', {}, None, True]},
        'key<>&"\\': "unmodified value",
    }
    markdown = _delivery_context_markdown(context)
    restored, _ = reconstruct(markdown)
    assert json.dumps(restored, ensure_ascii=False) == json.dumps(context, ensure_ascii=False)
    assert list(restored) == list(context)
    assert type(restored["boolean"]) is bool and type(restored["count"]) is int
    assert type(restored["fraction"]) is float
    assert restored["null"] is None


def test_public_delivery_manifest_exact_and_all_issue_literals_present_in_pdf(tmp_path):
    context = {
        "issues": [
            "Repeated source diagnostic.",
            "Repeated source diagnostic.",
            'Literal "quote" and <tag> with trailing qualifier.',
        ],
        "token_usage": {"requests": 3, "failed_requests": 1, "unavailable_usage_requests": 1},
        "anonymity_policy": "unspecified",
    }
    original = FinalReview(paper_key="context presentation", run_id="offline")
    frozen = original.model_dump(mode="json")
    outputs = write_review(original, tmp_path, presentation="layered", **context)
    assert not any(k.endswith("_error") for k in outputs)
    appendix = Path(outputs["appendix_markdown"]).read_text("utf-8")
    tail = appendix.split("Delivery context (complete; ordered JSON entries):", 1)[1]
    restored, _ = reconstruct(tail)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert restored == manifest["delivery_context"]
    for key, value in context.items():
        assert restored[key] == value
    assert original.model_dump(mode="json") == frozen
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(outputs["appendix_pdf"]).pages)
    assert '"index": 0' in pdf_text and '"index": 1' in pdf_text and '"index": 2' in pdf_text
    assert "trailing qualifier" in pdf_text


@pytest.mark.parametrize(
    "warnings", [[], ["duplicate", "duplicate", "another"], ["Reference lookup diagnostics. " * 650]]
)
def test_overview_keeps_accounting_facts_while_warning_literals_remain_complete(tmp_path, warnings):
    context = {
        "issues": ["other issue"],
        "token_usage": {
            "warnings": warnings,
            "requests": 6,
            "failed_requests": 2,
            "unavailable_usage_requests": 2,
            "input_tokens": 150,
        },
    }
    before = json.dumps(context, ensure_ascii=False)
    outputs = write_review(
        FinalReview(paper_key="warning scope", run_id="offline"),
        tmp_path,
        presentation="layered",
        render_pdf=False,
        **context,
    )
    main = Path(outputs["markdown"]).read_text("utf-8")
    appendix = Path(outputs["appendix_markdown"]).read_text("utf-8")
    assert f"Accounting warnings: {len(warnings)}." in main
    assert "2 model call attempt" in main and "unavailable for 2 call attempt" in main
    for warning in warnings:
        assert warning not in main
        assert _text(warning) in appendix
    restored, entries = reconstruct(
        appendix.split("Delivery context (complete; ordered JSON entries):", 1)[1]
    )
    assert restored["token_usage"] == context["token_usage"]
    assert json.dumps(restored["token_usage"], ensure_ascii=False) == json.dumps(
        context["token_usage"], ensure_ascii=False
    )
    assert [row["value"] for row in entries if row.get("key") == "warnings"] == warnings
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert manifest["delivery_context"]["token_usage"] == context["token_usage"]
    assert json.dumps(context, ensure_ascii=False) == before


def test_original_full_report_warning_behavior_remains_unchanged(tmp_path):
    warnings = ["Recorded original warning with full qualifier."]
    output = write_review(
        FinalReview(paper_key="full", run_id="offline"),
        tmp_path,
        token_usage={"warnings": warnings},
        render_pdf=False,
    )
    assert _text(warnings[0]) in Path(output["markdown"]).read_text("utf-8")


def test_json_literal_html_and_long_url_stay_visible_without_external_pdf_actions(tmp_path):
    url = "https://invalid.example/" + "long-path-segment/" * 12 + "?value=1&next=2"
    issue = '<a href="' + url + '">original label</a> ![image](https://invalid.example/image.png) `literal` '
    issue += '"quotes" <>& remaining scientific qualifier.'
    output = write_review(
        FinalReview(paper_key="literal safety", run_id="offline"),
        tmp_path,
        presentation="layered",
        issues=[issue],
        token_usage={"warnings": [issue]},
    )
    assert not any(k.endswith("_error") for k in output)
    appendix = Path(output["appendix_markdown"]).read_text("utf-8")
    restored, _ = reconstruct(appendix.split("Delivery context (complete; ordered JSON entries):", 1)[1])
    assert restored["issues"] == [issue] and restored["token_usage"]["warnings"] == [issue]
    pdf = PdfReader(output["appendix_pdf"])
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "remaining scientific qualifier" in text and "original label" in text
    assert url in "".join(text.split())
    for page in pdf.pages:
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if "/A" in annotation:
                assert annotation["/A"]["/S"] == "/GoTo"
