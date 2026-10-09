"""Public RefCopilot objects are mocked; legacy output and provenance stay separate."""


import pytest

# The FactReview wrapper installs the vendored public library path.
# isort: off
from fact_generation.refcheck.refcheck import check_references_with_records
from refcopilot import RefCopilotPipeline
from refcopilot.models import (
    Backend,
    CheckedReference,
    ExternalRecord,
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Report,
    ReportSummary,
    Severity,
    SourceFormat,
    Verdict,
)
from refcopilot.report import to_factreview_dict
# isort: on


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("No live reference/model/Docker operation is allowed")

    monkeypatch.setattr("requests.sessions.Session.request", fail)
    monkeypatch.setattr("httpx.Client.send", fail)
    monkeypatch.setattr("subprocess.run", fail)


def report(raw=None, *, doi="10.1000/one", title="A grounded paper"):
    source = ExternalRecord(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="record-1",
        title=title,
        authors=["Ada Author"],
        year=2020,
        venue="Journal of Examples",
        doi=doi,
        url="https://www.semanticscholar.org/paper/record-1",
    )
    merged = MergedRecord(
        title=title,
        authors=source.authors,
        year=2020,
        venue=source.venue,
        doi=doi,
        url=source.url,
        sources=[source],
        provenance={k: Backend.SEMANTIC_SCHOLAR for k in ("title", "authors", "year", "venue", "doi", "url")},
    )
    ref = Reference(
        raw=raw or f"Ada Author. 2019. {title}. doi:{doi}",
        source_format=SourceFormat.TEXT,
        title=title,
        authors=["Ada Author"],
        year=2019,
        doi=doi,
    )
    checked = CheckedReference(
        reference=ref,
        merged=merged,
        verdict=Verdict.WARNING,
        issues=[
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="year_mismatch",
                message="Retrieved year is 2020.",
            )
        ],
    )
    return Report(checked=[checked], summary=ReportSummary(total_refs=1, warnings=1))


def invoke(monkeypatch, tmp_path, value):
    calls = []
    path = tmp_path / "bibliography.txt"
    text = "\n\n".join(c.reference.raw for c in value.checked)
    path.write_text(text, "utf-8")

    def run(self, paper):
        calls.append(paper)
        return value

    monkeypatch.setattr(RefCopilotPipeline, "run", run)
    result = check_references_with_records(paper=str(path), enable_parallel=False)
    assert calls == [text]
    return result, path


def test_one_public_run_keeps_export_and_full_report_exact(monkeypatch, tmp_path):
    value = report()
    result, _ = invoke(monkeypatch, tmp_path, value)
    assert result.payload == to_factreview_dict(value)
    assert result.records["report"] == value.model_dump(mode="json")
    assert result.records["bindings"][0]["reference_index"] == 0
    assert result.records["bindings"][0]["issue_index"] == 0
    assert result.records["bindings"][0]["exported_issue"] == result.payload["issues"][0]


def test_duplicate_titles_multiple_issues_and_unverified_keep_indices(monkeypatch, tmp_path):
    value = report()
    other = report(doi="10.1000/two").checked[0]
    other.issues.append(other.issues[0].model_copy(update={"code": "venue_mismatch"}))
    unverified = CheckedReference(reference=other.reference.model_copy(), verdict=Verdict.UNVERIFIED)
    value.checked.extend([other, unverified])
    value.summary.total_refs = 3
    result, _ = invoke(monkeypatch, tmp_path, value)
    assert result.payload == to_factreview_dict(value)
    assert [(r["reference_index"], r["issue_index"]) for r in result.records["bindings"]] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, None),
    ]


def test_full_bibtex_survives_legacy_truncation(monkeypatch, tmp_path):
    value = report(title="A " + "Long title " * 500)
    result, _ = invoke(monkeypatch, tmp_path, value)
    binding = result.records["bindings"][0]
    assert len(binding["corrected_bibtex"]) > 4000
    assert binding["export_truncated"]
    assert "...(truncated)" in result.payload["issues"][0]["corrected_bibtex"]
    assert result.payload == to_factreview_dict(value)


@pytest.mark.parametrize("bad", ["raise", "invalid", "changed_input"])
def test_run_failure_never_becomes_empty_success(monkeypatch, tmp_path, bad):
    path = tmp_path / "bibliography.txt"
    path.write_text("original", "utf-8")

    def run(self, paper):
        if bad == "raise":
            raise RuntimeError("backend unavailable")
        if bad == "invalid":
            return {"checked": []}
        path.write_text("changed", "utf-8")
        return report()

    monkeypatch.setattr(RefCopilotPipeline, "run", run)
    result = check_references_with_records(str(path))
    assert result.payload["ok"] is False
    assert result.payload["error_message"]
    assert result.records is None
