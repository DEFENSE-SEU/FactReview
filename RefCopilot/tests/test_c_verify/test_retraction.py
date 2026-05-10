"""Retraction detection."""

from __future__ import annotations

from refcopilot.models import (
    IssueCategory,
    MergedRecord,
    Reference,
    Severity,
    SourceFormat,
)
from refcopilot.verify.retraction import detect


def _ref(**kw):
    defaults = dict(raw="x", source_format=SourceFormat.BIBTEX)
    defaults.update(kw)
    return Reference(**defaults)


def _merged(**kw):
    defaults = dict(
        title="t",
        authors=["a"],
        year=2020,
        venue=None,
        doi=None,
        arxiv_id=None,
        latest_arxiv_version=None,
        arxiv_versions=[],
        is_retracted=False,
        url="",
        provenance={},
        sources=[],
    )
    defaults.update(kw)
    return MergedRecord(**defaults)


def test_no_merged_returns_empty():
    assert detect(_ref(title="x"), None) == []


def test_not_retracted_returns_empty():
    assert detect(_ref(title="x"), _merged(is_retracted=False)) == []


def test_retracted_emits_error_issue():
    issues = detect(_ref(title="x", doi="10.1109/access.2020.3018326"),
                    _merged(is_retracted=True, doi="10.1109/access.2020.3018326"))
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.ERROR
    assert issue.category == IssueCategory.RETRACTED
    assert issue.code == "is_retracted"
    assert "retracted" in issue.message.lower()
    assert issue.suggestion is not None
