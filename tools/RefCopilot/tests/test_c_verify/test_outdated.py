"""Outdated reference detection."""

from __future__ import annotations

from refcopilot.models import Backend, ExternalRecord, MergedRecord, Reference, SourceFormat
from refcopilot.verify.outdated import detect


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
        withdrawn=False,
        url="",
        provenance={},
        sources=[],
    )
    defaults.update(kw)
    return MergedRecord(**defaults)


def test_arxiv_published_at_venue():
    ref = _ref(arxiv_id="1706.03762", venue=None)
    merged = _merged(venue="NeurIPS 2017", arxiv_id="1706.03762")
    issues = detect(ref, merged)
    codes = [i.code for i in issues]
    assert "arxiv_published" in codes


def test_arxiv_only_no_warning():
    ref = _ref(arxiv_id="2501.12948")
    merged = _merged(arxiv_id="2501.12948", venue=None)
    issues = detect(ref, merged)
    assert all(i.code != "arxiv_published" for i in issues)


def test_publication_venue_filter_excludes_corr():
    ref = _ref(arxiv_id="9999.99999")
    merged = _merged(arxiv_id="9999.99999", venue="CoRR")
    issues = detect(ref, merged)
    assert all(i.code != "arxiv_published" for i in issues)


def test_old_arxiv_version():
    ref = _ref(arxiv_id="1706.03762", arxiv_version=1)
    merged = _merged(arxiv_id="1706.03762", latest_arxiv_version=7, venue="NeurIPS")
    issues = detect(ref, merged)
    codes = [i.code for i in issues]
    assert "old_version" in codes


def test_same_version_no_warning():
    ref = _ref(arxiv_id="1706.03762", arxiv_version=7)
    merged = _merged(arxiv_id="1706.03762", latest_arxiv_version=7)
    issues = detect(ref, merged)
    assert all(i.code != "old_version" for i in issues)


def test_workshop_to_conference():
    ref = _ref(venue="ICML Workshop on FooBar", arxiv_id="2401.12345")
    merged = _merged(venue="ICML 2024", arxiv_id="2401.12345")
    issues = detect(ref, merged)
    assert any(i.code == "workshop_promoted" for i in issues)


def test_withdrawn_paper():
    ref = _ref(arxiv_id="1234.5678")
    merged = _merged(arxiv_id="1234.5678", withdrawn=True)
    issues = detect(ref, merged)
    assert any(i.code == "withdrawn" for i in issues)


def test_no_merged_returns_empty():
    assert detect(_ref(title="x"), None) == []
