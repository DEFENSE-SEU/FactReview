"""Incomplete reference detection."""

from __future__ import annotations

from refcopilot.models import MergedRecord, Reference, SourceFormat
from refcopilot.verify.completeness import detect


def _ref(**kw):
    defaults = dict(raw="x", source_format=SourceFormat.BIBTEX)
    defaults.update(kw)
    return Reference(**defaults)


def _merged(**kw):
    defaults = dict(
        title="t",
        authors=[],
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


def test_missing_doi():
    ref = _ref(title="t", doi=None)
    merged = _merged(doi="10.x/y")
    issues = detect(ref, merged)
    codes = {i.code for i in issues}
    assert "missing_doi" in codes


def test_missing_arxiv():
    ref = _ref(arxiv_id=None, title="t")
    merged = _merged(arxiv_id="1706.03762")
    codes = {i.code for i in detect(ref, merged)}
    assert "missing_arxiv_id" in codes


def test_missing_year():
    ref = _ref(year=None, title="t")
    merged = _merged(year=2017)
    codes = {i.code for i in detect(ref, merged)}
    assert "missing_year" in codes


def test_missing_venue():
    ref = _ref(venue=None, title="t")
    merged = _merged(venue="NeurIPS")
    codes = {i.code for i in detect(ref, merged)}
    assert "missing_venue" in codes


def test_truncated_authors():
    ref = _ref(authors=["A. Smith", "et al."], title="t")
    merged = _merged(authors=["A. Smith", "B. Jones", "C. Doe", "D. Lee"])
    codes = {i.code for i in detect(ref, merged)}
    assert "truncated_authors" in codes


def test_complete_authors_no_warning():
    ref = _ref(authors=["A. Smith", "B. Jones"], title="t")
    merged = _merged(authors=["A. Smith", "B. Jones"])
    codes = {i.code for i in detect(ref, merged)}
    assert "truncated_authors" not in codes


def test_abbreviated_venue():
    ref = _ref(venue="NeurIPS", title="t")
    merged = _merged(venue="Neural Information Processing Systems")
    codes = {i.code for i in detect(ref, merged)}
    assert "abbreviated_venue" in codes


def test_doi_already_present_no_warning():
    ref = _ref(doi="10.x/y", title="t")
    merged = _merged(doi="10.x/y")
    codes = {i.code for i in detect(ref, merged)}
    assert "missing_doi" not in codes


def test_no_merged_returns_empty():
    assert detect(_ref(title="x"), None) == []


# ---------------------------------------------------------------------------
# canonical_title_mismatch — fires when cited title is a typo of the canonical title
# ---------------------------------------------------------------------------


def test_canonical_title_mismatch_fires_for_typo():
    ref = _ref(
        title="Math-arena: Evaluating llms on uncontaminated math competitions",
        authors=["Mislav Balunovic", "Jasper Dekoninck"],
    )
    merged = _merged(
        title="MathArena: Evaluating LLMs on Uncontaminated Math Competitions",
        authors=["Mislav Balunovic", "Jasper Dekoninck", "Ivo Petrov"],
    )
    issues = detect(ref, merged)
    codes = {i.code for i in issues}
    assert "canonical_title_mismatch" in codes
    issue = next(i for i in issues if i.code == "canonical_title_mismatch")
    assert issue.suggestion and "MathArena" in issue.suggestion


def test_canonical_title_mismatch_silent_when_normalized_titles_equal():
    """Differences only in case/punctuation that the normalizer collapses must not fire."""
    ref = _ref(
        title="ATTENTION IS ALL YOU NEED!!!",
        authors=["A. Vaswani"],
    )
    merged = _merged(
        title="Attention is all you need",
        authors=["A. Vaswani"],
    )
    codes = {i.code for i in detect(ref, merged)}
    assert "canonical_title_mismatch" not in codes


def test_canonical_title_mismatch_silent_without_author_overlap():
    """Same title spelled differently is not enough — must also share authors."""
    ref = _ref(title="Math-arena", authors=["Random Person"])
    merged = _merged(title="MathArena", authors=["Mislav Balunovic"])
    codes = {i.code for i in detect(ref, merged)}
    assert "canonical_title_mismatch" not in codes
