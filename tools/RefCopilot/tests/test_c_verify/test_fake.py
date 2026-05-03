"""Test C.1 — fake (hallucination) detection."""

from __future__ import annotations

from refcopilot.merge import merge_records
from refcopilot.models import (
    Backend,
    ExternalRecord,
    HallucinationVerdict,
    Reference,
    SourceFormat,
)
from refcopilot.verify import hallucination
from refcopilot.verify.text_match import author_overlap, is_garbled_title, title_similarity


def _ref(**kw):
    defaults = dict(raw="x", source_format=SourceFormat.BIBTEX)
    defaults.update(kw)
    return Reference(**defaults)


def _record(**kw):
    defaults = dict(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="x",
        title="placeholder",
        url="https://example.com",
    )
    defaults.update(kw)
    return ExternalRecord(**defaults)


# ---- title_similarity & author_overlap sanity checks ----------------------


def test_title_similarity_exact():
    assert title_similarity("Attention Is All You Need", "attention is all you need") > 0.95


def test_title_similarity_unrelated():
    assert title_similarity("Quantum Bagels", "Attention Is All You Need") < 0.5


def test_author_overlap_exact():
    assert author_overlap(["Ashish Vaswani"], ["Ashish Vaswani"]) == 1.0


def test_author_overlap_last_token_match():
    assert author_overlap(["A. Vaswani"], ["Ashish Vaswani"]) >= 0.5


def test_author_overlap_team_prefix():
    # Cited "DeepSeek-AI" should match retrieved "DeepSeek-AI Authors"
    assert author_overlap(["DeepSeek-AI"], ["DeepSeek-AI Liu"]) >= 0.5


def test_author_overlap_disjoint():
    assert author_overlap(["A. Smith"], ["B. Jones"]) == 0.0


# ---- garbled detection ----------------------------------------------------


def test_garbled_empty():
    assert is_garbled_title("") is True
    assert is_garbled_title(None) is True


def test_garbled_starts_mid_word():
    # First word "ay" is a short lowercase fragment that doesn't match a stopword.
    title = "ay we should think about this more carefully"
    raw = "#" + title  # signals no-author entry
    assert is_garbled_title(title, raw) is True


def test_garbled_normal_title_passes():
    assert is_garbled_title("Attention Is All You Need") is False


# ---- pre_screen --------------------------------------------------------


def test_fake_no_match_anywhere():
    ref = _ref(title="Quantum Bagel Theorems", authors=["X. Fake"])
    verdict = hallucination.pre_screen(ref, matches=[], merged=None)
    assert verdict == HallucinationVerdict.LIKELY


def test_no_match_with_url_is_uncertain():
    ref = _ref(title="X", authors=["A"], url="https://example.com")
    verdict = hallucination.pre_screen(ref, matches=[], merged=None)
    assert verdict == HallucinationVerdict.UNCERTAIN


def test_low_title_similarity_is_likely():
    ref = _ref(title="Quantum Bagel Theorems", authors=["X. Fake"])
    record = _record(title="Attention Is All You Need", authors=["A. Vaswani"])
    matches = [record]
    merged = merge_records(matches)
    assert hallucination.pre_screen(ref, matches=matches, merged=merged) == HallucinationVerdict.LIKELY


def test_low_author_overlap_no_url_is_likely():
    ref = _ref(
        title="Attention Is All You Need",
        authors=["X. Fake", "Y. Imaginary"],
    )
    record = _record(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
    )
    verdict = hallucination.pre_screen(ref, matches=[record], merged=None)
    assert verdict == HallucinationVerdict.LIKELY


def test_high_similarity_is_unlikely():
    ref = _ref(title="Attention Is All You Need", authors=["A. Vaswani", "N. Shazeer"])
    record = _record(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"])
    verdict = hallucination.pre_screen(ref, matches=[record], merged=None)
    assert verdict == HallucinationVerdict.UNLIKELY


def test_garbled_title_returns_uncertain():
    ref = _ref(title="ay we should think about this more carefully", raw="#blah blah")
    verdict = hallucination.pre_screen(ref, matches=[], merged=None)
    assert verdict == HallucinationVerdict.UNCERTAIN


def test_off_by_one_year_does_not_flag_fake():
    # Year is part of merge but doesn't enter pre_screen for fake detection
    # directly. Make sure that a title/author match with off-by-one year still
    # produces UNLIKELY.
    ref = _ref(title="Attention Is All You Need", authors=["A. Vaswani"], year=2017)
    record = _record(title="Attention Is All You Need", authors=["Ashish Vaswani"], year=2018)
    verdict = hallucination.pre_screen(ref, matches=[record], merged=None)
    assert verdict == HallucinationVerdict.UNLIKELY


# ---- to_issue ---------------------------------------------------------


def test_to_issue_emits_no_match():
    ref = _ref(title="Quantum Bagel Theorems")
    issue = hallucination.to_issue(HallucinationVerdict.LIKELY, ref, matches=[])
    assert issue is not None
    assert issue.code == "no_match"


def test_to_issue_emits_title_mismatch():
    ref = _ref(title="Quantum Bagel Theorems", authors=["X."])
    record = _record(title="Attention Is All You Need", authors=["Vaswani"])
    issue = hallucination.to_issue(HallucinationVerdict.LIKELY, ref, matches=[record])
    assert issue is not None
    assert issue.code == "title_mismatch"


def test_to_issue_emits_author_mismatch():
    ref = _ref(title="Attention Is All You Need", authors=["X. Fake"])
    record = _record(title="Attention Is All You Need", authors=["Ashish Vaswani"])
    issue = hallucination.to_issue(HallucinationVerdict.LIKELY, ref, matches=[record])
    assert issue is not None
    assert issue.code == "author_mismatch"


def test_to_issue_returns_none_for_unlikely():
    ref = _ref(title="X")
    assert hallucination.to_issue(HallucinationVerdict.UNLIKELY, ref, matches=[]) is None
    assert hallucination.to_issue(HallucinationVerdict.UNCERTAIN, ref, matches=[]) is None
