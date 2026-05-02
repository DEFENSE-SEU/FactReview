"""Minimal tests for the BibTeX lookup helper."""

from __future__ import annotations

import pytest

from fact_generation.positioning.bibtex import _norm_title, lookup_bibtex, title_similarity

# ── Unit: normalisation ──


def test_norm_title_basic():
    assert _norm_title("  Attention-Is All You Need! ") == "attention is all you need"


def test_norm_title_unicode_dashes():
    assert _norm_title("Self\u2013Supervised Learning") == "self supervised learning"


# ── Unit: similarity ──


def test_similarity_identical():
    assert title_similarity("Attention Is All You Need", "Attention Is All You Need") == 1.0


def test_similarity_case_insensitive():
    assert title_similarity("attention is all you need", "ATTENTION IS ALL YOU NEED") == 1.0


def test_similarity_prefers_close():
    a = "Attention Is All You Need"
    assert title_similarity(a, a) > title_similarity(a, "Some Other Paper")


# ── Unit: lookup with mock ──


def test_lookup_empty_title():
    result = lookup_bibtex("")
    assert result["bibtex"] == ""
    assert result["matched_title"] == ""


def test_lookup_exact_match(monkeypatch):
    def fake_http_get_json(url, headers, timeout_s=20, retries=4):
        if "/paper/search" in url:
            return {"data": [{"title": "Attention Is All You Need", "paperId": "PID1"}]}
        if "/paper/PID1" in url:
            return {
                "citationStyles": {
                    "bibtex": "@article{vaswani2017,\n  title={Attention Is All You Need}\n}\n"
                }
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("fact_generation.positioning.bibtex._http_get_json", fake_http_get_json)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")

    result = lookup_bibtex("Attention Is All You Need")
    assert result["exact"] is True
    assert "@article" in result["bibtex"]
    assert result["matched_title"] == "Attention Is All You Need"


def test_lookup_fuzzy_fallback(monkeypatch):
    def fake_http_get_json(url, headers, timeout_s=20, retries=4):
        if "/paper/search" in url:
            return {
                "data": [
                    {"title": "Attention Is All You Need", "paperId": "PID1"},
                    {"title": "Other Paper", "paperId": "PID2"},
                ]
            }
        if "/paper/PID1" in url:
            return {"citationStyles": {"bibtex": "@article{vaswani2017}\n"}}
        if "/paper/PID2" in url:
            return {"citationStyles": {"bibtex": "@article{other}\n"}}
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("fact_generation.positioning.bibtex._http_get_json", fake_http_get_json)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")

    result = lookup_bibtex("Attention Is All U Need")  # typo triggers fuzzy
    assert result["bibtex"] != ""
    assert result["matched_title"] != ""


def test_lookup_missing_api_key(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        lookup_bibtex("Some Title")
