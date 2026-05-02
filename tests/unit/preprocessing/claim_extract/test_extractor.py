"""Unit tests for :mod:`fact_extraction.extractor`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from preprocessing.claim_extract.extractor import (
    _merge_claims,
    _parse_llm_claims,
    extract_facts,
)
from schemas.claim import Claim, ClaimLocation, ClaimType
from schemas.config import ClaimExtractCfg, LLMCfg
from schemas.paper import Paper, PaperMetadata, Section, Table


def _synthetic_paper() -> Paper:
    """A miniature paper with one body section and one results table."""
    return Paper(
        metadata=PaperMetadata(paper_key="mini", title="Mini"),
        pdf_path=Path("mini.pdf"),
        sections=[
            Section(
                id="sec_1",
                title="Introduction",
                text=(
                    "We propose MiniModel, a novel approach. "
                    "Our method outperforms baselines on FB15k-237 with MRR of 0.355. "
                    "Source code is available at http://github.com/example/mini."
                ),
                char_start=0,
            ),
        ],
        tables=[
            Table(
                id="table_1",
                caption="Link prediction results on FB15k-237.",
                rows=[
                    ["Method", "MRR", "Hits@10"],
                    ["TransE", "0.294", "46.5"],
                    ["MiniModel", "0.355", "53.5"],
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# _parse_llm_claims
# ---------------------------------------------------------------------------


class TestParseLLMClaims:
    def test_happy_path(self) -> None:
        raw = [
            {
                "id": "claim_01",
                "text": "We propose MiniModel.",
                "type": "methodological",
                "scope": "local",
                "datasets": [],
                "baselines": [],
                "metrics": [],
                "location": {"section_id": "sec_1", "char_start": 0, "char_end": 24},
                "evidence_targets": ["literature.neighbor_family"],
            }
        ]
        claims = _parse_llm_claims(raw)
        assert len(claims) == 1
        assert claims[0].type == ClaimType.METHODOLOGICAL
        assert claims[0].location.section_id == "sec_1"
        assert claims[0].evidence_targets == ["literature.neighbor_family"]

    def test_unknown_type_defaults_to_empirical(self) -> None:
        raw = [{"id": "claim_01", "text": "t", "type": "folkloric"}]
        claims = _parse_llm_claims(raw)
        assert claims[0].type == ClaimType.EMPIRICAL

    def test_missing_text_dropped(self) -> None:
        raw = [{"id": "claim_01", "text": "", "type": "empirical"}]
        assert _parse_llm_claims(raw) == []

    def test_non_dict_items_ignored(self) -> None:
        raw = ["not a dict", 42, None]
        assert _parse_llm_claims(raw) == []


# ---------------------------------------------------------------------------
# _merge_claims
# ---------------------------------------------------------------------------


class TestMergeClaims:
    def test_empty_heuristic_passes_through(self) -> None:
        llm = [Claim(id="claim_01", text="x", type=ClaimType.EMPIRICAL, location=ClaimLocation())]
        assert _merge_claims(llm, []) is llm

    def test_empty_llm_returns_heuristic(self) -> None:
        heur = [Claim(id="claim_01", text="x", type=ClaimType.EMPIRICAL, location=ClaimLocation())]
        assert _merge_claims([], heur) is heur

    def test_reproducibility_is_always_kept(self) -> None:
        llm = [
            Claim(
                id="claim_01",
                text="We propose MiniModel, a novel method.",
                type=ClaimType.METHODOLOGICAL,
                location=ClaimLocation(section_id="sec_1"),
            ),
        ]
        heur = [
            Claim(
                id="claim_02",
                text="Source code is available at github.com/mini.",
                type=ClaimType.REPRODUCIBILITY,
                location=ClaimLocation(section_id="sec_1"),
            ),
        ]
        merged = _merge_claims(llm, heur)
        assert any(c.type == ClaimType.REPRODUCIBILITY for c in merged)

    def test_duplicates_are_suppressed(self) -> None:
        llm = [
            Claim(
                id="claim_01",
                text="our method outperforms baselines on FB15k-237",
                type=ClaimType.EMPIRICAL,
                location=ClaimLocation(),
            ),
        ]
        heur = [
            Claim(
                id="claim_99",
                text="Our method outperforms baselines on FB15k-237.",
                type=ClaimType.EMPIRICAL,
                location=ClaimLocation(),
            ),
        ]
        merged = _merge_claims(llm, heur)
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# extract_facts (integration with all sub-modules, but LLM patched)
# ---------------------------------------------------------------------------


class TestExtractFacts:
    def test_heuristic_mode_no_llm_call(self) -> None:
        paper = _synthetic_paper()
        cfg = ClaimExtractCfg(mode="heuristic", decompose_broad_claims=True)
        result = extract_facts(paper, cfg=cfg, llm_cfg=None)

        assert result.backend == "heuristic"
        assert result.claims, "heuristic should find at least one claim"
        # Reported results come from the table regardless of mode.
        assert result.reported_results
        assert any(r.metric == "MRR" and r.value == 0.355 for r in result.reported_results)

    def test_auto_mode_falls_back_when_llm_cfg_missing(self) -> None:
        paper = _synthetic_paper()
        # No llm_cfg passed — auto mode should silently backfill heuristic.
        result = extract_facts(paper, cfg=ClaimExtractCfg(mode="auto"))
        assert result.backend == "auto:heuristic-fallback"
        assert result.claims

    def test_auto_mode_with_llm_success(self) -> None:
        paper = _synthetic_paper()
        fake_llm_claims = [
            Claim(
                id="claim_01",
                text="MiniModel achieves MRR 0.355 on FB15k-237.",
                type=ClaimType.EMPIRICAL,
                scope="local",
                datasets=["FB15k-237"],
                metrics=["MRR"],
                location=ClaimLocation(section_id="sec_1"),
            )
        ]
        with patch(
            "preprocessing.claim_extract.extractor._call_llm_for_claims",
            return_value=fake_llm_claims,
        ):
            result = extract_facts(
                paper,
                cfg=ClaimExtractCfg(mode="auto"),
                llm_cfg=LLMCfg(provider="openai", model="stub"),
            )
        assert result.backend == "auto:llm+heuristic"
        assert any(c.id == "claim_01" for c in result.claims)

    def test_llm_mode_returns_empty_if_llm_fails(self) -> None:
        paper = _synthetic_paper()
        with patch(
            "preprocessing.claim_extract.extractor._call_llm_for_claims",
            return_value=None,
        ):
            result = extract_facts(
                paper,
                cfg=ClaimExtractCfg(mode="llm"),
                llm_cfg=LLMCfg(provider="openai"),
            )
        assert result.backend == "llm"
        assert result.claims == []

    def test_decomposition_applied_when_enabled(self) -> None:
        # Custom broad claim from a hand-rolled LLM result.
        broad = Claim(
            id="claim_01",
            text="Gains on link prediction and node classification on FB15k-237 and WN18RR.",
            type=ClaimType.EMPIRICAL,
            scope="broad",
            datasets=["FB15k-237", "WN18RR"],
            metrics=["MRR"],
            location=ClaimLocation(section_id="sec_1"),
        )
        with patch(
            "preprocessing.claim_extract.extractor._call_llm_for_claims",
            return_value=[broad],
        ):
            result = extract_facts(
                _synthetic_paper(),
                cfg=ClaimExtractCfg(mode="llm", decompose_broad_claims=True),
                llm_cfg=LLMCfg(provider="openai"),
            )
        c = next(c for c in result.claims if c.id == "claim_01")
        assert len(c.subclaims) == 4  # 2 tasks × 2 datasets × 1 metric

    def test_decomposition_skipped_when_disabled(self) -> None:
        broad = Claim(
            id="claim_01",
            text="Gains on link prediction and node classification on FB15k-237 and WN18RR.",
            type=ClaimType.EMPIRICAL,
            scope="broad",
            datasets=["FB15k-237", "WN18RR"],
            metrics=["MRR"],
            location=ClaimLocation(),
        )
        with patch(
            "preprocessing.claim_extract.extractor._call_llm_for_claims",
            return_value=[broad],
        ):
            result = extract_facts(
                _synthetic_paper(),
                cfg=ClaimExtractCfg(mode="llm", decompose_broad_claims=False),
                llm_cfg=LLMCfg(provider="openai"),
            )
        assert result.claims[0].subclaims == []


# ---------------------------------------------------------------------------
# Prompt template loading (package-data integrity check)
# ---------------------------------------------------------------------------


def test_prompt_template_loads_and_formats() -> None:
    from preprocessing.claim_extract.extractor import _load_prompt_template

    tmpl = _load_prompt_template()
    assert "{title}" in tmpl
    assert "{paper_key}" in tmpl
    assert "{sections}" in tmpl
    assert "{reported_summary}" in tmpl
    # Format does not raise given all four slots.
    formatted = tmpl.format(title="t", paper_key="k", sections="s", reported_summary="r")
    assert "t" in formatted and "k" in formatted


@pytest.mark.parametrize("mode", ["heuristic", "auto"])
def test_reported_results_always_populated(mode: str) -> None:
    paper = _synthetic_paper()
    result = extract_facts(paper, cfg=ClaimExtractCfg(mode=mode))
    assert result.reported_results
    assert any(r.method == "MiniModel" for r in result.reported_results)
