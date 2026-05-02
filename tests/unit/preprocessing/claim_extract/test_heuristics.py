"""Unit tests for :mod:`fact_extraction.heuristics`.

Uses synthetic :class:`Paper` objects rather than CompGCN — these tests
must stay generic and remain meaningful even if the CompGCN fixture is
later replaced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from preprocessing.claim_extract.heuristics import (
    _classify_sentence,
    _extract_datasets,
    _extract_metrics,
    _infer_scope,
    extract_claims_heuristic,
)
from schemas.claim import ClaimType
from schemas.paper import Paper, PaperMetadata, Section


def _paper_with(sections: list[Section]) -> Paper:
    return Paper(
        metadata=PaperMetadata(paper_key="synthetic"),
        pdf_path=Path("synthetic.pdf"),
        sections=sections,
    )


class TestClassifySentence:
    def test_empirical_trigger(self) -> None:
        assert (
            _classify_sentence("Our method outperforms prior work on link prediction.") == ClaimType.EMPIRICAL
        )

    def test_numeric_claim_is_empirical(self) -> None:
        assert _classify_sentence("The model achieves an MRR of 0.355 on FB15k-237.") == ClaimType.EMPIRICAL

    def test_methodological_trigger(self) -> None:
        assert (
            _classify_sentence("We propose a novel framework for multi-relational graphs.")
            == ClaimType.METHODOLOGICAL
        )

    def test_theoretical_trigger(self) -> None:
        assert (
            _classify_sentence("We prove that our method generalizes prior work (Proposition 4.1).")
            == ClaimType.THEORETICAL
        )

    def test_reproducibility_trigger(self) -> None:
        assert (
            _classify_sentence("Source code is available at http://github.com/example/repo.")
            == ClaimType.REPRODUCIBILITY
        )

    def test_non_claim_returns_none(self) -> None:
        assert _classify_sentence("Figure 1 depicts the architecture.") is None


class TestExtractDatasetsAndMetrics:
    def test_known_benchmarks(self) -> None:
        ds = _extract_datasets("Results on FB15k-237 and WN18RR show gains.")
        assert ds == ["FB15k-237", "WN18RR"]

    def test_fallback_uppercase_token(self) -> None:
        ds = _extract_datasets("We evaluate on CUSTOMDS42 and BENCH9.")
        assert "CUSTOMDS42" in ds and "BENCH9" in ds

    def test_metrics_various_forms(self) -> None:
        m = _extract_metrics("MRR of 0.355 and Hits@10 of 53.5%.")
        assert "MRR" in m and "Hits@10" in m


class TestScope:
    def test_broad_when_multiple_datasets(self) -> None:
        s = "Our method outperforms baselines on FB15k-237 and WN18RR."
        assert _infer_scope(s, ClaimType.EMPIRICAL) == "broad"

    def test_broad_when_across_keyword(self) -> None:
        s = "Gains hold across multiple benchmarks."
        assert _infer_scope(s, ClaimType.EMPIRICAL) == "broad"

    def test_local_otherwise(self) -> None:
        s = "The model achieves 0.355 MRR on FB15k-237."
        assert _infer_scope(s, ClaimType.EMPIRICAL) == "local"


class TestExtractClaimsHeuristic:
    def test_end_to_end_on_mini_paper(self) -> None:
        sec_intro = Section(
            id="sec_1",
            title="Introduction",
            text=(
                "We propose COMPGCN, a novel framework for multi-relational graphs. "
                "Our method outperforms baselines on FB15k-237 and WN18RR. "
                "The source code is available at http://github.com/example/compgcn."
            ),
            char_start=0,
        )
        sec_refs = Section(
            id="sec_refs",
            title="References",
            text="Kipf & Welling (2016). Semi-Supervised Classification with GCNs.",
            char_start=2000,
        )
        paper = _paper_with([sec_intro, sec_refs])

        claims = extract_claims_heuristic(paper)

        # References section must be skipped.
        for c in claims:
            assert c.location.section_id != "sec_refs"

        types = {c.type for c in claims}
        assert ClaimType.METHODOLOGICAL in types
        assert ClaimType.EMPIRICAL in types
        assert ClaimType.REPRODUCIBILITY in types

        # Empirical claim should surface datasets.
        empirical = [c for c in claims if c.type == ClaimType.EMPIRICAL]
        assert empirical
        assert set(empirical[0].datasets) >= {"FB15k-237", "WN18RR"}
        assert empirical[0].scope == "broad"

    def test_stable_ids(self) -> None:
        sec = Section(
            id="sec_1",
            title="Intro",
            text="We propose X. Our method outperforms prior work.",
            char_start=0,
        )
        claims = extract_claims_heuristic(_paper_with([sec]))
        ids = [c.id for c in claims]
        assert ids == sorted(ids)
        assert all(cid.startswith("claim_") for cid in ids)

    def test_max_claims_bound(self) -> None:
        # Generate a long section with many triggers.
        body = " ".join(["We propose a new method."] * 200)
        sec = Section(id="sec_1", title="Intro", text=body, char_start=0)
        claims = extract_claims_heuristic(_paper_with([sec]), max_claims=5)
        assert len(claims) == 5


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("We prove a theorem.", ClaimType.THEORETICAL),
        ("Our approach is equivalent to prior work.", ClaimType.THEORETICAL),
        ("We release the code publicly.", ClaimType.REPRODUCIBILITY),
        ("Relative improvement of 2.3 points over the baseline.", ClaimType.EMPIRICAL),
    ],
)
def test_classify_parametrised(sentence: str, expected: ClaimType) -> None:
    assert _classify_sentence(sentence) == expected
