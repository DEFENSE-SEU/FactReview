"""Unit tests for :mod:`fact_extraction.decomposer`."""

from __future__ import annotations

from preprocessing.claim_extract.decomposer import (
    _detect_tasks,
    _own_method_name,
    decompose_claim,
    decompose_claims,
)
from schemas.claim import Claim, ClaimLocation, ClaimType
from schemas.paper import ReportedResult


def _broad_empirical_claim() -> Claim:
    return Claim(
        id="claim_01",
        text=(
            "We demonstrate that COMPGCN outperforms baselines on link prediction, "
            "node classification, and graph classification over FB15k-237 and WN18RR."
        ),
        type=ClaimType.EMPIRICAL,
        scope="broad",
        datasets=["FB15k-237", "WN18RR"],
        baselines=["TransE", "ConvE"],
        metrics=["MRR"],
        location=ClaimLocation(section_id="sec_1"),
    )


def test_detect_tasks_multiple() -> None:
    tasks = _detect_tasks("Evaluated on link prediction, node classification, and graph classification.")
    assert tasks == ["link_prediction", "node_classification", "graph_classification"]


def test_detect_tasks_none() -> None:
    assert _detect_tasks("A purely theoretical result.") == []


def test_own_method_name_after_propose() -> None:
    claim = Claim(
        id="claim_01",
        text="We propose COMPGCN, a new framework.",
        type=ClaimType.METHODOLOGICAL,
        location=ClaimLocation(),
    )
    assert _own_method_name(claim) == "COMPGCN"


def test_own_method_name_after_our() -> None:
    claim = Claim(
        id="claim_01",
        text="Our BERT-style encoder achieves strong results.",
        type=ClaimType.EMPIRICAL,
        location=ClaimLocation(),
    )
    assert _own_method_name(claim) == "BERT-style"


def test_own_method_name_absent() -> None:
    claim = Claim(
        id="claim_01",
        text="Performance is strong across benchmarks.",
        type=ClaimType.EMPIRICAL,
        location=ClaimLocation(),
    )
    assert _own_method_name(claim) is None


def test_decompose_broad_generates_cartesian_product() -> None:
    claim = _broad_empirical_claim()
    # No reported results yet → expected_value stays None.
    out = decompose_claim(claim, reported=[])

    # 3 tasks × 2 datasets × 1 metric = 6 subclaims.
    assert len(out.subclaims) == 6
    # Ids are stable and dense.
    ids = [s.id for s in out.subclaims]
    assert ids == [f"claim_01.sub_{i:02d}" for i in range(1, 7)]

    # Every subclaim carries its coordinates.
    coords = {(s.task, s.dataset, s.metric) for s in out.subclaims}
    assert ("link_prediction", "FB15k-237", "MRR") in coords
    assert ("graph_classification", "WN18RR", "MRR") in coords


def test_decompose_local_claim_is_identity() -> None:
    claim = Claim(
        id="claim_01",
        text="MRR of 0.355 on FB15k-237.",
        type=ClaimType.EMPIRICAL,
        scope="local",
        datasets=["FB15k-237"],
        metrics=["MRR"],
        location=ClaimLocation(),
    )
    out = decompose_claim(claim, reported=[])
    assert out.subclaims == []
    assert out is claim or out.model_dump() == claim.model_dump()


def test_decompose_binds_reported_value() -> None:
    claim = _broad_empirical_claim()
    reported = [
        ReportedResult(
            id="t4.r1.c1",
            metric="MRR",
            value=0.355,
            dataset="FB15k-237",
            task="link_prediction",
            method="COMPGCN",
            table_id="table_4",
            row_index=1,
            col_index=1,
        ),
        ReportedResult(
            id="t4.r2.c1",
            metric="MRR",
            value=0.479,
            dataset="WN18RR",
            task="link_prediction",
            method="COMPGCN",
            table_id="table_4",
            row_index=2,
            col_index=1,
        ),
    ]
    out = decompose_claim(claim, reported=reported)

    by_coord = {(s.task, s.dataset, s.metric): s.expected_value for s in out.subclaims}
    assert by_coord[("link_prediction", "FB15k-237", "MRR")] == 0.355
    assert by_coord[("link_prediction", "WN18RR", "MRR")] == 0.479
    # Non-link-prediction tasks have no match → expected_value stays None.
    assert by_coord[("node_classification", "FB15k-237", "MRR")] is None


def test_decompose_method_name_tiebreak() -> None:
    claim = _broad_empirical_claim()
    # Two rows at the same (metric, dataset); method match wins.
    reported = [
        ReportedResult(
            id="t.1",
            metric="MRR",
            value=0.300,
            dataset="FB15k-237",
            task="link_prediction",
            method="TransE",
            table_id="t",
            row_index=1,
            col_index=0,
        ),
        ReportedResult(
            id="t.2",
            metric="MRR",
            value=0.355,
            dataset="FB15k-237",
            task="link_prediction",
            method="COMPGCN",
            table_id="t",
            row_index=2,
            col_index=0,
        ),
    ]
    out = decompose_claim(claim, reported=reported)
    match = next(s for s in out.subclaims if s.task == "link_prediction" and s.dataset == "FB15k-237")
    assert match.expected_value == 0.355


def test_decompose_claims_preserves_order() -> None:
    claims = [
        _broad_empirical_claim(),
        Claim(
            id="claim_02",
            text="Local numeric claim.",
            type=ClaimType.EMPIRICAL,
            scope="local",
            location=ClaimLocation(),
        ),
    ]
    out = decompose_claims(claims, reported=[])
    assert [c.id for c in out] == ["claim_01", "claim_02"]
    assert out[0].subclaims  # broad claim decomposed
    assert out[1].subclaims == []  # local claim untouched
