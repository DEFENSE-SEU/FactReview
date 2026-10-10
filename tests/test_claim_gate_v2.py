from __future__ import annotations

import pytest

from assessment import assess_claims, gate_claims
from schemas.claim import Claim, ClaimStatus, Evidence, EvidencePointer, ExecutionProvenance

S, U, Q, F = ClaimStatus.SUPPORTED, ClaimStatus.UNVERIFIED, ClaimStatus.QUESTIONED, ClaimStatus.FLAWED

POINTERS = {
    "paper_internal": EvidencePointer(locator="paper.pdf", quote="Table 2 reports 91.2.", page=4),
    "literature": EvidencePointer(locator="arxiv:2101.00001", quote="The same method reaches 91.2."),
    "theory": EvidencePointer(locator="paper.pdf", quote="By Lemma 1 the bound holds.", page=3),
    "code": EvidencePointer(locator="configs/train.yaml", line=12),
    "execution": EvidencePointer(locator="runs/eval/metrics.json", key="accuracy"),
}


def evidence(source: str, *, direction: str = "support", sufficient: bool = True, **extra) -> Evidence:
    if source == "execution":
        extra.setdefault("aligned", True)
    return Evidence(
        source=source,
        pointer=POINTERS[source],
        covered=["c1"],
        direction=direction,
        sufficient=sufficient,
        **extra,
    )


def claim(needs: list[str], items: list[Evidence], *, core: bool = True) -> Claim:
    return Claim(
        id="claim-1",
        text="The method reaches 91.2 accuracy.",
        loc={"page": 1},
        conditions=[{"id": "c1", "dataset": "ImageNet", "metric": "accuracy"}],
        needs=needs,
        importance="core" if core else "secondary",
        evidence=items,
    )


def tiers(record: Claim) -> dict[str, ClaimStatus]:
    (gated,) = gate_claims(assess_claims([record]))
    return {tier: outcome.status for tier, outcome in gated.gate_status.items()}


CASES = [
    # needs, evidence sources, core, expected (lenient, default, strict)
    (["Experiments"], ["paper_internal"], True, (S, U, U)),
    (["Experiments"], ["paper_internal"], False, (S, S, U)),
    (["Experiments"], ["execution"], True, (S, S, S)),
    (["Experiments"], ["execution"], False, (S, S, S)),
    (["Experiments"], ["paper_internal", "code"], True, (S, S, U)),
    (["Experiments"], ["code"], False, (S, S, S)),
    (["Code"], ["execution"], True, (S, S, S)),
    (["Literature"], ["literature"], True, (S, S, S)),
    (["Literature", "Theory"], ["literature"], True, (S, U, U)),
    (["Literature", "Theory"], ["literature"], False, (S, S, U)),
    (["Literature", "Theory"], ["literature", "theory"], False, (S, S, S)),
    (["Code", "Experiments"], ["code", "execution"], True, (S, S, S)),
    (["Code", "Experiments"], ["code", "paper_internal"], True, (S, S, U)),
    (["Code", "Experiments"], ["code", "paper_internal"], False, (S, S, S)),
    ([], ["paper_internal"], True, (S, S, S)),
]


@pytest.mark.parametrize(("needs", "sources", "core", "expected"), CASES)
def test_gate_tiers(needs, sources, core, expected):
    assert tiers(claim(needs, [evidence(source) for source in sources], core=core)) == dict(
        zip(("lenient", "default", "strict"), expected, strict=True)
    )


def test_insufficient_or_non_affecting_independent_support_does_not_satisfy_gate():
    items = [
        evidence("paper_internal"),
        evidence("literature", sufficient=False),
        evidence("execution", affects_claim=False),
    ]
    assert tiers(claim(["Literature", "Experiments"], items)) == {"lenient": S, "default": U, "strict": U}


def test_released_artifact_recomputation_satisfies_strict_experiments():
    provenance = ExecutionProvenance(
        released_artifact=True,
        artifact_kind="data",
        environment_explanation_possible=False,
        artifact_path="data/predictions.jsonl",
        artifact_sha256="a" * 64,
        repository="repo",
        recomputation_pointer="runs/eval/recompute.json",
    )
    items = [evidence("execution", provenance=provenance)]
    for core in (True, False):
        assert tiers(claim(["Experiments"], items, core=core)) == {"lenient": S, "default": S, "strict": S}


def test_strict_reports_missing_execution_for_core_experiments():
    (gated,) = gate_claims(assess_claims([claim(["Experiments"], [evidence("code")])]), "strict")
    assert gated.status == U and gated.gate_status["default"].status == S
    assert gated.gate_status["strict"].reasons == [
        "no aligned execution or released data/logs recomputation for core Experiments claim"
    ]


@pytest.mark.parametrize(
    ("items", "status"),
    [
        ([evidence("paper_internal", direction="flaw", overturnable=False)], F),
        ([evidence("paper_internal", direction="flaw")], Q),
        ([evidence("paper_internal", sufficient=False)], U),
    ],
)
def test_non_supported_statuses_pass_every_tier(items, status):
    assert tiers(claim(["Experiments"], items)) == {"lenient": status, "default": status, "strict": status}


@pytest.mark.parametrize(("level", "status"), [("lenient", S), ("default", U), ("strict", U)])
def test_selected_tier_sets_status_and_records_reason(level, status):
    (gated,) = gate_claims(assess_claims([claim(["Experiments"], [evidence("paper_internal")])]), level)
    assert gated.status == status
    assert gated.gate_status["lenient"].status == S
    assert gated.gate_status["default"].reasons == [
        "no sufficient independent Experiments support (sources execution/code)"
    ]
    assert any(note.startswith(f"Gate ({level}):") for note in gated.notes) == (status == U)
    assert Claim.model_validate(gated.model_dump(mode="json")) == gated
