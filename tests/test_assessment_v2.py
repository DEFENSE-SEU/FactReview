import pytest

from assessment import assess_claim
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidencePointer, ExecutionProvenance


def evidence(direction="support", covered=("a", "b"), sufficient=True, **kwargs):
    return Evidence(
        source="paper_internal",
        pointer=EvidencePointer(locator="paper.md", page=1, quote="Result"),
        direction=direction,
        covered=list(covered),
        sufficient=sufficient,
        **kwargs,
    )


def claim(items):
    return Claim(
        id="c1",
        text="Improves two settings",
        loc=ClaimLocation(page=1),
        conditions=[Condition(id=x, description=x) for x in ("a", "b")],
        needs=[],
        evidence=items,
    )


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([evidence(), evidence("flaw", overturnable=False)], "questioned"),
        ([evidence("flaw", overturnable=False)], "flawed"),
        ([evidence("flaw", sufficient=False, concern=True)], "questioned"),
        ([evidence()], "supported"),
        ([], "unverified"),
        ([evidence(covered=("a",))], "unverified"),
        ([evidence(covered=("a",)), evidence("flaw", ("b",), False, concern=True)], "questioned"),
        ([evidence(covered=("a",)), evidence(covered=("b",))], "supported"),
        ([evidence(covered=("a",)), evidence("flaw", ("b",), overturnable=False)], "flawed"),
        ([evidence("flaw", sufficient=False, concern=False)], "unverified"),
        (
            [
                evidence(),
                evidence(
                    "flaw",
                    sufficient=False,
                    concern=True,
                    affects_claim=False,
                    note="Large gap; variance was not reported",
                ),
            ],
            "supported",
        ),
        ([evidence("flaw", overturnable=True)], "questioned"),
    ],
)
def test_ordered_rules_partial_coverage_and_concerns(items, expected):
    record = claim(items)
    assert assess_claim(record).status == expected
    assert record.status == "unverified"


def test_paper_internal_support_remains_visible_and_notes_are_idempotent():
    record = claim(
        [
            evidence(),
            evidence(
                "flaw",
                sufficient=False,
                concern=True,
                affects_claim=False,
                note="Large gap; variance not reported",
            ),
        ]
    )
    result = assess_claim(record)
    assert result.status == "supported"
    assert result.evidence[0].source == "paper_internal"
    assert result.notes == ["Large gap; variance not reported"]
    assert assess_claim(result) == result
    assert record.notes == []


def execution(*, aligned, sufficient=False, provenance=None, overturnable=True):
    return Evidence(
        source="execution",
        pointer=EvidencePointer(locator="run.json", key="metrics"),
        direction="flaw",
        covered=["a"],
        sufficient=sufficient,
        concern=True,
        aligned=aligned,
        overturnable=overturnable,
        provenance=provenance,
        note="Metric mismatch",
    )


@pytest.mark.parametrize("aligned", [False, None])
def test_unaligned_execution_never_becomes_concern_or_evidence(aligned):
    result = assess_claim(claim([execution(aligned=aligned)]))
    assert result.status == "unverified"
    assert result.notes == ["Metric mismatch"]


def test_aligned_mismatch_requires_artifact_provenance_for_flawed():
    candidate = execution(aligned=True, sufficient=True, overturnable=False)
    assert assess_claim(claim([candidate])).status == "questioned"
    candidate.provenance = ExecutionProvenance(
        released_artifact=True, artifact_kind="logs", environment_explanation_possible=False
    )
    assert assess_claim(claim([candidate])).status == "questioned"
    candidate.provenance.artifact_path = "released-results.json"
    candidate.provenance.artifact_sha256 = "a" * 64
    candidate.provenance.repository = "released-repo"
    candidate.provenance.recomputation_pointer = "run.json#calculation"
    assert assess_claim(claim([candidate])).status == "flawed"
