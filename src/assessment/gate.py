"""Three-tier gate applied after the aggregation rules to Supported claims only.

lenient keeps the rule status. default requires, for each need of a core claim,
one sufficient support item from that need's independent source; strict applies
the same requirement to every claim. A Supported claim missing such evidence is
reported as Unverified with the missing needs; other statuses pass unchanged.
"""

from schemas.claim import Claim, ClaimStatus, EvidenceNeed, GateLevel, GateOutcome

GATE_LEVELS: tuple[GateLevel, ...] = ("lenient", "default", "strict")

# paper_internal is the manuscript itself and never counts as independent.
# Experiments are satisfied only by execution evidence, and sufficient execution
# evidence is always aligned (Evidence validator). Recomputation from the
# authors' released data or logs is also source "execution", marked by
# provenance.released_artifact with artifact_kind "data"/"logs"
# (fact_generation/execution/v2.py, prediction_measurement.py), so strict's
# "aligned execution or released-artifact recomputation" for core Experiments
# claims is exactly this requirement.
INDEPENDENT_SOURCE = {
    EvidenceNeed.LITERATURE: "literature",
    EvidenceNeed.THEORY: "theory",
    EvidenceNeed.CODE: "code",
    EvidenceNeed.EXPERIMENTS: "execution",
}


def _missing(claim: Claim) -> list[str]:
    sources = {
        item.source
        for item in claim.evidence
        if item.affects_claim and item.direction == "support" and item.sufficient
    }
    return [
        f"no sufficient independent {need.value} support (source {INDEPENDENT_SOURCE[need]})"
        for need in claim.needs
        if INDEPENDENT_SOURCE[need] not in sources
    ]


def gate_claims(claims: list[Claim], level: GateLevel = "default") -> list[Claim]:
    """Record all three tier outcomes and make ``level`` the reported status."""
    gated = []
    for claim in claims:
        result = claim.model_copy(deep=True)
        missing = _missing(claim) if claim.status == ClaimStatus.SUPPORTED else []
        outcomes = {"lenient": GateOutcome(status=claim.status)}
        for tier, applies in (("default", claim.importance == "core"), ("strict", True)):
            reasons = missing if applies else []
            outcomes[tier] = GateOutcome(
                status=ClaimStatus.UNVERIFIED if reasons else claim.status, reasons=reasons
            )
        result.gate_status = outcomes
        result.status = outcomes[level].status
        result.notes.extend(f"Gate ({level}): {reason}" for reason in outcomes[level].reasons)
        gated.append(result)
    return gated
