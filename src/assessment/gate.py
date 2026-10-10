"""Three-tier gate applied after the aggregation rules to Supported claims only.

lenient keeps the rule status. default requires, for each need of a core claim,
one sufficient support item from a source that need accepts as independent;
non-core claims pass as in lenient. strict applies that requirement to every
claim and additionally requires a core Experiments claim to have aligned
execution support or a recomputation from the authors' released data or logs.
A Supported claim missing such evidence is reported as Unverified with the
missing needs; other statuses pass unchanged.
"""

from schemas.claim import Claim, ClaimStatus, EvidenceNeed, GateLevel, GateOutcome

GATE_LEVELS: tuple[GateLevel, ...] = ("lenient", "default", "strict")

# paper_internal is the manuscript itself and never counts as independent.
INDEPENDENT_SOURCES = {
    EvidenceNeed.LITERATURE: ("literature",),
    EvidenceNeed.THEORY: ("theory",),
    EvidenceNeed.CODE: ("code", "execution"),
    EvidenceNeed.EXPERIMENTS: ("execution", "code"),
}


def _support(claim: Claim) -> list:
    return [
        item
        for item in claim.evidence
        if item.affects_claim and item.direction == "support" and item.sufficient
    ]


def _missing(claim: Claim) -> list[str]:
    sources = {item.source for item in _support(claim)}
    return [
        f"no sufficient independent {need.value} support (sources {'/'.join(INDEPENDENT_SOURCES[need])})"
        for need in claim.needs
        if not sources.intersection(INDEPENDENT_SOURCES[need])
    ]


def _missing_execution(claim: Claim) -> list[str]:
    if claim.importance != "core" or EvidenceNeed.EXPERIMENTS not in claim.needs:
        return []
    for item in _support(claim):
        if item.source != "execution":
            continue
        released = item.provenance is not None and item.provenance.released_artifact
        if item.aligned is True or (released and item.provenance.artifact_kind in ("data", "logs")):
            return []
    return ["no aligned execution or released data/logs recomputation for core Experiments claim"]


def gate_claims(claims: list[Claim], level: GateLevel = "default") -> list[Claim]:
    """Record all three tier outcomes and make ``level`` the reported status."""
    gated = []
    for claim in claims:
        result = claim.model_copy(deep=True)
        supported = claim.status == ClaimStatus.SUPPORTED
        missing = _missing(claim) if supported else []
        tier_reasons = {
            "default": missing if claim.importance == "core" else [],
            "strict": missing + (_missing_execution(claim) if supported else []),
        }
        outcomes = {"lenient": GateOutcome(status=claim.status)}
        for tier, reasons in tier_reasons.items():
            outcomes[tier] = GateOutcome(
                status=ClaimStatus.UNVERIFIED if reasons else claim.status, reasons=reasons
            )
        result.gate_status = outcomes
        result.status = outcomes[level].status
        result.notes.extend(f"Gate ({level}): {reason}" for reason in outcomes[level].reasons)
        gated.append(result)
    return gated
