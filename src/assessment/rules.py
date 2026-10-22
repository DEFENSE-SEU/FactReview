"""The five ordered aggregation rules from method specification §6.3."""

from schemas.claim import Claim, ClaimStatus, Evidence


def _decisive_flaw(item: Evidence) -> bool:
    if item.direction != "flaw" or not item.sufficient or item.overturnable:
        return False
    if item.source != "execution":
        return True
    provenance = item.provenance
    return bool(
        provenance
        and provenance.released_artifact
        and provenance.artifact_kind in {"data", "logs"}
        and not provenance.environment_explanation_possible
        and provenance.artifact_path
        and provenance.artifact_sha256
        and provenance.repository
        and provenance.recomputation_pointer
    )


def assess_claim(claim: Claim) -> Claim:
    """Return a fresh assessed record without model calls or mutation of inputs.

    Producer branches validate source contents before constructing evidence.
    Unaligned execution outcomes explain missing verification and cannot change
    a status, including through the concern rule.
    """
    assessed = claim.model_copy(deep=True)
    usable = []
    for item in assessed.evidence:
        if not item.affects_claim or (item.source == "execution" and item.aligned is not True):
            note = item.note or f"Non-decisive {item.source} observation: {item.pointer.locator}"
            if note not in assessed.notes:
                assessed.notes.append(note)
            continue
        usable.append(item)
    supports = [item for item in usable if item.direction == "support" and item.sufficient]
    flaws = [item for item in usable if item.direction == "flaw" and item.sufficient]
    support_coverage = {part for item in supports for part in item.covered}
    flaw_coverage = {part for item in flaws for part in item.covered}
    if support_coverage & flaw_coverage:
        assessed.status = ClaimStatus.QUESTIONED
    elif any(_decisive_flaw(item) for item in flaws):
        assessed.status = ClaimStatus.FLAWED
    elif any(item.concern or (item.direction == "flaw" and item.sufficient) for item in usable):
        assessed.status = ClaimStatus.QUESTIONED
    elif {condition.id for condition in assessed.conditions}.issubset(support_coverage):
        assessed.status = ClaimStatus.SUPPORTED
    else:
        assessed.status = ClaimStatus.UNVERIFIED
    return assessed


def assess_claims(claims: list[Claim]) -> list[Claim]:
    return [assess_claim(claim) for claim in claims]
