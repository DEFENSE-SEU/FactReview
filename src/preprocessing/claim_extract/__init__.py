"""Stage §3.1b (claim_extract): extract review-relevant :class:`Claim` objects.

Turns the structured :class:`schemas.paper.Paper` produced by the parse
stage into a list of :class:`schemas.claim.Claim` objects — optionally
decomposed into :class:`SubClaim` entries for broad claims, and paired
with :class:`ReportedResult` values extracted from tables.
"""

from preprocessing.claim_extract.decomposer import decompose_claim, decompose_claims
from preprocessing.claim_extract.extractor import ExtractionResult, extract_facts
from preprocessing.claim_extract.heuristics import extract_claims_heuristic
from preprocessing.claim_extract.results_parser import extract_reported_results

__all__ = [
    "ExtractionResult",
    "decompose_claim",
    "decompose_claims",
    "extract_claims_heuristic",
    "extract_facts",
    "extract_reported_results",
]
