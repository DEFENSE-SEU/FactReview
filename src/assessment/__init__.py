"""Deterministic evidence-bounded claim assessment."""

from assessment.gate import GATE_LEVELS, gate_claims
from assessment.rules import assess_claim, assess_claims

__all__ = ["GATE_LEVELS", "assess_claim", "assess_claims", "gate_claims"]
