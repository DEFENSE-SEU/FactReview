"""RefCopilot — reference-accuracy checker for academic papers."""

from refcopilot.models import (
    CheckedReference,
    ExternalRecord,
    HallucinationVerdict,
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Report,
    Severity,
    Verdict,
)
from refcopilot.pipeline import RefCopilotPipeline

__all__ = [
    "CheckedReference",
    "ExternalRecord",
    "HallucinationVerdict",
    "Issue",
    "IssueCategory",
    "MergedRecord",
    "Reference",
    "RefCopilotPipeline",
    "Report",
    "Severity",
    "Verdict",
]

__version__ = "0.1.0"
