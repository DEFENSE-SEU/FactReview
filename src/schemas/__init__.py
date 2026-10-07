"""Cross-stage Pydantic contracts.

Every data structure that crosses a module boundary in the project
is defined here. Internal per-module types stay local.
"""

from __future__ import annotations

from schemas.claim import (
    AuthorQuestion,
    Claim,
    ClaimLocation,
    ClaimStatus,
    Condition,
    Evidence,
    EvidenceNeed,
    EvidencePointer,
    ExecutionPlan,
    ExecutionProvenance,
    ExecutionTask,
    Finding,
)
from schemas.execution import (
    ExecutionEvidence,
    ExecutionExitStatus,
    ExecutionPayload,
    ExecutionStageStatus,
    RunArtifact,
    Task,
)
from schemas.paper import Figure, Paper, PaperMetadata, Section, Table
from schemas.positioning import LiteratureContext, NeighborMethod, NoveltyType
from schemas.review import FinalReview
from schemas.stage import StageResult, StageStatus

__all__ = [
    "AuthorQuestion",
    "Claim",
    "ClaimLocation",
    "ClaimStatus",
    "Condition",
    "Evidence",
    "EvidenceNeed",
    "EvidencePointer",
    "ExecutionEvidence",
    "ExecutionExitStatus",
    "ExecutionPayload",
    "ExecutionPlan",
    "ExecutionProvenance",
    "ExecutionStageStatus",
    "ExecutionTask",
    "Figure",
    "FinalReview",
    "Finding",
    "LiteratureContext",
    "NeighborMethod",
    "NoveltyType",
    "Paper",
    "PaperMetadata",
    "RunArtifact",
    "Section",
    "StageResult",
    "StageStatus",
    "Table",
    "Task",
]
