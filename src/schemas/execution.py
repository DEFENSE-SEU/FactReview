"""Execution schemas — the output of §3.3 execution-based verification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.stage import StageStatus

ExecutionExitStatus = Literal["success", "inconclusive", "failed", "skipped"]
# Same set of values as ``StageStatus`` — kept as an alias rather than a
# duplicate Literal so that adding a new status only needs one edit.
ExecutionStageStatus = StageStatus


class ExecutionPayload(BaseModel):
    """Canonical shape of ``stages/fact_generation/execution/execution.json``.

    Both the run path (``run_execution_stage``) and the skipped path
    (``pipeline_full`` when ``--run-execution`` is off) write this shape so
    downstream consumers (the report stage) don't need to defend against two
    schemas. A skipped payload sets ``status="skipped"`` and leaves run-time
    fields at their defaults — distinguishable via ``status`` rather than
    presence/absence of keys.
    """

    model_config = ConfigDict(extra="ignore")

    paper_key: str
    paper_pdf: str
    status: ExecutionStageStatus
    exit_status: ExecutionExitStatus
    success: bool = False
    run_dir: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    alignment: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A single reproducibility task derived from the paper/repo."""

    model_config = ConfigDict(extra="ignore")

    id: str  # e.g. "eval_fb237_conve"
    name: str
    command: list[str] = Field(default_factory=list)
    workdir: str = "."
    timeout_sec: int = 3600
    expects_metrics: list[str] = Field(default_factory=list)
    enabled: bool = True
    claim_ids: list[str] = Field(default_factory=list)  # which Claims this task serves
    description: str = ""


class RunArtifact(BaseModel):
    """A file/folder produced by executing a task."""

    path: str  # relative to run directory
    kind: str  # "log", "metric", "checkpoint", "image", …
    size_bytes: int | None = None


class ExecutionEvidence(BaseModel):
    """Per-task evidence record consumed by §3.4 review."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    claim_ids: list[str] = Field(default_factory=list)
    success: bool
    return_code: int | None = None
    duration_sec: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[RunArtifact] = Field(default_factory=list)
    # Aligned results vs paper claims ("supported", "partial", "conflict", "unknown").
    alignment: dict[str, str] = Field(default_factory=dict)
    error: str = ""
    notes: str = ""
