"""Uniform return type for pipeline stages.

Every ``run_<stage>_stage(...)`` entry point returns a :class:`StageResult`.
``pipeline_full.run_full_pipeline`` reads ``status`` to drive the run summary,
``outputs`` to surface artifact paths to users, and ``extra`` for the
stage-specific bits (job ids, refcheck payload, …).

Conventions:

* ``outputs["main"]`` is the primary on-disk artifact this stage produced.
  For most stages this is the per-stage JSON summary
  (``stages/<group>/<stage>/<name>.json``); for report it is the rendered
  review markdown; for teaser it is the generated image when present, else
  the teaser prompt.
* ``outputs["main"]`` MUST point to a real file on disk whenever
  ``status == "ok"``. The contract for ``status in ("failed", "skipped")``
  is softer: ``main`` may be present (e.g. an error-detail JSON) or absent
  (when the stage couldn't produce any artifact at all).
* Granular keys (``markdown``, ``pdf``, ``audit_json``, ``prompt``,
  ``image``, …) name additional artifacts when a stage emits more than one;
  they may be absent when the corresponding file wasn't produced.
* ``status`` values are deliberately coarse — finer-grained outcomes
  (e.g. teaser's ``prompt_only`` vs ``generated``) live in ``extra``.
* ``error`` is a short, user-facing one-liner explaining *why* the stage
  failed. Stages MUST populate it whenever ``status == "failed"`` so
  ``pipeline_full.run_full_pipeline`` can surface it under
  ``summary["stage_errors"]`` without users having to open per-stage JSON.
  For ``ok`` / ``skipped`` it is empty by default; ``inconclusive`` may
  optionally carry context (e.g. the orchestrator's exit reason).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StageStatus = Literal["ok", "skipped", "failed", "inconclusive"]


class StageResult(BaseModel):
    """Common shape returned by every pipeline stage.

    ``model_config = extra="forbid"`` — typos in keyword arguments raise
    ``ValidationError`` instead of being silently dropped. This is what
    catches mistakes like ``StageResult(..., extras={...})`` (note the
    trailing ``s``) which would otherwise lose the metadata silently.
    """

    model_config = ConfigDict(extra="forbid")

    status: StageStatus
    outputs: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    error: str = ""

    def get_output(self, key: str, default: str = "") -> str:
        return self.outputs.get(key, default)


__all__ = ["StageResult", "StageStatus"]
