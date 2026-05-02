"""Tests for the uniform StageResult schema.

These guard the contract that pipeline_full.py and every stage_runner depends
on: a coarse Literal status, an outputs dict with a canonical "main" key, and
a stage-specific extra dict. Without these tests the only thing catching a
regression (e.g. a stage returning ``status="prompt_only"``) is a real
pipeline run.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schemas.stage import StageResult, StageStatus


def test_status_accepts_all_literal_values() -> None:
    for status in ("ok", "skipped", "failed", "inconclusive"):
        StageResult(status=status)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_status",
    [
        "prompt_only",  # the teaser bug we just fixed
        "generated",
        "success",
        "OK",
        "",
    ],
)
def test_status_rejects_non_literal_strings(bad_status: str) -> None:
    with pytest.raises(ValidationError):
        StageResult(status=bad_status)  # type: ignore[arg-type]


def test_default_outputs_extra_error() -> None:
    r = StageResult(status="ok")
    assert r.outputs == {}
    assert r.extra == {}
    assert r.error == ""


def test_round_trip_preserves_payload() -> None:
    original = StageResult(
        status="ok",
        outputs={"main": "/run/x.json", "pdf": "/run/x.pdf"},
        extra={"job_id": "abc-123", "nested": {"a": 1}},
        error="",
    )
    blob = json.dumps(original.model_dump(), ensure_ascii=False)
    restored = StageResult.model_validate_json(blob)
    assert restored == original


def test_get_output_helper_returns_default() -> None:
    r = StageResult(status="ok", outputs={"main": "/x"})
    assert r.get_output("main") == "/x"
    assert r.get_output("missing") == ""
    assert r.get_output("missing", default="fallback") == "fallback"


def test_stage_status_alias_matches_model_field() -> None:
    """If StageStatus drifts from StageResult.status, downstream
    ``status: StageStatus`` annotations in stage_runners would silently
    accept stale values."""
    field_annotation = StageResult.model_fields["status"].annotation
    assert field_annotation is StageStatus


def test_error_field_round_trips_when_populated() -> None:
    """Stages must populate ``error`` whenever ``status='failed'`` so
    pipeline_full.py can surface failures in summary['stage_errors'] without
    users having to open per-stage JSON. This guards the round-trip so a
    refactor that drops the field at serialization is caught immediately."""
    r = StageResult(status="failed", error="MinerU markdown not produced")
    assert r.error == "MinerU markdown not produced"
    blob = json.dumps(r.model_dump(), ensure_ascii=False)
    restored = StageResult.model_validate_json(blob)
    assert restored.error == r.error


def test_error_is_orthogonal_to_status() -> None:
    """The schema does not enforce ``error`` non-empty when status is failed
    (the contract is *strongly recommended* but not validated). This test
    pins that intentional looseness — if we later tighten it, this test
    flips and forces us to update every stage_runner accordingly."""
    StageResult(status="failed", error="")  # legal: a stage may still be migrating
    StageResult(status="ok", error="non-fatal warning")  # legal: ok stages may carry context
    StageResult(status="inconclusive", error="max_attempts reached")  # legal: inconclusive context


def test_unknown_field_typo_is_rejected_not_silently_dropped() -> None:
    """``extra="forbid"`` makes typos like ``extras=`` (vs ``extra=``)
    or ``output=`` (vs ``outputs=``) raise instead of silently swallowing
    the data — guards against losing whole payloads to a one-letter typo.
    """
    with pytest.raises(ValidationError):
        StageResult(status="ok", extras={"job_id": "abc"})  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        StageResult(status="ok", output={"main": "/x"})  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        StageResult(status="ok", err="oops")  # type: ignore[call-arg]
