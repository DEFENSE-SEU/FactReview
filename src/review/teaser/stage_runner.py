"""Review teaser sub-stage.

Reads the report sub-stage's clean markdown (without the refcheck section, so
teaser prompts stay on the actual review content) and produces a teaser figure
prompt + image under ``stages/review/teaser/``.

When ``GEMINI_API_KEY`` is set and ``TEASER_USE_GEMINI`` is not ``false`` the
teaser image is generated via Gemini; otherwise only the prompt is written.
"""

from __future__ import annotations

from pathlib import Path

from common.pipeline_context import (
    ensure_full_pipeline_context,
    execution_stage_dir,
    read_json_file,
    report_stage_dir,
    teaser_stage_dir,
    write_json_file,
)
from review.teaser.teaser import _env_true, generate_teaser_figure
from schemas.stage import StageResult


def run_teaser_stage(
    *,
    run_dir: Path,
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="teaser")

    report_dir = report_stage_dir(run_dir)
    clean_md = report_dir / "final_review_clean.md"
    final_md = report_dir / "final_review.md"
    source_md = clean_md if clean_md.exists() else final_md

    out_dir = teaser_stage_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teaser_json_path = out_dir / "teaser_figure.json"

    if not source_md.exists():
        # The report stage failed to produce a markdown — propagate as a hard
        # failure so callers and CI signal correctly instead of silently skipping.
        error_msg = (
            f"no review markdown produced by the report stage (checked {clean_md.name} and {final_md.name})"
        )
        payload = {
            "status": "failed",
            "message": error_msg,
            "source_markdown_path": "",
            "prompt_path": "",
            "image_path": "",
        }
        write_json_file(teaser_json_path, payload)
        return StageResult(
            status="failed",
            outputs={"main": str(teaser_json_path), "json": str(teaser_json_path)},
            error=error_msg,
        )

    execution_payload = read_json_file(execution_stage_dir(run_dir) / "execution.json")
    execution_skipped = execution_payload.get("status") == "skipped"

    use_gemini = _env_true("TEASER_USE_GEMINI", default=True)
    teaser_result = generate_teaser_figure(
        source_md,
        output_dir=out_dir,
        generate_image=use_gemini,
        execution_skipped=execution_skipped,
    )
    payload = {
        "status": teaser_result.status,
        "message": teaser_result.message,
        "clipboard_copied": teaser_result.clipboard_copied,
        "used_gemini_api": teaser_result.used_gemini_api,
        "model": teaser_result.model,
        "source_markdown_path": teaser_result.source_markdown_path,
        "prompt_path": teaser_result.prompt_path,
        "prompt": teaser_result.prompt,
        "image_path": teaser_result.image_path,
        "response_path": teaser_result.response_path,
    }
    write_json_file(teaser_json_path, payload)

    # Also append the teaser payload to the review JSON so a single artifact
    # captures the final review (report + teaser) outcome.
    review_json_path = report_dir / "final_review.json"
    review_payload = read_json_file(review_json_path)
    if review_payload:
        review_payload["teaser_figure"] = payload
        write_json_file(review_json_path, review_payload)

    # ``main`` is the canonical user-facing artifact: the generated image when
    # Gemini was used, otherwise the prompt that the user pastes into the
    # Gemini web app to produce one.
    outputs: dict[str, str] = {"json": str(teaser_json_path)}
    if teaser_result.prompt_path:
        outputs["prompt"] = str(teaser_result.prompt_path)
    if teaser_result.image_path:
        outputs["image"] = str(teaser_result.image_path)
    outputs["main"] = outputs.get("image") or outputs.get("prompt") or outputs["json"]
    # generate_teaser_figure() returns "prompt_only" or "generated"; both are
    # successful outcomes for this stage. The granular value is kept in the
    # teaser_figure payload for callers that care.
    return StageResult(
        status="ok",
        outputs=outputs,
        extra={"teaser_figure": payload},
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
