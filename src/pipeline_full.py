from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.env import load_env_file
from common.pipeline_context import execution_stage_dir, init_full_pipeline_context
from fact_generation.execution.stage_runner import run_execution_stage
from fact_generation.positioning.stage_runner import run_positioning_stage
from fact_generation.refcheck.stage_runner import run_refcheck_stage
from llm.provider_capabilities import is_codex_provider
from preprocessing.claim_extract.stage_runner import run_claim_extract_stage
from preprocessing.parse.stage_runner import run_parse_stage
from review.report.stage_runner import run_report_stage
from review.teaser.stage_runner import run_teaser_stage
from schemas.execution import ExecutionPayload
from schemas.stage import StageResult
from util.paper_input import infer_paper_key, materialize_paper_pdf
from util.run_layout import build_run_dir, ensure_run_subdirs, make_run_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_TOTAL_STAGES = 7


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def _run_stage(index: int, name: str, fn: Callable[[], StageResult]) -> StageResult:
    _log(f"[{index}/{_TOTAL_STAGES}] {name}: starting...")
    t0 = time.monotonic()
    result = fn()
    dt = time.monotonic() - t0
    status = result.status
    err = (result.error or "").strip()
    suffix = f" — {err}" if err else ""
    _log(f"[{index}/{_TOTAL_STAGES}] {name}: {status} ({dt:.1f}s){suffix}")
    return result


def _set_env_if_value(name: str, value: str | None) -> None:
    token = str(value or "").strip()
    if token:
        os.environ[name] = token


def _apply_cli_env_overrides(args: argparse.Namespace) -> None:
    """Mirror CLI overrides into ``os.environ`` and invalidate the settings cache.

    Must be called *before* the first ``get_settings()`` in the same process,
    because ``get_settings`` is ``lru_cache``-d. The trailing ``cache_clear()``
    only handles the case where ``get_settings()`` was called *during* this
    function (e.g. inside ``_set_env_if_value``); subsequent in-process callers
    that import ``common.config.get_settings`` directly will already see the
    new env values on first call.
    """
    llm_provider = str(getattr(args, "llm_provider", "") or "").strip()
    if llm_provider:
        os.environ["MODEL_PROVIDER"] = llm_provider
        os.environ["EXECUTION_MODEL_PROVIDER"] = llm_provider
    _set_env_if_value("MINERU_API_TOKEN", getattr(args, "mineru_api_token", ""))
    _set_env_if_value("GEMINI_API_KEY", getattr(args, "gemini_api_key", ""))

    llm_model = str(getattr(args, "llm_model", "") or "").strip()
    if llm_model:
        os.environ["AGENT_MODEL"] = llm_model
        os.environ["EXECUTION_OPENAI_MODEL"] = llm_model
        provider = str(getattr(args, "llm_provider", "") or os.getenv("MODEL_PROVIDER") or "").strip()
        if is_codex_provider(provider):
            os.environ["OPENAI_CODEX_MODEL"] = llm_model

    teaser_mode = str(getattr(args, "teaser_mode", "auto") or "auto").strip().lower()
    if teaser_mode == "prompt":
        os.environ["TEASER_USE_GEMINI"] = "false"
    elif teaser_mode == "api":
        os.environ["TEASER_USE_GEMINI"] = "true"

    get_settings.cache_clear()


def run_full_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    settings = get_settings()
    paper_source = str(args.paper_pdf or "").strip()
    paper_key = (args.paper_key or "").strip() or infer_paper_key(paper_source)
    run_id = make_run_id()
    run_dir = build_run_dir(args.run_root, paper_key, run_id)
    layout = ensure_run_subdirs(run_dir)
    paper_input = materialize_paper_pdf(
        paper_source,
        layout["inputs"] / "source_pdf",
        paper_key=paper_key,
    )
    paper_pdf = paper_input.path
    init_full_pipeline_context(run_dir=run_dir)
    run_execution = bool(getattr(args, "run_execution", False))

    _log("FactReview pipeline starting")
    _log(f"  paper_key : {paper_key}")
    _log(f"  paper_pdf : {paper_pdf}")
    _log(f"  run_dir   : {run_dir}")

    parse_result = _run_stage(
        1,
        "parse",
        lambda: run_parse_stage(
            repo_root=repo_root,
            run_dir=run_dir,
            paper_pdf=paper_pdf,
            paper_key=paper_key,
            reuse_job_id=str(args.reuse_job_id or "").strip(),
            materialize_execution_extract=run_execution,
        ),
    )
    claim_extract_result = _run_stage(
        2,
        "claim_extract",
        lambda: run_claim_extract_stage(
            repo_root=repo_root,
            run_dir=run_dir,
        ),
    )
    enable_refcheck = bool(getattr(args, "enable_refcheck", False) or settings.reference_check_enabled)
    refcheck_result = _run_stage(
        3,
        "refcheck",
        lambda: run_refcheck_stage(
            repo_root=repo_root,
            run_dir=run_dir,
            paper_pdf=paper_pdf,
            paper_key=paper_key,
            enable_refcheck=enable_refcheck,
        ),
    )
    positioning_result = _run_stage(
        4,
        "positioning",
        lambda: run_positioning_stage(
            repo_root=repo_root,
            run_dir=run_dir,
        ),
    )
    if not run_execution:
        _log(f"[5/{_TOTAL_STAGES}] execution: skipped (use --run-execution to enable)")
        skipped_payload = ExecutionPayload(
            paper_key=paper_key,
            paper_pdf=str(paper_pdf),
            status="skipped",
            exit_status="skipped",
        )
        execution_out = execution_stage_dir(run_dir) / "execution.json"
        _write_json(execution_out, skipped_payload.model_dump())
        execution_result = StageResult(
            status="skipped",
            outputs={"main": str(execution_out)},
            extra={"run_dir": ""},
        )
    else:
        execution_result = _run_stage(
            5,
            "execution",
            lambda: run_execution_stage(
                run_dir=run_dir,
                paper_pdf=paper_pdf,
                paper_key=paper_key,
                paper_extracted_dir=str(
                    (parse_result.extra.get("shared_execution_extract") or {}).get("paper_extracted_dir")
                    or ""
                ),
                max_attempts=int(args.max_attempts),
                no_pdf_extract=bool(args.no_pdf_extract),
            ),
        )
    report_result = _run_stage(
        6,
        "report",
        lambda: run_report_stage(
            repo_root=repo_root,
            run_dir=run_dir,
        ),
    )
    teaser_result = _run_stage(7, "teaser", lambda: run_teaser_stage(run_dir=run_dir))

    results: dict[str, StageResult] = {
        "parse": parse_result,
        "claim_extract": claim_extract_result,
        "refcheck": refcheck_result,
        "positioning": positioning_result,
        "execution": execution_result,
        "report": report_result,
        "teaser": teaser_result,
    }
    statuses = {name: r.status for name, r in results.items()}
    stage_errors = {name: r.error for name, r in results.items() if r.error}
    # PDF render is best-effort within the report stage (markdown is canonical),
    # so the stage stays status="ok" even when the PDF fails. Surface the cause
    # at summary level so users don't have to open per-stage extras to find it.
    report_pdf_error = str(report_result.extra.get("pdf_render_error") or "").strip()
    if report_pdf_error and "report" not in stage_errors:
        stage_errors["report_pdf"] = report_pdf_error

    outputs: dict[str, str] = {}
    for name in ("parse", "claim_extract", "refcheck", "positioning", "execution"):
        main = results[name].outputs.get("main")
        if main:
            outputs[name] = main
    granular = (
        ("refcheck_md", refcheck_result.outputs.get("markdown")),
        ("report_json", report_result.outputs.get("json")),
        ("report_md", report_result.outputs.get("markdown")),
        ("report_audit_json", report_result.outputs.get("audit_json")),
        ("report_pdf", report_result.outputs.get("pdf")),
        ("teaser_figure_prompt", teaser_result.outputs.get("prompt")),
        ("teaser_figure_image", teaser_result.outputs.get("image")),
    )
    for key, value in granular:
        if value:
            outputs[key] = value

    summary = {
        "paper_key": paper_key,
        "paper_source": paper_input.source,
        "paper_source_type": paper_input.source_type,
        "paper_pdf": str(paper_pdf),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "job_id": parse_result.extra.get("job_id"),
        "job_dir": parse_result.extra.get("job_dir"),
        "stages": statuses,
        "stage_errors": stage_errors,
        "outputs": outputs,
        "reference_check": refcheck_result.extra.get("reference_check") or {"enabled": enable_refcheck},
        "teaser_figure": teaser_result.extra.get("teaser_figure") or {},
    }

    summary_path = run_dir / "full_pipeline_summary.json"
    _write_json(summary_path, summary)

    failed = [name for name, status in statuses.items() if status == "failed"]
    if failed:
        _log(f"Pipeline finished with failures in: {', '.join(failed)}")
        for name in failed:
            reason = stage_errors.get(name) or "(no reason recorded)"
            _log(f"  - {name}: {reason}")
    else:
        _log("Pipeline finished successfully.")
    review_md = outputs.get("report_md")
    if review_md:
        _log(f"Final review: {review_md}")
    _log(f"Summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("factreview_full_pipeline")
    p.add_argument("paper_pdf", type=str, help="Path or URL to a paper PDF")
    p.add_argument("--paper-key", type=str, default="")
    p.add_argument("--run-root", type=str, default="runs")
    p.add_argument("--reuse-job-id", type=str, default="", help="Reuse an existing runtime job snapshot")
    p.add_argument(
        "--llm-provider",
        type=str,
        default="",
        help="LLM provider override. Default is openai-codex after `codex login`.",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="LLM model override for the selected provider.",
    )
    p.add_argument(
        "--mineru-api-token",
        type=str,
        default="",
        help="MinerU API token override. Prefer MINERU_API_TOKEN in .env for routine use.",
    )
    p.add_argument(
        "--gemini-api-key",
        type=str,
        default="",
        help="Optional Gemini API key override for teaser image generation.",
    )
    p.add_argument(
        "--teaser-mode",
        choices=("auto", "prompt", "api"),
        default="auto",
        help="Teaser figure mode: auto attempts Gemini when a key exists, prompt saves/copies the prompt, api attempts the configured image API.",
    )
    p.add_argument(
        "--enable-refcheck",
        action="store_true",
        help="Run RefChecker reference-accuracy validation and append warning/error results to the final report.",
    )
    p.add_argument(
        "--run-execution",
        action="store_true",
        help="Run the repository execution stage. Disabled by default.",
    )
    p.add_argument("--max-attempts", type=int, default=5, help="Execution-stage max fix loop attempts")
    p.add_argument(
        "--no-pdf-extract",
        action="store_true",
        help="Pass through to external execution stage (skip MinerU in execution prepare).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    _apply_cli_env_overrides(args)
    summary = run_full_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
