"""The v2 stage sequence, using existing external-service infrastructure."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from assessment import assess_claims
from common import run_stats
from fact_generation.execution.v2 import execute_plans
from fact_generation.execution.v2_config import ExecutionConfig
from preprocessing.materials import index_repository, parse_materials
from review.report.v2 import write_review
from review.teaser.v2 import write_teaser
from schemas.review import FinalReview
from screening.stage import screen_paper
from util.paper_input import infer_paper_key, materialize_paper_pdf
from util.run_layout import build_run_dir, make_run_id
from util.submission_cutoff import resolve_arxiv_first_submission
from verification.dispatch import verify_claims

STAGES = ("materials", "screening", "verification", "execution", "assessment", "report", "teaser")
STATS_MODULES = {
    "materials": "parse",
    "screening": "analysis",
    "verification": "analysis",
    "execution": "execution",
    "assessment": "analysis",
    "report": "report_generation",
    "teaser": "teaser_figure",
}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _repository(url: str, destination: Path) -> Path:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Repository URL must be an HTTPS URL without embedded credentials")
    if destination.exists():
        raise ValueError("Repository snapshot destination already exists")
    response = subprocess.run(
        ["git", "clone", "--depth", "1", "--", url, str(destination)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if response.returncode:
        raise RuntimeError(f"Repository snapshot failed: {response.stderr.strip()}")
    return destination


def _interactive_approval(plan, estimated_cost):
    print(plan.model_dump_json(indent=2), flush=True)
    print(f"Estimated cost: {estimated_cost}", flush=True)
    return input("Approve this execution plan? [y/N] ").strip().lower() in {"y", "yes"}


def run_v2_pipeline(
    args,
    *,
    parser=None,
    call=None,
    reference_checker=None,
    branches=None,
    global_literature=None,
    runner=None,
    approver=None,
    repairer=None,
    render_pdf=True,
) -> dict:
    source = str(args.paper_pdf)
    key = str(getattr(args, "paper_key", "") or infer_paper_key(source))
    run_id = f"{make_run_id()}_{uuid.uuid4().hex[:8]}"
    root = build_run_dir(getattr(args, "run_root", "runs"), key, run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with run_stats.run_scope(root / "run_stats.json"):
        summary = {
            "method_version": "v2",
            "paper_key": key,
            "run_id": run_id,
            "run_dir": str(root),
            "paper_source": source,
            "stages": dict.fromkeys(STAGES, "pending"),
            "stage_durations_sec": {},
            "stage_errors": {},
            "outputs": {},
            "issues": [],
        }
        current_stage = "materials"
        started = time.perf_counter()
        stage_started = started

        def record_stage_duration(name, duration):
            # Reference checking runs inside screening and has its own stats row.
            # Keep module totals exclusive while the screening stage retains wall time.
            nested = (
                run_stats.read()["modules"]["reference_check"]["duration_sec"] if name == "screening" else 0
            )
            run_stats.record_duration(STATS_MODULES[name], max(0, duration - nested))

        def stage(name, function):
            nonlocal current_stage, stage_started
            current_stage = name
            begin = time.perf_counter()
            stage_started = begin
            print(f"[{STAGES.index(name) + 1}/{len(STAGES)}] {name}: starting", flush=True)
            stats_module = STATS_MODULES[name]
            with run_stats.module_scope(stats_module):
                value = function()
            duration = time.perf_counter() - begin
            summary["stage_durations_sec"][name] = duration
            summary["stages"][name] = "ok"
            record_stage_duration(name, duration)
            run_stats.record_module_status(stats_module, "ok")
            _save(root / "full_pipeline_summary.json", summary)
            return value

        try:
            unsupported = [
                name
                for name in (
                    "reuse_job_id",
                    "execution_auto_tasks",
                    "execution_auto_tasks_force",
                    "execution_paper_budget_sec",
                    "no_cutoff",
                )
                if getattr(args, name, None)
            ]
            if getattr(args, "teaser_mode", "auto") == "api":
                unsupported.append("teaser_mode=api (v2 generates a deterministic SVG and image prompt)")
            if unsupported:
                raise ValueError("Unsupported v2 options: " + ", ".join(unsupported))
            from util.cutoff_date import concurrent_window_start, parse_cutoff, parse_submission_deadline

            deadline = str(getattr(args, "submission_deadline", "") or "").strip()
            provenance = {"source": "unresolved", "value": None, "venue_deadline_known": False}
            if deadline:
                parsed_deadline = parse_submission_deadline(deadline)
                provenance = {
                    "source": "submission_deadline",
                    "value": deadline,
                    "venue_deadline_known": True,
                }
            elif str(getattr(args, "cutoff_date", "") or "").strip():
                explicit = parse_cutoff(args.cutoff_date)
                if explicit.precision == "day":
                    deadline, parsed_deadline = explicit.to_string(), explicit
                    provenance.update(source="explicit_cutoff", value=deadline)
                else:
                    summary["issues"].append(
                        "V2 requires a day-level cutoff; the explicit coarse cutoff remains unresolved."
                    )
            elif getattr(args, "derive_cutoff_from_arxiv", False):
                try:
                    parsed_deadline, provenance = asyncio.run(
                        resolve_arxiv_first_submission(str(getattr(args, "arxiv_id", "") or source))
                    )
                    deadline = parsed_deadline.to_string()
                    summary["issues"].append(
                        f"Literature cutoff uses arXiv first submission ({deadline}); the venue deadline is unknown."
                    )
                except Exception as exc:
                    provenance["error"] = f"{type(exc).__name__}: {exc}"
                    summary["issues"].append(f"arXiv cutoff fallback unresolved: {exc}")
            if deadline:
                summary["submission_deadline"] = deadline
                summary["concurrent_start"] = concurrent_window_start(parsed_deadline).isoformat()
            else:
                summary["issues"].append("Submission deadline missing; Literature cannot assess prior work.")
            summary["cutoff"] = provenance
            _save(root / "cutoff.json", provenance)
            summary["outputs"]["cutoff"] = str(root / "cutoff.json")
            paper = materialize_paper_pdf(source, root / "inputs" / "source_pdf", paper_key=key)
            repository_root = str(getattr(args, "repository_root", "") or "")
            repository_url = str(getattr(args, "repository_url", "") or "")

            def materials_stage():
                materials = asyncio.run(
                    parse_materials(
                        paper_pdf=paper.path,
                        output_dir=root / "materials",
                        paper_key=key,
                        repo_root=Path(repository_root) if repository_root else None,
                        parser=parser,
                    )
                )
                remote_url = repository_url
                if materials.repository is None and not remote_url:
                    urls = set(
                        re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", materials.markdown)
                    )
                    if len(urls) == 1:
                        remote_url = next(iter(urls)).rstrip(".")
                    elif len(urls) > 1:
                        materials.issues.append(
                            "Multiple repository URLs; specify --repository-url for the released source."
                        )
                if materials.repository is None and remote_url:
                    try:
                        snapshot = _repository(remote_url, root / "materials" / "repository")
                        materials.repository = index_repository(snapshot)
                        summary["repository_url"] = remote_url
                    except Exception as exc:
                        materials.issues.append(str(exc))
                (root / "materials" / "materials.json").write_text(
                    materials.model_dump_json(indent=2), encoding="utf-8"
                )
                return materials

            materials = stage("materials", materials_stage)
            summary["outputs"]["materials"] = str(root / "materials" / "materials.json")
            summary["issues"].extend(materials.issues)
            screening = stage(
                "screening",
                lambda: screen_paper(
                    materials, root / "screening", call=call, reference_checker=reference_checker
                ),
            )
            summary["issues"].extend(screening.issues)
            summary["outputs"]["screening"] = str(root / "screening" / "screening.json")
            verification = stage(
                "verification",
                lambda: asyncio.run(
                    verify_claims(
                        screening.claims,
                        materials,
                        root / "verification",
                        submission_deadline=deadline,
                        branches=branches,
                        global_literature=global_literature,
                        call=call,
                    )
                ),
            )
            summary["issues"].extend(verification.issues)
            summary["outputs"]["verification"] = str(root / "verification" / "verification.json")
            claims, ledger = verification.claims, []
            if getattr(args, "run_execution", False):
                current_stage = "execution"
                stage_started = time.perf_counter()
                if getattr(args, "execution_no_docker", False):
                    raise ValueError("V2 execution requires Docker")
                config_path = str(getattr(args, "execution_config", "") or "")
                config_data = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
                overrides = getattr(args, "_execution_overrides", set())
                for argument, field, default in (
                    ("max_attempts", "max_attempts", 3),
                    ("approval_mode", "approval_mode", "auto"),
                    ("training_budget", "training_budget", 0),
                    ("execution_docker_build_timeout_sec", "docker_build_timeout_seconds", 3600),
                ):
                    if not config_path or argument in overrides:
                        value = getattr(args, argument, default)
                        config_data[field] = (
                            default if field == "docker_build_timeout_seconds" and value == 0 else value
                        )
                if not config_path or "execution_no_llm" in overrides:
                    config_data["refine_with_llm"] = not getattr(args, "execution_no_llm", False)
                config = ExecutionConfig.model_validate(config_data)
                execution = stage(
                    "execution",
                    lambda: execute_plans(
                        verification.plans,
                        claims,
                        materials,
                        root / "execution",
                        config=config,
                        runner=runner,
                        repairer=repairer,
                        approver=approver
                        or (_interactive_approval if config.approval_mode == "interactive" else None),
                    ),
                )
                claims, ledger = execution.claims, execution.ledger
                summary["issues"].extend(execution.issues)
            else:
                summary["stages"]["execution"] = "skipped"
                run_stats.record_module_status("execution", "skipped")
                summary["issues"].append("Execution disabled; released artifacts were not run.")
            claims = stage("assessment", lambda: assess_claims(claims))
            review = FinalReview(
                paper_key=key,
                run_id=run_id,
                claims=claims,
                findings=screening.findings + verification.findings,
                ledger=ledger,
            )
            usage = run_stats.with_totals(run_stats.read(root / "run_stats.json"))["total"]
            token_usage = (
                {**usage["token_usage"], "estimated": usage["estimated"]}
                if usage["token_usage"]["requests"]
                else None
            )
            outputs = stage(
                "report",
                lambda: write_review(
                    review,
                    root / "review" / "report",
                    issues=summary["issues"],
                    render_pdf=render_pdf,
                    token_usage=token_usage,
                ),
            )
            summary["outputs"].update(
                {f"report_{name}": path for name, path in outputs.items() if name != "pdf_error"}
            )
            if outputs.get("pdf_error"):
                summary["stage_errors"]["report_pdf"] = outputs["pdf_error"]
            outputs = stage("teaser", lambda: write_teaser(review, root / "review" / "teaser"))
            summary["outputs"].update({f"teaser_{name}": path for name, path in outputs.items()})
            summary["counts"] = {status.value: count for status, count in review.summary_counts.items()}
        except Exception as exc:
            summary["stages"][current_stage] = "failed"
            summary["stage_errors"][current_stage] = f"{type(exc).__name__}: {exc}"
            duration = time.perf_counter() - stage_started
            summary["stage_durations_sec"][current_stage] = duration
            record_stage_duration(current_stage, duration)
            run_stats.record_module_status(STATS_MODULES[current_stage], "failed", warning=str(exc))
        finally:
            summary["stages"] = {
                name: "skipped" if value == "pending" else value for name, value in summary["stages"].items()
            }
            summary["duration_seconds"] = time.perf_counter() - started
            for module, record in run_stats.read(root / "run_stats.json")["modules"].items():
                if record["status"] == "pending":
                    run_stats.record_module_status(module, "skipped")
            run_stats.set_pipeline_duration(summary["duration_seconds"])
            summary["run_stats"] = run_stats.with_totals(run_stats.read(root / "run_stats.json"))
            _save(root / "run_stats.json", summary["run_stats"])
            summary["outputs"]["run_stats"] = str(root / "run_stats.json")
            _save(root / "full_pipeline_summary.json", summary)
        return summary
