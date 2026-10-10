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
from fact_generation.execution.recovery import recover_execution_records
from fact_generation.execution.v2 import ExecutionResult, execute_plans
from fact_generation.execution.v2_config import ExecutionConfig
from preprocessing.materials import index_repository, parse_materials
from review.delivery import checked_delivery
from review.recovery import finalize_teaser_review, interrupted_report_history, write_recovery_review
from review.report.advice import generate_advice
from review.report.v2 import verification_limitations, write_review
from review.teaser.v2 import teaser_payload, write_teaser
from schemas.limitations import VerificationLimitation
from schemas.review import DeliveryCheck, FinalReview
from screening.stage import ScreeningFailure, screen_paper
from util.paper_input import infer_paper_key, materialize_paper_pdf
from util.run_layout import build_run_dir, make_run_id
from util.submission_cutoff import resolve_arxiv_first_submission
from verification.dispatch import VerificationResult, verify_claims

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


def _model_usage(stats):
    total = stats["total"]
    usage = {
        **total["token_usage"],
        "estimated": total["estimated"],
        **{
            key: total[key]
            for key in ("failed_requests", "unavailable_usage_requests", "image_count", "warnings")
        },
    }
    if not usage["requests"] or (
        usage["unavailable_usage_requests"] == usage["requests"] and not usage["estimated_requests"]
    ):
        usage["unavailable"] = True
    return usage


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
            "report_presentation": getattr(args, "report_presentation", "full"),
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

        def stage(name, function, *, recover=None):
            nonlocal current_stage, stage_started
            current_stage = name
            begin = time.perf_counter()
            stage_started = begin
            summary["stages"][name] = "running"
            _save(root / "full_pipeline_summary.json", summary)
            print(f"[{STAGES.index(name) + 1}/{len(STAGES)}] {name}: starting", flush=True)
            stats_module = STATS_MODULES[name]
            failure = None
            try:
                with run_stats.module_scope(stats_module):
                    value = function()
            except Exception as exc:
                if recover is None:
                    raise
                value = recover(exc)
                failure = f"{type(exc).__name__}: {exc}"
                summary["stage_errors"][name] = failure
                summary["issues"].append(f"{name} stage failed: {failure}")
            duration = time.perf_counter() - begin
            summary["stage_durations_sec"][name] = duration
            summary["stages"][name] = "failed" if failure else "ok"
            record_stage_duration(name, duration)
            failed = any(summary["stages"][s] == "failed" for s in STAGES if STATS_MODULES[s] == stats_module)
            run_stats.record_module_status(stats_module, "failed" if failed else "ok", warning=failure)
            _save(root / "full_pipeline_summary.json", summary)
            return value

        def recover_screening(exc):
            if not isinstance(exc, ScreeningFailure):
                raise exc
            return exc.result

        def skip_stage(name, reason):
            summary["stages"][name] = "skipped"
            summary["issues"].append(reason)
            if not any(
                summary["stages"][s] in {"ok", "failed"}
                for s in STAGES
                if STATS_MODULES[s] == STATS_MODULES[name]
            ):
                run_stats.record_module_status(STATS_MODULES[name], "skipped")

        try:
            if summary["report_presentation"] not in ("full", "layered"):
                raise ValueError("Unknown report presentation; choose full or layered")
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
                    urls = sorted(
                        set(
                            re.findall(
                                r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", materials.markdown
                            )
                        )
                    )
                    if urls:
                        summary["repository_candidates"] = [url.rstrip(".") for url in urls]
                        materials.issues.append(
                            "Manuscript repository URLs have not been bound to the authors' released source; "
                            "specify --repository-url or --repository-root. Third-party dependencies and "
                            "baseline links cannot establish release identity."
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
                    materials,
                    root / "screening",
                    call=call,
                    reference_checker=reference_checker,
                    anonymity_policy=getattr(args, "anonymity_policy", "unspecified"),
                    claim_coverage_window_chars=getattr(args, "claim_coverage_window_chars", 24000),
                    claim_coverage_review_calls=getattr(args, "claim_coverage_review_calls", 12),
                    claim_coverage_followup_calls=getattr(args, "claim_coverage_followup_calls", 12),
                    claim_coverage_validation_calls=getattr(args, "claim_coverage_validation_calls", 12),
                ),
                recover=recover_screening,
            )
            summary["issues"].extend(screening.issues)
            summary["figure_coverage"] = screening.figure_coverage
            summary["figure_context_coverage"] = screening.figure_context_coverage
            summary["table_coverage"] = screening.table_coverage
            summary["table_context_coverage"] = screening.table_context_coverage
            summary["writing_coverage"] = screening.writing_coverage
            summary["claim_coverage"] = screening.claim_coverage
            summary["anonymity_policy"] = screening.anonymity_policy
            summary["outputs"]["screening"] = str(root / "screening" / "screening.json")
            if screening.claim_coverage.get("audit_path"):
                summary["outputs"]["claim_coverage"] = screening.claim_coverage["audit_path"]
            extracted = [c.model_copy(deep=True) for c in screening.claims]

            def recover_verification(exc):
                retained = [c.model_copy(deep=True) for c in extracted]
                for claim in retained:
                    claim.notes.append(
                        f"Verification stage failed; no result was adopted: {type(exc).__name__}: {exc}"
                    )
                    claim.verification_limitations.append(
                        VerificationLimitation(
                            claim_id=claim.id,
                            condition_ids=[c.id for c in claim.conditions],
                            stage="verification",
                            kind="stage_failed",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
                return VerificationResult(claims=retained)

            if screening.claim_extraction_status == "failed":
                skip_stage("verification", "Verification skipped because claim extraction failed.")
                verification = VerificationResult(claims=[])
            else:
                verification = stage(
                    "verification",
                    lambda: asyncio.run(
                        verify_claims(
                            [c.model_copy(deep=True) for c in extracted],
                            materials,
                            root / "verification",
                            submission_deadline=deadline,
                            branches=branches,
                            global_literature=global_literature,
                            call=call,
                            blocked_claim_ids=screening.blocked_claim_ids,
                        )
                    ),
                    recover=recover_verification,
                )
            summary["issues"].extend(verification.issues)
            if summary["stages"]["verification"] == "ok":
                summary["outputs"]["verification"] = str(root / "verification" / "verification.json")
            claims, ledger = verification.claims, []
            verified = [c.model_copy(deep=True) for c in claims]

            def recover_execution(exc):
                retained = [c.model_copy(deep=True) for c in verified]
                for claim in retained:
                    claim.notes.append(
                        f"Execution stage failed; no new execution result was adopted: {type(exc).__name__}: {exc}"
                    )
                    claim.verification_limitations.append(
                        VerificationLimitation(
                            claim_id=claim.id,
                            condition_ids=[c.id for c in claim.conditions],
                            stage="execution",
                            kind="stage_failed",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
                records, issues = recover_execution_records(root / "execution", verification.plans)
                return ExecutionResult(claims=retained, ledger=records, issues=issues)

            def execution_stage():
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
                return execute_plans(
                    [plan.model_copy(deep=True) for plan in verification.plans],
                    [c.model_copy(deep=True) for c in verified],
                    materials,
                    root / "execution",
                    config=config,
                    runner=runner,
                    repairer=repairer,
                    approver=approver
                    or (_interactive_approval if config.approval_mode == "interactive" else None),
                )

            if summary["stages"]["verification"] != "ok":
                skip_stage("execution", "Execution skipped because verification did not complete.")
            elif getattr(args, "run_execution", False):
                execution = stage(
                    "execution",
                    execution_stage,
                    recover=recover_execution,
                )
                claims, ledger = execution.claims, execution.ledger
                summary["issues"].extend(execution.issues)
            else:
                skip_stage("execution", "Execution disabled; released artifacts were not run.")
            if screening.claim_extraction_status == "failed":
                skip_stage("assessment", "Assessment skipped because claim extraction failed.")
            else:
                claims = stage("assessment", lambda: assess_claims(claims))
            incomplete = [name for name in STAGES if summary["stages"][name] == "failed"]
            review = FinalReview(
                paper_key=key,
                run_id=run_id,
                claims=claims,
                findings=screening.findings + verification.findings,
                ledger=ledger,
                run_status="partial" if incomplete else "completed",
                incomplete_stages=incomplete,
                execution_requested=bool(getattr(args, "run_execution", False)),
            )
            review = checked_delivery(
                review, stages=summary["stages"], extraction_status=screening.claim_extraction_status,
                additional_checks=[*screening.delivery_checks, *verification.delivery_checks],
                **{name: summary[name] for name in (
                    "claim_coverage", "writing_coverage", "figure_coverage", "table_coverage",
                    "figure_context_coverage", "table_context_coverage",
                )},
            )
            _save(root / "assessment" / "assessed_review.json", review.model_dump(mode="json"))
            summary["outputs"]["assessment_snapshot"] = str(root / "assessment" / "assessed_review.json")

            def report_stage():
                nonlocal review
                review.advice_requested = True
                advice = generate_advice(review, root / "review" / "advice", call=call)
                review = advice.review
                review.advice_requested = True
                review = checked_delivery(review)
                summary["issues"].extend(advice.issues)
                summary["advice"] = advice.counts
                if (root / "review" / "advice").is_dir():
                    summary["outputs"]["advice"] = str(root / "review" / "advice")
                summary["model_usage"] = _model_usage(
                    run_stats.with_totals(run_stats.read(root / "run_stats.json"))
                )
                summary["issues"] = verification_limitations(
                    issues=summary["issues"],
                    figure_coverage=summary["figure_coverage"],
                    figure_context_coverage=summary["figure_context_coverage"],
                    table_coverage=summary["table_coverage"],
                    table_context_coverage=summary["table_context_coverage"],
                    writing_coverage=summary["writing_coverage"],
                    claim_coverage=summary["claim_coverage"],
                    anonymity_policy=summary["anonymity_policy"],
                    token_usage=summary["model_usage"],
                )
                rendered = write_review(
                    review,
                    root / "review" / "report",
                    issues=summary["issues"],
                    render_pdf=render_pdf,
                    token_usage=summary["model_usage"],
                    figure_coverage=summary["figure_coverage"],
                    figure_context_coverage=summary["figure_context_coverage"],
                    table_coverage=summary["table_coverage"],
                    table_context_coverage=summary["table_context_coverage"],
                    writing_coverage=summary["writing_coverage"],
                    claim_coverage=summary["claim_coverage"],
                    anonymity_policy=summary["anonymity_policy"],
                    presentation=summary["report_presentation"],
                )
                # Static rendering revalidates advice against the saved sources.
                # Use that persisted result for counts and downstream delivery.
                review = FinalReview.model_validate_json(Path(rendered["json"]).read_text(encoding="utf-8"))
                summary["advice"] = {
                    state: sum(c.advice is not None and c.advice.state == state for c in review.claims)
                    for state in ("generated", "unavailable")
                }
                return rendered

            def recovery_context():
                return {
                    "issues": summary["issues"], "token_usage": summary.get("model_usage"),
                    **{name: summary.get(name) for name in (
                        "figure_coverage", "figure_context_coverage", "table_coverage",
                        "table_context_coverage", "writing_coverage", "claim_coverage", "anonymity_policy",
                    )},
                }

            def recover_report(exc):
                nonlocal review
                history_errors = {}
                history = interrupted_report_history(root / "review" / "report", errors=history_errors)
                for key, artifact in history.items():
                    summary["outputs"]["history_report_" + key] = artifact["path"]
                checks = [DeliveryCheck(
                    stage="report", component="report_writer", state="failed",
                    reason=f"Report writer raised {type(exc).__name__}. Audit: {root / 'full_pipeline_summary.json'}#/stage_errors/report",
                )]
                if history_errors:
                    checks.append(DeliveryCheck(
                        stage="report", component="report_history", state="unavailable",
                        reason="Interrupted report history could not be completely read; per-file errors are retained in artifact_history_errors.json.",
                    ))
                if any(key.endswith("pdf") for key in (*history, *history_errors)):
                    checks.append(DeliveryCheck(
                        stage="report", component="pdf_delivery_finalization", state="incomplete",
                        reason="PDF files written before the report failure are retained as interrupted history; final delivery export is incomplete.",
                    ))
                review = checked_delivery(review, stages={"report": "failed"}, additional_checks=checks)
                review, recovered = write_recovery_review(
                    review, root / "review" / "report_recovery", render_narrative=False, **recovery_context(),
                )
                if history:
                    path = root / "review" / "report_recovery" / "artifact_history.json"
                    _save(path, history)
                    summary["outputs"]["history_report_artifacts"] = str(path)
                if history_errors:
                    path = root / "review" / "report_recovery" / "artifact_history_errors.json"
                    _save(path, history_errors)
                    summary["outputs"]["history_report_errors"] = str(path)
                return recovered

            outputs = stage("report", report_stage, recover=recover_report)
            summary["outputs"].update(
                {f"report_{name}": path for name, path in outputs.items() if not name.endswith("_error")}
            )
            for name, detail in outputs.items():
                if name.endswith("_error") and detail:
                    summary["stage_errors"][f"report_{name.removesuffix('_error')}"] = detail
            def recover_teaser(exc):
                nonlocal review
                checks = [DeliveryCheck(
                    stage="teaser", component="teaser_writer", state="failed",
                    reason=f"Teaser writer raised {type(exc).__name__}. Audit: {root / 'full_pipeline_summary.json'}#/stage_errors/teaser",
                )]
                review = checked_delivery(review, stages={"teaser": "failed"}, additional_checks=checks)
                review, recovered = finalize_teaser_review(
                    review, root / "review" / "delivery_recovery",
                    prior_outputs=summary["outputs"], report_succeeded=summary["stages"]["report"] == "ok",
                    presentation=summary["report_presentation"], **recovery_context(),
                )
                # Retain every successful earlier artifact under an explicit
                # historical role; the new paths carry final partial metadata.
                for key in list(summary["outputs"]):
                    if key.startswith("report_"):
                        summary["outputs"]["history_" + key] = summary["outputs"].pop(key)
                summary["outputs"].update({"report_" + key: value for key, value in recovered.items()
                                           if not key.endswith("_error")})
                summary["stage_errors"].update({"report_" + key.removesuffix("_error"): value
                                                for key, value in recovered.items() if key.endswith("_error")})
                payload = root / "review" / "teaser_recovery" / "teaser.json"
                _save(payload, teaser_payload(review))
                return {"json": str(payload)}

            outputs = stage("teaser", lambda: write_teaser(review, root / "review" / "teaser"), recover=recover_teaser)
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
            for directory in (
                "visual_calls",
                "code_scopes",
                "claim_extraction",
                "experiment_scope",
            ):
                if (root / directory).is_dir():
                    summary["outputs"][directory] = str(root / directory)
            if (root / "verification" / "theory_derivations").is_dir():
                summary["outputs"]["theory_derivations"] = str(root / "verification" / "theory_derivations")
            summary["stages"] = {
                name: "skipped" if value == "pending" else value for name, value in summary["stages"].items()
            }
            if "review" in locals():
                final_delivery = checked_delivery(review, stages=summary["stages"])
                summary["advice"] = {
                    state: sum(c.advice is not None and c.advice.state == state for c in final_delivery.claims)
                    for state in ("generated", "unavailable")
                }
                summary["run_status"] = final_delivery.run_status
                summary["incomplete_stages"] = final_delivery.incomplete_stages
                summary["delivery_checks"] = [item.model_dump(mode="json") for item in final_delivery.delivery_checks]
            else:
                summary["incomplete_stages"] = [name for name in STAGES if summary["stages"][name] == "failed"]
                summary["run_status"] = "partial" if summary["incomplete_stages"] else "completed"
                summary["delivery_checks"] = []
            summary["duration_seconds"] = time.perf_counter() - started
            for module, record in run_stats.read(root / "run_stats.json")["modules"].items():
                if record["status"] == "pending":
                    run_stats.record_module_status(module, "skipped")
            run_stats.set_pipeline_duration(summary["duration_seconds"])
            summary["run_stats"] = run_stats.with_totals(run_stats.read(root / "run_stats.json"))
            summary["model_usage"] = _model_usage(summary["run_stats"])
            summary["issues"] = verification_limitations(
                issues=summary["issues"],
                figure_coverage=summary.get("figure_coverage"),
                figure_context_coverage=summary.get("figure_context_coverage"),
                table_coverage=summary.get("table_coverage"),
                table_context_coverage=summary.get("table_context_coverage"),
                writing_coverage=summary.get("writing_coverage"),
                claim_coverage=summary.get("claim_coverage"),
                anonymity_policy=summary.get("anonymity_policy"),
                token_usage=summary["model_usage"],
            )
            _save(root / "run_stats.json", summary["run_stats"])
            summary["outputs"]["run_stats"] = str(root / "run_stats.json")
            _save(root / "full_pipeline_summary.json", summary)
        return summary
