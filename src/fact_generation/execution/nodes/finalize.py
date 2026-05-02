from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from util.fs import ensure_dir, write_text
from util.meta import index_artifacts
from util.recorder import append_event, write_issues_md


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def _md_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = [str(x).replace("\n", " ").strip() for x in row]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _task_family(task: Any) -> str:
    """Read family from structured field first; fallback to id-prefix heuristic."""
    if isinstance(task, dict) and task.get("family"):
        return str(task["family"]).strip()
    task_id = str(task.get("id") if isinstance(task, dict) else task or "")
    if task_id.startswith("train_"):
        return "train"
    if task_id.startswith("eval_"):
        return "eval"
    if "prepare" in task_id or "setup" in task_id:
        return "prepare"
    if "smoke" in task_id:
        return "smoke"
    return "other"


def _task_dataset(task: Any) -> str:
    """Read dataset from structured field first; fallback to id-substring heuristic."""
    if isinstance(task, dict) and task.get("dataset"):
        return str(task["dataset"]).strip()
    task_id = str(task.get("id") if isinstance(task, dict) else task or "").lower()
    if "fb15k" in task_id:
        return "FB15k-237"
    if "wn18rr" in task_id:
        return "WN18RR"
    return ""


def _task_variant(task_id: str) -> str:
    s = task_id
    for prefix in ("train_", "eval_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    for suffix in ("_fb15k_237", "_wn18rr"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _summarize_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(tasks)
    passed = sum(1 for t in tasks if t.get("success") and not t.get("skipped"))
    skipped = sum(1 for t in tasks if t.get("skipped"))
    failed = sum(1 for t in tasks if not t.get("success"))
    dry_run = sum(1 for t in tasks if t.get("dry_run"))
    by_family: dict[str, int] = {}
    by_dataset: dict[str, dict[str, int]] = {}
    for t in tasks:
        fam = _task_family(t)
        ds = _task_dataset(t) or "n/a"
        by_family[fam] = by_family.get(fam, 0) + 1
        bucket = by_dataset.setdefault(ds, {"train": 0, "eval": 0, "other": 0})
        if fam in {"train", "eval"}:
            bucket[fam] += 1
        else:
            bucket["other"] += 1
    return {
        "total": total,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
        "by_family": by_family,
        "by_dataset": by_dataset,
    }


def _judge_highlights(
    judge: dict[str, Any], run_result: dict[str, Any]
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """
    Four reviewer-facing labels:
      verified         – baseline checks passed deterministically
      inconclusive     – run OK but no/insufficient baseline checks
      deviated         – run OK but some baseline checks failed (numeric mismatch)
      execution_failed – tasks did not complete successfully
    """
    results = judge.get("results") or []
    notes: list[str] = []
    baseline_checks: list[dict[str, Any]] = []

    run_ok = bool(run_result.get("success"))
    has_inconclusive = isinstance(results, list) and any(
        isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline" for r in results
    )

    if not run_ok:
        label = "execution_failed"
        failed_task = run_result.get("failed_task") or ""
        if failed_task:
            notes.append(f"Execution stopped at task `{failed_task}`.")
    elif judge.get("passed") is True:
        label = "verified"
    elif has_inconclusive:
        label = "inconclusive"
        notes.append(
            "No deterministic baseline checks are defined yet, so the run cannot be judged against paper claims."
        )
    else:
        label = "deviated"
        notes.append("Run completed but one or more deterministic checks did not pass.")

    for r in results if isinstance(results, list) else []:
        if not isinstance(r, dict):
            continue
        if r.get("type") == "llm_judge":
            resp = r.get("response") or {}
            why = resp.get("why") or []
            if isinstance(why, list):
                notes.extend([str(x) for x in why[:6]])
            checks = resp.get("suggested_baseline_checks") or []
            if isinstance(checks, list):
                baseline_checks.extend([c for c in checks if isinstance(c, dict)])
        elif r.get("type") == "paper_table_alignment":
            matched = int(r.get("matched") or 0)
            failed_n = int(r.get("failed_n") or 0)
            notes.append(
                f"Deterministic paper-table alignment matched {matched} targets with {failed_n} mismatches."
            )
        elif r.get("type") == "reference_check":
            errs = int(r.get("errors") or 0)
            warns = int(r.get("warnings") or 0)
            total = int(r.get("total_refs") or 0)
            if errs:
                notes.append(
                    f"Reference check found {errs} error(s) and {warns} warning(s) across {total} references."
                )
            elif warns:
                notes.append(f"Reference check found {warns} warning(s) across {total} references.")
            elif total:
                notes.append(f"Reference check: all {total} references verified without warnings.")
        elif r.get("passed") is False and r.get("type") not in {"inconclusive_no_baseline", "llm_judge"}:
            notes.append(
                f"Check `{r.get('type')}` on `{r.get('path', '')}`: expected={r.get('expected')}, observed={r.get('observed')}."
            )

    deduped: list[str] = []
    seen = set()
    for n in notes:
        if not n or n in seen:
            continue
        seen.add(n)
        deduped.append(n)
    return label, deduped, baseline_checks


def _artifact_highlights(artifacts_index: dict[str, Any]) -> list[str]:
    files = artifacts_index.get("files") or []
    if not isinstance(files, list) or not files:
        return []
    return [str(x) for x in files[:12]]


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    run_info = state.get("run", {})
    run_dir = Path(run_info.get("dir") or "")
    artifacts_dir = Path(run_info.get("artifacts_dir") or (run_dir / "artifacts"))

    # 1) update issues.md from state history (authoritative, reproducible)
    write_issues_md(run_dir, state.get("history", []))

    # Make final status explicit
    if state.get("judge", {}).get("passed") is True and state.get("status") != "failed":
        state["status"] = "success"
    else:
        # If we cannot judge due to missing baseline but the run itself succeeded, mark as inconclusive.
        results = (state.get("judge", {}) or {}).get("results") or []
        run_ok = bool((state.get("run_result", {}) or {}).get("success"))
        if (
            state.get("status") != "failed"
            and run_ok
            and isinstance(results, list)
            and any(isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline" for r in results)
        ):
            state["status"] = "inconclusive"

    # 2) write a deterministic summary.json in run dir
    artifacts_index = index_artifacts(artifacts_dir)
    summary = {
        "run_id": run_info.get("id"),
        "status": state.get("status"),
        "attempts": state.get("attempt", 0),
        "run_result": state.get("run_result", {}),
        "judge": state.get("judge", {}),
        "artifacts": artifacts_index,
    }
    write_text(run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    # 3) keep reviewer-facing report and diff summaries inside this execution run.
    paper_key = str((state.get("config") or {}).get("paper_key") or "paper")
    reports_dir = ensure_dir(run_dir / "reports")
    diffs_dir = ensure_dir(run_dir / "diffs")

    cfg = state.get("config") or {}
    run_result = state.get("run_result") or {}
    judge = state.get("judge") or {}
    tasks = run_result.get("tasks") or []
    tasks = tasks if isinstance(tasks, list) else []
    task_summary = _summarize_tasks(tasks)
    judge_label, judge_notes, suggested_checks = _judge_highlights(judge, run_result)
    artifact_files = _artifact_highlights(artifacts_index)

    # main report (human-readable, reviewer-oriented)
    md_lines: list[str] = []
    md_lines.append(f"# Reproduction Review Report: {paper_key}")
    md_lines.append("")
    md_lines.append(f"Run ID: `{run_info.get('id')}`")
    md_lines.append("")
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(f"- Final status: `{state.get('status')}`")
    md_lines.append(f"- Judge outcome: `{judge_label}`")
    md_lines.append(f"- Workflow attempts: `{state.get('attempt', 0)}`")
    md_lines.append(f"- Total tasks in this run: `{task_summary['total']}`")
    md_lines.append(
        f"- Task execution summary: `{task_summary['passed']} passed`, `{task_summary['skipped']} skipped`, `{task_summary['failed']} failed`"
    )
    if task_summary["dry_run"]:
        md_lines.append(f"- Dry-run tasks: `{task_summary['dry_run']}`")
    if not judge_notes:
        md_lines.append("- Overall assessment: evidence was collected without major reported issues.")
    else:
        md_lines.append("- Overall assessment:")
        for n in judge_notes[:4]:
            md_lines.append(f"  - {n}")
    md_lines.append("")

    md_lines.append("## Run Configuration")
    md_lines.append("")
    md_lines.append(f"- Paper key: `{paper_key}`")
    md_lines.append(f"- Source root: `{cfg.get('paper_root') or ''}`")
    md_lines.append(f"- Tasks file: `{cfg.get('tasks_path') or ''}`")
    md_lines.append(f"- Baseline file: `{cfg.get('baseline_path') or ''}`")
    md_lines.append(f"- PDF path: `{cfg.get('paper_pdf') or ''}`")
    md_lines.append(f"- Repo URL: `{cfg.get('paper_repo_url') or ''}`")
    md_lines.append(f"- Dry run: `{bool(cfg.get('dry_run'))}`")
    md_lines.append(f"- LLM provider: `{cfg.get('llm_provider') or ''}`")
    md_lines.append(f"- LLM model: `{cfg.get('llm_model') or ''}`")
    md_lines.append("")

    md_lines.append("## Experiment Coverage")
    md_lines.append("")
    fam_rows = [[k, str(v)] for k, v in sorted(task_summary["by_family"].items())]
    if fam_rows:
        md_lines.extend(_md_table(["Category", "Count"], fam_rows))
        md_lines.append("")
    dataset_rows = []
    for ds, counts in sorted(task_summary["by_dataset"].items()):
        dataset_rows.append(
            [ds, str(counts.get("train", 0)), str(counts.get("eval", 0)), str(counts.get("other", 0))]
        )
    if dataset_rows:
        md_lines.extend(_md_table(["Dataset", "Train", "Eval", "Other"], dataset_rows))
        md_lines.append("")

    matrix_rows: list[list[str]] = []
    for t in tasks:
        task_id = str(t.get("id") or "")
        fam = _task_family(t)
        ds = _task_dataset(t)
        if fam not in {"train", "eval"}:
            continue
        status = "skipped" if t.get("skipped") else ("ok" if t.get("success") else "failed")
        if t.get("dry_run"):
            status = "dry-run"
        matrix_rows.append([fam, ds or "n/a", _task_variant(task_id), status])
    if matrix_rows:
        md_lines.append("### Task Matrix")
        md_lines.append("")
        md_lines.extend(_md_table(["Type", "Dataset", "Variant", "Status"], matrix_rows[:60]))
        md_lines.append("")

    md_lines.append("## Validation Outcome")
    md_lines.append("")
    md_lines.append(f"- Deterministic pass/fail: `{bool(judge.get('passed'))}`")
    md_lines.append(f"- Review verdict: **{judge_label}**")
    verdict_desc = {
        "verified": "All deterministic baseline checks passed within tolerance.",
        "inconclusive": "Run succeeded but no/insufficient baseline checks to verify paper claims.",
        "deviated": "Run succeeded but some quantitative checks exceeded tolerance.",
        "execution_failed": "Task execution failed before producing sufficient evidence.",
    }
    md_lines.append(f"- Meaning: {verdict_desc.get(judge_label, '')}")
    if judge_notes:
        md_lines.append("- Key findings:")
        for note in judge_notes:
            md_lines.append(f"  - {note}")
    else:
        md_lines.append("- No additional findings were recorded.")
    md_lines.append("")

    # Evidence Table: per-check observed vs expected
    evidence_rows: list[list[str]] = []
    for r in judge.get("results") or []:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type", "")
        if rtype in {"file_exists", "json_value", "csv_agg"}:
            verdict_str = "PASS" if r.get("passed") else "FAIL"
            evidence_rows.append(
                [
                    rtype,
                    str(r.get("path", "")),
                    str(r.get("expected", "")),
                    str(r.get("observed", "")),
                    str(r.get("tolerance", "")),
                    verdict_str,
                ]
            )
    if evidence_rows:
        md_lines.append("### Evidence Table")
        md_lines.append("")
        md_lines.extend(
            _md_table(["Check", "Path", "Expected", "Observed", "Tolerance", "Result"], evidence_rows)
        )
        md_lines.append("")

    if suggested_checks:
        md_lines.append("## Recommended Baseline Checks")
        md_lines.append("")
        md_lines.append(
            "These checks were suggested so future runs can be judged deterministically against paper claims."
        )
        md_lines.append("")
        for chk in suggested_checks[:12]:
            md_lines.append(f"- `{json.dumps(chk, ensure_ascii=False)}`")
        md_lines.append("")

    md_lines.append("## Available Evidence")
    md_lines.append("")
    md_lines.append(f"- Run directory: `{run_dir}`")
    md_lines.append(f"- Report: `{reports_dir / (str(run_info.get('id')) + '.md')}`")
    md_lines.append(f"- Issues log: `{run_dir / 'issues.md'}`")
    md_lines.append(f"- Artifact file count: `{len(artifacts_index.get('files') or [])}`")
    if artifact_files:
        md_lines.append("- Sample artifact paths:")
        for p in artifact_files:
            md_lines.append(f"  - `{p}`")
    else:
        md_lines.append("- No archived artifacts were found for this run.")
    md_lines.append("")

    # Optional: deterministic paper alignment report (if produced by judge/run)
    alignment_md = artifacts_dir / "alignment" / "alignment.md"
    alignment_json = artifacts_dir / "alignment" / "alignment.json"
    if alignment_md.exists() or alignment_json.exists():
        md_lines.append("## Paper alignment (deterministic)")
        md_lines.append("")
        if alignment_md.exists():
            try:
                md_lines.append(alignment_md.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        else:
            md_lines.append(f"- alignment_json: {alignment_json}")
            md_lines.append("")

    tables_index = _load_json_if_exists(artifacts_dir / "tables" / "index.json")
    if tables_index:
        md_lines.append("## Metrics Tables")
        md_lines.append("")
        md_lines.append("Auto-generated metrics tables are available under `artifacts/tables/`.")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(tables_index, ensure_ascii=False, indent=2))
        md_lines.append("```")
        md_lines.append("")

    reference_check = next(
        (
            r
            for r in (judge.get("results") or [])
            if isinstance(r, dict) and r.get("type") == "reference_check"
        ),
        None,
    )
    if isinstance(reference_check, dict):
        try:
            from fact_generation.refcheck.refcheck import format_reference_check_markdown

            reference_check_md = format_reference_check_markdown(reference_check)
        except Exception:
            reference_check_md = ""
        if reference_check_md.strip():
            md_lines.append(reference_check_md.rstrip())
            md_lines.append("")

    md_lines.append("## Reviewer-Facing Template")
    md_lines.append("")
    md_lines.append(
        "Use the following scaffold when turning this run into a reviewer-facing reproduction summary."
    )
    md_lines.append("")
    md_lines.append("### 1. Claim")
    md_lines.append("")
    md_lines.append(
        "- The submitted code was executed through the `execution` workflow on the provided repository snapshot."
    )
    md_lines.append(
        "- The workflow attempted to reproduce the paper's reported experiments and collect machine-readable outputs."
    )
    md_lines.append("")
    md_lines.append("### 2. What Was Actually Run")
    md_lines.append("")
    md_lines.append(
        "- Report the task matrix above, emphasizing datasets, variants, and whether the run was a dry run or a real execution."
    )
    md_lines.append(
        "- If some experiments were not executed, state that explicitly rather than implying full coverage."
    )
    md_lines.append("")
    md_lines.append("### 3. Quantitative Comparison")
    md_lines.append("")
    md_lines.append(
        "- Insert aligned paper-vs-run metric tables here once baseline checks and artifact metrics are available."
    )
    md_lines.append("- Prefer MRR, MR, Hits@1, Hits@3, and Hits@10 when those metrics exist.")
    md_lines.append("")
    md_lines.append("### 4. Deviations and Risks")
    md_lines.append("")
    if judge_notes:
        for note in judge_notes[:6]:
            md_lines.append(f"- {note}")
    else:
        md_lines.append("- No major deviations were recorded in this run.")
    md_lines.append("")
    md_lines.append("### 5. Recommendation")
    md_lines.append("")
    if judge_label == "verified":
        md_lines.append(
            "- The run provides enough evidence to support a positive reproduction statement, subject to reviewer interpretation."
        )
    elif judge_label == "inconclusive":
        md_lines.append(
            "- The current run is not yet sufficient for a final reproduction claim; baseline checks and/or real metric artifacts are still needed."
        )
    elif judge_label == "deviated":
        md_lines.append(
            "- The run completed but some quantitative checks deviated from paper claims. Review the Evidence Table for specific deltas."
        )
    else:
        md_lines.append(
            "- Execution failed before producing sufficient evidence. Fix the execution errors before drawing reproduction conclusions."
        )
    md_lines.append("")

    report_path = reports_dir / f"{run_info.get('id')}.md"
    write_text(report_path, "\n".join(md_lines) + "\n")

    # diff artifacts
    write_text(diffs_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    # 4) "facts pack" for final review writing.
    review_root = run_dir / "review_pack"
    ensure_dir(review_root)

    # Deterministic actionable hints (no LLM)
    suggestions: list[dict[str, Any]] = []
    # missing baseline => can't conclude
    if any(
        isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline"
        for r in (judge.get("results") or [])
    ):
        suggestions.append(
            {
                "type": "define_baseline",
                "why": "No baseline checks defined; cannot verify paper results.",
                "next": "Define execution checks, or for bundled demos edit demos/<paper_key>/execution/checks.json; ensure artifact_paths collect required outputs.",
            }
        )
    # common deps issue
    if isinstance(run_result, dict) and "difflib" in str(run_result.get("stderr_tail") or "").lower():
        suggestions.append(
            {
                "type": "fix_requirements_stdlib",
                "why": "requirements.txt contains Python stdlib module (difflib); pip cannot install it.",
                "next": "Use this run's logs/requirements.cleaned.txt or patch the repo requirements file; update tasks install_deps to point to cleaned file.",
            }
        )
    # LLM config issues
    if any(isinstance(h, dict) and h.get("kind") == "fix_llm_error" for h in (state.get("history") or [])):
        suggestions.append(
            {
                "type": "llm_config",
                "why": "LLM-based fix step failed due to missing/invalid provider config.",
                "next": "Rerun with --no-llm, set MODEL_PROVIDER=openai-codex after `codex login`, or set MODEL_PROVIDER/OPENAI_API_KEY (or Claude keys) and a valid model.",
            }
        )

    # ── Build structured evidence summaries for reviewer consumption ──

    # Checks summary: list each deterministic check with observed vs expected
    checks_summary: list[dict[str, Any]] = []
    alignment_summary: dict[str, Any] = {}
    for r in judge.get("results") or []:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type", "")
        if rtype in {"file_exists", "json_value", "csv_agg"}:
            checks_summary.append(
                {
                    "type": rtype,
                    "path": r.get("path", ""),
                    "passed": r.get("passed"),
                    "expected": r.get("expected"),
                    "observed": r.get("observed"),
                    "tolerance": r.get("tolerance"),
                }
            )
        elif rtype == "reference_check":
            checks_summary.append(
                {
                    "type": rtype,
                    "passed": r.get("passed"),
                    "total_refs": r.get("total_refs", 0),
                    "errors": r.get("errors", 0),
                    "warnings": r.get("warnings", 0),
                    "unverified": r.get("unverified", 0),
                    "error_details": r.get("error_details", []),
                    "warning_details": r.get("warning_details", []),
                    "unverified_details": r.get("unverified_details", []),
                    "report_file": r.get("report_file", ""),
                }
            )
        elif rtype == "paper_table_alignment":
            alignment_summary = {
                "matched": r.get("matched", 0),
                "passed": r.get("passed_n", 0),
                "failed": r.get("failed_n", 0),
                "unmatched_run_metrics": r.get("unmatched_run_metrics", []),
                "critiques": r.get("critiques_n", 0),
                "artifact": r.get("alignment_artifact", ""),
            }

    # Coverage gaps: identify what's missing for a complete reproduction verdict
    coverage_gaps: list[str] = []
    if any(
        isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline"
        for r in (judge.get("results") or [])
    ):
        coverage_gaps.append("No baseline checks defined — cannot verify paper claims quantitatively.")
    if not checks_summary:
        coverage_gaps.append("No deterministic checks (file_exists/json_value/csv_agg) were evaluated.")
    if not alignment_summary:
        coverage_gaps.append(
            "No paper-table alignment data available (paper_extracted/tables/ may be missing)."
        )
    eval_tasks = [t for t in tasks if _task_family(t) == "eval"]
    if not eval_tasks:
        coverage_gaps.append("No evaluation tasks were defined or executed; only smoke/prepare tasks ran.")

    # ── Optional: BibTeX enrichment for paper claims (--enable-bibtex) ──
    bibtex_entries: list[dict[str, Any]] = []
    if cfg.get("enable_bibtex"):
        try:
            from ..tools.bibtex import lookup_bibtex

            # Collect unique claim titles from tasks
            seen_titles: set = set()
            for t in tasks:
                for _claim in t.get("claims") or []:
                    # Claims look like "Table 4: TransE+CompGCN(...) on FB15k-237, MRR=0.335"
                    # Try to extract a paper title from the paper_key or config
                    pass
            # Use the paper's own title if available from extracted metadata
            paper_title = str(cfg.get("paper_title") or "").strip()
            if not paper_title:
                # Try to derive from paper_key
                paper_title = paper_key.replace("_", " ").replace("-", " ").strip()
            if paper_title and paper_title not in seen_titles:
                seen_titles.add(paper_title)
                r = lookup_bibtex(paper_title)
                if r.get("bibtex"):
                    bibtex_entries.append(
                        {
                            "query_title": paper_title,
                            "matched_title": r["matched_title"],
                            "bibtex": r["bibtex"],
                            "exact": r["exact"],
                        }
                    )
        except Exception:
            pass  # bibtex enrichment is best-effort

    facts = {
        "paper_key": paper_key,
        "run_id": run_info.get("id"),
        "status": state.get("status"),
        "judge_label": judge_label,
        "repo_url": cfg.get("paper_repo_url") or "",
        "paper_pdf": cfg.get("paper_pdf") or "",
        "paper_pdf_extracted_md": cfg.get("paper_pdf_extracted_md") or "",
        "paper_root": cfg.get("paper_root") or "",
        "local_source_path": cfg.get("local_source_path") or "",
        "conda_prefix": cfg.get("conda_prefix") or "",
        "tasks_path": cfg.get("tasks_path") or "",
        "baseline_path": cfg.get("baseline_path") or "",
        "artifacts_index": artifacts_index,
        "run_result": run_result,
        "judge": judge,
        "task_summary": task_summary,
        "checks_summary": checks_summary,
        "alignment_summary": alignment_summary,
        "coverage_gaps": coverage_gaps,
        "paths": {
            "run_dir": str(run_dir),
            "issues_jsonl": str(run_dir / "issues.jsonl"),
            "issues_md": str(run_dir / "issues.md"),
            "report": str(report_path),
            "diff_dir": str(diffs_dir),
        },
        "suggestions": suggestions,
        "bibtex": bibtex_entries if bibtex_entries else [],
    }
    write_text(review_root / "facts.json", json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
    write_text(
        review_root / "README.md",
        "# Review Facts Pack\n\n"
        "This folder contains **facts-only** artifacts to support writing a final review.\n\n"
        f"- `facts.json`: structured evidence and pointers (paper_key={paper_key}, run_id={run_info.get('id')})\n",
    )

    append_event(run_dir, "finalize", {"report": str(report_path), "diff_dir": str(diffs_dir)})
    state.setdefault("history", []).append({"kind": "finalize", "data": {"report": str(report_path)}})
    return state
