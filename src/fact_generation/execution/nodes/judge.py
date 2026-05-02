from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.client import llm_json, resolve_llm_config
from util.fs import write_text
from util.meta import index_artifacts
from util.recorder import append_event

from ..tools.alignment import run_alignment
from ..tools.baseline_checks import Baseline
from ..tools.metrics import compute_check


def _read_optional(path: str, max_chars: int = 14000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n...(truncated)\n"
        return txt
    except Exception:
        return ""


def _llm_judge_enabled(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("llm_judge_mode") or "").strip().lower()
    if mode in {"assist", "verdict"}:
        return mode
    return "off"


def judge_node(state: dict[str, Any]) -> dict[str, Any]:
    run_info = state.get("run", {})
    run_dir = Path(run_info.get("dir") or "")
    artifacts_dir = Path(run_info.get("artifacts_dir") or (run_dir / "artifacts"))
    logs_dir = Path(run_info.get("logs_dir") or (run_dir / "logs"))

    baseline_raw = state.get("baseline") or {}
    baseline = Baseline(raw=baseline_raw if isinstance(baseline_raw, dict) else {})

    checks = baseline.checks
    results: list[dict[str, Any]] = []
    passed = True
    run_ok = bool(state.get("run_result", {}).get("success"))
    cfg = state.get("config", {}) or {}

    # ── Evidence source 1: Deterministic baseline checks ──
    if not checks:
        passed = False
        results.append({"type": "inconclusive_no_baseline", "passed": False, "run_success": run_ok})
    else:
        for chk in checks:
            r = compute_check(str(artifacts_dir), chk)
            results.append(r)
            if not r.get("passed"):
                passed = False

    # ── Evidence source 2: Paper-table alignment (always, independent of baseline) ──
    try:
        configured_tables_dir = str(cfg.get("paper_extracted_tables_dir") or "").strip()
        # prepare_node sets cfg['paper_extracted_tables_dir']; if it is missing or
        # the directory does not exist, skip alignment evidence silently.
        paper_tables_dir = Path(configured_tables_dir).resolve() if configured_tables_dir else None
        if paper_tables_dir is not None and paper_tables_dir.exists():
            ar = run_alignment(
                cfg=cfg,
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                paper_extracted_tables_dir=paper_tables_dir,
            )
            results.append(
                {
                    "type": "paper_table_alignment",
                    "passed": bool(ar.matched > 0 and ar.failed == 0 and run_ok),
                    "matched": ar.matched,
                    "passed_n": ar.passed,
                    "failed_n": ar.failed,
                    "unmatched_run_metrics": ar.unmatched_run_metrics,
                    "critiques_n": len(ar.critiques or []),
                    "alignment_artifact": "alignment/alignment.json",
                }
            )
    except Exception as e:
        results.append(
            {"type": "paper_table_alignment", "passed": False, "error": f"{type(e).__name__}: {e}"}
        )

    # ── Evidence source 3: LLM judge (advisory by default) ──
    llm_mode = _llm_judge_enabled(cfg)
    if llm_mode != "off" and (not bool(cfg.get("no_llm"))):
        extracted_md = str(cfg.get("paper_pdf_extracted_md") or "").strip()
        evidence = {
            "paper_key": str(cfg.get("paper_key") or ""),
            "paper_pdf": str(cfg.get("paper_pdf") or ""),
            "paper_root": str(cfg.get("paper_root") or ""),
            "repo_url": str(cfg.get("paper_repo_url") or ""),
            "run_id": str(run_info.get("id") or ""),
            "run_success": run_ok,
            "run_result": state.get("run_result") or {},
            "artifacts_index": index_artifacts(artifacts_dir),
            "paper_extracted_md_excerpt": _read_optional(extracted_md, max_chars=14000),
            "baseline_current": baseline_raw if isinstance(baseline_raw, dict) else {},
        }
        system = (
            "You are judging whether a paper reproduction run matches claimed results.\n"
            "Return JSON only. Do not include prose outside JSON.\n"
            "If evidence is insufficient, keep verdict as inconclusive and propose concrete baseline checks.\n"
        )
        prompt = json.dumps(
            {
                "mode": llm_mode,
                "evidence": evidence,
                "output_schema": {
                    "verdict": "pass|fail|inconclusive",
                    "confidence": 0.0,
                    "why": ["short strings"],
                    "suggested_artifacts": ["paths or patterns to collect"],
                    "suggested_baseline_checks": [
                        {"type": "file_exists", "path": "relative/to/artifacts"},
                        {
                            "type": "json_value",
                            "path": "relative/to/artifacts",
                            "json_path": ["key", 0, "subkey"],
                            "expected": 0.0,
                            "tolerance": 0.0,
                        },
                        {
                            "type": "csv_agg",
                            "path": "relative/to/artifacts",
                            "expr": {"groupby": ["col"], "agg": {"metric": "mean"}},
                            "expected": [{"col": "x", "metric": 0.0}],
                            "tolerance": 0.0,
                        },
                    ],
                },
            },
            ensure_ascii=False,
        )
        llm_cfg = resolve_llm_config(
            str(cfg.get("llm_provider") or ""),
            str(cfg.get("llm_model") or ""),
            str(cfg.get("llm_base_url") or ""),
        )
        resp = llm_json(prompt=prompt, system=system, cfg=llm_cfg)
        try:
            write_text(logs_dir / "judge_llm_prompt.json", prompt + "\n")
            write_text(
                logs_dir / "judge_llm_response.json", json.dumps(resp, ensure_ascii=False, indent=2) + "\n"
            )
        except Exception:
            pass

        verdict = str(resp.get("verdict") or "").strip().lower() if isinstance(resp, dict) else ""
        conf = resp.get("confidence") if isinstance(resp, dict) else None
        results.append(
            {
                "type": "llm_judge",
                "mode": llm_mode,
                "passed": (verdict == "pass") if llm_mode == "verdict" else False,
                "verdict": verdict or "inconclusive",
                "confidence": conf,
                "response": resp,
            }
        )
        if llm_mode == "verdict" and verdict in {"pass", "fail"}:
            passed = verdict == "pass"

    # ── Evidence source 4: Reference accuracy check (optional, --enable-refcheck) ──
    if cfg.get("enable_refcheck"):
        paper_pdf = str(cfg.get("paper_pdf") or "").strip()
        if paper_pdf and Path(paper_pdf).exists():
            try:
                from fact_generation.refcheck.refcheck import check_references

                rc = check_references(
                    paper=paper_pdf,
                    output_file=str(artifacts_dir / "reference_check_details.txt"),
                    debug=False,
                )
                results.append(
                    {
                        "type": "reference_check",
                        "passed": rc.get("ok", False) and rc.get("errors", 0) == 0,
                        "total_refs": rc.get("total_refs", 0),
                        "errors": rc.get("errors", 0),
                        "warnings": rc.get("warnings", 0),
                        "unverified": rc.get("unverified", 0),
                        "error_message": rc.get("error_message", ""),
                        "issues": rc.get("issues", []),
                        "error_details": rc.get("error_details", []),
                        "warning_details": rc.get("warning_details", []),
                        "unverified_details": rc.get("unverified_details", []),
                        "report_file": rc.get("report_file", ""),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "type": "reference_check",
                        "passed": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    judge = {"passed": passed, "results": results}
    state["judge"] = judge

    append_event(run_dir, "judge", {"passed": passed, "results": results})
    state.setdefault("history", []).append({"kind": "judge", "data": {"passed": passed, "results": results}})

    # Preserve failed status from earlier nodes; do not overwrite with "running"
    if state.get("status") != "failed":
        state["status"] = "running"
    return state
