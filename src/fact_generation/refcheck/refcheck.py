"""Reference-checking adapter.

By default this dispatches to RefCopilot (`tools/RefCopilot/`). The legacy
`refchecker` submodule under `tools/refchecker/` is kept for comparison and
can be selected via the env flag::

    FACTREVIEW_USE_REFCOPILOT=0  → use the original refchecker submodule.
    FACTREVIEW_USE_REFCOPILOT=1  → use RefCopilot (default).

Both backends expose the same `check_references()` and
`format_reference_check_markdown()` API, so the rest of the FactReview pipeline
(stage_runner, execution nodes) does not need to know which is in use.

Usage (library)::

    from fact_generation.refcheck.refcheck import check_references

    result = check_references(
        paper="2401.12345",          # arXiv ID, URL, or local PDF/tex path
        output_file="refs_out.txt",  # optional
    )
    # result -> {"total_refs": 42, "errors": 3, "warnings": 1, ...}

Usage (CLI)::

    python -m reference_check.refcheck --paper 2401.12345
    python -m reference_check.refcheck --paper ./paper.pdf --output-file results.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Locate the two candidate backends under tools/.
# src/fact_generation/refcheck/refcheck.py -> repo_root/tools/{refchecker,RefCopilot}/src
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFCHECKER_SRC = _REPO_ROOT / "tools" / "refchecker" / "src"
_REFCOPILOT_SRC = _REPO_ROOT / "tools" / "RefCopilot" / "src"

if _REFCOPILOT_SRC.exists() and str(_REFCOPILOT_SRC) not in sys.path:
    sys.path.insert(0, str(_REFCOPILOT_SRC))
if _REFCHECKER_SRC.exists() and str(_REFCHECKER_SRC) not in sys.path:
    sys.path.insert(0, str(_REFCHECKER_SRC))


def _use_refcopilot() -> bool:
    """Read the FACTREVIEW_USE_REFCOPILOT flag (default: True)."""
    raw = (os.environ.get("FACTREVIEW_USE_REFCOPILOT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_checker(
    *,
    api_key: str | None = None,
    db_path: str | None = None,
    output_file: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    debug: bool = False,
    enable_parallel: bool = True,
    max_workers: int = 4,
):
    """Construct an ArxivReferenceChecker with the given options."""
    from refchecker import ArxivReferenceChecker

    llm_config = None
    if llm_provider:
        llm_config = {
            "provider": llm_provider,
            "model": llm_model,
        }

    ss_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")

    return ArxivReferenceChecker(
        semantic_scholar_api_key=ss_key,
        db_path=db_path,
        output_file=output_file,
        llm_config=llm_config,
        debug_mode=debug,
        enable_parallel=enable_parallel,
        max_workers=max_workers,
    )


def check_references(
    paper: str,
    *,
    api_key: str | None = None,
    db_path: str | None = None,
    output_file: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    debug: bool = False,
    enable_parallel: bool = True,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run reference checking on *paper* and return a summary dict.

    Parameters
    ----------
    paper : str
        ArXiv ID (e.g. ``"2401.12345"``), arXiv URL, or local file path
        (``.pdf`` / ``.tex`` / ``.bib``).
    api_key : str, optional
        Semantic Scholar API key. Falls back to env vars.
    db_path : str, optional
        Path to a local Semantic Scholar SQLite database (legacy refchecker only).
    output_file : str, optional
        If given, write the error report to this path.
    debug : bool
        Enable verbose logging.

    Returns
    -------
    dict with keys:
        ok              – True if the run completed (even if errors found)
        total_refs      – number of references processed
        errors          – number of errors found
        warnings        – number of warnings found
        unverified      – number of unverifiable references
        error_message   – non-empty string if the run itself failed
    """
    if _use_refcopilot():
        try:
            from refcopilot.compat import check_references as _refcopilot_check  # type: ignore
        except Exception as exc:
            logger.warning("RefCopilot import failed (%s); falling back to refchecker", exc)
        else:
            return _refcopilot_check(
                paper,
                api_key=api_key,
                db_path=db_path,
                output_file=output_file,
                llm_provider=llm_provider,
                llm_model=llm_model,
                debug=debug,
                enable_parallel=enable_parallel,
                max_workers=max_workers,
            )

    try:
        checker = _build_checker(
            api_key=api_key,
            db_path=db_path,
            output_file=output_file,
            llm_provider=llm_provider,
            llm_model=llm_model,
            debug=debug,
            enable_parallel=enable_parallel,
            max_workers=max_workers,
        )

        # Determine whether paper is a local file or an arXiv specifier.
        paper_path = Path(paper)
        is_local = paper_path.exists() and paper_path.is_file()

        if is_local:
            checker.run(debug_mode=debug, local_pdf_path=str(paper_path))
        else:
            checker.run(debug_mode=debug, specific_paper_id=paper)

        issues = _extract_issues(checker)
        return {
            "ok": True,
            "total_refs": getattr(checker, "total_references_processed", 0),
            "errors": getattr(checker, "total_errors_found", 0),
            "warnings": getattr(checker, "total_warnings_found", 0),
            "unverified": getattr(checker, "total_unverified_refs", 0),
            "error_message": "",
            "issues": issues,
            "error_details": [issue for issue in issues if issue.get("severity") == "error"],
            "warning_details": [issue for issue in issues if issue.get("severity") == "warning"],
            "unverified_details": [issue for issue in issues if issue.get("severity") == "unverified"],
            "report_file": str(output_file or ""),
        }
    except Exception as exc:
        logger.exception("refchecker run failed")
        return {
            "ok": False,
            "total_refs": 0,
            "errors": 0,
            "warnings": 0,
            "unverified": 0,
            "error_message": str(exc),
            "issues": [],
            "error_details": [],
            "warning_details": [],
            "unverified_details": [],
            "report_file": str(output_file or ""),
        }


def _clean_text(value: Any, *, max_chars: int | None = None) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[: max(0, max_chars - 16)].rstrip() + "\n...(truncated)"
    return text


def _source_issue_entries(entry: dict[str, Any]) -> list[dict[str, Any]]:
    original_errors = entry.get("_original_errors")
    if isinstance(original_errors, list):
        entries = [item for item in original_errors if isinstance(item, dict)]
        if entries:
            return entries
    return [entry]


def _issue_type(entry: dict[str, Any]) -> str:
    return (
        _clean_text(entry.get("error_type") or entry.get("warning_type") or entry.get("info_type"))
        or "unknown"
    )


def _issue_details(entry: dict[str, Any]) -> str:
    return _clean_text(
        entry.get("error_details") or entry.get("warning_details") or entry.get("info_details"),
        max_chars=4000,
    )


def _issue_severity(entry: dict[str, Any]) -> str:
    normalized = _issue_type(entry).strip().lower()
    if normalized == "unverified":
        return "unverified"
    if _clean_text(entry.get("warning_type")):
        return "warning"
    if _clean_text(entry.get("info_type")):
        return "info"
    return "error"


def _extract_issues(checker: Any) -> list[dict[str, str]]:
    raw_issues = getattr(checker, "errors", []) or []
    issues: list[dict[str, str]] = []
    for entry in raw_issues:
        if not isinstance(entry, dict):
            continue
        for source_issue in _source_issue_entries(entry):
            issues.append(
                {
                    "severity": _issue_severity(source_issue),
                    "type": _issue_type(source_issue),
                    "reference_title": _clean_text(entry.get("ref_title"), max_chars=500),
                    "reference_year": _clean_text(entry.get("ref_year_cited"), max_chars=80),
                    "cited_url": _clean_text(entry.get("ref_url_cited"), max_chars=1000),
                    "verified_url": _clean_text(entry.get("ref_verified_url"), max_chars=1000),
                    "details": _issue_details(source_issue) or _issue_details(entry),
                    "raw_reference": _clean_text(entry.get("ref_raw_text"), max_chars=4000),
                    "corrected_plaintext": _clean_text(entry.get("ref_corrected_plaintext"), max_chars=4000),
                    "corrected_bibtex": _clean_text(entry.get("ref_corrected_bibtex"), max_chars=4000),
                    "corrected_bibitem": _clean_text(entry.get("ref_corrected_bibitem"), max_chars=4000),
                }
            )
    return issues


def _issue_title(issue: dict[str, Any]) -> str:
    title = _clean_text(issue.get("reference_title"), max_chars=160)
    if title:
        return title.replace("\n", " ")
    raw = _clean_text(issue.get("raw_reference"), max_chars=160)
    if raw:
        return raw.replace("\n", " ")
    return "Untitled reference"


def _append_issue_lines(lines: list[str], issue: dict[str, Any], index: int) -> None:
    issue_type = _clean_text(issue.get("type"), max_chars=80) or "unknown"
    lines.append(f"{index}. **{_issue_title(issue)}** (`{issue_type}`)")
    details = _clean_text(issue.get("details"), max_chars=1600)
    if details:
        lines.append(f"   - Details: {details.replace(chr(10), chr(10) + '     ')}")
    cited_url = _clean_text(issue.get("cited_url"), max_chars=500)
    if cited_url:
        lines.append(f"   - Cited URL: {cited_url}")
    verified_url = _clean_text(issue.get("verified_url"), max_chars=500)
    if verified_url:
        lines.append(f"   - Verified URL: {verified_url}")
    raw_reference = _clean_text(issue.get("raw_reference"), max_chars=1200)
    if raw_reference:
        lines.append("   - Raw reference:")
        lines.append("")
        lines.append("     ```text")
        for raw_line in raw_reference.replace("```", "` ` `").splitlines():
            lines.append(f"     {raw_line}")
        lines.append("     ```")
    corrected = _clean_text(issue.get("corrected_plaintext"), max_chars=1200)
    if corrected:
        lines.append("   - Suggested correction:")
        lines.append("")
        lines.append("     ```text")
        for corrected_line in corrected.replace("```", "` ` `").splitlines():
            lines.append(f"     {corrected_line}")
        lines.append("     ```")


def format_reference_check_markdown(result: dict[str, Any], *, max_issues: int = 20) -> str:
    """Render a reviewer-facing markdown section for refchecker output."""
    if not isinstance(result, dict):
        return ""

    lines: list[str] = []
    lines.append("## Reference Check")
    lines.append("")
    if not result.get("ok"):
        message = (
            _clean_text(result.get("error_message"), max_chars=1200) or "Reference check did not complete."
        )
        lines.append("RefChecker was enabled, but the run did not complete successfully.")
        lines.append("")
        lines.append(f"- Error: {message}")
        report_file = _clean_text(result.get("report_file"), max_chars=500)
        if report_file:
            lines.append(f"- Detail file: `{report_file}`")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    total_refs = int(result.get("total_refs") or 0)
    error_count = int(result.get("errors") or 0)
    warning_count = int(result.get("warnings") or 0)
    unverified_count = int(result.get("unverified") or 0)
    lines.append(
        "RefChecker validates bibliography entries against external scholarly metadata sources. "
        "This section is included only when reference checking is explicitly enabled."
    )
    lines.append("")
    lines.append(
        f"- References processed: `{total_refs}`; errors: `{error_count}`; "
        f"warnings: `{warning_count}`; unverified: `{unverified_count}`."
    )
    report_file = _clean_text(result.get("report_file"), max_chars=500)
    if report_file:
        lines.append(f"- Detail file: `{report_file}`")
    lines.append("")

    issues = result.get("issues") if isinstance(result.get("issues"), list) else []
    grouped = {
        "Errors": [i for i in issues if isinstance(i, dict) and i.get("severity") == "error"],
        "Warnings": [i for i in issues if isinstance(i, dict) and i.get("severity") == "warning"],
        "Unverified References": [
            i for i in issues if isinstance(i, dict) and i.get("severity") == "unverified"
        ],
    }
    rendered_any = False
    for heading, rows in grouped.items():
        if not rows:
            continue
        rendered_any = True
        lines.append(f"### {heading}")
        lines.append("")
        for index, issue in enumerate(rows[:max_issues], start=1):
            _append_issue_lines(lines, issue, index)
        if len(rows) > max_issues:
            lines.append(f"- {len(rows) - max_issues} additional item(s) omitted from this summary.")
        lines.append("")

    if not rendered_any:
        lines.append("No warning or error details were returned by RefChecker.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _cli_main() -> int:
    p = argparse.ArgumentParser(
        prog="refcheck",
        description="Check references in an academic paper (adapter for refchecker).",
    )
    p.add_argument("--paper", required=True, help="ArXiv ID, URL, or local PDF/TeX path")
    p.add_argument("--db-path", default=None, help="Local Semantic Scholar DB path")
    p.add_argument("--output-file", default=None, help="Write error report to this path")
    p.add_argument("--llm-provider", default=None, help="LLM provider (openai, anthropic, ...)")
    p.add_argument("--llm-model", default=None, help="LLM model name")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    result = check_references(
        paper=args.paper,
        db_path=args.db_path,
        output_file=args.output_file,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        debug=args.debug,
        max_workers=args.max_workers,
    )

    if result["ok"]:
        print(f"References processed: {result['total_refs']}")
        print(
            f"Errors: {result['errors']}, Warnings: {result['warnings']}, Unverified: {result['unverified']}"
        )
        return 0
    else:
        print(f"ERROR: {result['error_message']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
