"""Reference-checking adapter.

Thin shim around RefCopilot (`RefCopilot/`). The Markdown summary included in
the final FactReview review lists fabricated references (errors) and
metadata-warning rows that carry an actionable BibTeX replacement, so users
can paste the corrected entry directly into their bibliography. Unverified
entries (no match on either backend) remain in ``reference_check.json`` only
and surface through RefCopilot's standalone CLI.

Usage (library)::

    from fact_generation.refcheck.refcheck import check_references

    result = check_references(
        paper="2401.12345",          # arXiv ID, URL, or local PDF/tex path
        output_file="refs_out.txt",  # optional
    )
    # result -> {"total_refs": 42, "errors": 3, "warnings": 1, ...}

Usage (CLI)::

    python -m fact_generation.refcheck.refcheck --paper 2401.12345
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFCOPILOT_SRC = _REPO_ROOT / "RefCopilot" / "src"

if _REFCOPILOT_SRC.exists() and str(_REFCOPILOT_SRC) not in sys.path:
    sys.path.insert(0, str(_REFCOPILOT_SRC))


def record_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class ReferenceCheckBundle:
    """The unchanged legacy export and independently saved public Report records."""

    payload: dict[str, Any]
    records: dict[str, Any] | None


def reference_records(report, payload: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    """Bind exported issues by public object order; never join truncated titles."""
    from refcopilot import Report, Severity, Verdict
    from refcopilot.bibtex_suggest import suggest_bibtex
    from refcopilot.report import to_factreview_dict

    if not isinstance(report, Report):
        raise ValueError("RefCopilot did not return a public Report")
    report = Report.model_validate(report.model_dump(mode="json"))
    if report.summary.total_refs != len(report.checked):
        raise ValueError("RefCopilot Report coverage does not match its checked records")
    bindings = []
    for reference_index, checked in enumerate(report.checked):
        indices = list(range(len(checked.issues)))
        if not indices and checked.verdict == Verdict.UNVERIFIED:
            indices = [None]
        for issue_index in indices:
            one = checked.model_copy(deep=True)
            one.issues = [] if issue_index is None else [checked.issues[issue_index]]
            rows = to_factreview_dict(Report(checked=[one]))["issues"]
            if len(rows) != 1 or len(bindings) >= len(payload["issues"]):
                raise ValueError("RefCopilot export cannot be bound to its original issues")
            exported = payload["issues"][len(bindings)]
            if rows[0] != exported:
                raise ValueError("RefCopilot export changed while binding issue records")
            full = ""
            if issue_index is not None and checked.issues[issue_index].severity == Severity.WARNING:
                full = suggest_bibtex(checked.reference, checked.merged)
            bindings.append(
                {
                    "reference_index": reference_index,
                    "issue_index": issue_index,
                    "exported_issue": exported,
                    "checked_sha256": record_digest(checked.model_dump(mode="json")),
                    "corrected_bibtex": full,
                    "bibtex_sha256": hashlib.sha256(full.encode()).hexdigest(),
                    "export_truncated": full != exported.get("corrected_bibtex", ""),
                }
            )
    if len(bindings) != len(payload["issues"]):
        raise ValueError("RefCopilot export contains unbound issues")
    raw_report = report.model_dump(mode="json")
    return {
        "version": 1,
        "input_sha256": input_sha256,
        "payload_sha256": record_digest(payload),
        "report_sha256": record_digest(raw_report),
        "report": raw_report,
        "bindings": bindings,
    }


def check_references_with_records(
    paper: str,
    *,
    api_key: str | None = None,
    output_file: str | None = None,
    debug: bool = False,
    enable_parallel: bool = True,
    max_workers: int = 4,
) -> ReferenceCheckBundle:
    """Run the public pipeline once, preserving full records outside the legacy export.

    The optional v2 detail file uses the library's public Markdown serializer.
    Legacy check_references/CLI behavior is unchanged.
    """
    from refcopilot import RefCopilotPipeline
    from refcopilot.report import to_factreview_dict, to_markdown

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    try:
        candidate = Path(paper) if len(paper) < 1024 and "\n" not in paper else None
        source = candidate if candidate is not None and candidate.is_file() else None
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source else None
        if source and source.suffix.lower() in {".txt", ".tex"}:
            paper = source.read_text(encoding="utf-8-sig")
        pipeline = RefCopilotPipeline(
            s2_api_key=api_key if api_key is not None else os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
            use_llm_verify=True,
            max_workers=max_workers if enable_parallel else 1,
        )
        report = pipeline.run(paper)
        if source and hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Reference input file changed during the check")
        payload = to_factreview_dict(report, report_file=str(output_file or ""))
        records = reference_records(report, payload, hashlib.sha256(paper.encode()).hexdigest())
        if output_file:
            destination = Path(output_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(to_markdown(report), encoding="utf-8")
        return ReferenceCheckBundle(payload, records)
    except Exception as exc:
        return ReferenceCheckBundle(
            {
                "ok": False,
                "total_refs": 0,
                "errors": 0,
                "warnings": 0,
                "unverified": 0,
                "error_message": f"{type(exc).__name__}: {exc}",
                "issues": [],
                "error_details": [],
                "warning_details": [],
                "unverified_details": [],
                "report_file": str(output_file or ""),
            },
            None,
        )


def check_references(
    paper: str,
    *,
    api_key: str | None = None,
    output_file: str | None = None,
    debug: bool = False,
    enable_parallel: bool = True,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run reference checking on *paper* (arXiv ID, URL, or local PDF/TeX/BibTeX path).

    Returns the dict written to ``reference_check.json``. See
    :func:`refcopilot.factreview.check_references` for the full schema.
    """
    from refcopilot.factreview import check_references as _check  # type: ignore

    # RefCopilot's TEXT input consumes literal text. Its detector recognizes a
    # .txt/.tex path, but the pipeline does not open that path before extraction.
    # Keep PDF/BibTeX/URL handling in the library and resolve text files here.
    candidate = Path(paper)
    if candidate.suffix.lower() in {".txt", ".tex"} and candidate.is_file():
        paper = candidate.read_text(encoding="utf-8-sig")

    return _check(
        paper,
        api_key=api_key if api_key is not None else os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        output_file=output_file,
        debug=debug,
        enable_parallel=enable_parallel,
        max_workers=max_workers,
    )


def format_reference_check_markdown(result: dict[str, Any], *, max_issues: int = 20) -> str:
    """Render the Markdown summary embedded in the final FactReview report.

    Includes errors (fabricated references) and warnings, the latter rendered
    with an inline corrected-BibTeX block (with data-source comments) so the
    user can copy-paste a fix. Unverified entries are omitted from this
    embedded summary; they remain in ``reference_check.json``.
    """
    from refcopilot.factreview import format_factreview_markdown  # type: ignore

    return format_factreview_markdown(result, max_issues=max_issues, include_warnings=True)


def _cli_main() -> int:
    p = argparse.ArgumentParser(
        prog="refcheck",
        description="Check references in an academic paper using RefCopilot.",
    )
    p.add_argument("--paper", required=True, help="ArXiv ID, URL, or local PDF/TeX path")
    p.add_argument("--output-file", default=None, help="Write the per-reference text report to this path")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    result = check_references(
        paper=args.paper,
        output_file=args.output_file,
        debug=args.debug,
        max_workers=args.max_workers,
    )

    if result["ok"]:
        print(f"References processed: {result['total_refs']}")
        print(
            f"Errors: {result['errors']}, Warnings: {result['warnings']}, Unverified: {result['unverified']}"
        )
        return 0
    print(f"ERROR: {result['error_message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
