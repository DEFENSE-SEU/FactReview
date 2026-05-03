"""Reference-checking adapter.

Thin shim around RefCopilot (`tools/RefCopilot/`). The Markdown summary
included in the final FactReview review lists only fabricated references
(``severity == "error"``) so the appendix stays compact; warnings and
unverified entries remain in ``reference_check.json`` and are shown in full
when RefCopilot is invoked through its standalone CLI.

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
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFCOPILOT_SRC = _REPO_ROOT / "tools" / "RefCopilot" / "src"

if _REFCOPILOT_SRC.exists() and str(_REFCOPILOT_SRC) not in sys.path:
    sys.path.insert(0, str(_REFCOPILOT_SRC))


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

    return _check(
        paper,
        api_key=api_key,
        output_file=output_file,
        debug=debug,
        enable_parallel=enable_parallel,
        max_workers=max_workers,
    )


def format_reference_check_markdown(result: dict[str, Any], *, max_issues: int = 20) -> str:
    """Render the Markdown summary embedded in the final FactReview report.

    Errors only — the appendix is meant to flag fabricated citations rather
    than every metadata nit. Reviewers who need the full breakdown can read
    ``reference_check.json`` or run RefCopilot's standalone CLI.
    """
    from refcopilot.factreview import format_factreview_markdown  # type: ignore

    return format_factreview_markdown(result, max_issues=max_issues)


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
            f"Errors: {result['errors']}, Warnings: {result['warnings']}, "
            f"Unverified: {result['unverified']}"
        )
        return 0
    print(f"ERROR: {result['error_message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
