"""Command-line entry point for RefCopilot."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import SourceFormat
from refcopilot.pipeline import RefCopilotPipeline
from refcopilot.report import to_factreview_dict, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="refcopilot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="Check references in an input.")
    check.add_argument("input", help=".bib / .pdf / URL / arxiv:ID / plain text")
    check.add_argument(
        "--input-type",
        choices=[s.value for s in SourceFormat] + ["auto"],
        default="auto",
    )
    check.add_argument("--output", choices=["json", "markdown", "both"], default="both")
    check.add_argument("--output-dir", default="./refcopilot-out")
    check.add_argument("--no-llm-verify", action="store_true")
    check.add_argument("--cache-dir", default=None)
    check.add_argument("--cache-ttl-days", type=int, default=30)
    check.add_argument("--no-cache", action="store_true")
    check.add_argument("--max-refs", type=int, default=None, help="cap the number of references checked")
    check.add_argument("--debug", action="store_true")

    prune = sub.add_parser("cache", help="Cache management.")
    prune_sub = prune.add_subparsers(dest="cache_cmd", required=True)
    prune_p = prune_sub.add_parser("prune", help="Delete stale cache entries.")
    prune_p.add_argument("--cache-dir", default=None)
    prune_p.add_argument("--ttl-days", type=int, default=30)

    args = parser.parse_args(argv)

    if args.cmd == "check":
        return _run_check(args)
    if args.cmd == "cache" and args.cache_cmd == "prune":
        return _run_cache_prune(args)
    parser.error("unknown command")
    return 2


def _run_check(args) -> int:
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RefCopilotPipeline(
        cache_dir=args.cache_dir,
        cache_enabled=not args.no_cache,
        cache_ttl_days=args.cache_ttl_days,
        s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        s2_base_url=os.environ.get("SEMANTIC_SCHOLAR_BASE_URL"),
        use_llm_verify=not args.no_llm_verify,
    )

    input_kind = None if args.input_type == "auto" else SourceFormat(args.input_type)
    report = pipeline.run(args.input, input_type=input_kind, max_refs=args.max_refs)

    if args.output in ("json", "both"):
        summary_path = out_dir / "reference_check.json"
        summary_path.write_text(
            json.dumps(
                to_factreview_dict(report, report_file=str(out_dir / "details.txt")),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        full_json_path = out_dir / "reference_check.full.json"
        full_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"wrote {summary_path}", file=sys.stderr)
        print(f"wrote {full_json_path}", file=sys.stderr)

    if args.output in ("markdown", "both"):
        md_path = out_dir / "reference_check.md"
        md_path.write_text(to_markdown(report), encoding="utf-8")
        print(f"wrote {md_path}", file=sys.stderr)

    s = report.summary
    print(
        f"refs={s.total_refs} errors={s.errors} warnings={s.warnings} "
        f"unverified={s.unverified}"
    )
    return 0 if s.errors == 0 else 1


def _run_cache_prune(args) -> int:
    cache_dir = args.cache_dir or (Path.home() / ".cache" / "refcopilot")
    c = DiskCache(cache_dir, ttl_days=args.ttl_days)
    n = c.prune()
    print(f"removed {n} stale cache files from {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
