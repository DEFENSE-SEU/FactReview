from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fact_generation.execution.stage_runner import run_execution_stage
from util.paper_input import infer_paper_key, materialize_paper_pdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("factreview_stage_execution")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory to write stage outputs")
    p.add_argument("--paper-pdf", type=str, default="", help="Optional explicit paper PDF path or URL")
    p.add_argument("--paper-key", type=str, default="")
    p.add_argument(
        "--paper-extracted-dir", type=str, default="", help="Optional run-local MinerU extract snapshot"
    )
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--no-pdf-extract", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    paper_key = str(args.paper_key or "").strip() or (
        infer_paper_key(args.paper_pdf) if str(args.paper_pdf or "").strip() else ""
    )
    paper_pdf = (
        materialize_paper_pdf(
            args.paper_pdf,
            run_dir / "inputs" / "source_pdf",
            paper_key=paper_key,
        ).path
        if str(args.paper_pdf or "").strip()
        else None
    )
    result = run_execution_stage(
        run_dir=run_dir,
        paper_pdf=paper_pdf,
        paper_key=paper_key,
        paper_extracted_dir=str(args.paper_extracted_dir or "").strip(),
        max_attempts=int(args.max_attempts),
        no_pdf_extract=bool(args.no_pdf_extract),
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
