from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.parse.stage_runner import run_parse_stage
from util.paper_input import infer_paper_key, materialize_paper_pdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("factreview_stage_parse")
    p.add_argument("paper_pdf", type=str, help="Path or URL to a paper PDF")
    p.add_argument("--paper-key", type=str, default="")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory to write stage outputs")
    p.add_argument("--reuse-job-id", type=str, default="", help="Reuse an existing runtime job snapshot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    paper_key = (args.paper_key or "").strip() or infer_paper_key(args.paper_pdf)
    paper_input = materialize_paper_pdf(
        args.paper_pdf, run_dir / "inputs" / "source_pdf", paper_key=paper_key
    )
    result = run_parse_stage(
        repo_root=ROOT,
        run_dir=run_dir,
        paper_pdf=paper_input.path,
        paper_key=paper_key,
        reuse_job_id=str(args.reuse_job_id or "").strip(),
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
