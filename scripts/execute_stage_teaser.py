from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from review.teaser.stage_runner import run_teaser_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("factreview_stage_teaser")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory to write stage outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    result = run_teaser_stage(run_dir=run_dir)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
