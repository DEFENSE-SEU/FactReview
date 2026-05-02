from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


def _find_latest_ckpt(ckpt_dir: Path, prefix: str) -> Path | None:
    if not prefix:
        return None
    pats = [
        str(ckpt_dir / f"{prefix}*"),
    ]
    matches = []
    for pat in pats:
        matches.extend([Path(p) for p in glob.glob(pat)])
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="./checkpoints", help="checkpoint directory (default: ./checkpoints)")
    ap.add_argument("--prefix", required=True, help="checkpoint file prefix (before timestamp suffix)")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt = _find_latest_ckpt(ckpt_dir, args.prefix)
    if ckpt is None:
        print(json.dumps({"ok": False, "error": "ckpt_not_found", "ckpt_dir": str(ckpt_dir), "prefix": args.prefix}))
        return 2

    state = torch.load(str(ckpt), map_location="cpu")
    saved_args = state.get("args") or {}
    # Build a params namespace compatible with Runner.
    p = SimpleNamespace(**saved_args)

    # Avoid accidental GPU usage in environments without CUDA.
    try:
        p.gpu = "-1"
    except Exception:
        pass

    # Import Runner from repo's run.py
    from run import Runner  # type: ignore

    runner = Runner(p)
    runner.model.load_state_dict(state["state_dict"])
    try:
        runner.optimizer.load_state_dict(state["optimizer"])
    except Exception:
        pass

    # Evaluate
    res = runner.evaluate(args.split, epoch=-1)

    out = {
        "ok": True,
        "split": args.split,
        "dataset": getattr(p, "dataset", ""),
        "score_func": getattr(p, "score_func", ""),
        "opn": getattr(p, "opn", ""),
        "ckpt": str(ckpt),
        "mrr": res.get("mrr"),
        "mr": res.get("mr"),
        "hits@1": res.get("hits@1"),
        "hits@3": res.get("hits@3"),
        "hits@10": res.get("hits@10"),
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8", errors="ignore")
    print(json.dumps({"ok": True, "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


