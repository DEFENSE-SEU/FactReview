"""Opt-in live VLM probes; synthetic inputs cannot establish review accuracy.

Run ``python scripts/check_v2_visual.py`` after configuring the usual model or
VLM overrides. Transport codes exist only in the image, never in the prompt.
Figure probes use fixture parser output and the production material/check path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import uuid
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf

from common import run_stats
from common.env import load_env_file
from preprocessing.materials import build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from screening.checks import ask
from screening.figures import check_figures

FIGURE_CASES = (
    (
        "clear",
        "Figure 1. Panel (a) shows measured processing rate over time.",
        "Figure 1(a) shows the rate rising from 10 to 40 items per second between time 0 and 3 seconds.",
        None,
    ),
    (
        "missing_panel",
        "Figure 1. Panel (a) shows measured processing rate over time.",
        "Figure 1(c) shows the processing rate over time.",
        "text_figure_consistency",
    ),
    (
        "missing_labels",
        "Figure 1. Measurement results.",
        "Figure 1 shows the measurements.",
        "self_containedness",
    ),
    (
        "tiny_labels",
        "Figure 1. Panel (a) shows measured processing rate over time.",
        "Figure 1(a) shows the rate rising from 10 to 40 items per second between time 0 and 3 seconds.",
        "legibility",
    ),
)


def transport_probe(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    code = f"{secrets.randbelow(1000000):06d}"
    image = directory / "pixels.png"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=120)
        page.insert_text((35, 80), code, fontsize=48)
        page.get_pixmap(dpi=96).save(image)
    response = ask(
        'Read the six-digit code in the attached image. Return JSON {"code": "six digits"}.',
        {},
        module="screening_figures.transport_probe",
        images=[str(image)],
    )
    return {
        "kind": "pixel_transport",
        "passed": response.get("code") == code,
        "expected": code,
        "response": response,
        "input": str(image),
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }


def figure_probe(directory: Path, name: str, caption: str, reference: str, expected: str | None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    pdf = directory / "source.pdf"
    # Keep the layout and content fixed; only the printed label size changes.
    # At 96 dpi, 2 pt text has fewer than three vertical pixels per em.
    label_size = 2 if name == "tiny_labels" else 11
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=350)
        page.draw_line((65, 200), (345, 200), color=(0, 0, 0))
        page.draw_line((65, 200), (65, 40), color=(0, 0, 0))
        for index in range(4):
            page.insert_text((61 + index * 90, 217), str(index), fontsize=label_size)
            page.insert_text((43, 198 - index * 45), str((index + 1) * 10), fontsize=label_size)
        page.insert_text((68, 28), "(a)", fontsize=13)
        if name != "missing_labels":
            axis_size = 2 if name == "tiny_labels" else 12
            page.insert_text((165, 236), "Time (s)", fontsize=axis_size)
            page.insert_text((25, 180), "Rate (items/s)", fontsize=axis_size, rotate=90)
        points = [(65, 195), (155, 150), (245, 105), (335, 60)]
        for start, end in pairwise(points):
            page.draw_line(start, end, color=(0, 0, 0.8), width=2)
        page.draw_line((85, 48), (113, 48), color=(0, 0, 0.8), width=2)
        page.insert_text((120, 52), "Measured", fontsize=label_size)
        page.insert_textbox((20, 264, 382, 304), caption, fontsize=10)
        page.insert_textbox((20, 305, 382, 345), reference, fontsize=10)
        document.save(pdf)
    rows = [
        {
            "type": "image",
            "page_idx": 0,
            "image_caption": caption,
            "bbox": [15, 15, 370, 245],
            "bbox_space": "pdf_points",
        },
        {"type": "text", "page_idx": 0, "text": reference},
    ]
    parsed = MineruParseResult(
        caption + "\n\n" + reference, rows, None, "controlled_fixture", {}, "controlled_fixture"
    )
    materials = build_materials(parsed, paper_pdf=pdf, output_dir=directory / "materials", paper_key=name)
    records = []
    findings, issues = check_figures(materials, recover_errors=True, records=records)
    categories = {item.level for item in findings}
    inspected = len(records) == 1 and all(
        record.status == "checked" and record.printed_size_verified for record in records
    )
    return {
        "kind": "controlled_figure",
        "case": name,
        "expected_category": expected,
        "inputs": [
            {
                "image": figure.printed_crop_path,
                "sha256": hashlib.sha256(Path(figure.printed_crop_path).read_bytes()).hexdigest(),
                "printed_dpi": figure.printed_dpi,
                "bbox_points": figure.bbox_points,
            }
            for figure in materials.figures
        ],
        "passed": inspected and not issues and categories == ({expected} if expected else set()),
        "findings": [item.model_dump(mode="json") for item in findings],
        "issues": issues,
        "records": [item.model_dump() for item in records],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "transport", "figures"), default="all")
    parser.add_argument(
        "--figure-cases",
        nargs="+",
        choices=[case[0] for case in FIGURE_CASES],
        help="Run only these figure cases; transport probes still follow --mode.",
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "v2_visual")
    args = parser.parse_args(argv)
    if args.figure_cases and args.mode == "transport":
        parser.error("--figure-cases requires --mode figures or all")
    load_env_file(ROOT / ".env")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = (args.run_root / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "boundary": "Real configured VLM; synthetic images; figure parser is a fixture. No review accuracy estimate.",
        "mode": args.mode,
        "cases": [],
        "passed": False,
    }
    probes = []
    if args.mode in {"all", "transport"}:
        probes.extend((f"transport_{i}", lambda path: transport_probe(path)) for i in range(2))
    if args.mode in {"all", "figures"}:
        probes.extend(
            (case[0], lambda path, case=case: figure_probe(path, *case))
            for case in FIGURE_CASES
            if args.figure_cases is None or case[0] in args.figure_cases
        )
    with run_stats.run_scope(output / "run_stats.json"):
        for name, probe in probes:
            try:
                record = probe(output / name)
            except Exception as exc:
                record = {"case": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["cases"].append(record)
            (output / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({"case": name, "passed": record["passed"]}), flush=True)
    result["passed"] = all(case["passed"] for case in result["cases"])
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
