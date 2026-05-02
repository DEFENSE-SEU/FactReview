from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from common.env import load_env_file

_SECTION_RE = re.compile(
    r"(?ims)^##\s+(?P<title>(?:\*\*)?\d+\.\s+.+?(?:\*\*)?)\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)
_BULLET_FIELD_RE = re.compile(r"^\s*[-*•]\s*(?P<key>[A-Za-z][A-Za-z\s]+?)\s*:\s*(?P<value>.+?)\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_LOCATION_RE = re.compile(r"(?im)^\s*Location\s*:\s*(?P<value>.+?)\s*$")
_LABEL_BLOCK_RE = re.compile(
    r"(?ims)^\s*(?P<label>Main Result|Ablation Result|Strengths|Weaknesses)\s*:?\s*$"
    r"(?P<body>.*?)(?=^\s*(?:Main Result|Ablation Result|Strengths|Weaknesses)\s*:?\s*$|\Z)"
)


@dataclass(frozen=True)
class TableBlock:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class TeaserFigurePayload:
    title: str
    task: str
    status_legend: list[str]
    technical_positioning_caption: str
    technical_positioning_image: str
    technical_positioning_table: TableBlock | None
    claims_table: TableBlock | None
    selected_claim_rows: list[dict[str, str]]
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    experiment_main_location: str
    experiment_main_table: TableBlock | None
    experiment_ablation_location: str
    experiment_ablation_table: TableBlock | None


@dataclass(frozen=True)
class TeaserFigureGenerationResult:
    status: str
    prompt: str
    prompt_path: str
    image_path: str
    response_path: str
    model: str
    message: str
    clipboard_copied: bool
    used_gemini_api: bool
    source_markdown_path: str


@dataclass(frozen=True)
class TemplateAnchor:
    name: str
    bbox: tuple[float, float, float, float]
    text: str


_TEMPLATE_REGION_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "title_banner": (0.04, 0.04, 0.97, 0.15),
    "task_badge": (0.75, 0.04, 0.98, 0.10),
    "status_badges": (0.67, 0.08, 0.98, 0.145),
    "delta_badges": (0.77, 0.12, 0.98, 0.175),
    "main_canvas": (0.04, 0.15, 0.97, 0.92),
    "technical_panel": (0.05, 0.17, 0.56, 0.56),
    "claims_panel": (0.56, 0.17, 0.80, 0.56),
    "summary_panel": (0.80, 0.17, 0.96, 0.92),
    "strengths_panel": (0.80, 0.58, 0.96, 0.75),
    "weaknesses_panel": (0.80, 0.75, 0.96, 0.92),
    "experiments_panel": (0.05, 0.58, 0.79, 0.90),
}

_TEMPLATE_REGION_PROMPT_HINTS: dict[str, str] = {
    "title_banner": "Restore the dark top banner spanning nearly the full width, with the title left-aligned inside it, followed by the summary text — do not prefix the summary with 'TL;DR' or any other label.",
    "task_badge": "Place the task label as a rounded badge right-aligned at the top-right corner, directly above the status badge row. Its width must auto-fit the text content — do not fix or extend the left edge to fill the full bbox width. Right edge stays pinned at the right margin; only the left edge floats with content length.",
    "status_badges": "Preserve the top-right status badge strip as a tight horizontal run of rounded badges with the original ordering and spacing.",
    "delta_badges": "Keep the Improvement and Reduction badges on their own lower row directly beneath the status badges.",
    "main_canvas": "Preserve the large light-gray body shell under the header; do not switch to a flat white or fully reflowed canvas.",
    "technical_panel": "Keep the technical-positioning block on the left side of the body, occupying the largest panel width.",
    "claims_panel": "Keep the claim/evidence rows in the middle-right column rather than moving them below the figure or into the summary panel.",
    "summary_panel": "Keep the summary column docked on the far right with stacked Strengths and Weaknesses blocks.",
    "strengths_panel": "Restore the green-tinted Strengths background in the upper part of the right summary column.",
    "weaknesses_panel": "Restore the pink-tinted Weaknesses background in the lower part of the right summary column.",
    "experiments_panel": "Keep the experiment/result tables across the lower-left and lower-middle area, below the technical and claims panels.",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _copy_text_to_clipboard(text: str) -> bool:
    if not str(text or ""):
        return False

    commands: list[list[str]] = []
    if shutil.which("pbcopy"):
        commands.append(["pbcopy"])
    if shutil.which("wl-copy"):
        commands.append(["wl-copy"])
    if shutil.which("xclip"):
        commands.append(["xclip", "-selection", "clipboard"])
    if shutil.which("xsel"):
        commands.append(["xsel", "--clipboard", "--input"])
    if os.name == "nt" and shutil.which("clip"):
        commands.append(["clip"])

    for command in commands:
        try:
            subprocess.run(
                command,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            continue
    return False


def _prompt_only_message(reason: str, *, clipboard_copied: bool) -> str:
    copy_sentence = (
        "Prompt was also copied to the clipboard."
        if clipboard_copied
        else "Automatic clipboard copy was unavailable; open the prompt file and copy it manually."
    )
    return (
        f"{reason} Prompt was written to disk. {copy_sentence} "
        "Paste it into the Gemini web app to generate the teaser figure manually."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _template_png_path() -> Path:
    return _repo_root() / "demos" / "compgcn" / "teaser_reference.png"


def _template_reference_png_bytes(scale: float = 0.9) -> bytes | None:
    png_path = _template_png_path()
    if not png_path.exists():
        return None
    try:
        if scale <= 0 or abs(scale - 1.0) < 0.001:
            return png_path.read_bytes()
        with Image.open(png_path) as image:
            image = image.convert("RGB")
            width = max(1, round(image.width * scale))
            height = max(1, round(image.height * scale))
            resized = image.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return None


def _template_layout_signature(max_panels: int = 14) -> str:
    _ = max_panels
    png_path = _template_png_path()
    if not png_path.exists():
        return "Template PNG missing."
    try:
        with Image.open(png_path) as image:
            width = max(float(image.width or 1), 1.0)
            height = max(float(image.height or 1), 1.0)
            try:
                relative = png_path.relative_to(_repo_root())
            except ValueError:
                relative = png_path
            return f"Canvas size {int(width)}x{int(height)} (~{width / height:.3f}:1), reference image: {relative}"
    except Exception as exc:
        return f"Template signature extraction failed: {type(exc).__name__}: {exc}"


def _format_bbox(bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    return f"({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f})"


def _template_visual_anchors() -> list[TemplateAnchor]:
    return []


def _template_visual_anchor_summary() -> str:
    anchors = _template_visual_anchors()
    if not anchors:
        try:
            relative = _template_png_path().relative_to(_repo_root())
        except ValueError:
            relative = _template_png_path()
        return f"Reference template image: {relative}; rely on the attached template image and fixed module constraints."
    lines = [
        "Exact visual anchors extracted from the template image (normalized coordinates on a 16:9 canvas):"
    ]
    for anchor in anchors:
        lines.append(f"- {anchor.name}: bbox={_format_bbox(anchor.bbox)}; text={anchor.text}")
    return "\n".join(lines)


def _template_region_constraints() -> str:
    lines = ["Lock these structural regions to the template's geometry:"]
    for name, bbox in _TEMPLATE_REGION_BBOXES.items():
        hint = _TEMPLATE_REGION_PROMPT_HINTS.get(name, "")
        lines.append(f"- {name}: bbox={_format_bbox(bbox)}. {hint}".strip())
    return "\n".join(lines)


def _template_reference_image() -> Image.Image | None:
    template_png = _template_reference_png_bytes(scale=1.0)
    if not template_png:
        return None
    try:
        return Image.open(io.BytesIO(template_png)).convert("RGB")
    except Exception:
        return None


def _crop_normalized(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = max(0, min(width, round(bbox[0] * width)))
    top = max(0, min(height, round(bbox[1] * height)))
    right = max(left + 1, min(width, round(bbox[2] * width)))
    bottom = max(top + 1, min(height, round(bbox[3] * height)))
    return image.crop((left, top, right, bottom))


def _grid_similarity(
    template_image: Image.Image, generated_image: Image.Image, *, cols: int, rows: int
) -> float:
    template_grid = np.asarray(template_image.resize((cols, rows), Image.BILINEAR), dtype=np.float32)
    generated_grid = np.asarray(generated_image.resize((cols, rows), Image.BILINEAR), dtype=np.float32)
    mae = float(np.abs(template_grid - generated_grid).mean() / 255.0)
    return max(0.0, min(1.0, 1.0 - mae))


def _edge_similarity(
    template_image: Image.Image, generated_image: Image.Image, *, cols: int, rows: int
) -> float:
    template_gray = (
        np.asarray(template_image.convert("L").resize((cols, rows), Image.BILINEAR), dtype=np.float32) / 255.0
    )
    generated_gray = (
        np.asarray(generated_image.convert("L").resize((cols, rows), Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    template_edges = np.concatenate(
        [
            np.abs(np.diff(template_gray, axis=1)).reshape(-1),
            np.abs(np.diff(template_gray, axis=0)).reshape(-1),
        ]
    )
    generated_edges = np.concatenate(
        [
            np.abs(np.diff(generated_gray, axis=1)).reshape(-1),
            np.abs(np.diff(generated_gray, axis=0)).reshape(-1),
        ]
    )
    if template_edges.size == 0 or generated_edges.size == 0:
        return 0.0
    mae = float(np.abs(template_edges - generated_edges).mean())
    return max(0.0, min(1.0, 1.0 - mae))


def _region_edge_density(image: Image.Image, bbox: tuple[float, float, float, float]) -> float:
    region = _crop_normalized(image, bbox).convert("L")
    arr = np.asarray(region.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.0
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    merged = np.concatenate([gx.reshape(-1), gy.reshape(-1)])
    if merged.size == 0:
        return 0.0
    return float(np.mean(merged))


def _build_validation_feedback(weak_regions: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for region in weak_regions[:4]:
        name = str(region.get("name") or "").strip()
        hint = _TEMPLATE_REGION_PROMPT_HINTS.get(name)
        if hint and hint not in hints:
            hints.append(hint)
    if not hints:
        hints.append(
            "The generated figure still drifts from the template. Match the template's module geometry, colored panels, and badge placement more literally."
        )
    return hints


def _validate_generated_teaser_image(image_path: str | Path) -> dict[str, Any]:
    template_image = _template_reference_image()
    if template_image is None:
        return {
            "passed": True,
            "score": 1.0,
            "threshold": 0.0,
            "color_similarity": 1.0,
            "edge_similarity": 1.0,
            "region_scores": [],
            "prompt_feedback": [],
            "reason": "Template image unavailable; skipped strict validation.",
        }

    generated = (
        Image.open(_coerce_path(image_path)).convert("RGB").resize(template_image.size, Image.BILINEAR)
    )
    color_similarity = _grid_similarity(template_image, generated, cols=24, rows=14)
    edge_similarity = _edge_similarity(template_image, generated, cols=24, rows=14)
    region_scores: list[dict[str, Any]] = []
    for name, bbox in _TEMPLATE_REGION_BBOXES.items():
        template_region = _crop_normalized(template_image, bbox)
        generated_region = _crop_normalized(generated, bbox)
        similarity = _grid_similarity(template_region, generated_region, cols=10, rows=6)
        region_scores.append({"name": name, "bbox": _format_bbox(bbox), "similarity": round(similarity, 4)})
    region_scores.sort(key=lambda item: float(item["similarity"]))

    overall_score = (0.68 * color_similarity) + (0.32 * edge_similarity)
    threshold = float(os.getenv("TEASER_TEMPLATE_SIMILARITY_THRESHOLD") or "0.78")
    core_regions = {"title_banner", "status_badges", "summary_panel", "experiments_panel", "main_canvas"}
    core_floor = float(os.getenv("TEASER_TEMPLATE_CORE_REGION_MIN") or "0.66")
    weak_regions = [item for item in region_scores if float(item["similarity"]) < max(0.74, threshold - 0.04)]
    core_ok = all(
        float(item["similarity"]) >= core_floor for item in region_scores if str(item["name"]) in core_regions
    )
    region_similarity = {str(item["name"]): float(item["similarity"]) for item in region_scores}
    required_region_mins = {
        "summary_panel": float(os.getenv("TEASER_REQUIRED_SUMMARY_MIN") or "0.72"),
        "strengths_panel": float(os.getenv("TEASER_REQUIRED_STRENGTHS_MIN") or "0.70"),
        "weaknesses_panel": float(os.getenv("TEASER_REQUIRED_WEAKNESSES_MIN") or "0.70"),
        "technical_panel": float(os.getenv("TEASER_REQUIRED_TECHNICAL_MIN") or "0.70"),
    }
    required_regions_ok = all(
        region_similarity.get(name, 0.0) >= float(min_v) for name, min_v in required_region_mins.items()
    )

    summary_density = _region_edge_density(generated, _TEMPLATE_REGION_BBOXES["summary_panel"])
    strengths_density = _region_edge_density(generated, _TEMPLATE_REGION_BBOXES["strengths_panel"])
    weaknesses_density = _region_edge_density(generated, _TEMPLATE_REGION_BBOXES["weaknesses_panel"])
    density_floor = float(os.getenv("TEASER_SUMMARY_DENSITY_MIN") or "0.035")
    summary_presence_ok = (
        summary_density >= density_floor
        and strengths_density >= density_floor * 0.85
        and weaknesses_density >= density_floor * 0.85
    )

    passed = overall_score >= threshold and core_ok and required_regions_ok and summary_presence_ok
    prompt_feedback = _build_validation_feedback(weak_regions)
    if not summary_presence_ok:
        prompt_feedback.append(
            "Summary module is mandatory: render the right Summary panel with visible text, and include non-empty Strengths and Weaknesses blocks."
        )
    if region_similarity.get("technical_panel", 0.0) < required_region_mins["technical_panel"]:
        prompt_feedback.append(
            "Use the provided technical-positioning image as the exact visual anchor in the technical panel; do not replace it with a different diagram."
        )
    return {
        "passed": passed,
        "score": round(overall_score, 4),
        "threshold": threshold,
        "color_similarity": round(color_similarity, 4),
        "edge_similarity": round(edge_similarity, 4),
        "region_scores": region_scores,
        "required_region_mins": required_region_mins,
        "summary_density": round(summary_density, 4),
        "strengths_density": round(strengths_density, 4),
        "weaknesses_density": round(weaknesses_density, 4),
        "summary_presence_ok": bool(summary_presence_ok),
        "required_regions_ok": bool(required_regions_ok),
        "prompt_feedback": prompt_feedback,
        "reason": "passed" if passed else "template_similarity_below_threshold",
    }


def _int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _ensure_env_loaded() -> None:
    load_env_file(_repo_root() / ".env")


def _env_true(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _strip_inline_markup(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"</?span[^>]*>", "", value, flags=re.IGNORECASE)
    value = value.replace("**", "").replace("`", "")
    value = re.sub(r"<[^>]+>", "", value)
    return value.strip()


def _extract_sections(markdown_text: str) -> dict[str, str]:
    def _canonical_title(raw_title: str) -> str:
        plain = _strip_inline_markup(raw_title).lower()
        plain = re.sub(r"^\d+\.\s*", "", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if "metadata" in plain:
            return "1. Metadata"
        if "technical positioning" in plain:
            return "2. Technical Positioning"
        if "claims" in plain:
            return "3. Claims"
        if "summary" in plain:
            return "4. Summary"
        if "experiment" in plain:
            return "5. Experiment"
        return str(raw_title or "").strip()

    sections: dict[str, str] = {}
    for match in _SECTION_RE.finditer(markdown_text or ""):
        title = str(match.group("title") or "").strip()
        body = str(match.group("body") or "").strip()
        sections[_canonical_title(title)] = body
    return sections


def _parse_markdown_table(block: str) -> TableBlock | None:
    lines = [ln.rstrip() for ln in (block or "").splitlines()]
    table_lines = [ln for ln in lines if ln.strip().startswith("|") and ln.strip().endswith("|")]
    if len(table_lines) < 2:
        return None

    headers = [_strip_inline_markup(cell) for cell in table_lines[0].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for line in table_lines[1:]:
        cells = [_strip_inline_markup(cell) for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        rows.append(cells[: len(headers)])
    return TableBlock(headers=headers, rows=rows)


def _table_to_markdown(table: TableBlock | None) -> str:
    if table is None:
        return "Not found in manuscript"
    head = "| " + " | ".join(table.headers) + " |"
    sep = "| " + " | ".join(["---"] * len(table.headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in table.rows)
    return "\n".join([head, sep, body]).strip()


def _normalize_header_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _find_header_index(table: TableBlock | None, candidates: tuple[str, ...]) -> int:
    if table is None:
        return -1
    normalized = [_normalize_header_token(h) for h in table.headers]
    for idx, header in enumerate(normalized):
        for token in candidates:
            token_norm = _normalize_header_token(token)
            if token_norm and token_norm in header:
                return idx
    return -1


def _first_number(value: str) -> float | None:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _status_rank(value: str) -> int:
    text = _normalize_header_token(_strip_inline_markup(value))
    if "supported" in text and "paper" not in text and "partial" not in text:
        return 4
    if "paper-supported" in text or "paper supported" in text:
        return 3
    if "inconclusive" in text or "partial" in text or "⚠" in value:
        return 2
    if "in conflict" in text or "conflict" in text or "✗" in value:
        return 1
    return 0


def _metric_is_lower_better(metric_value: str) -> bool:
    s = _normalize_header_token(metric_value)
    # MRR must be checked before MR to avoid substring false-match.
    if "mrr" in s:
        return False
    if s in {"mr", "mean rank"} or "mean rank" in s:
        return True
    lower_tokens = ("loss", "error", "wer", "cer", "perplexity")
    return any(tok in s for tok in lower_tokens)


def _annotate_diff(raw: str, *, improved: bool | None) -> str:
    clean = raw.strip()
    if improved is None:
        return clean
    return f"[GREEN]{clean}[/GREEN]" if improved else f"[RED]{clean}[/RED]"


def _colorize_main_diff(diff_text: str, metric_text: str) -> str:
    delta = _first_number(diff_text)
    if delta is None or abs(delta) <= 1e-12:
        return diff_text.strip()
    lower_better = _metric_is_lower_better(metric_text)
    improved = delta < 0 if lower_better else delta > 0
    return _annotate_diff(diff_text, improved=improved)


def _colorize_ablation_diff(diff_text: str) -> str:
    delta = _first_number(diff_text)
    if delta is None or abs(delta) <= 1e-12:
        return diff_text.strip()
    return _annotate_diff(diff_text, improved=delta < 0)


def _experiment_table_to_markdown(table: TableBlock | None, *, is_ablation: bool) -> str:
    if table is None:
        return "Not found in manuscript"
    diff_idx = _find_header_index(table, ("difference", "delta", "Δ"))
    metric_idx = _find_header_index(table, ("metric",))
    dimension_idx = _find_header_index(table, ("ablation dimension", "dimension")) if is_ablation else -1
    head = "| " + " | ".join(table.headers) + " |"
    sep = "| " + " | ".join(["---"] * len(table.headers)) + " |"
    rows_md: list[str] = []
    for row in table.rows:
        colored = list(row)
        if 0 <= diff_idx < len(colored):
            if is_ablation and _is_ablation_anchor_row(row, dimension_idx):
                pass  # Optimal Setup is the reference row; ablation delta semantics don't apply.
            elif is_ablation:
                colored[diff_idx] = _colorize_ablation_diff(colored[diff_idx])
            else:
                metric_text = row[metric_idx] if 0 <= metric_idx < len(row) else ""
                colored[diff_idx] = _colorize_main_diff(colored[diff_idx], metric_text)
        rows_md.append("| " + " | ".join(colored) + " |")
    return "\n".join([head, sep] + rows_md).strip()


def _main_result_row_value(
    row: list[str],
    *,
    task_idx: int,
    dataset_idx: int,
    metric_idx: int,
    baseline_idx: int,
    paper_idx: int,
    diff_idx: int,
    status_idx: int,
) -> tuple[float, int, float]:
    metric_text = row[metric_idx] if 0 <= metric_idx < len(row) else ""
    lower_better = _metric_is_lower_better(metric_text)
    delta = _first_number(row[diff_idx]) if 0 <= diff_idx < len(row) else None
    baseline = _first_number(row[baseline_idx]) if 0 <= baseline_idx < len(row) else None
    paper = _first_number(row[paper_idx]) if 0 <= paper_idx < len(row) else None
    if delta is None and baseline is not None and paper is not None:
        delta = (baseline - paper) if lower_better else (paper - baseline)
    if delta is None:
        if paper is not None:
            delta = -paper if lower_better else paper
        else:
            delta = float("-inf")
    status_score = _status_rank(row[status_idx]) if 0 <= status_idx < len(row) else 0
    return (float(delta), int(status_score), abs(float(delta)) if delta != float("-inf") else 0.0)


def _compress_main_result_table(table: TableBlock | None) -> TableBlock | None:
    if table is None or not table.rows:
        return table
    task_idx = _find_header_index(table, ("task",))
    dataset_idx = _find_header_index(table, ("dataset",))
    metric_idx = _find_header_index(table, ("metric",))
    baseline_idx = _find_header_index(table, ("best baseline", "baseline"))
    paper_idx = _find_header_index(table, ("paper result", "result"))
    diff_idx = _find_header_index(table, ("difference", "delta", "Δ"))
    status_idx = _find_header_index(table, ("evaluation status", "status"))

    if dataset_idx < 0 and task_idx < 0:
        return table

    grouped: dict[tuple[str, str], list[str]] = {}
    for row in table.rows:
        task_value = row[task_idx].strip() if 0 <= task_idx < len(row) else ""
        dataset_value = row[dataset_idx].strip() if 0 <= dataset_idx < len(row) else ""
        key = (_normalize_header_token(task_value) or "_", _normalize_header_token(dataset_value) or "_")
        prev = grouped.get(key)
        if prev is None:
            grouped[key] = row
            continue
        current_score = _main_result_row_value(
            row,
            task_idx=task_idx,
            dataset_idx=dataset_idx,
            metric_idx=metric_idx,
            baseline_idx=baseline_idx,
            paper_idx=paper_idx,
            diff_idx=diff_idx,
            status_idx=status_idx,
        )
        prev_score = _main_result_row_value(
            prev,
            task_idx=task_idx,
            dataset_idx=dataset_idx,
            metric_idx=metric_idx,
            baseline_idx=baseline_idx,
            paper_idx=paper_idx,
            diff_idx=diff_idx,
            status_idx=status_idx,
        )
        # Prefer higher value improvement, then stronger status, then larger absolute effect.
        if current_score > prev_score:
            grouped[key] = row

    selected_rows = [grouped[key] for key in grouped]
    return TableBlock(headers=table.headers, rows=selected_rows)


def _ablation_row_effect(row: list[str], *, full_idx: int, paper_idx: int, diff_idx: int) -> float:
    delta = _first_number(row[diff_idx]) if 0 <= diff_idx < len(row) else None
    if delta is not None:
        return abs(float(delta))
    full_v = _first_number(row[full_idx]) if 0 <= full_idx < len(row) else None
    paper_v = _first_number(row[paper_idx]) if 0 <= paper_idx < len(row) else None
    if full_v is not None and paper_v is not None:
        return abs(float(paper_v - full_v))
    return float("-inf")


def _ablation_reference_full_model(
    table: TableBlock,
    *,
    full_idx: int,
    config_idx: int,
) -> float | None:
    if full_idx < 0:
        return None
    ref_from_base: float | None = None
    values: list[float] = []
    for row in table.rows:
        full_v = _first_number(row[full_idx]) if full_idx < len(row) else None
        if full_v is None:
            continue
        values.append(float(full_v))
        if config_idx >= 0 and config_idx < len(row):
            cfg = _normalize_header_token(row[config_idx])
            if any(tok in cfg for tok in ("base", "full model", "default", "baseline")):
                ref_from_base = float(full_v)
    if ref_from_base is not None:
        return ref_from_base
    if not values:
        return None
    # Fallback to mode (most frequent full-model value), then first seen.
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    mode_value = max(counts.items(), key=lambda kv: (kv[1], -values.index(kv[0])))[0]
    return float(mode_value)


def _ablation_row_effect_with_reference(
    row: list[str],
    *,
    full_idx: int,
    paper_idx: int,
    diff_idx: int,
    reference_full_model: float | None,
) -> float:
    delta = _first_number(row[diff_idx]) if 0 <= diff_idx < len(row) else None
    if delta is not None:
        return abs(float(delta))
    paper_v = _first_number(row[paper_idx]) if 0 <= paper_idx < len(row) else None
    if paper_v is not None and reference_full_model is not None:
        return abs(float(paper_v - reference_full_model))
    # Legacy fallback only when reference cannot be resolved.
    return _ablation_row_effect(row, full_idx=full_idx, paper_idx=paper_idx, diff_idx=diff_idx)


def _is_ablation_anchor_row(row: list[str], dimension_idx: int) -> bool:
    dim = row[dimension_idx].strip().lower() if dimension_idx < len(row) else ""
    return dim == "optimal setup"


def _compress_ablation_table(table: TableBlock | None) -> TableBlock | None:
    if table is None or not table.rows:
        return table
    dimension_idx = _find_header_index(table, ("ablation dimension", "dimension"))
    if dimension_idx < 0:
        return table
    config_idx = _find_header_index(table, ("configuration", "config"))
    full_idx = _find_header_index(table, ("full model",))
    paper_idx = _find_header_index(table, ("paper result", "result"))
    diff_idx = _find_header_index(table, ("difference", "delta", "Δ"))
    status_idx = _find_header_index(table, ("evaluation status", "status"))
    reference_full_model = _ablation_reference_full_model(
        table,
        full_idx=full_idx,
        config_idx=config_idx,
    )

    anchor_rows: list[list[str]] = []
    grouped: dict[str, list[str]] = {}
    for row in table.rows:
        # Always preserve the "Optimal setup" anchor row — never compress it away.
        if _is_ablation_anchor_row(row, dimension_idx):
            anchor_rows.append(row)
            continue
        dim = row[dimension_idx].strip() if dimension_idx < len(row) else ""
        key = _normalize_header_token(dim) or "_"
        prev = grouped.get(key)
        if prev is None:
            grouped[key] = row
            continue
        current_effect = _ablation_row_effect_with_reference(
            row,
            full_idx=full_idx,
            paper_idx=paper_idx,
            diff_idx=diff_idx,
            reference_full_model=reference_full_model,
        )
        prev_effect = _ablation_row_effect_with_reference(
            prev,
            full_idx=full_idx,
            paper_idx=paper_idx,
            diff_idx=diff_idx,
            reference_full_model=reference_full_model,
        )
        current_status = _status_rank(row[status_idx]) if 0 <= status_idx < len(row) else 0
        prev_status = _status_rank(prev[status_idx]) if 0 <= status_idx < len(prev) else 0
        # Prefer larger absolute effect, then better status.
        if (current_effect, current_status) > (prev_effect, prev_status):
            grouped[key] = row

    selected_rows = anchor_rows + [grouped[key] for key in grouped]
    return TableBlock(headers=table.headers, rows=selected_rows)


def _extract_first_table(text: str) -> TableBlock | None:
    blocks: list[str] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    for block in blocks:
        table = _parse_markdown_table(block)
        if table is not None:
            return table
    return None


def _extract_metadata(body: str) -> tuple[str, str]:
    title = ""
    task = ""
    for line in (body or "").splitlines():
        match = _BULLET_FIELD_RE.match(_strip_inline_markup(line))
        if not match:
            continue
        key = str(match.group("key") or "").strip().lower()
        value = str(match.group("value") or "").strip()
        if key == "title":
            title = value
        elif key == "task":
            task = value
    return title, task


def _extract_status_legend(text: str) -> list[str]:
    match = re.search(r"(?ims)\(Status legend:\s*(?P<body>.*?)\)", text or "")
    if not match:
        return []
    body = re.sub(r"\s+", " ", str(match.group("body") or "")).strip()
    parts = [part.strip(" .") for part in re.split(r"[,;]", body) if part.strip()]
    return parts


def _extract_technical_positioning(body: str) -> tuple[str, str, TableBlock | None]:
    lines = [line.strip() for line in (body or "").splitlines() if line.strip()]
    caption = ""
    for line in lines:
        plain = _strip_inline_markup(line)
        if plain.lower().startswith("figure "):
            caption = plain
            break
    if not caption:
        for line in lines:
            if line.startswith("![") or (line.startswith("|") and line.endswith("|")):
                continue
            caption = _strip_inline_markup(line)
            break
    image_match = _MARKDOWN_IMAGE_RE.search(body or "")
    image_src = str(image_match.group("src") or "").strip() if image_match else ""
    table = _extract_first_table(body)
    return caption, image_src, table


def _extract_claims(body: str) -> tuple[TableBlock | None, list[str]]:
    table = _extract_first_table(body)
    status_legend = _extract_status_legend(body)
    return table, status_legend


def _extract_labeled_block(body: str, label: str) -> str:
    patterns = [
        re.compile(
            rf"(?ims)^\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*(?P<content>.*?)(?=^\s*(?:\*\*)?(?:Strengths|Weaknesses)(?:\*\*)?\s*:|\Z)"
        ),
        re.compile(
            rf"(?ims)^\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*$\n(?P<content>.*?)(?=^\s*(?:\*\*)?(?:Strengths|Weaknesses)(?:\*\*)?\s*$|\Z)"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(body or "")
        if match:
            return str(match.group("content") or "").strip()
    return ""


def _extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for line in (text or "").splitlines():
        stripped = _strip_inline_markup(line)
        if stripped.startswith(("- ", "* ", "• ")):
            bullets.append(stripped[2:].strip())
    return bullets


def _extract_summary(body: str) -> tuple[str, list[str], list[str]]:
    summary_lines: list[str] = []
    strengths_lines: list[str] = []
    weaknesses_lines: list[str] = []
    mode = "summary"

    for raw_line in (body or "").splitlines():
        plain = _strip_inline_markup(raw_line)
        match_strengths = re.match(r"(?i)^\s*strengths\s*:\s*(.*)$", plain)
        if match_strengths:
            mode = "strengths"
            tail = str(match_strengths.group(1) or "").strip()
            if tail:
                strengths_lines.append(tail if tail.startswith(("- ", "* ", "• ")) else f"- {tail}")
            continue
        match_weaknesses = re.match(r"(?i)^\s*weaknesses\s*:\s*(.*)$", plain)
        if match_weaknesses:
            mode = "weaknesses"
            tail = str(match_weaknesses.group(1) or "").strip()
            if tail:
                weaknesses_lines.append(tail if tail.startswith(("- ", "* ", "• ")) else f"- {tail}")
            continue

        if mode == "summary":
            summary_lines.append(raw_line)
        elif mode == "strengths":
            strengths_lines.append(raw_line)
        else:
            weaknesses_lines.append(raw_line)

    summary_text = re.sub(r"\n{2,}", "\n\n", "\n".join(summary_lines)).strip()
    strengths = _extract_bullets("\n".join(strengths_lines))
    weaknesses = _extract_bullets("\n".join(weaknesses_lines))
    return summary_text, strengths, weaknesses


def _extract_experiment_subsection(body: str, label: str) -> tuple[str, TableBlock | None]:
    match = re.search(
        rf"(?ims)^\s*(?:###\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:?\s*$\n(?P<content>.*?)(?=^\s*(?:###\s*)?(?:\*\*)?(?:Main Result|Ablation Result)(?:\*\*)?\s*:?\s*$|\Z)",
        body or "",
    )
    if not match:
        return "", None
    content = str(match.group("content") or "").strip()
    location = ""
    for line in content.splitlines():
        location_match = _LOCATION_RE.search(_strip_inline_markup(line))
        if location_match:
            location = str(location_match.group("value") or "").strip()
            break
    table = _extract_first_table(content)
    return location, table


def _row_to_dict(table: TableBlock, row: list[str]) -> dict[str, str]:
    normalized = row[: len(table.headers)] + [""] * max(0, len(table.headers) - len(row))
    return {
        header: re.sub(r"\s+", " ", str(value or "")).strip() or "Not found in manuscript"
        for header, value in zip(table.headers, normalized, strict=False)
    }


def _select_claim_rows(table: TableBlock | None, limit: int = 3) -> list[dict[str, str]]:
    if table is None:
        return []
    claim_idx = next((idx for idx, header in enumerate(table.headers) if "claim" in header.lower()), -1)
    evidence_idx = next((idx for idx, header in enumerate(table.headers) if "evidence" in header.lower()), -1)
    status_idx = next((idx for idx, header in enumerate(table.headers) if "status" in header.lower()), -1)
    selected: list[dict[str, str]] = []
    for row in table.rows:
        claim_text = row[claim_idx].strip() if claim_idx >= 0 and claim_idx < len(row) else ""
        evidence_text = row[evidence_idx].strip() if evidence_idx >= 0 and evidence_idx < len(row) else ""
        status_text = row[status_idx].strip() if status_idx >= 0 and status_idx < len(row) else ""
        if not any([claim_text, evidence_text, status_text]):
            continue
        selected.append(
            {
                "Claim": claim_text or "Not found in manuscript",
                "Evidence": evidence_text or "Not found in manuscript",
                "Status": status_text or "Not found in manuscript",
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _derive_claims_aggregate_status(rows: list[dict[str, str]]) -> str:
    statuses = [_strip_inline_markup(row.get("Status", "")).lower() for row in rows]
    has_reproduced_supported = any(
        s.startswith("supported") or s.startswith("✓") or "✓" in s
        for s in statuses
    )
    has_reproduced_conflict = any(
        "in conflict" in s or s.startswith("✗") or "✗" in s
        for s in statuses
    )
    if has_reproduced_supported and has_reproduced_conflict:
        return "⚠ Partially supported"
    if has_reproduced_supported:
        return "✓ Supported"
    if has_reproduced_conflict:
        return "✗ In conflict"
    return "⚠ Inconclusive"


def _format_selected_claims(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "1. **Claim:** **Not found in manuscript**"
    return "\n".join(
        f"{idx}. **Claim:** **{row.get('Claim', 'Not found in manuscript')}**; "
        f"**Evidence:** {row.get('Evidence', 'Not found in manuscript')}; "
        f"**Status:** {row.get('Status', 'Not found in manuscript')}"
        for idx, row in enumerate(rows, start=1)
    )


def extract_teaser_figure_payload(markdown_text: str) -> TeaserFigurePayload:
    sections = _extract_sections(markdown_text)
    metadata = sections.get("1. Metadata", "")
    technical = sections.get("2. Technical Positioning", "")
    claims = sections.get("3. Claims", "")
    summary = sections.get("4. Summary", "")
    experiment = sections.get("5. Experiment", "")

    title, task = _extract_metadata(metadata)
    tp_caption, tp_image, tp_table = _extract_technical_positioning(technical)
    claims_table, claims_status = _extract_claims(claims)
    summary_text, strengths, weaknesses = _extract_summary(summary)
    main_location, main_table = _extract_experiment_subsection(experiment, "Main Result")
    ablation_location, ablation_table = _extract_experiment_subsection(experiment, "Ablation Result")
    main_table = _compress_main_result_table(main_table)
    ablation_table = _compress_ablation_table(ablation_table)
    selected_claim_rows = _select_claim_rows(claims_table, limit=3)

    return TeaserFigurePayload(
        title=title or "Not found in manuscript",
        task=task or "Not found in manuscript",
        status_legend=claims_status,
        technical_positioning_caption=tp_caption or "Not found in manuscript",
        technical_positioning_image=tp_image or "Not found in manuscript",
        technical_positioning_table=tp_table,
        claims_table=claims_table,
        selected_claim_rows=selected_claim_rows,
        summary=summary_text or "Not found in manuscript",
        strengths=strengths,
        weaknesses=weaknesses,
        experiment_main_location=main_location or "Not found in manuscript",
        experiment_main_table=main_table,
        experiment_ablation_location=ablation_location or "Not found in manuscript",
        experiment_ablation_table=ablation_table,
    )


def extract_teaser_figure_payload_from_latest_extraction(
    latest_extraction_path: str | Path,
) -> TeaserFigurePayload:
    path = Path(latest_extraction_path)
    return extract_teaser_figure_payload(_read_text(path))


def build_teaser_figure_prompt(
    payload: TeaserFigurePayload,
    *,
    correction_hints: list[str] | None = None,
    attempt_index: int = 1,
    execution_skipped: bool = False,
) -> str:
    status_text = "; ".join(payload.status_legend) if payload.status_legend else "Not found in manuscript"
    strengths_text = (
        "\n".join(f"- {item}" for item in payload.strengths)
        if payload.strengths
        else "- Not found in manuscript"
    )
    weaknesses_text = (
        "\n".join(f"- {item}" for item in payload.weaknesses)
        if payload.weaknesses
        else "- Not found in manuscript"
    )
    selected_claims_text = _format_selected_claims(payload.selected_claim_rows)
    anchor_summary = _template_visual_anchor_summary()
    region_constraints = _template_region_constraints()
    retry_text = ""
    if correction_hints:
        retry_lines = [
            "[Retry Corrections]",
            f"- This is retry attempt {attempt_index}. The previous image did not match the template closely enough.",
        ]
        retry_lines.extend(f"- {hint}" for hint in correction_hints)
        retry_text = "\n".join(retry_lines) + "\n\n"

    return (
        "Create a single polished teaser figure for an ML paper review summary.\n"
        "The output should read like a presentation-quality overview graphic, not a raw markdown rendering.\n"
        "Use the extracted report content below as authoritative content to place into the figure.\n"
        "Preserve factual wording, numeric values, and status labels from the source.\n"
        "Treat the following layout/style instructions as fixed constraints derived from the reference demos/compgcn/teaser_reference.png.\n"
        "Treat the attached reference image as a hard layout-and-style target, not as loose inspiration.\n"
        "If any conflict appears between content length and layout fidelity, preserve layout fidelity first and shrink or wrap text.\n"
        "Keep colors unchanged and keep the relative positions of all modules unchanged; only adjust module width/height "
        "slightly based on content length.\n"
        "All text should use Times New Roman.\n"
        "There is no strict text-length limit inside each module; automatically adjust font sizes, line breaks, spacing, "
        "and box sizes for the most visually balanced result.\n"
        "Do not invent any extra claims, metrics, or statuses.\n"
        "The Summary module is mandatory and must be visible on the right side with both Strengths and Weaknesses content.\n"
        "The Technical Positioning figure must reuse the provided technical reference image, not a substituted architecture image.\n"
        "Use final_review content as canonical source text and preserve wording exactly; do not paraphrase, merge, or drop any required module content.\n"
        "\n"
        "[Fixed Layout Instructions]\n"
        "- Keep the overall teaser layout structure and relative module positions consistent with the reference design.\n"
        "- The top-right area contains status badges; preserve their relative placement and badge style.\n"
        "- All status badges use rounded-rectangle backgrounds.\n"
        "- The lower row includes Improvement and Reduction badges; preserve their relative placement.\n"
        "- The right-side summary panel keeps Strengths above Weaknesses, with the specified bottom background colors.\n"
        "- Claim rows should be laid out adaptively based on content, with no fixed per-line text limit.\n"
        "\n"
        "[Template Geometry]\n"
        f"{region_constraints}\n"
        "\n"
        "[Template Visual Anchors]\n"
        f"{anchor_summary}\n"
        "\n"
        "[Fixed Badge Styles]\n"
        "- Task: text color RGB(30,40,80); rounded-rectangle background RGB(235,238,248); "
        "right-aligned directly above the status badge row; width auto-fits content (do not stretch to full row width).\n"
        "- Supported: text color RGB(88,144,78); left icon is a check mark with RGB(0,150,100); rounded-rectangle "
        "background RGB(172,215,142).\n"
        "- Partially supported / Inconclusive (top-right area only — do NOT use this combined label in claims rows): "
        "text color RGB(182,140,2); left icon is a triangular warning symbol (⚠) — do NOT use a circle, question mark, "
        "or any other icon; the triangle border is RGB(184,134,11) with a white internal exclamation mark; "
        "rounded-rectangle background RGB(254,230,149).\n"
        "- Partially supported (claims row badge): text color RGB(182,140,2); left icon is a triangular warning "
        "symbol (⚠) — same icon as above; rounded-rectangle background RGB(254,230,149). Label reads '⚠ Partially supported'.\n"
        "- Inconclusive (claims row badge): text color RGB(182,140,2); left icon is a triangular warning "
        "symbol (⚠) — same icon as above; rounded-rectangle background RGB(254,230,149). Label reads '⚠ Inconclusive'.\n"
        "- In conflict: text color RGB(200,29,49); left icon is an X with RGB(139,0,0); rounded-rectangle background "
        "RGB(239,148,158).\n"
        "- Improvement: text color RGB(86,133,44); rounded-rectangle background RGB(117,189,66).\n"
        "- Reduction: text color RGB(133,19,44); rounded-rectangle background RGB(229,76,94).\n"
        "\n"
        "[Fixed Content Rules]\n"
        "- The task label badge must always appear right-aligned directly above the status badge row, "
        "using fixed colors (background RGB(235,238,248), text RGB(30,40,80)); "
        "its width auto-fits the text content — do not fix the left edge or stretch the badge to fill the row width; "
        "its text is extracted from the 'Task' field in [Report Content].\n"
        "- The top-right status badge area must always show exactly three fixed badges in this order: "
        "'✓ Supported', '⚠ Partially supported / Inconclusive', '✗ In conflict'. "
        "These are fixed template elements; do not derive, replace, or omit any of them based on "
        "claim row statuses or execution results. "
        "The second badge must literally read '⚠ Partially supported / Inconclusive' with a triangular ⚠ icon. "
        "All three badges are right-aligned as a compact horizontal group; Supported shifts right accordingly.\n"
        "- The Improvement and Reduction badges must always appear below the status badges with fixed "
        "labels ('Improvement', 'Reduction'), fixed colors, and fixed positions — do not modify their "
        "text or derive them from execution results.\n"
        "- The claims section should show exactly 3 claim rows, and they must be dynamically extracted from the report's claims table using the Claim, Evidence, and Status information.\n"
        "- In claims rows, each claim's status badge must display the exact status label from the claim's Status field "
        "(e.g., '⚠ Inconclusive', '⚠ Partially supported', '✓ Supported', '✗ In conflict') — "
        "do NOT substitute the combined top-right badge label '⚠ Partially supported / Inconclusive' for individual claim row badges.\n"
        "- In the claims module, each claim sentence must be visually bold in the figure.\n"
        "- Each claim row has no fixed text-length requirement; wrap and resize based on content for the cleanest layout.\n"
        "- The technical positioning module must directly use the extracted figure/image reference and table content.\n"
        "- The technical positioning visual must reuse the provided technical reference image faithfully (same subject/structure).\n"
        "- The experiment module must directly use the extracted main-result and ablation tables below.\n"
        "- For the Strengths section, use bottom background color RGB(200,229,179).\n"
        "- For the Weaknesses section, use bottom background color RGB(245,183,191).\n"
        "- The Summary column is required: if Summary/Strengths/Weaknesses is missing or empty, the output is invalid.\n"
        "- All extracted report content below must be represented in the final figure modules; missing or truncated modules are invalid outputs.\n"
        "- Do not alter any extracted factual text/value: keep wording, numbers, status labels, and signs exactly as provided.\n"
        "- In experiment tables, difference values annotated [GREEN]...[/GREEN] must be rendered in green text "
        "(RGB(86,133,44)); values annotated [RED]...[/RED] must be rendered in red text (RGB(200,29,49)). "
        "Strip the [GREEN]/[RED] annotation markers from the displayed text — show only the numeric value in color.\n"
        + (
            "- No experiment execution was performed; the experiment tables do not contain an "
            "Evaluation Status column — do not add, infer, or synthesize one.\n"
            if execution_skipped else ""
        )
        + "\n"
        "[Rendering Guidance]\n"
        "- Make the figure as aesthetically balanced as possible.\n"
        "- Use adaptive typography, spacing, and box scaling automatically, but do not alter the fixed colors or the "
        "relative placement of modules.\n"
        "- Use clear visual hierarchy, concise labels, table-like alignment where needed, and publication-style spacing.\n"
        "- Match the original canvas composition literally: same dark header, same light body shell, same right-side stacked summary column, same lower experiments band.\n"
        "- This is a dynamic pipeline: for each run, first extract the teaser-display fields from the provided latest_extraction markdown, then compose the final teaser figure prompt from those extracted fields and the fixed style constraints above.\n"
        "- Only inject content that is meant to be displayed in the teaser figure; do not add extra extracted fields that are not part of the visible teaser modules.\n"
        "\n"
        f"{retry_text}"
        "[Report Content]\n"
        f"Title: {payload.title}\n"
        f"Task: {payload.task}\n"
        f"Status legend: {status_text}\n"
        "\n"
        "[Technical Positioning]\n"
        f"Caption: {payload.technical_positioning_caption}\n"
        f"Image reference: {payload.technical_positioning_image}\n"
        "Table:\n"
        f"{_table_to_markdown(payload.technical_positioning_table)}\n"
        "\n"
        "[Claims]\n"
        "Selected 3 claim rows for direct layout use:\n"
        f"{selected_claims_text}\n"
        "\n"
        "Full claims table:\n"
        f"{_table_to_markdown(payload.claims_table)}\n"
        "\n"
        "[Summary]\n"
        f"{payload.summary}\n"
        "Strengths:\n"
        f"{strengths_text}\n"
        "Weaknesses:\n"
        f"{weaknesses_text}\n"
        "\n"
        "[Experiments]\n"
        f"Main result location: {payload.experiment_main_location}\n"
        "Main result table:\n"
        f"{_experiment_table_to_markdown(payload.experiment_main_table, is_ablation=False)}\n"
        "\n"
        f"Ablation result location: {payload.experiment_ablation_location}\n"
        "Ablation result table:\n"
        f"{_experiment_table_to_markdown(payload.experiment_ablation_table, is_ablation=True)}\n"
    )


def build_teaser_figure_prompt_from_latest_extraction(latest_extraction_path: str | Path) -> str:
    payload = extract_teaser_figure_payload_from_latest_extraction(latest_extraction_path)
    return build_teaser_figure_prompt(payload)


def _default_teaser_output_dir(latest_extraction_path: Path) -> Path:
    parent = latest_extraction_path.parent.resolve()
    if parent.name == "output":
        return parent / "teaser_figure"
    return parent / "teaser_figure"


def _coerce_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_inline_image_bytes(node: Any) -> bytes | None:
    if isinstance(node, dict):
        for key in ("imageBytes", "bytesBase64Encoded", "image_bytes", "bytes_base64_encoded", "b64_json"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    return base64.b64decode(value)
                except Exception:
                    continue
        image_url = node.get("image_url") or node.get("imageUrl") or {}
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:image") and "," in url:
                try:
                    return base64.b64decode(url.split(",", 1)[1])
                except Exception:
                    pass
        for value in node.values():
            decoded = _extract_inline_image_bytes(value)
            if decoded is not None:
                return decoded
        return None
    if isinstance(node, list):
        for item in node:
            decoded = _extract_inline_image_bytes(item)
            if decoded is not None:
                return decoded
    return None


def _resolve_image_request(
    *,
    model_override: str | None,
    api_key_override: str | None,
    timeout_override: int | None,
) -> tuple[str, str, int, str]:
    _ensure_env_loaded()
    base_url = str(os.getenv("GEMINI_BASE_URL") or "").strip().rstrip("/")
    api_key = str(api_key_override or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key and base_url:
        api_key = str(
            os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
        ).strip()
    model = str(
        model_override
        or os.getenv("GEMINI_IMAGE_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "imagen-4.0-generate-001"
    ).strip()
    timeout_seconds = int(timeout_override or int(os.getenv("GEMINI_TIMEOUT_SECONDS") or "120"))
    return api_key, model, timeout_seconds, base_url


def _call_gemini_image_api(
    *,
    prompt: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
    base_url: str = "",
    template_image_png_bytes: bytes | None = None,
    technical_image_png_bytes: bytes | None = None,
) -> dict[str, Any]:
    if base_url:
        user_content: Any
        if template_image_png_bytes or technical_image_png_bytes:
            content_items: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if template_image_png_bytes:
                content_items.append(
                    {
                        "type": "text",
                        "text": "Reference image A: Template layout/style target (must match geometry and colors).",
                    }
                )
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(template_image_png_bytes).decode("ascii")
                        },
                    }
                )
            if technical_image_png_bytes:
                content_items.append(
                    {
                        "type": "text",
                        "text": "Reference image B: Technical-positioning figure to be reused in the technical panel.",
                    }
                )
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(technical_image_png_bytes).decode("ascii")
                        },
                    }
                )
            user_content = content_items
        else:
            user_content = prompt
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": user_content}],
                "modalities": ["image", "text"],
                "stream": False,
                "image_config": {
                    "aspect_ratio": "16:9",
                    "image_size": "2K",
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini/OpenRouter image API returned a non-object JSON payload.")
        return payload

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict"
    response = requests.post(
        endpoint,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini image API returned a non-object JSON payload.")
    return payload


def _resolve_technical_reference_image_bytes(
    *,
    latest_path: Path,
    payload: TeaserFigurePayload,
) -> bytes | None:
    token = str(payload.technical_positioning_image or "").strip()
    if not token or token.lower() == "not found in manuscript":
        return None
    raw_path = Path(token).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((latest_path.parent / raw_path).resolve())
        candidates.append((_repo_root() / raw_path).resolve())
        candidates.append((latest_path.parent / "overview_figure.jpg").resolve())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            with Image.open(candidate) as img:
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            continue
    return None


def generate_teaser_figure(
    latest_extraction_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    timeout_seconds: int = 120,
    generate_image: bool = True,
    execution_skipped: bool = False,
) -> TeaserFigureGenerationResult:
    _ensure_env_loaded()
    latest_path = _coerce_path(latest_extraction_path).resolve()
    teaser_payload = extract_teaser_figure_payload_from_latest_extraction(latest_path)
    prompt = build_teaser_figure_prompt(teaser_payload, execution_skipped=execution_skipped)
    final_output_dir = (
        _coerce_path(output_dir).resolve()
        if output_dir is not None
        else _default_teaser_output_dir(latest_path)
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = final_output_dir / "teaser_figure_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    if not generate_image:
        clipboard_copied = _copy_text_to_clipboard(prompt)
        _, model, _, _ = _resolve_image_request(
            model_override=gemini_model,
            api_key_override=gemini_api_key,
            timeout_override=timeout_seconds,
        )
        return TeaserFigureGenerationResult(
            status="prompt_only",
            prompt=prompt,
            prompt_path=str(prompt_path),
            image_path="",
            response_path="",
            model=model,
            message=_prompt_only_message("Image generation disabled.", clipboard_copied=clipboard_copied),
            clipboard_copied=clipboard_copied,
            used_gemini_api=False,
            source_markdown_path=str(latest_path),
        )

    api_key, model, timeout_seconds, base_url = _resolve_image_request(
        model_override=gemini_model,
        api_key_override=gemini_api_key,
        timeout_override=timeout_seconds,
    )
    response_path = final_output_dir / "teaser_figure_gemini_response.json"
    image_path = final_output_dir / "teaser_figure.png"

    if not api_key:
        clipboard_copied = _copy_text_to_clipboard(prompt)
        return TeaserFigureGenerationResult(
            status="prompt_only",
            prompt=prompt,
            prompt_path=str(prompt_path),
            image_path="",
            response_path="",
            model=model,
            message=_prompt_only_message(
                "No teaser image API key configured.",
                clipboard_copied=clipboard_copied,
            ),
            clipboard_copied=clipboard_copied,
            used_gemini_api=False,
            source_markdown_path=str(latest_path),
        )

    strict_template_mode = _env_true("TEASER_TEMPLATE_STRICT", default=True)
    max_attempts = max(1, _int_env("TEASER_TEMPLATE_MAX_ATTEMPTS", 3 if strict_template_mode else 1))
    template_image_png_bytes = (
        _template_reference_png_bytes(scale=0.9)
        if _env_true("TEASER_GEMINI_INCLUDE_TEMPLATE_IMAGE", default=True)
        else None
    )
    technical_image_png_bytes = (
        _resolve_technical_reference_image_bytes(latest_path=latest_path, payload=teaser_payload)
        if _env_true("TEASER_GEMINI_INCLUDE_TECHNICAL_IMAGE", default=True)
        else None
    )
    validation_path = final_output_dir / "teaser_figure_validation.json"
    attempt_summaries: list[dict[str, Any]] = []
    correction_hints: list[str] = []
    best_attempt: dict[str, Any] | None = None

    for attempt_index in range(1, max_attempts + 1):
        attempt_prompt = build_teaser_figure_prompt(
            teaser_payload,
            correction_hints=correction_hints,
            attempt_index=attempt_index,
        )
        attempt_prompt_path = final_output_dir / f"teaser_figure_prompt_attempt_{attempt_index}.txt"
        attempt_prompt_path.write_text(attempt_prompt, encoding="utf-8")

        attempt_response = _call_gemini_image_api(
            prompt=attempt_prompt,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            base_url=base_url,
            template_image_png_bytes=template_image_png_bytes,
            technical_image_png_bytes=technical_image_png_bytes,
        )
        attempt_response_path = (
            final_output_dir / f"teaser_figure_gemini_response_attempt_{attempt_index}.json"
        )
        _write_json(attempt_response_path, attempt_response)

        image_bytes = _extract_inline_image_bytes(attempt_response)
        if image_bytes is None:
            raise RuntimeError(
                "Gemini returned successfully, but no image bytes were found in the response payload."
            )
        attempt_image_path = final_output_dir / f"teaser_figure_attempt_{attempt_index}.png"
        attempt_image_path.write_bytes(image_bytes)

        validation = (
            _validate_generated_teaser_image(attempt_image_path)
            if strict_template_mode
            else {
                "passed": True,
                "score": 1.0,
                "threshold": 0.0,
                "color_similarity": 1.0,
                "edge_similarity": 1.0,
                "region_scores": [],
                "prompt_feedback": [],
                "reason": "strict_template_mode_disabled",
            }
        )
        attempt_summary = {
            "attempt": attempt_index,
            "prompt_path": str(attempt_prompt_path),
            "image_path": str(attempt_image_path),
            "response_path": str(attempt_response_path),
            "validation": validation,
        }
        attempt_summaries.append(attempt_summary)

        if best_attempt is None or float(validation.get("score", 0.0)) > float(
            best_attempt["validation"].get("score", 0.0)
        ):
            best_attempt = {
                "attempt": attempt_index,
                "prompt": attempt_prompt,
                "prompt_path": attempt_prompt_path,
                "image_path": attempt_image_path,
                "response_path": attempt_response_path,
                "response_payload": attempt_response,
                "validation": validation,
            }

        if bool(validation.get("passed")):
            break
        correction_hints = list(validation.get("prompt_feedback") or [])

    if best_attempt is None:
        raise RuntimeError("Teaser figure generation did not produce any attempts.")

    prompt = str(best_attempt["prompt"])
    prompt_path.write_text(prompt, encoding="utf-8")
    shutil.copyfile(best_attempt["image_path"], image_path)
    shutil.copyfile(best_attempt["response_path"], response_path)

    _write_json(
        validation_path,
        {
            "strict_template_mode": strict_template_mode,
            "max_attempts": max_attempts,
            "best_attempt": int(best_attempt["attempt"]),
            "best_score": float(best_attempt["validation"].get("score", 0.0)),
            "passed": bool(best_attempt["validation"].get("passed")),
            "attempts": attempt_summaries,
        },
    )

    final_validation = best_attempt["validation"]
    message = (
        "Teaser figure image generated via Gemini API. "
        f"Best template similarity {float(final_validation.get('score', 0.0)):.3f} "
        f"after {int(best_attempt['attempt'])} attempt(s)."
    )
    if strict_template_mode and not bool(final_validation.get("passed")):
        message += " Validation did not fully pass; the best attempt was kept and feedback was written to teaser_figure_validation.json."

    return TeaserFigureGenerationResult(
        status="generated",
        prompt=prompt,
        prompt_path=str(prompt_path),
        image_path=str(image_path),
        response_path=str(response_path),
        model=model,
        message=message,
        clipboard_copied=False,
        used_gemini_api=True,
        source_markdown_path=str(latest_path),
    )


def payload_to_dict(payload: TeaserFigurePayload) -> dict[str, Any]:
    return {
        "title": payload.title,
        "task": payload.task,
        "status_legend": payload.status_legend,
        "technical_positioning_caption": payload.technical_positioning_caption,
        "technical_positioning_image": payload.technical_positioning_image,
        "technical_positioning_table": None
        if payload.technical_positioning_table is None
        else {
            "headers": payload.technical_positioning_table.headers,
            "rows": payload.technical_positioning_table.rows,
        },
        "claims_table": None
        if payload.claims_table is None
        else {
            "headers": payload.claims_table.headers,
            "rows": payload.claims_table.rows,
        },
        "selected_claim_rows": payload.selected_claim_rows,
        "summary": payload.summary,
        "strengths": payload.strengths,
        "weaknesses": payload.weaknesses,
        "experiment_main_location": payload.experiment_main_location,
        "experiment_main_table": None
        if payload.experiment_main_table is None
        else {
            "headers": payload.experiment_main_table.headers,
            "rows": payload.experiment_main_table.rows,
        },
        "experiment_ablation_location": payload.experiment_ablation_location,
        "experiment_ablation_table": None
        if payload.experiment_ablation_table is None
        else {
            "headers": payload.experiment_ablation_table.headers,
            "rows": payload.experiment_ablation_table.rows,
        },
    }
