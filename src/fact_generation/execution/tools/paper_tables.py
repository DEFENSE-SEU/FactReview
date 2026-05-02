from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperMetricTarget:
    """
    A best-effort, paper-extracted numeric target.

    This is intentionally "loose": different papers expose different schemas.
    We focus on a small common subset and keep enough provenance to audit.
    """

    paper_table_id: str
    paper_table_md_path: str
    dataset: str
    scoring_function: str  # e.g. "TransE"
    method: str  # e.g. "X + CoMPGCN (Sub)"
    metrics: dict[str, float]  # e.g. {"mrr":0.335, "mr":194, "hits@10":0.514}


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _strip_md(s: str) -> str:
    # Remove simple markdown emphasis/backticks.
    t = (s or "").strip()
    t = t.replace("`", "")
    t = re.sub(r"[*_]+", "", t)
    return _norm_space(t)


def _split_md_row(line: str) -> list[str]:
    # Markdown tables: | a | b | c |
    s = (line or "").strip()
    if not s.startswith("|"):
        return []
    # Remove leading/trailing pipes and split.
    s = s.strip("|")
    return [_strip_md(x) for x in s.split("|")]


def _is_sep_row(cells: list[str]) -> bool:
    if not cells:
        return False
    # e.g. ["---", "---:", "---"]
    return all(re.fullmatch(r":?-{3,}:?", (c or "").strip()) for c in cells)


def _to_float(s: str) -> float | None:
    t = (s or "").strip()
    if not t:
        return None
    # Handle ".294" style.
    if re.fullmatch(r"\.\d+", t):
        t = "0" + t
    # Remove commas and stray symbols.
    t = t.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _metric_key(cell: str) -> str:
    c = (cell or "").strip().lower().replace(" ", "")
    if c in {"mrr"}:
        return "mrr"
    if c in {"mr"}:
        return "mr"
    # H@10 / H@ 10 / hits@10 ...
    c = c.replace("h@", "hits@")
    if c.startswith("hits@"):
        return c
    return c


def _extract_compgcn_table_004(
    md_text: str, *, paper_table_id: str, paper_table_md_path: str
) -> list[PaperMetricTarget]:
    """
    Extract targets from COMPGCN Table 4-like markdown (multi-header layout).

    Expected shape (from mineru extraction):
    - header row: "Scoring Function (=X) → | TransE | DistMult | ConvE | ..."
    - header row: "Methods ↓ | MRR | MR | H@ 10 | MRR | MR | H@10 | ..."
    - rows: "X + CoMPGCN (Sub) | 0.335 | 194 | 0.514 | ..."
    """
    lines = [ln.rstrip("\n") for ln in (md_text or "").splitlines()]
    # Find a markdown table block that includes "Scoring Function" and "Methods".
    start = None
    for i, ln in enumerate(lines):
        if "Scoring Function" in ln and ln.strip().startswith("|"):
            start = i
            break
    if start is None:
        return []

    # Collect contiguous table lines.
    block: list[str] = []
    for ln in lines[start:]:
        if not ln.strip().startswith("|"):
            break
        block.append(ln)
    if len(block) < 4:
        return []

    header1 = _split_md_row(block[0])
    # block[1] is usually separator row
    header2 = []
    # Find header2: first non-separator after header1
    for ln in block[1:3]:
        c = _split_md_row(ln)
        if c and not _is_sep_row(c):
            header2 = c
            break
    if not header1 or not header2:
        return []

    # Scoring functions are in header1 after first cell; ignore empties.
    scoring_funcs = [c for c in header1[1:] if c]
    if not scoring_funcs:
        return []

    # Metrics are in header2 after first cell (repeated).
    metric_cells = [c for c in header2[1:] if c]
    if not metric_cells:
        return []
    metrics = [_metric_key(c) for c in metric_cells]

    # We expect a repeating group like [mrr, mr, hits@10] for each scoring func.
    group_size = 3
    max_cols = min(len(scoring_funcs) * group_size, len(metrics))

    targets: list[PaperMetricTarget] = []
    for ln in block:
        row = _split_md_row(ln)
        if not row or _is_sep_row(row):
            continue
        if len(row) < 2:
            continue
        method = row[0]
        if not method or method.lower().startswith("methods"):
            continue

        vals = row[1:]
        # Trim/pad to max cols
        vals = vals[:max_cols]
        if len(vals) < max_cols:
            # pad missing with empty
            vals = vals + [""] * (max_cols - len(vals))

        for j in range(0, max_cols, group_size):
            sf_idx = j // group_size
            sf = scoring_funcs[sf_idx] if sf_idx < len(scoring_funcs) else ""
            if not sf:
                continue
            m: dict[str, float] = {}
            for k in range(group_size):
                key = metrics[j + k] if (j + k) < len(metrics) else ""
                v = _to_float(vals[j + k])
                if key and v is not None:
                    m[key] = v
            if not m:
                continue
            targets.append(
                PaperMetricTarget(
                    paper_table_id=paper_table_id,
                    paper_table_md_path=paper_table_md_path,
                    dataset="FB15k-237",
                    scoring_function=sf,
                    method=method,
                    metrics=m,
                )
            )
    return targets


def extract_paper_metric_targets(paper_extracted_tables_dir: Path) -> list[PaperMetricTarget]:
    """
    Best-effort extraction of numeric targets from paper_extracted tables.

    This is designed to be conservative:
    - if we can't parse, we return an empty list rather than guessing.
    - we keep provenance (table id + md path) for audit.
    """
    tables_dir = Path(paper_extracted_tables_dir)
    idx = tables_dir / "index.json"
    if not idx.exists():
        return []

    try:
        items = json.loads(_read_text(idx) or "[]")
    except Exception:
        items = []
    if not isinstance(items, list):
        return []

    out: list[PaperMetricTarget] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        table_id = str(it.get("id") or "").strip()
        md_path = str(it.get("path_md") or it.get("md_path") or "").strip()
        if not md_path:
            continue
        p = Path(md_path)
        if not p.is_absolute():
            # Some index.json uses relative paths; treat them as relative to tables_dir.
            p = tables_dir / md_path
        if not p.exists():
            continue
        md_text = _read_text(p)

        # Currently we only ship a robust parser for COMPGCN-style Table 4.
        # This still improves the framework because it demonstrates how to do
        # deterministic, auditable alignment; other table types can be added later.
        if "Scoring Function" in md_text and "CoMPGCN" in md_text:
            out.extend(
                _extract_compgcn_table_004(
                    md_text, paper_table_id=table_id or p.stem, paper_table_md_path=str(p)
                )
            )

    return out
