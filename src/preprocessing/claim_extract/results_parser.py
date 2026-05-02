"""Parse paper tables into :class:`ReportedResult` entries.

The §3.4 review writer and the §3.3 execution-alignment step both need
structured access to every numeric value the paper claims. This module
walks :class:`Paper.tables` and produces one :class:`ReportedResult`
per (row, column) cell that parses as a number, attaching as much
context (dataset, task, method, metric) as we can recover from the
caption and column headers.

The parser is intentionally generic: it relies on column-header tokens
("MRR", "Accuracy", …) and dataset-mention heuristics, not on any
paper-specific table layout.
"""

from __future__ import annotations

import re

from schemas.paper import Paper, ReportedResult, Table

from .heuristics import _DATASET_RE, _extract_datasets, _extract_metrics

_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
_METRIC_HEADER_RE = re.compile(
    r"MRR|MR|Hits?@\d+|Accuracy|Acc\.?|F1|BLEU|ROUGE|AUC|AUROC|AUPRC|"
    r"Precision|Recall|EM|Perplexity|PPL|Rouge-?\d+|mAP",
    re.IGNORECASE,
)


def _parse_number(cell: str) -> float | None:
    """Return the first number in *cell*, normalising ``%`` to a 0-100 float."""
    if not cell:
        return None
    s = cell.strip().replace(",", "")
    # Strip common decorations: boldface markdown, superscripts, ± noise.
    s = re.sub(r"\*+|\$", "", s)
    s = re.sub(r"\\pm\s*\d+(?:\.\d+)?", "", s)
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _infer_task_from_caption(caption: str) -> str | None:
    """Very light heuristic — the LLM pass can override."""
    c = (caption or "").lower()
    if "link prediction" in c:
        return "link_prediction"
    if "node classification" in c:
        return "node_classification"
    if "graph classification" in c:
        return "graph_classification"
    if "image classification" in c or "imagenet" in c:
        return "image_classification"
    if "question answering" in c or "qa" in c:
        return "question_answering"
    return None


def _infer_dataset_from_caption(caption: str) -> str | None:
    ds = _extract_datasets(caption or "")
    return ds[0] if ds else None


def _classify_headers(row: list[str]) -> tuple[list[int], list[str]]:
    """Identify metric columns in a header row.

    Returns a (``indices``, ``metric_names``) pair: for each column
    whose header text matches a known metric token, record its index
    and the canonicalised metric name.
    """
    idxs: list[int] = []
    names: list[str] = []
    for j, cell in enumerate(row):
        if cell is None:
            continue
        metrics = _extract_metrics(str(cell))
        if metrics:
            idxs.append(j)
            names.append(metrics[0])
    return idxs, names


def _first_text_cell(row: list[str]) -> str:
    for c in row:
        s = (c or "").strip()
        if s and not _NUMBER_RE.fullmatch(s):
            return s
    return ""


def extract_reported_results(paper: Paper) -> list[ReportedResult]:
    """Walk every table on *paper* and produce a flat list of results."""
    out: list[ReportedResult] = []
    for table in paper.tables:
        out.extend(_extract_from_table(table))
    return out


def _extract_from_table(table: Table) -> list[ReportedResult]:
    rows = table.rows or []
    if len(rows) < 2:
        return []

    # Locate a metric header row within the first three rows. Header rows
    # sometimes live below a group-name row, so we try each in turn.
    header_idx = -1
    metric_cols: list[int] = []
    metric_names: list[str] = []
    for r in range(min(3, len(rows))):
        idxs, names = _classify_headers(rows[r])
        if idxs:
            header_idx = r
            metric_cols = idxs
            metric_names = names
            break
    if header_idx < 0:
        return []

    caption_task = _infer_task_from_caption(table.caption)
    caption_ds = _infer_dataset_from_caption(table.caption)

    results: list[ReportedResult] = []
    seen = 0
    for r in range(header_idx + 1, len(rows)):
        row = rows[r]
        if not row:
            continue
        method = _first_text_cell(row)
        if not method:
            continue
        for col_idx, metric in zip(metric_cols, metric_names, strict=False):
            if col_idx >= len(row):
                continue
            value = _parse_number(row[col_idx])
            if value is None:
                continue
            seen += 1
            results.append(
                ReportedResult(
                    id=f"{table.id}.row{r}.col{col_idx}",
                    metric=metric.upper() if metric.lower() != "hits@10" else "Hits@10",
                    value=value,
                    dataset=_dataset_for_cell(table, row, col_idx, caption_ds),
                    task=caption_task,
                    method=method,
                    table_id=table.id,
                    row_index=r,
                    col_index=col_idx,
                    context=(table.caption or "").strip()[:300],
                )
            )
    return results


def _dataset_for_cell(table: Table, row: list[str], col_idx: int, caption_ds: str | None) -> str | None:
    """Try to recover a per-column dataset annotation above *col_idx*.

    Many paper tables have a two-level header: dataset names on one row,
    metric names on the next. If the cell at ``(0, col_idx)`` mentions a
    known benchmark, prefer it over the caption-level dataset.
    """
    if table.rows:
        top = table.rows[0] if len(table.rows) > 0 else []
        if col_idx < len(top):
            hit = _DATASET_RE.search(str(top[col_idx]))
            if hit:
                return hit.group(0)
    return caption_ds
