from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from util.fs import ensure_dir, write_text


def _load_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.5f}"
    except Exception:
        return ""


def _md_table(rows: list[dict[str, Any]], *, caption: str, columns: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join([c[0] for c in columns]) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in range(len(columns) - 1)]) + "|")
    for r in rows:
        vals = []
        for _, key in columns:
            if key in {"mrr", "mr"} or key.startswith("hits@"):
                vals.append(_fmt(r.get(key)))
            else:
                vals.append(str(r.get(key) or ""))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _html_table(rows: list[dict[str, Any]], *, caption: str, columns: list[tuple[str, str]]) -> str:
    cap = f"<caption>{caption}</caption>" if caption else ""
    head = "<tr>" + "".join([f"<th>{c[0]}</th>" for c in columns]) + "</tr>"
    body_rows = []
    for r in rows:
        tds = []
        for _, key in columns:
            if key in {"mrr", "mr"} or key.startswith("hits@"):
                tds.append(f"<td>{_fmt(r.get(key))}</td>")
            else:
                tds.append(f"<td>{r.get(key) or ''!s}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<table>\n"
        f"{cap}\n"
        "<thead>\n"
        f"{head}\n"
        "</thead>\n"
        "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
        "</table>\n"
    )


def _group_key(r: dict[str, Any]) -> tuple[str, str]:
    # dataset + split are common across papers; if absent, collapse into one table.
    return (str(r.get("dataset") or ""), str(r.get("split") or ""))


def maybe_summarize_metrics_tables(*, cfg: dict, run_dir: Path, artifacts_dir: Path) -> None:
    """
    Generic post-run summarizer:
    - If <artifacts>/metrics/*.json exist, generate baseline-like tables under
      <artifacts>/tables/{md,html}/ and <artifacts>/tables/index.json.

    The JSON schema is intentionally loose. Recommended keys:
    - dataset, split, model/variant fields (e.g., score_func/opn), and numeric metrics (mrr, hits@k, etc.)
    """
    metrics_dir = artifacts_dir / "metrics"
    if not metrics_dir.exists():
        return

    rows: list[dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        d = _load_json(p)
        if not d:
            continue
        r = dict(d)
        r["_file"] = str(p.name)
        rows.append(r)
    if not rows:
        return

    paper_key = str(cfg.get("paper_key") or "paper").strip()
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(_group_key(r), []).append(r)

    tables_dir = artifacts_dir / "tables"
    md_dir = tables_dir / "md"
    html_dir = tables_dir / "html"
    ensure_dir(md_dir)
    ensure_dir(html_dir)

    # Prefer common metric columns if present.
    columns: list[tuple[str, str]] = [
        ("variant", "variant"),
        ("mrr", "mrr"),
        ("mr", "mr"),
        ("hits@1", "hits@1"),
        ("hits@3", "hits@3"),
        ("hits@10", "hits@10"),
    ]

    index: list[dict[str, Any]] = []
    table_idx = 1
    for (dataset, split), grp in sorted(by_group.items()):
        # Best-effort variant label: score_func/opn or model/seed, etc.
        normalized: list[dict[str, Any]] = []
        for r in grp:
            rr = dict(r)
            if not rr.get("variant"):
                sf = str(rr.get("score_func") or "").strip()
                opn = str(rr.get("opn") or "").strip()
                if sf or opn:
                    rr["variant"] = f"{sf}/{opn}".strip("/")
                else:
                    rr["variant"] = str(rr.get("model") or rr.get("name") or rr.get("_file") or "")
            normalized.append(rr)
        grp_sorted = sorted(normalized, key=lambda x: str(x.get("variant") or ""))

        cap_parts = [paper_key, "reproduction"]
        if split:
            cap_parts.append(str(split))
        if dataset:
            cap_parts.append(f"on {dataset}")
        caption = " ".join([p for p in cap_parts if p]).strip()

        md_text = _md_table(grp_sorted, caption=caption, columns=columns)
        html_text = _html_table(grp_sorted, caption=caption, columns=columns)

        md_path = md_dir / f"table_{table_idx:03d}.md"
        html_path = html_dir / f"table_{table_idx:03d}.html"
        write_text(md_path, md_text)
        write_text(html_path, html_text)

        index.append(
            {
                "id": f"table_{table_idx:03d}",
                "caption": caption,
                "dataset": dataset,
                "split": split,
                "html_path": str(html_path.relative_to(tables_dir)).replace("\\", "/"),
                "md_path": str(md_path.relative_to(tables_dir)).replace("\\", "/"),
                "source_metrics": [str(r.get("_file") or "") for r in grp_sorted],
            }
        )
        table_idx += 1

    (tables_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8", errors="ignore"
    )
    write_text(
        artifacts_dir / "tables" / "README.md",
        "This folder is auto-generated from artifacts/metrics/*.json to make run outputs easy to compare against paper_extracted tables.\n",
    )
