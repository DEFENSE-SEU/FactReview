from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from util.fs import ensure_dir, write_text

from .paper_tables import PaperMetricTarget, extract_paper_metric_targets


def _read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")
    except Exception:
        return {}


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "")


def _score_func_display(score_func: str) -> str:
    m = {
        "transe": "TransE",
        "distmult": "DistMult",
        "conve": "ConvE",
        "complex": "ComplEx",
        "rotate": "RotatE",
    }
    return m.get(_norm(score_func), score_func)


def _opn_display(opn: str) -> str:
    m = {"sub": "Sub", "mult": "Mult", "corr": "Corr"}
    return m.get(_norm(opn), opn)


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class AlignmentTolerance:
    mrr: float = 0.01
    hits_at_10: float = 0.02
    mr: float = 30.0


@dataclass(frozen=True)
class AlignmentMatch:
    run_metrics_file: str
    dataset: str
    split: str
    score_func: str
    opn: str
    expected: dict[str, float]
    observed: dict[str, float]
    delta: dict[str, float]
    within_tolerance: dict[str, bool]
    passed: bool
    paper_table_id: str
    paper_table_md_path: str
    paper_row_label: str
    paper_scoring_function: str


@dataclass(frozen=True)
class AlignmentResult:
    extracted_targets: int
    matched: int
    passed: int
    failed: int
    unmatched_run_metrics: list[str]
    critiques: list[dict[str, Any]]
    matches: list[dict[str, Any]]
    notes: list[str]


def _pick_target(
    targets: list[PaperMetricTarget], *, dataset: str, score_func: str, opn: str
) -> PaperMetricTarget | None:
    ds = _norm(dataset)
    sf = _score_func_display(score_func)
    op = _opn_display(opn)

    cand = [t for t in targets if _norm(t.dataset) == ds and _norm(t.scoring_function) == _norm(sf)]
    if not cand:
        return None

    # Prefer method rows that mention compgcn and the opn (Sub/Mult/Corr).
    # This is intentionally conservative for COMPGCN.
    preferred = [t for t in cand if ("compgcn" in _norm(t.method)) and (op.lower() in _norm(t.method))]
    if preferred:
        return preferred[0]

    # Fallback: any row mentioning opn.
    op_only = [t for t in cand if op.lower() in _norm(t.method)]
    if op_only:
        return op_only[0]

    return cand[0] if cand else None


def _calc_delta(obs: dict[str, float], exp: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, exp_v in exp.items():
        obs_v = obs.get(k)
        if obs_v is None:
            continue
        out[k] = float(obs_v) - float(exp_v)
    return out


def _within_tol(delta: dict[str, float], tol: AlignmentTolerance) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for k, d in delta.items():
        if k == "mrr":
            out[k] = abs(d) <= float(tol.mrr)
        elif k in {"hits@10", "hits@10."}:
            out[k] = abs(d) <= float(tol.hits_at_10)
        elif k == "mr":
            out[k] = abs(d) <= float(tol.mr)
        else:
            # Unknown metric keys: treat as not checked.
            out[k] = False
    return out


def _extract_run_metrics_row(d: dict[str, Any]) -> tuple[str, str, str, str, dict[str, float]]:
    dataset = str(d.get("dataset") or "").strip()
    split = str(d.get("split") or "").strip()
    score_func = str(d.get("score_func") or d.get("scoring_function") or "").strip()
    opn = str(d.get("opn") or "").strip()
    metrics: dict[str, float] = {}
    for k in ["mrr", "mr", "hits@10", "hits@3", "hits@1"]:
        v = _as_float(d.get(k))
        if v is not None:
            metrics[k] = float(v)
    return dataset, split, score_func, opn, metrics


def run_alignment(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    artifacts_dir: Path,
    paper_extracted_tables_dir: Path,
) -> AlignmentResult:
    """
    Deterministic alignment between:
    - run artifacts metrics/*.json (observed)
    - paper_extracted tables/*.md (expected, best-effort)
    """
    tol = AlignmentTolerance()
    try:
        t = cfg.get("alignment_tolerance") or {}
        if isinstance(t, dict):
            tol = AlignmentTolerance(
                mrr=float(t.get("mrr") or tol.mrr),
                hits_at_10=float(t.get("hits@10") or t.get("hits_at_10") or tol.hits_at_10),
                mr=float(t.get("mr") or tol.mr),
            )
    except Exception:
        tol = AlignmentTolerance()

    targets = extract_paper_metric_targets(paper_extracted_tables_dir)

    metrics_dir = Path(artifacts_dir) / "metrics"
    metrics_files = sorted(metrics_dir.glob("*.json")) if metrics_dir.exists() else []

    matches: list[AlignmentMatch] = []
    unmatched: list[str] = []
    notes: list[str] = []
    critiques: list[dict[str, Any]] = []

    if not targets:
        notes.append(
            "No parseable paper targets found in paper_extracted tables (deterministic alignment skipped)."
        )

    for mf in metrics_files:
        d = _read_json(mf)
        dataset, split, score_func, opn, obs_metrics = _extract_run_metrics_row(d)
        if not (dataset and score_func and opn and obs_metrics):
            unmatched.append(mf.name)
            continue

        if not targets:
            unmatched.append(mf.name)
            continue

        tgt = _pick_target(targets, dataset=dataset, score_func=score_func, opn=opn)
        if tgt is None:
            unmatched.append(mf.name)
            continue

        # Compare only overlapping keys (paper table might not include hits@1/hits@3).
        exp = dict(tgt.metrics)
        obs = {k: v for k, v in obs_metrics.items() if k in exp}
        # If paper has mrr/mr/hits@10 but run doesn't, keep it unmatched.
        if not obs:
            unmatched.append(mf.name)
            continue

        delta = _calc_delta(obs, exp)
        within = _within_tol(delta, tol)
        passed = all(within.values()) if within else False

        matches.append(
            AlignmentMatch(
                run_metrics_file=mf.name,
                dataset=dataset,
                split=split,
                score_func=score_func,
                opn=opn,
                expected=exp,
                observed=obs,
                delta=delta,
                within_tolerance=within,
                passed=passed,
                paper_table_id=tgt.paper_table_id,
                paper_table_md_path=tgt.paper_table_md_path,
                paper_row_label=tgt.method,
                paper_scoring_function=tgt.scoring_function,
            )
        )

        if not passed:
            # Severity is heuristic: large deltas on key metrics are "high".
            sev = "low"
            dmrr = abs(delta.get("mrr", 0.0)) if "mrr" in delta else 0.0
            dh10 = abs(delta.get("hits@10", 0.0)) if "hits@10" in delta else 0.0
            dmr = abs(delta.get("mr", 0.0)) if "mr" in delta else 0.0
            if (dmrr > 0.05) or (dh10 > 0.06) or (dmr > 200):
                sev = "high"
            elif (
                (dmrr > float(tol.mrr) * 2) or (dh10 > float(tol.hits_at_10) * 2) or (dmr > float(tol.mr) * 2)
            ):
                sev = "medium"
            critiques.append(
                {
                    "type": "paper_alignment_mismatch",
                    "severity_level": sev,
                    "run_metrics_file": mf.name,
                    "paper_table_id": tgt.paper_table_id,
                    "paper_row": tgt.method,
                    "paper_scoring_function": tgt.scoring_function,
                    "dataset": dataset,
                    "score_func": score_func,
                    "opn": opn,
                    "expected": exp,
                    "observed": obs,
                    "delta": delta,
                    "tolerance": {"mrr": tol.mrr, "mr": tol.mr, "hits@10": tol.hits_at_10},
                }
            )

    passed_n = sum(1 for m in matches if m.passed)
    failed_n = sum(1 for m in matches if (not m.passed))

    result = AlignmentResult(
        extracted_targets=len(targets),
        matched=len(matches),
        passed=passed_n,
        failed=failed_n,
        unmatched_run_metrics=unmatched,
        critiques=critiques,
        matches=[asdict(m) for m in matches],
        notes=notes,
    )

    # Persist under artifacts/alignment/
    out_dir = ensure_dir(Path(artifacts_dir) / "alignment")
    write_text(out_dir / "alignment.json", json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")

    # Human-readable snippet (used by finalize report)
    md_lines: list[str] = []
    md_lines.append("# Paper alignment (deterministic)")
    md_lines.append("")
    md_lines.append(f"- extracted_targets: {result.extracted_targets}")
    md_lines.append(f"- matched: {result.matched}")
    md_lines.append(f"- passed: {result.passed}")
    md_lines.append(f"- failed: {result.failed}")
    if result.unmatched_run_metrics:
        md_lines.append(f"- unmatched_run_metrics: {len(result.unmatched_run_metrics)}")
    md_lines.append("")
    if result.notes:
        md_lines.append("## Notes")
        md_lines.append("")
        for n in result.notes:
            md_lines.append(f"- {n}")
        md_lines.append("")
    if matches:
        md_lines.append("## Matches")
        md_lines.append("")
        for m in matches:
            md_lines.append(f"### {m.run_metrics_file} ({m.dataset} {m.score_func}/{m.opn})")
            md_lines.append(f"- paper_table: {m.paper_table_id}")
            md_lines.append(f"- paper_row: {m.paper_row_label}")
            md_lines.append("")
            md_lines.append("```json")
            md_lines.append(json.dumps(asdict(m), ensure_ascii=False, indent=2))
            md_lines.append("```")
            md_lines.append("")

    if critiques:
        md_lines.append("## Critiques (mismatches)")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(critiques, ensure_ascii=False, indent=2))
        md_lines.append("```")
        md_lines.append("")
    write_text(out_dir / "alignment.md", "\n".join(md_lines) + "\n")

    return result
