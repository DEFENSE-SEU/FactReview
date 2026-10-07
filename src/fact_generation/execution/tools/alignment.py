from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from util.fs import ensure_dir, write_text

from ..v2_config import TOLERANCES, metric_tolerance
from .paper_tables import PaperMetricTarget, _metric_key, extract_paper_metric_targets


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
    mrr: float = TOLERANCES["alignment"]["mrr"]
    hits_at_10: float = TOLERANCES["alignment"]["rate"]
    mr: float = TOLERANCES["alignment"]["mr"]


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
    paper_metric_source: str = ""
    paper_claim: str = ""


@dataclass(frozen=True)
class AlignmentComparison:
    """One paper-vs-run metric comparison, flattened for report/audit consumers."""

    paper_key: str
    observed_key: str
    metric: str
    dataset: str
    split: str
    paper_value: float
    observed_value: float
    paper_value_raw: float
    observed_value_raw: float
    delta: float
    delta_abs: float
    delta_pct: float
    tolerance: float
    within_tolerance: bool
    passed: bool
    direction: str
    run_metrics_file: str
    paper_table_id: str
    paper_table_md_path: str
    paper_row_label: str
    paper_scoring_function: str
    paper_metric_source: str = ""
    paper_claim: str = ""


@dataclass(frozen=True)
class AlignmentResult:
    extracted_targets: int
    matched: int
    passed: int
    failed: int
    unmatched_run_metrics: list[str]
    critiques: list[dict[str, Any]]
    matches: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]
    notes: list[str]


def _tokens(*parts: str) -> set[str]:
    text = " ".join(str(p or "") for p in parts).lower()
    return {t for t in re_split_nonword(text) if len(t) >= 3}


def re_split_nonword(text: str) -> list[str]:
    import re

    return re.split(r"[^a-z0-9@]+", text.lower())


def _target_from_dict(raw: dict[str, Any]) -> PaperMetricTarget | None:
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        return None
    clean_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        fv = _as_float(v)
        if fv is not None:
            clean_metrics[_metric_key(str(k))] = float(fv)
    if not clean_metrics:
        return None
    return PaperMetricTarget(
        paper_table_id=str(raw.get("paper_table_id") or raw.get("id") or "baseline_target"),
        paper_table_md_path=str(raw.get("paper_table_md_path") or raw.get("source") or ""),
        dataset=str(raw.get("dataset") or ""),
        scoring_function=str(raw.get("scoring_function") or ""),
        method=str(raw.get("method") or raw.get("paper_claim") or ""),
        metrics=clean_metrics,
        metric_source=str(raw.get("metric_source") or "baseline"),
        paper_claim=str(raw.get("paper_claim") or ""),
    )


def _pick_target(
    targets: list[PaperMetricTarget],
    *,
    dataset: str,
    score_func: str,
    opn: str,
    run_descriptor: str,
    metric_keys: set[str],
) -> PaperMetricTarget | None:
    ds = _norm(dataset)
    sf = _score_func_display(score_func)
    op = _opn_display(opn)

    scored: list[tuple[float, PaperMetricTarget]] = []
    run_tokens = _tokens(dataset, score_func, opn, run_descriptor)
    for target in targets:
        target_metric_keys = set(target.metrics)
        overlap = metric_keys & target_metric_keys
        if not overlap:
            continue
        score = float(len(overlap) * 5)

        target_ds = _norm(target.dataset)
        if ds and target_ds:
            if ds == target_ds:
                score += 6
            else:
                score -= 8
        elif ds or target_ds:
            score += 0.5

        if sf and target.scoring_function and _norm(sf) == _norm(target.scoring_function):
            score += 5
        if op and op.lower() in _norm(target.method):
            score += 3
        if score_func and _norm(score_func) in _norm(target.method):
            score += 2

        target_tokens = _tokens(target.dataset, target.scoring_function, target.method, target.paper_claim)
        token_overlap = run_tokens & target_tokens
        score += min(4, len(token_overlap))

        # Avoid arbitrary matching when there is no shared context at all and
        # the paper has many candidate targets.
        has_context = bool((ds and target_ds and ds == target_ds) or token_overlap or target.scoring_function)
        if not has_context and len(targets) > 1:
            continue

        if score > 0:
            scored.append((score, target))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _scale_pair(metric: str, observed: float, expected: float) -> tuple[float, float]:
    key = _metric_key(metric)
    bounded = key in {
        "mrr",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "map",
        "ndcg",
    } or key.startswith("hits@")
    if bounded:
        if observed <= 1.0 and 1.0 < expected <= 100.0:
            return observed, expected / 100.0
        if expected <= 1.0 and 1.0 < observed <= 100.0:
            return observed / 100.0, expected
    return observed, expected


def _calc_delta(obs: dict[str, float], exp: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, exp_v in exp.items():
        obs_v = obs.get(k)
        if obs_v is None:
            continue
        scaled_obs, scaled_exp = _scale_pair(k, float(obs_v), float(exp_v))
        out[k] = scaled_obs - scaled_exp
    return out


def _within_tol(delta: dict[str, float], tol: AlignmentTolerance) -> dict[str, bool]:
    return {key: abs(value) <= _metric_tolerance(key, tol, delta=value) for key, value in delta.items()}


def _metric_tolerance(metric: str, tol: AlignmentTolerance, *, delta: float | None = None) -> float:
    key = _metric_key(metric)
    if key == "mrr":
        return float(tol.mrr)
    if key in {"hits@10", "hits@10."}:
        return float(tol.hits_at_10)
    if key == "mr":
        return float(tol.mr)
    return metric_tolerance(key, profile="alignment", delta=delta)


def _metric_direction(metric: str) -> str:
    key = _metric_key(metric)
    lower_is_better = {
        "mr",
        "error_rate",
        "mae",
        "rmse",
        "mse",
        "perplexity",
        "loss",
        "fid",
        "wer",
        "cer",
        "latency",
        "time",
    }
    return "lower_is_better" if key in lower_is_better else "higher_is_better"


def _comparison_label(*, dataset: str, row_label: str, scoring_function: str, metric: str) -> str:
    parts = [dataset, row_label, scoring_function, metric]
    return " / ".join(str(p).strip() for p in parts if str(p).strip())


def _build_comparisons_for_match(match: AlignmentMatch, *, tol: AlignmentTolerance) -> list[AlignmentComparison]:
    rows: list[AlignmentComparison] = []
    for metric, _delta in sorted(match.delta.items()):
        if metric not in match.expected or metric not in match.observed:
            continue
        paper_raw = float(match.expected[metric])
        observed_raw = float(match.observed[metric])
        observed_cmp, paper_cmp = _scale_pair(metric, observed_raw, paper_raw)
        delta_cmp = float(observed_cmp) - float(paper_cmp)
        tolerance = _metric_tolerance(metric, tol, delta=delta_cmp)
        within = bool(match.within_tolerance.get(metric, abs(delta_cmp) <= tolerance))
        paper_key = _comparison_label(
            dataset=match.dataset,
            row_label=match.paper_row_label,
            scoring_function=match.paper_scoring_function or match.score_func,
            metric=metric,
        )
        observed_key = _comparison_label(
            dataset=match.dataset,
            row_label=match.score_func or match.paper_row_label,
            scoring_function=match.opn,
            metric=metric,
        )
        rows.append(
            AlignmentComparison(
                paper_key=paper_key,
                observed_key=observed_key or f"{match.run_metrics_file}:{metric}",
                metric=metric,
                dataset=match.dataset,
                split=match.split,
                paper_value=float(paper_cmp),
                observed_value=float(observed_cmp),
                paper_value_raw=paper_raw,
                observed_value_raw=observed_raw,
                delta=delta_cmp,
                delta_abs=abs(delta_cmp),
                delta_pct=(abs(delta_cmp) / max(abs(float(paper_cmp)), 1e-9) * 100.0),
                tolerance=tolerance,
                within_tolerance=within,
                passed=within,
                direction=_metric_direction(metric),
                run_metrics_file=match.run_metrics_file,
                paper_table_id=match.paper_table_id,
                paper_table_md_path=match.paper_table_md_path,
                paper_row_label=match.paper_row_label,
                paper_scoring_function=match.paper_scoring_function,
                paper_metric_source=match.paper_metric_source,
                paper_claim=match.paper_claim,
            )
        )
    return rows


def _is_metric_key(key: str) -> bool:
    canonical = _metric_key(key)
    return canonical in {
        "mrr",
        "mr",
        "accuracy",
        "error_rate",
        "f1",
        "precision",
        "recall",
        "auc",
        "bleu",
        "rouge-l",
        "rouge-1",
        "rouge-2",
        "map",
        "ndcg",
        "mae",
        "rmse",
        "mse",
        "perplexity",
        "loss",
        "fid",
        "inception_score",
        "r2",
    } or canonical.startswith("hits@")


def _collect_metrics(d: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def add_from(obj: dict[str, Any]) -> None:
        for k, v in obj.items():
            if isinstance(v, bool):
                continue
            fv = _as_float(v)
            if fv is None or not _is_metric_key(str(k)):
                continue
            metrics[_metric_key(str(k))] = float(fv)

    add_from(d)
    nested = d.get("metrics")
    if isinstance(nested, dict):
        add_from(nested)
    return metrics


def _extract_run_metrics_row(d: dict[str, Any]) -> tuple[str, str, str, str, str, dict[str, float]]:
    dataset = str(d.get("dataset") or "").strip()
    split = str(d.get("split") or "").strip()
    score_func = str(d.get("score_func") or d.get("scoring_function") or "").strip()
    opn = str(d.get("opn") or "").strip()
    descriptor = " ".join(
        str(d.get(k) or "")
        for k in ("method", "model", "variant", "name", "task", "claim", "paper_claim")
    ).strip()
    claims = d.get("claims")
    if isinstance(claims, list):
        descriptor = (descriptor + " " + " ".join(str(x) for x in claims)).strip()
    return dataset, split, score_func, opn, descriptor, _collect_metrics(d)


def _find_metrics_files(artifacts_dir: Path) -> list[Path]:
    files = []
    for p in sorted(Path(artifacts_dir).rglob("*.json")):
        rel = str(p.relative_to(artifacts_dir)).replace("\\", "/")
        if rel.startswith(("alignment/", "tables/")):
            continue
        if p.stat().st_size > 5_000_000:
            continue
        files.append(p)
    return files


def run_alignment(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    artifacts_dir: Path,
    paper_extracted_tables_dir: Path,
    paper_metric_targets: list[dict[str, Any]] | None = None,
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

    targets: list[PaperMetricTarget] = []
    for raw_target in paper_metric_targets or []:
        if isinstance(raw_target, dict):
            target = _target_from_dict(raw_target)
            if target is not None:
                targets.append(target)
    if not targets:
        targets = extract_paper_metric_targets(paper_extracted_tables_dir)

    metrics_files = _find_metrics_files(artifacts_dir)

    matches: list[AlignmentMatch] = []
    comparisons: list[AlignmentComparison] = []
    unmatched: list[str] = []
    notes: list[str] = []
    critiques: list[dict[str, Any]] = []

    if not targets:
        notes.append(
            "No parseable paper targets found in paper_extracted tables (deterministic alignment skipped)."
        )

    for mf in metrics_files:
        d = _read_json(mf)
        dataset, split, score_func, opn, descriptor, obs_metrics = _extract_run_metrics_row(d)
        rel_mf = str(mf.relative_to(artifacts_dir)).replace("\\", "/")
        if not obs_metrics:
            unmatched.append(rel_mf)
            continue

        if not targets:
            unmatched.append(rel_mf)
            continue

        tgt = _pick_target(
            targets,
            dataset=dataset,
            score_func=score_func,
            opn=opn,
            run_descriptor=descriptor,
            metric_keys=set(obs_metrics),
        )
        if tgt is None:
            unmatched.append(rel_mf)
            continue

        # Compare only overlapping keys (paper table might not include hits@1/hits@3).
        exp = dict(tgt.metrics)
        obs = {k: v for k, v in obs_metrics.items() if k in exp}
        # If paper has mrr/mr/hits@10 but run doesn't, keep it unmatched.
        if not obs:
            unmatched.append(rel_mf)
            continue

        delta = _calc_delta(obs, exp)
        within = _within_tol(delta, tol)
        passed = all(within.values()) if within else False

        match = AlignmentMatch(
            run_metrics_file=rel_mf,
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
            paper_metric_source=tgt.metric_source,
            paper_claim=tgt.paper_claim,
        )
        matches.append(match)
        comparisons.extend(_build_comparisons_for_match(match, tol=tol))

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
                    "run_metrics_file": rel_mf,
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
        comparisons=[asdict(c) for c in comparisons],
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

    if comparisons:
        md_lines.append("## Metric comparisons")
        md_lines.append("")
        md_lines.append("| Dataset | Metric | Paper | Reproduced | Delta | Tolerance | Result |")
        md_lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for c in comparisons:
            result_label = "PASS" if c.passed else "FAIL"
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        c.dataset or "n/a",
                        c.metric,
                        f"{c.paper_value:.6g}",
                        f"{c.observed_value:.6g}",
                        f"{c.delta:.6g}",
                        f"{c.tolerance:.6g}",
                        result_label,
                    ]
                )
                + " |"
            )
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
