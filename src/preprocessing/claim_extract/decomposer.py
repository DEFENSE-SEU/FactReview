"""Split broad claims into per-(task, dataset, metric) sub-claims.

Paper §3.1b calls out the "broad claim" problem explicitly: a single
sentence such as *"our method outperforms prior work on link prediction,
node classification, and graph classification"* hides three testable
assertions. The execution and review stages work at the sub-claim
granularity, so we decompose broad claims up-front.

When :class:`ReportedResult` entries are available (from the tables),
we also try to bind each sub-claim to the most likely numeric target
— that becomes its ``expected_value`` and makes §3.4 labelling
dramatically more precise.
"""

from __future__ import annotations

import re
from itertools import product

from schemas.claim import Claim, SubClaim
from schemas.paper import ReportedResult

_TASK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("link_prediction", re.compile(r"\blink\s+prediction\b", re.IGNORECASE)),
    ("node_classification", re.compile(r"\bnode\s+classification\b", re.IGNORECASE)),
    ("graph_classification", re.compile(r"\bgraph\s+classification\b", re.IGNORECASE)),
    ("image_classification", re.compile(r"\bimage\s+classification\b", re.IGNORECASE)),
    ("question_answering", re.compile(r"\bquestion\s+answering\b|\bQA\b", re.IGNORECASE)),
    ("machine_translation", re.compile(r"\bmachine\s+translation\b", re.IGNORECASE)),
    ("language_modeling", re.compile(r"\blanguage\s+modeling\b", re.IGNORECASE)),
    ("text_classification", re.compile(r"\btext\s+classification\b", re.IGNORECASE)),
]


def _detect_tasks(text: str) -> list[str]:
    found: list[str] = []
    for name, pat in _TASK_PATTERNS:
        if pat.search(text or "") and name not in found:
            found.append(name)
    return found


def _cross(
    tasks: list[str],
    datasets: list[str],
    metrics: list[str],
) -> list[tuple[str | None, str | None, str | None]]:
    """Cartesian product with ``None`` preserved when a dimension is empty."""
    t_axis: list[str | None] = tasks or [None]
    d_axis: list[str | None] = datasets or [None]
    m_axis: list[str | None] = metrics or [None]
    return list(product(t_axis, d_axis, m_axis))


def _match_reported(
    task: str | None,
    dataset: str | None,
    metric: str | None,
    method_name: str | None,
    reported: list[ReportedResult],
) -> ReportedResult | None:
    """Find the best-matching :class:`ReportedResult` for a sub-claim.

    Matching is on (dataset, metric) — the two dimensions we can usually
    extract cleanly from both the claim text and the table headers. If
    a candidate also matches the paper's *own* method name it wins the
    tie-break.
    """
    if not reported or not metric:
        return None

    def _metric_match(r: ReportedResult) -> bool:
        return (r.metric or "").lower() == metric.lower()

    def _dataset_match(r: ReportedResult) -> bool:
        if dataset is None:
            return True
        return (r.dataset or "").lower() == dataset.lower()

    candidates = [r for r in reported if _metric_match(r) and _dataset_match(r)]
    if not candidates:
        return None

    if task:
        task_matches = [r for r in candidates if (r.task or "") == task]
        if not task_matches:
            # Task was asked for but no table row carries it — don't silently
            # bind to an unrelated task's number.
            return None
        candidates = task_matches

    if method_name:
        owned = [r for r in candidates if (r.method or "").lower() == method_name.lower()]
        if owned:
            return owned[0]

    # As a last resort, prefer the largest value on MRR/Hits@/Accuracy/F1
    # (the paper's own result is usually the best in the row set).
    maximising = {"mrr", "accuracy", "acc.", "acc", "f1", "auc", "bleu", "em", "map"}
    if (metric or "").lower() in maximising or (metric or "").lower().startswith("hits@"):
        return max(candidates, key=lambda r: r.value)
    return candidates[0]


def _own_method_name(claim: Claim) -> str | None:
    """Crude guess at the paper's own method name from the claim text.

    Looks for an ``ALLCAPS``/``MixedCase`` token after "we propose / our"
    — good enough for binding claim → ReportedResult when the paper uses
    a consistent codename (e.g. "COMPGCN", "ConvE", "BERT").
    """
    m = re.search(
        r"\b(?:we\s+propose|we\s+introduce|we\s+present|our)\s+([A-Z][A-Za-z0-9\-]{2,})",
        claim.text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return None


def decompose_claim(claim: Claim, reported: list[ReportedResult]) -> Claim:
    """Return a copy of *claim* with ``subclaims`` populated when broad."""
    if claim.scope != "broad":
        return claim

    tasks = _detect_tasks(claim.text)
    datasets = list(claim.datasets)
    metrics = list(claim.metrics)

    combos = _cross(tasks, datasets, metrics)
    # A single (None, None, None) means we cannot split meaningfully.
    if len(combos) <= 1:
        return claim

    method_name = _own_method_name(claim)

    subclaims: list[SubClaim] = []
    for i, (task, dataset, metric) in enumerate(combos, start=1):
        hit = _match_reported(task, dataset, metric, method_name, reported)
        subclaims.append(
            SubClaim(
                id=f"{claim.id}.sub_{i:02d}",
                text=_render_subclaim_text(claim, task, dataset, metric),
                task=task,
                dataset=dataset,
                metric=metric,
                expected_value=hit.value if hit else None,
            )
        )

    return claim.model_copy(update={"subclaims": subclaims})


def _render_subclaim_text(
    claim: Claim,
    task: str | None,
    dataset: str | None,
    metric: str | None,
) -> str:
    parts = [f"[derived from {claim.id}]"]
    if task:
        parts.append(f"task={task}")
    if dataset:
        parts.append(f"dataset={dataset}")
    if metric:
        parts.append(f"metric={metric}")
    return " ".join(parts)


def decompose_claims(claims: list[Claim], reported: list[ReportedResult]) -> list[Claim]:
    """Batch variant over a list of claims; preserves order."""
    return [decompose_claim(c, reported) for c in claims]
