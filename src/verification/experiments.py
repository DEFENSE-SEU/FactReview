"""Paper-only experiment checks and the sole claim-linked execution-plan producer."""

from __future__ import annotations

import json
import math
import re
import uuid
from html.parser import HTMLParser
from itertools import product
from typing import Literal

from pydantic import Field

from schemas.claim import (
    AuthorQuestion,
    Claim,
    Contract,
    Evidence,
    ExecutionPlan,
    ExecutionTask,
    NonEmpty,
)
from schemas.materials import SharedMaterials
from screening.checks import ask
from verification.contracts import BranchResult, RejectedPlan
from verification.theory import (
    FULL_SUPPORT_DESCRIPTION,
    _covered,
    _fully_supported,
    _paper_pointer,
    _support_note,
)


class PaperNumber(Contract):
    block_id: NonEmpty
    quote: NonEmpty = Field(
        description="Verbatim contiguous substring of the cited block's text; preserve math and whitespace."
    )
    token: NonEmpty
    # A precise sentence/row/cell context may disambiguate a larger quoted table.
    value_context: str = Field(
        default="", description="Exact contiguous substring of quote; preserve math and whitespace."
    )


def _has_label(label: str | None, quote: str) -> bool:
    return bool(label and re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", quote, re.I))


def _number(materials: SharedMaterials, item: PaperNumber) -> float:
    _paper_pointer(materials, item.block_id, item.quote)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?", item.token):
        raise ValueError("A target must quote a finite numeric token from the paper")
    if not re.search(r"(?<![\w.])" + re.escape(item.token) + r"(?![\w%]|\.\d)", item.quote):
        raise ValueError("Reported number token is absent from the quoted paper passage")
    # Preserve the paper's metric scale: 89% remains 89, alongside its quote.
    value = float(item.token.rstrip("%"))
    if not math.isfinite(value):
        raise ValueError("Reported target must be finite")
    return value


def _target_ambiguity(claim: Claim, target: PlanTarget, value: float) -> str:
    """Conservatively block target assignment when a quote contains several values."""
    reported = target.reported
    context = reported.value_context or reported.quote
    if context not in reported.quote:
        raise ValueError("value_context must be an exact substring of the quoted paper passage")
    condition = next(item for item in claim.conditions if item.id == target.condition_id)
    if not _has_label(condition.metric, context) or (
        condition.dataset and not _has_label(condition.dataset, context)
    ):
        return "Exact value context does not identify the target dataset and metric"
    for setting, setting_value in condition.settings.items():
        if not _has_label(str(setting_value), context):
            return f"Exact value context does not identify the target setting {setting}={setting_value}"
    other_datasets = {item.dataset for item in claim.conditions if item.dataset != condition.dataset}
    if any(_has_label(dataset, context) for dataset in other_datasets):
        return "Exact value context contains multiple target datasets"
    # Dataset and metric labels can contain digits (FB15k-237, Hits@10).
    # Exclude those labels before counting independent reported numbers.
    numeric_context = context
    for label in (condition.dataset, condition.metric):
        if label:
            numeric_context = re.sub(
                r"(?<!\w)" + re.escape(label) + r"(?!\w)", " ", numeric_context, flags=re.I
            )
    for setting, setting_value in condition.settings.items():
        if isinstance(setting_value, (int, float)) and not isinstance(setting_value, bool):
            pattern = (
                r"(?<!\w)"
                + re.escape(setting)
                + r"\s*[:=]?\s*"
                + re.escape(str(setting_value))
                + r"(?![\w.]|\.\d)"
            )
            numeric_context, count = re.subn(pattern, " ", numeric_context, flags=re.I)
            if not count:
                return f"Numeric setting {setting}={setting_value} is not explicitly bound in the context"
    tokens = re.findall(
        r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?(?!\w|\.\d)", numeric_context
    )
    values = {float(token.rstrip("%")) for token in tokens}
    if values != {value}:
        return "Exact value context does not uniquely bind the reported number to this condition"
    return ""


def _comparison_units(number: PaperNumber, metric: str) -> set[str]:
    """Retain explicit scales; do not guess a conversion for unlabelled values."""
    aliases = {
        "percent": "%",
        "percentage": "%",
        "second": "s",
        "seconds": "s",
        "sec": "s",
        "millisecond": "ms",
        "milliseconds": "ms",
        "microsecond": "us",
        "microseconds": "us",
        "µs": "us",
        "μs": "us",
        "minute": "min",
        "minutes": "min",
        "hour": "h",
        "hours": "h",
    }
    units = ["%"] if number.token.endswith("%") else []
    # A narrowed value context may omit the table header. Keep every explicit
    # scale for this metric from the grounded full quote.
    units.extend(
        unit.strip() for unit in re.findall(re.escape(metric) + r"\s*\(([^)]+)\)", number.quote, re.I)
    )
    suffixes = (
        []
        if number.token.endswith("%")
        else re.findall(
            r"(?<![\w.])" + re.escape(number.token) + r"\s*(%|[A-Za-zµμ]+(?:/[A-Za-zµμ]+)?)(?!\w)",
            number.quote,
            re.I,
        )
    )
    connective_words = {
        "on",
        "for",
        "with",
        "and",
        "from",
        "which",
        "where",
        "using",
        "in",
        "at",
    }
    units.extend(unit for unit in suffixes if unit.casefold() not in connective_words)
    return {aliases.get(unit.casefold(), unit) for unit in units}


class ExperimentItem(Contract):
    aspect: Literal["correspondence", "fairness", "isolation", "stability", "consistency"]
    kind: Literal[
        "paper_support",
        "missing_control",
        "missing_ablation",
        "missing_statistic",
        "small_gap_without_statistics",
        "large_gap_no_variance",
        "text_table_contradiction",
    ]
    block_id: NonEmpty
    quote: NonEmpty = Field(
        description="Verbatim contiguous substring of the cited block's text; preserve math and whitespace."
    )
    covered: list[NonEmpty] = Field(
        min_length=1,
        description="Distinct exact condition IDs from allowed_condition_ids for this claim. No claim IDs, block IDs, or labels.",
        json_schema_extra={"uniqueItems": True},
    )
    fully_supported_conditions: list[NonEmpty] = Field(
        default_factory=list, description=FULL_SUPPORT_DESCRIPTION, json_schema_extra={"uniqueItems": True}
    )
    detail: NonEmpty
    # Contradiction pairs must be the same metric/setting; units are explicit.
    comparison: list[PaperNumber] = Field(default_factory=list, max_length=2)


class PlanTarget(Contract):
    condition_id: NonEmpty = Field(
        description="One exact ID from allowed_condition_ids; target IDs must be distinct within each plan."
    )
    reported: PaperNumber


class PlanCandidate(Contract):
    targets: list[PlanTarget] = Field(min_length=1)
    entry_script: str | None = None
    config: str | None = None
    run_mode: Literal["evaluation", "analysis", "training"]
    feasibility: Literal["ready", "blocked"]
    blocker: str = ""
    priority: Literal["high", "medium", "low"]
    estimated_cost: str = "unknown"
    data_paths: list[str] = Field(default_factory=list)
    weight_paths: list[str] = Field(default_factory=list)


class ExperimentsOutput(Contract):
    checked_aspects: list[Literal["correspondence", "fairness", "isolation", "stability", "consistency"]]
    items: list[ExperimentItem]
    plans: list[PlanCandidate] = Field(default_factory=list, max_length=1)
    issues: list[str] = Field(default_factory=list)


class ScopePassage(Contract):
    block_id: NonEmpty
    quote: NonEmpty


class ConditionScope(Contract):
    condition_id: NonEmpty
    claim_quote: NonEmpty
    assertion: Literal[
        "descriptive",
        "controlled_comparison",
        "causal_attribution",
        "statistical_generalization",
        "unresolved",
    ]
    matched_controls_required: bool
    credited_component: str = ""
    uncertainty_sensitive: bool
    relation: Literal["none", "gt", "ge", "lt", "le", "eq", "difference"]
    subject: str
    comparator: str
    subject_setting: str = ""
    comparator_setting: str = ""
    difference_direction: Literal["signed", "increase", "decrease"] = "signed"
    difference_direction_quote: str = ""
    required_cases: list[str] = Field(default_factory=list)
    rationale: NonEmpty


class TableCell(Contract):
    """Zero-based expanded HTML table coordinates, checked against source bytes."""

    table: int = Field(default=0, ge=0)
    row: int = Field(ge=0)
    column: int = Field(ge=0)


class ScopeComparison(Contract):
    case: str = ""
    settings: dict[str, str] = Field(default_factory=dict)
    left: PaperNumber
    right: PaperNumber
    left_label: NonEmpty
    right_label: NonEmpty
    metric_label: NonEmpty
    relation: Literal["gt", "ge", "lt", "le", "eq", "difference"]
    difference: PaperNumber | None = None
    difference_mode: Literal["absolute", "relative_percent", "percentage_points", "unresolved"] = "unresolved"
    left_cell: TableCell | None = None
    right_cell: TableCell | None = None
    context: list[ScopePassage] = Field(min_length=1)


class ItemScope(Contract):
    item_index: int = Field(ge=0)
    condition_id: NonEmpty
    applicability: Literal["applicable", "not_applicable", "unverified"]
    grounds: list[ScopePassage] = Field(min_length=1)
    rationale: NonEmpty
    full_support: bool = False
    qualifiers_complete: bool = False
    comparison_objects: Literal["matched", "unmatched", "unresolved", "not_comparative"] = "unresolved"
    unresolved_qualifiers: list[str] = Field(default_factory=list)
    comparisons: list[ScopeComparison] = Field(default_factory=list)


class ExperimentScopeReview(Contract):
    conditions: list[ConditionScope]
    items: list[ItemScope]


class _TableGrid(HTMLParser):
    """Read native HTML tables without adding a parser dependency or executing HTML."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tables, self.rows, self.row, self.cell = [], None, None, None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.rows is not None:
                raise ValueError("Nested tables have ambiguous numerical coordinates")
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            attrs = dict(attrs)
            spans = [int(attrs.get(name, "1")) for name in ("rowspan", "colspan")]
            if any(value < 1 or value > 100 for value in spans):
                raise ValueError("Unsupported table span")
            self.cell = [[], *spans]

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[0].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(("".join(self.cell[0]).strip(), *self.cell[1:]))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            grid = {}
            for row_index, row in enumerate(self.rows):
                column = 0
                for text, rowspan, colspan in row:
                    while (row_index, column) in grid:
                        column += 1
                    for y in range(row_index, row_index + rowspan):
                        for x in range(column, column + colspan):
                            if (y, x) in grid:
                                raise ValueError("Overlapping table spans")
                            grid[y, x] = text
                    column += colspan
            self.tables.append(grid)
            self.rows = None


def _table_value(number, cell, label, metric, settings):
    tables = _TableGrid(number.quote).tables
    if cell.table >= len(tables):
        raise ValueError("Selected numerical table does not exist in the exact quote")
    grid = tables[cell.table]
    text = grid.get((cell.row, cell.column), "")
    if text != number.token:
        raise ValueError("Selected table cell does not uniquely contain the reported token")
    row_labels = " ".join(v for (r, c), v in grid.items() if r == cell.row and c < cell.column)
    column_labels = " ".join(v for (r, c), v in grid.items() if c == cell.column and r < cell.row)
    label_in_row, label_in_column = _has_label(label, row_labels), _has_label(label, column_labels)
    if not label_in_row and not label_in_column:
        raise ValueError("Table row/column does not identify the selected subject/comparator")
    # A task-specific column can inherit a common metric from the exact table
    # caption/context. This also permits transposed native tables.
    axes = row_labels + " " + column_labels
    spans = list(re.finditer(r"<table\b.*?</table>", number.quote, re.I | re.S))
    if cell.table >= len(spans):
        raise ValueError("Table caption boundaries are unresolved")
    prior_end = spans[cell.table - 1].end() if cell.table else 0
    caption = number.quote[prior_end : spans[cell.table].start()]
    if not _has_label(metric, axes):
        if any(_has_label(metric, value) for value in grid.values()):
            raise ValueError("Selected table coordinates use a different explicit metric header")
        opposite_axis = column_labels if label_in_row else row_labels
        if not _has_label(metric, caption) or not any(
            _has_label(value, opposite_axis) for value in settings.values()
        ):
            raise ValueError("Table coordinates do not bind the metric or comparison case")
    return axes, caption, tuple(grid.values())


def _side_settings(condition, scope, selected_settings, binding, *, axes=None, table_labels=()):
    def bound(label):
        if axes is not None and any(_has_label(label, value) for value in table_labels):
            return _has_label(label, axes)
        return _has_label(label, binding)

    if condition.dataset and not bound(condition.dataset):
        raise ValueError("A numerical side does not identify its own target dataset")
    # Subject/comparator identities are checked for each operand separately.
    role_keys = {
        "model",
        "method",
        "subject",
        "comparison",
        "comparator",
        "baseline",
        scope.subject_setting,
        scope.comparator_setting,
    }
    for key, value in condition.settings.items():
        if key in role_keys:
            continue
        selected = selected_settings[key] if isinstance(value, list) else str(value)
        if not bound(selected):
            raise ValueError(f"A numerical side does not bind its own condition setting {key}")


def _claim_passages(claim, condition_id):
    explicit_primary = [
        ref
        for ref in claim.source_refs
        if ref.source_block_id == claim.source_block_id and ref.source_quote == claim.source_quote
    ]
    if not explicit_primary or any(condition_id in ref.covered for ref in explicit_primary):
        yield claim.source_block_id, claim.source_quote or claim.text
    for ref in claim.source_refs:
        if condition_id in ref.covered:
            yield ref.source_block_id, ref.source_quote


def _numeric_scope_check(claim, condition, scope, decision, materials) -> list[str]:
    reasons = []
    if scope.relation == "none":
        if scope.assertion in {"controlled_comparison", "causal_attribution"} and condition.metric:
            reasons.append("A comparative metric condition needs an explicit numerical relation")
        if decision.comparison_objects in {"unmatched", "unresolved"}:
            reasons.append("The claimed subject/comparator identity has not been established")
        if condition.metric and any(
            key in condition.settings for key in ("comparison", "comparator", "baseline")
        ):
            reasons.append("An explicit comparator setting requires a numerical comparison relation")
        return reasons
    if decision.comparison_objects != "matched":
        reasons.append("The claimed subject/comparator identity has not been established")
    expected_cases = set(scope.required_cases or [""])
    # Explicit multi-setting conditions must be covered as written. A reviewer
    # cannot silently omit a task with a tie or opposite result.
    dimensions = {
        key: [str(value) for value in values]
        for key, values in condition.settings.items()
        if isinstance(values, list)
    }
    expected_settings = set(product(*dimensions.values())) if dimensions else set()
    observed_settings = []
    cases = [item.case for item in decision.comparisons]
    if len(cases) != len(set(cases)) or set(cases) != expected_cases:
        reasons.append("Numerical checks must cover each required comparison case exactly once")
    for comparison in decision.comparisons:
        try:
            settings = dict(comparison.settings)
            if len(dimensions) == 1 and not settings:
                settings[next(iter(dimensions))] = comparison.case
            if set(settings) != set(dimensions):
                raise ValueError("Comparison settings must select one value for each list-valued dimension")
            observed_settings.append(tuple(settings[key] for key in dimensions))
            for label, role, setting_keys in (
                (comparison.left_label, "subject", ("model", "method", "subject")),
                (comparison.right_label, "comparator", ("comparison", "comparator", "baseline")),
            ):
                setting_key = getattr(scope, f"{role}_setting")
                declared = getattr(scope, role)
                if setting_key:
                    if setting_key not in condition.settings:
                        raise ValueError(f"Unknown {role} setting binding")
                    declared = settings.get(setting_key, str(condition.settings[setting_key]))
                if not declared or label != declared:
                    raise ValueError(f"Numerical operand is not bound to the claim's {role}")
                for key in setting_keys:
                    value = condition.settings.get(key)
                    if value is not None and not isinstance(value, (dict, list)) and label != str(value):
                        raise ValueError(f"Numerical operand changes condition {role} setting {key}")
            if comparison.relation != scope.relation:
                raise ValueError("Numerical comparison weakens or changes the claim's relation")
            context = "\n".join(p.quote for p in comparison.context)
            for passage in comparison.context:
                _paper_pointer(materials, passage.block_id, passage.quote)
            if condition.dataset and not _has_label(condition.dataset, context):
                raise ValueError("Numerical context does not identify the claimed dataset")
            if not condition.metric or not _has_label(condition.metric, context):
                raise ValueError("Numerical context does not identify the exact claimed metric")
            if comparison.metric_label != condition.metric:
                raise ValueError("Numerical comparison changes the claimed metric")
            for key, value in condition.settings.items():
                candidates = [settings[key]] if isinstance(value, list) else [str(value)]
                if not all(_has_label(candidate, context) for candidate in candidates):
                    raise ValueError(f"Numerical context does not bind condition setting {key}")
            values = []
            for number, label, cell in (
                (comparison.left, comparison.left_label, comparison.left_cell),
                (comparison.right, comparison.right_label, comparison.right_cell),
            ):
                value = _number(materials, number)
                if cell is not None:
                    axes, caption, table_labels = _table_value(
                        number, cell, label, comparison.metric_label, settings
                    )
                    _side_settings(
                        condition, scope, settings, axes + " " + caption, axes=axes, table_labels=table_labels
                    )
                    values.append(value)
                    continue
                local = number.value_context or number.quote
                if (
                    local not in number.quote
                    or not _has_label(label, local)
                    or not _has_label(comparison.metric_label, local)
                ):
                    raise ValueError("A numerical side lacks its exact subject/comparator and metric context")
                _side_settings(condition, scope, settings, local)
                clean = local
                for part in (label, comparison.metric_label, condition.dataset, comparison.case):
                    if part:
                        clean = re.sub(re.escape(part), " ", clean, flags=re.I)
                numeric = re.findall(
                    r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?(?!\w|\.\d)", clean
                )
                if {float(token.rstrip("%")) for token in numeric} != {value}:
                    raise ValueError("A numerical side contains multiple unbound values")
                values.append(value)
            units = [
                _comparison_units(number, comparison.metric_label)
                for number in (comparison.left, comparison.right)
            ]
            if any(len(unit) > 1 for unit in units) or units[0] != units[1]:
                raise ValueError("Comparison units/scales do not match")
            left, right = values
            relation = comparison.relation
            if relation == "difference":
                if comparison.difference is None:
                    raise ValueError("An asserted difference needs a grounded expected difference")
                expected = _number(materials, comparison.difference)
                if not any(
                    comparison.difference.quote in quote
                    and (block_id is None or block_id == comparison.difference.block_id)
                    for block_id, quote in _claim_passages(claim, condition.id)
                ):
                    raise ValueError("Expected difference is not grounded in this claim's own source quote")
                difference_units = _comparison_units(comparison.difference, comparison.metric_label)
                if scope.difference_direction != "signed" and (
                    not scope.difference_direction_quote
                    or not any(
                        scope.difference_direction_quote in quote
                        for _, quote in _claim_passages(claim, condition.id)
                    )
                ):
                    raise ValueError(
                        "Difference direction needs an exact qualifier from this condition's source"
                    )
                delta = right - left if scope.difference_direction == "decrease" else left - right
                if comparison.difference_mode == "relative_percent":
                    if right == 0 or difference_units != {"%"}:
                        raise ValueError(
                            "Relative difference needs a nonzero baseline and explicit percent scale"
                        )
                    actual = delta / abs(right) * 100
                elif (comparison.difference_mode == "absolute" and difference_units == units[0]) or (
                    comparison.difference_mode == "percentage_points"
                    and units[0] == {"%"}
                    and difference_units in ({"%"}, {"pp"}, {"points"})
                ):
                    actual = delta
                else:
                    raise ValueError(
                        "Expected difference type or units/scales do not match the compared values"
                    )
                holds = math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
            else:
                holds = {
                    "gt": left > right,
                    "ge": left >= right,
                    "lt": left < right,
                    "le": left <= right,
                    "eq": left == right,
                }[relation]
            if not holds:
                raise ValueError(f"Numerical relation {relation} is false for {left} and {right}")
        except ValueError as exc:
            reasons.append(f"{comparison.case or condition.id}: {exc}")
    if dimensions and (
        len(observed_settings) != len(set(observed_settings)) or set(observed_settings) != expected_settings
    ):
        reasons.append("Numerical checks do not cover every joint combination of listed condition settings")
    return reasons


def _scope_review(claim, materials, output, *, call):
    """Independent semantic review; its grounded structural/numeric gates fail closed."""
    from common import run_stats

    payload = {
        "claim": claim.model_dump(mode="json"),
        "paper_blocks": [block.model_dump() for block in materials.blocks],
        "candidate_items": [item.model_dump() for item in output.items],
        "output_schema": ExperimentScopeReview.model_json_schema(),
    }
    audit = {"claim_id": claim.id, "input": payload}
    stats = run_stats.stats_path()
    audit_path = stats.parent / "experiment_scope" / f"{uuid.uuid4().hex}.json" if stats is not None else None
    try:
        response = ask(
            "Independently audit experiment candidate applicability and ENTIRE condition support. "
            "Do not defer to candidate detail or fully_supported_conditions. Return output_schema JSON, "
            "one conditions entry per claim condition and exactly one items entry per candidate item_index/covered condition. "
            "Ground claim_quote in claim.text, source_quote or a source_ref covering that condition, and all grounds/context in exact paper block quotes. "
            "Distinguish descriptive reports (reported scores, hardware or duration) from controlled comparisons, "
            "causal component attribution and statistical generalization. A leaderboard score/gap alone does not "
            "assert matched training data/budget/tuning or causal superiority. Missing ablations apply only to "
            "an actually credited component; do not demand hardware-count ablations for a resource report. "
            "Missing variance affects the claim only when uncertainty could overturn an asserted comparison or "
            "generalization; keep large-gap and descriptive-report limitations as notes. "
            "Set full_support only when every qualifier, subject, exact comparator, dataset, metric, setting and "
            "quantifier is established. A tie cannot establish strict improvement/degradation. A lower-ranked "
            "baseline cannot establish a comparison against the top system. Record unresolved_qualifiers explicitly. "
            "For comparative numerical support provide relation, all required_cases, and a numerical comparison "
            "for each case. For list-valued settings supply comparison.settings with one selected value per dimension, "
            "covering every joint combination. Declare subject/comparator exact names from the claim source, or bind them "
            "via subject_setting/comparator_setting to an exact condition setting key. left must be the claim's subject "
            "and right its comparator; never reverse them to make the comparison hold. Use empty names for noncomparative claims. "
            "Each side must quote one unambiguous value with its subject and exact metric label; "
            "value_context is a contiguous substring. For native HTML tables quote the table and provide left_cell/right_cell "
            "zero-based table/row/column after expanding rowspan/colspan; the selected cell must contain only the token, "
            "with subject and metric/task bound by its row/column headers. Ground difference in the current claim's source quote or covered source_refs; "
            "difference_mode must explicitly distinguish absolute deltas, relative_percent and percentage_points. "
            "For a stated positive increase/decrease magnitude, set condition difference_direction and copy its exact "
            "source qualifier into difference_direction_quote; decrease uses comparator minus subject, with the comparator "
            "as the relative baseline. The default signed mode preserves subject minus comparator. Keep operand identities fixed. "
            "Percentage points require explicit percent-scaled inputs; do not infer units from metric names. Preserve expected difference units. "
            "Context passages must identify the exact condition dataset/metric/settings and common units. "
            "Multi-value tables without uniquely bound values remain partial. Non-numerical descriptive support "
            "does not require invented numerical comparisons. Explain applicability with exact grounds; uncertainty "
            "or missing review information must remain unverified, without creating a new flaw.",
            payload,
            module="verification.experiments.scope",
            call=call,
        )
        audit["response"] = response
        review = ExperimentScopeReview.model_validate(response)
        conditions = {row.condition_id: row for row in review.conditions}
        if len(conditions) != len(review.conditions) or set(conditions) != {c.id for c in claim.conditions}:
            raise ValueError("Scope review must cover each claim condition exactly once")
        for row in review.conditions:
            passages = [claim.text, *(quote for _, quote in _claim_passages(claim, row.condition_id))]
            if not any(row.claim_quote in quote for quote in passages):
                raise ValueError("Scope review claim quote is not an exact part of the current claim")
            if row.credited_component and not any(row.credited_component in quote for quote in passages):
                raise ValueError("Credited component is absent from the current claim")
            for role in ("subject", "comparator"):
                label, key = getattr(row, role), getattr(row, f"{role}_setting")
                if label and not key and not any(_has_label(label, quote) for quote in passages):
                    raise ValueError(f"Scope {role} is absent from the current claim source")
        decisions = {(row.item_index, row.condition_id): row for row in review.items}
        expected = {
            (index, condition) for index, item in enumerate(output.items) for condition in item.covered
        }
        if len(decisions) != len(review.items) or set(decisions) != expected:
            raise ValueError("Scope review must cover every candidate/condition pair exactly once")
        for row in review.items:
            for passage in row.grounds:
                _paper_pointer(materials, passage.block_id, passage.quote)
        audit["validated"] = True
        audit["item_locations"] = [
            {
                "candidate_index": row.item_index,
                "condition_id": row.condition_id,
                "response_pointer": f"/response/items/{position}",
            }
            for position, row in enumerate(review.items)
        ]
        return conditions, decisions, None, audit_path
    except Exception as exc:
        issue = f"Experimental scope review unconfirmed: {type(exc).__name__}: {exc}"
        audit["error"] = issue
        return {}, {}, issue, audit_path
    finally:
        if audit_path is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def _scope_sources(decision, materials):
    """Keep reports compact while connecting supplemental grounds to the raw audit."""
    passages = list(decision.grounds)
    for comparison in decision.comparisons:
        passages.extend([comparison.left, comparison.right, *comparison.context])
        if comparison.difference is not None:
            passages.append(comparison.difference)
    blocks = {block.id: block for block in materials.blocks}
    sources = []
    for block_id in dict.fromkeys(passage.block_id for passage in passages):
        block = blocks.get(block_id)
        page = block.loc.page if block and block.loc else None
        sources.append(f"block={block_id}, page={page if page is not None else 'unavailable'}")
    return "; ".join(sources)


def _plan(claim: Claim, materials: SharedMaterials, candidate: PlanCandidate) -> ExecutionPlan:
    conditions = {condition.id: condition for condition in claim.conditions}
    ids = _covered(claim, [target.condition_id for target in candidate.targets])
    values = {target.condition_id: _number(materials, target.reported) for target in candidate.targets}
    target_issues = []
    for target in candidate.targets:
        condition = conditions[target.condition_id]
        # Exact quotes must identify the target metric/dataset, possibly across
        # a full table containing the header and dataset row.
        if not _has_label(condition.metric, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's metric")
        if condition.dataset and not _has_label(condition.dataset, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's dataset")
        ambiguity = _target_ambiguity(claim, target, values[target.condition_id])
        if ambiguity:
            target_issues.append(f"Unresolved paper target {target.condition_id}: {ambiguity}")
    repository = materials.repository
    files = {item.path for item in repository.files} if repository else set()
    entries = set(repository.entry_scripts) if repository else set()
    configs = set(repository.configs) if repository else set()
    if candidate.entry_script and candidate.entry_script not in entries:
        raise ValueError("Execution-plan entry script must come from the repository index")
    if candidate.config and candidate.config not in configs:
        raise ValueError("Execution-plan config must come from the repository index")
    if not set(candidate.data_paths + candidate.weight_paths).issubset(files):
        raise ValueError("Execution resources must be listed in the repository index")
    blockers = ([candidate.blocker] if candidate.blocker.strip() else []) + target_issues
    if repository is None or not candidate.entry_script:
        blockers.append("Candidate released code is missing")
    if not candidate.data_paths:
        blockers.append("Released data location is missing")
    if candidate.run_mode == "evaluation" and not candidate.weight_paths:
        blockers.append("Released weights are missing")
    if candidate.feasibility == "blocked" and not blockers:
        blockers.append("Plan is blocked; runtime requirements need clarification")
    # Training approval/budget is enforced by L3; preserve its candidate here.
    return ExecutionPlan(
        id=f"{claim.id}.plan",
        claim_id=claim.id,
        condition_ids=ids,
        target_conditions=[conditions[key] for key in ids],
        y_paper=values,
        task=ExecutionTask(entry_script=candidate.entry_script, config=candidate.config),
        run_mode=candidate.run_mode,
        feasibility="blocked" if blockers else "ready",
        blocker="; ".join(blockers),
        priority=candidate.priority,
        estimated_cost=candidate.estimated_cost,
    )


def verify_experiments(
    claim: Claim, materials: SharedMaterials, *, call=None, scope_call=None
) -> BranchResult:
    output = ExperimentsOutput.model_validate(
        ask(
            "Check this experimental claim from the paper alone in five aspects: correspondence "
            "(an experiment for every assertion), fairness (same data/budget/tuning), isolation "
            "(credited components ablated), stability (variance/seeds/significance for small gaps), "
            "consistency (abstract/text/table numbers). List all five in checked_aspects. Return output_schema JSON. Every item needs "
            "an exact located paper quote and a concrete detail. paper_support stays paper-internal. "
            "covered must be a nonempty list of distinct exact strings from allowed_condition_ids, the IDs "
            "in claim.conditions. Every plan target condition_id must also come from that list, with no "
            "duplicate targets. Never use claim.id, block IDs, datasets, or metric names as condition IDs. "
            "Only include conditions the item actually addresses. Omit items/plans that cannot be tied to "
            "allowed conditions and explain the limitation in issues; return empty items/plans when needed. "
            "For paper_support, explicitly list fully_supported_conditions only when this one item "
            "establishes the ENTIRE condition and every relevant claim qualifier, including its dataset, "
            "metric, comparisons, and settings. A setup description or one component's result provides "
            "partial support when the condition asserts more. Leave the list empty then and explain the "
            "uncovered parts in detail. The paper's assertion by itself does not establish its conclusion. "
            "Copy every quote verbatim as one contiguous substring of its selected paper block's text, "
            "and copy value_context verbatim from within quote. Preserve mathematical markup, whitespace, "
            "punctuation, and spelling exactly; do not normalize math, paraphrase, or join disjoint passages. "
            "large_gap_no_variance is a non-decisive note. text_table_contradiction requires two "
            "quoted numerical passages for the same target condition, dataset, metric, and every setting, "
            "on the same scale. Each value_context must identify those labels and one unique numeric value; "
            "preserve explicit units and percentage markers. Different splits/seeds/models, ambiguous "
            "multi-value tables, or unresolved unit conversions do not establish a contradiction. "
            "For each target claim with re-obtainable reported numbers, emit one plan. "
            "A plan must use the exact metric named by its target condition. Never replace an abstract "
            "quality or qualitative condition with a different numerical metric such as MRR. When no "
            "reported number matches the condition's metric, omit that target/plan and explain in issues. "
            "Include candidate entry script/config only from the supplied repository index; unknown commands/metric output "
            "remain for L3. Every plan target quotes the exact paper numeric token, metric, and dataset; "
            "copy full table headers when needed. For multi-value quotes, supply an exact value_context "
            "substring identifying the target dataset/metric and its unique reported value. Ambiguous "
            "whole-table references remain blocked until the target is resolved. Include actual indexed data/weight paths. Keep plans "
            "with missing code, data, weights, or budget as blocked with a reason. Priority follows "
            "the link to the paper's core contribution. Emit no execution evidence or final verdict.",
            {
                "claim": claim.model_dump(mode="json"),
                "allowed_condition_ids": [condition.id for condition in claim.conditions],
                "paper_blocks": [b.model_dump() for b in materials.blocks],
                "repository_index": materials.repository.model_dump() if materials.repository else None,
                "output_schema": ExperimentsOutput.model_json_schema(),
            },
            module="verification.experiments",
            call=call,
        )
    )
    if set(output.checked_aspects) != {"correspondence", "fairness", "isolation", "stability", "consistency"}:
        raise ValueError("Experiments must inspect all five required aspects")
    result = BranchResult(issues=output.issues)
    # Invalid first-pass source/coverage contracts still raise before an audit.
    for item in output.items:
        _paper_pointer(materials, item.block_id, item.quote)
        _fully_supported(_covered(claim, item.covered), item.fully_supported_conditions)
    scopes, decisions, scope_issue, scope_audit = (
        _scope_review(claim, materials, output, call=scope_call or call)
        if output.items
        else ({}, {}, None, None)
    )
    if scope_issue:
        result.issues.append(scope_issue)
    condition_map = {condition.id: condition for condition in claim.conditions}
    for index, item in enumerate(output.items):
        required_aspect = {
            "missing_ablation": "isolation",
            "missing_statistic": "stability",
            "small_gap_without_statistics": "stability",
            "large_gap_no_variance": "stability",
            "text_table_contradiction": "consistency",
        }.get(item.kind)
        if required_aspect and item.aspect != required_aspect:
            raise ValueError("Experimental concern is assigned to the wrong aspect")
        pointer = _paper_pointer(materials, item.block_id, item.quote)
        covered = _covered(claim, item.covered)
        full_support = _fully_supported(covered, item.fully_supported_conditions)
        contrary = item.kind != "paper_support"
        decisive = item.kind != "large_gap_no_variance"
        overturnable = True
        if item.kind == "text_table_contradiction":
            if len(item.comparison) != 2 or len(covered) != 1:
                raise ValueError("Text-table contradiction needs two values for one condition")
            left, right = [_number(materials, number) for number in item.comparison]
            condition = next(c for c in claim.conditions if c.id == covered[0])
            reasons = [
                reason
                for quoted, value in zip(item.comparison, (left, right), strict=True)
                if (
                    reason := _target_ambiguity(
                        claim, PlanTarget(condition_id=condition.id, reported=quoted), value
                    )
                )
            ]
            if not reasons:
                units = [_comparison_units(number, condition.metric) for number in item.comparison]
                if any(len(unit) > 1 for unit in units) or units[0] != units[1]:
                    reasons.append("Quoted values use different or incompletely specified units/scales")
            if left == right:
                reasons.append("Equal paper values cannot establish a text-table contradiction")
            if reasons:
                result.issues.append(
                    f"Text-table contradiction unconfirmed for {condition.id}: " + "; ".join(reasons)
                )
                continue
            # Different numbers alone can have rounding/protocol explanations;
            # preserve both pointers for the author to clarify the claim.
            detail = (
                item.detail + "; compared " + "; ".join(f"{n.block_id}: {n.quote}" for n in item.comparison)
            )
        else:
            detail = item.detail
        note = f"{item.aspect}/{item.kind}: {detail}"
        if not contrary:
            note = _support_note(note, item.fully_supported_conditions)
        accepted = []
        scope_notes = []
        for condition_id in covered:
            scope = scopes.get(condition_id)
            decision = decisions.get((index, condition_id))
            reasons = []
            if scope is None or decision is None:
                reasons.append("Independent scope/coverage review is unavailable")
            else:
                if scope.assertion == "unresolved" or decision.applicability != "applicable":
                    reasons.append(f"Scope applicability is {decision.applicability}; {decision.rationale}")
                if contrary:
                    if item.kind == "missing_ablation" and (
                        scope.assertion != "causal_attribution" or not scope.credited_component
                    ):
                        reasons.append(
                            "No component-performance attribution in this claim requires an ablation"
                        )
                    if (
                        item.kind == "missing_control"
                        and item.aspect == "fairness"
                        and (
                            not scope.matched_controls_required
                            or scope.assertion not in {"controlled_comparison", "causal_attribution"}
                        )
                    ):
                        reasons.append("Matched experimental controls are not part of this assertion")
                    if item.kind in {"missing_statistic", "small_gap_without_statistics"} and (
                        not scope.uncertainty_sensitive or scope.assertion == "descriptive"
                    ):
                        reasons.append("Unreported uncertainty does not bear on this descriptive assertion")
                else:
                    if (
                        not decision.full_support
                        or not decision.qualifiers_complete
                        or decision.unresolved_qualifiers
                    ):
                        reasons.append(
                            "Full condition support is unconfirmed: "
                            + "; ".join(decision.unresolved_qualifiers or [decision.rationale])
                        )
                    reasons.extend(
                        _numeric_scope_check(claim, condition_map[condition_id], scope, decision, materials)
                    )
                scope_notes.append(f"{condition_id}: {scope.assertion}; {decision.rationale}")
                scope_notes.append(
                    f"{condition_id} scope_source_refs=[{_scope_sources(decision, materials)}]"
                )
            if reasons:
                message = f"Experimental {item.kind} scope unconfirmed for {condition_id}: " + "; ".join(
                    reasons
                )
                result.issues.append(message)
                scope_notes.append(message)
            else:
                accepted.append(condition_id)
        note += "; independent_scope_review=" + " | ".join(scope_notes)
        if scope_audit is not None:
            note += f"; scope_audit={scope_audit}; candidate_index={index}; condition_ids={covered!r} (see audit item_locations)"
        rejected = [condition_id for condition_id in covered if condition_id not in accepted]
        if contrary:
            groups = [(accepted, decisive), (rejected, False)]
        elif full_support:
            groups = [(accepted, True), (rejected, False)]
        else:
            # Preserve the original coverage item when it never claimed full
            # support for every covered condition; an audit cannot upgrade it.
            groups = [(covered, False)]
        for group, sufficient in groups:
            if not group:
                continue
            result.evidence.append(
                Evidence(
                    source="paper_internal",
                    pointer=pointer,
                    covered=group,
                    direction="flaw" if contrary else "support",
                    sufficient=sufficient,
                    concern=contrary and sufficient,
                    affects_claim=sufficient if contrary else True,
                    overturnable=overturnable,
                    note=note,
                )
            )
        if contrary and decisive and accepted:
            result.questions.append(
                AuthorQuestion(
                    claim_id=claim.id, text=f"Could you clarify the {item.aspect} concern?", reason=detail
                )
            )
    try:
        result.plans = [_plan(claim, materials, candidate) for candidate in output.plans]
    except ValueError as exc:
        # All paper observations have passed their own strict checks. Expose
        # them to the orchestrator while rejecting the invalid plan explicitly.
        raise RejectedPlan(str(exc), result) from exc
    return result
