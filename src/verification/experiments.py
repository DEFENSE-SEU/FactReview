"""Paper-only experiment checks and the sole claim-linked execution-plan producer."""

from __future__ import annotations

import copy
import json
import math
import re
import uuid
from itertools import pairwise, product
from typing import Any, Literal

from pydantic import Field, model_validator

from schemas.claim import (
    AuthorQuestion,
    Claim,
    Contract,
    Evidence,
    ExecutionPlan,
    ExecutionTask,
    NonEmpty,
    PaperTargetSelector,
    PredictionProjection,
    ProjectionScopeDecision,
    SemanticPredictionProjection,
    SemanticProjectionScopeDecision,
)
from schemas.materials import SharedMaterials
from screening.checks import ask
from verification.contracts import BranchResult, RejectedPlan
from verification.experiment_binding_repair import (
    BindingContractError,
    repair_bindings,
)
from verification.experiment_catalog import (
    TableGrid as _TableGrid,
)
from verification.experiment_catalog import (
    build_catalog,
    catalog_prompt,
    resolve_case,
    resolve_cell,
    resolve_source,
)
from verification.experiment_sources import (
    joint_limit,
    prepare_joint_candidate,
    record_assertion,
    require_passage,
    revalidate_members,
    validate_source_uses,
)
from verification.experiment_targets import TargetBindingError, bind_execution_target
from verification.prose_numbers import (
    bind_pair,
    direct_pair_candidates,
    resolve_number,
    sentence_id,
    transition_endpoints,
)
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


_UNIT_ALIASES = {
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


def _canonical_unit(value: str) -> str | None:
    value = value.strip().casefold()
    return _UNIT_ALIASES.get(value, value if value in _UNIT_ALIASES.values() else None)


def _known_unit_setting(key: str, value: str) -> bool:
    return key in {"unit", "units"} and _canonical_unit(value) is not None


def _metric_name_format(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", re.sub(r"\s+", " ", value.strip())).casefold()


def _comparison_units(number: PaperNumber, metric: str) -> set[str]:
    """Retain explicit scales; do not guess a conversion for unlabelled values."""
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
    return {_canonical_unit(unit) or unit for unit in units}


def _table_header_rows(parsed, index, before_row):
    """Keep explicit headers and a finite structurally connected all-td prefix."""
    grid, meta, origins = parsed.tables[index], parsed.metadata[index], parsed.origins[index]
    accepted = set()
    numeric = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?")
    for row in range(before_row):
        fresh = [(r, c) for r, c in grid if r == row and origins[r, c] == (r, c)]
        if not fresh or any(numeric.fullmatch(grid[key]) for key in fresh):
            break
        semantic = all(
            (meta[key]["tag"] == "th" or meta[key]["thead"]) and meta[key]["scope"] not in {"row", "rowgroup"}
            for key in fresh
        )
        groups = [
            (origin, value)
            for origin, value in meta.items()
            if origins[origin] == origin
            and origin[0] in accepted
            and value["colspan"] > 1
            and origin[0] + value["rowspan"] == row
        ]
        grouped = bool(groups) and all(
            not grid[key].strip()
            or any(origin[1] <= key[1] < origin[1] + value["colspan"] for origin, value in groups)
            for key in fresh
        )
        if row == 0 or semantic or grouped:
            accepted.add(row)
        else:
            break
    return accepted


def _explicit_header_units(text):
    """Consume complete unit annotations; incompatible or compound scales fail closed."""
    annotations = list(re.finditer(r"\(([^()]*)\)", text))
    if not annotations:
        unit = _canonical_unit(text)
        return {unit} if unit else set()
    if text[annotations[-1].end() :].strip(" \t\r\n.,;:"):
        raise ValueError(
            "Comparison units/scales do not match: compound or trailing unit expression is unresolved"
        )
    if any(text[left.end() : right.start()].strip() for left, right in pairwise(annotations)):
        raise ValueError("Comparison units/scales do not match: intervening unit expression is unresolved")
    units = set()
    for annotation in annotations:
        unit = _canonical_unit(annotation.group(1))
        if unit is None:
            raise ValueError(
                "Comparison units/scales do not match: selected measurement-axis unit is unknown"
            )
        units.add(unit)
    return units


def _caption_unit_scope(body, quantity, dataset, settings):
    """Accept a finite, fully consumed unit declaration for this operand's scope."""
    quantity_match = re.search(r"(?<!\w)" + re.escape(quantity) + r"\s*(?=\()", body, re.I)
    if quantity_match is None:
        return False
    annotations = list(re.finditer(r"\([^()]*\)", body[quantity_match.end() :]))
    if not annotations:
        return False
    annotation_text = body[quantity_match.end() :]
    compound = re.compile(r"\s*(?:/|\bper\b|\*|×|÷|\^|\bdivided\b)", re.I)
    for left, right in pairwise(annotations):
        gap = annotation_text[left.end() : right.start()]
        if gap.strip() and not compound.match(gap):
            return False
    before = body[: quantity_match.start()]
    after = annotation_text[annotations[-1].end() :]
    # Eligible compound expressions continue to the existing unit parser, which
    # rejects them. A unit on an inapplicable caption never replaces axis units.
    if compound.match(after):
        after = ""
    before = re.sub(
        r"^\s*(?:results(?:\s+(?:for|of))?\s*[:.\-]?\s*)?(?:reported\s+)?", "", before, flags=re.I
    )

    def literal(value):
        return r"\s+".join(re.escape(part) for part in str(value).split())

    atoms = [literal(dataset)] if dataset else []
    values = [str(value) for value in settings.values()]
    for key, value in settings.items():
        key_pattern, value_pattern = literal(key), literal(value)
        if not key_pattern or not value_pattern:
            continue
        atoms.extend(
            (key_pattern + r"(?:\s*[:=]\s*|\s+)" + value_pattern, value_pattern + r"\s+" + key_pattern)
        )
        if values.count(str(value)) == 1 and not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(value)):
            atoms.append(value_pattern)
    atom = (
        re.compile(r"(?:" + "|".join(sorted(atoms, key=len, reverse=True)) + r")(?!\w)", re.I)
        if atoms
        else None
    )
    connector = re.compile(r"(?:[\s,;:.]+|\b(?:on|for|of|with|at|using|and)\b)", re.I)
    for residual in (before, after):
        offset = 0
        while offset < len(residual):
            match = (atom.match(residual, offset) if atom else None) or connector.match(residual, offset)
            if match is None or match.end() == offset:
                return False
            offset = match.end()
    return True


def _table_units(
    number, cell, label, metric, dataset=None, *, metric_binding=False, caption=None, caption_settings=None
):
    """Read explicit units on this cell's measurement axis, retaining native scale.

    Semantic th/thead headers retain their span coordinates. Native all-td tables
    use their first header row and structurally connected span children, or the
    identified measurement stub in a transposed row. Methods and data cells cannot
    donate a unit, even when they contain a percentage marker.
    """
    parsed = _TableGrid(number.quote)
    grid, meta = parsed.tables[cell.table], parsed.metadata[cell.table]
    row_role = any(
        c < cell.column and r == cell.row and _has_label(label, text) for (r, c), text in grid.items()
    )
    column_role = any(
        r < cell.row and c == cell.column and _has_label(label, text) for (r, c), text in grid.items()
    )
    if row_role == column_role:
        raise ValueError("Numerical unit has no unique measurement-axis orientation")
    header_rows = _table_header_rows(parsed, cell.table, cell.row)
    if row_role:
        axis = [
            text
            for (r, c), text in grid.items()
            if c == cell.column
            and r < cell.row
            and meta[r, c]["scope"] not in {"row", "rowgroup"}
            and (r in header_rows or meta[r, c]["thead"] or meta[r, c]["scope"] in {"col", "colgroup"})
        ]
    else:
        stub_fields = {"metric", "measure", "measurement", "dataset", "task"}
        stub_columns = {
            c for (r, c), text in grid.items() if r in header_rows and text.strip().casefold() in stub_fields
        }
        axis = [
            text
            for (r, c), text in grid.items()
            if r == cell.row
            and c < cell.column
            and meta[r, c]["scope"] not in {"col", "colgroup"}
            and (
                parsed.origins[cell.table][r, c][1] == 0
                or c in stub_columns
                or meta[r, c]["scope"] in {"row", "rowgroup"}
            )
        ]
    quantities = [metric, *([dataset] if metric_binding and dataset else [])]
    units = {"%"} if number.token.endswith("%") else set()
    axis_identified = any(_has_label(quantity, text) for quantity in quantities for text in axis)
    if axis_identified:
        for text in dict.fromkeys(axis):
            units.update(_explicit_header_units(text))
    # A located caption belongs only to the selected HTML table. Earlier table
    # captions and unrelated prefix paragraphs cannot donate units.
    spans = list(re.finditer(r"<table\b.*?</table>", number.quote, re.I | re.S))
    prefix = (
        number.quote[spans[cell.table - 1].end() if cell.table else 0 : spans[cell.table].start()]
        if caption is None
        else caption
    )
    names = list(re.finditer(r"(?:^|\n)\s*Table\s+\d+\s*[:.]", prefix, re.I))
    if names:
        caption = re.split(r"\n\s*\n", prefix[names[-1].start() :], maxsplit=1)[0]
        body = re.sub(r"^\s*Table\s+\d+\s*[:.]\s*", "", caption, flags=re.I).strip()
        for quantity in quantities:
            if not quantity or not _caption_unit_scope(body, quantity, dataset, caption_settings or {}):
                continue
            pattern = r"(?<!\w)" + re.escape(quantity) + r"\s*((?:\([^()]*\)\s*)+(?:/[^\s.,;]+)?)"
            for match in re.finditer(pattern, caption, re.I):
                if re.match(r"\s*(?:/|\bper\b|\*|×|÷|\^|\bdivided\b)", caption[match.end() :], re.I):
                    raise ValueError(
                        "Comparison units/scales do not match: compound caption unit is unresolved"
                    )
                units.update(_explicit_header_units(match.group(1)))
    return units


class ScopePassage(Contract):
    block_id: NonEmpty
    quote: NonEmpty


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
    additional_sources: list[ScopePassage] = Field(
        default_factory=list,
        description=(
            "Exact additional members of ONE joint paper_support candidate for ONE condition; "
            "[] means single-source. Together with block_id/quote, declare the complete supporting set: "
            "results, required definitions/setup, and the manuscript's exact numbered-table-reference "
            "passage linking any external metric/setup definition to the selected table. "
            "One exact member may serve multiple roles; do not duplicate it. "
            "A source in another partial item grants this candidate no access. Do not merge partial flags."
        ),
    )

    @model_validator(mode="after")
    def joint_contract(self):
        if self.additional_sources and (self.kind != "paper_support" or len(self.covered) != 1):
            raise ValueError("Joint additional_sources require paper_support for exactly one condition")
        return self


class PlanTarget(Contract):
    condition_id: NonEmpty = Field(
        description="One exact ID from allowed_condition_ids; target IDs must be distinct within each plan."
    )
    reported: PaperNumber
    selector: PaperTargetSelector | None = None
    # Keep malformed new proposals plan-local; the pure binder validates them.
    projection: PredictionProjection | SemanticPredictionProjection | Any | None = Field(default=None)


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


class ScopeSemantics(Contract):
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
    difference_direction: Literal["signed", "increase", "decrease"] = "signed"
    difference_direction_quote: str = ""
    required_cases: list[str] = Field(default_factory=list)
    setting_scopes: dict[str, Literal["subject", "comparator", "shared"]] = Field(
        default_factory=dict,
        description="Keys are the original BARE keys of condition.settings, such as split, method, from_setting or to_setting. Never put dataset, metric or settings.KEY here. For structural prose pair choices, use their canonical_setting_scopes. This differs from SourceBridge.condition_field, which uses settings.KEY.",
    )
    rationale: NonEmpty


class ConditionScope(ScopeSemantics):
    # Legacy strict responses retain this field. Catalog responses derive it
    # from the authoritative claim/condition instead of asking the model.
    endpoint_required: bool = False
    subject_setting: str = ""
    comparator_setting: str = ""


class TableCell(Contract):
    """Zero-based expanded HTML table coordinates, checked against source bytes."""

    table: int = Field(default=0, ge=0)
    row: int = Field(ge=0)
    column: int = Field(ge=0)


class SourceBridge(Contract):
    kind: Literal["subject", "comparator", "metric", "setting"]
    condition_field: NonEmpty = Field(
        description="The original field path: metric or settings.KEY (for example settings.procedure). This bridge path is distinct from the bare keys required in condition setting_scopes."
    )
    applies_to: Literal["subject", "comparator", "shared"]
    table_id: NonEmpty
    source_ids: list[NonEmpty] = Field(
        min_length=1,
        description=(
            "Verify each ID's resolved quote against this bridge. Include all required definition "
            "and manuscript table-reference passages; they may use different IDs. "
            "A passage mentioned only in explanation does not supply a missing source ID."
        ),
    )
    paper_label: str = Field(
        default="",
        description="For metric bridges: the exact original metric quantity (e.g. mIoU), never a dataset column label. For setup bridges: an exact treatment phrase shared by the selected axis and its definition, not necessarily the entire row label.",
    )
    explanation: NonEmpty


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
    bridges: list[SourceBridge] = Field(default_factory=list)
    left_catalog_id: str = ""
    right_catalog_id: str = ""
    left_number_id: str = ""
    right_number_id: str = ""
    difference_number_id: str = ""
    expected_left: PaperNumber | None = None
    expected_right: PaperNumber | None = None


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


class CatalogConditionScope(ScopeSemantics):
    # Identity and source bytes already exist in the authoritative input.
    claim_quote: str = ""
    subject: str = ""
    comparator: str = ""


class CatalogComparison(Contract):
    case_id: NonEmpty
    left_cell_id: str = ""
    right_cell_id: str = ""
    left_label_cell_id: str = ""
    right_label_cell_id: str = ""
    left_source_id: str = ""
    right_source_id: str = ""
    left_token: str = ""
    right_token: str = ""
    left_value_context: str = ""
    right_value_context: str = ""
    relation: Literal["gt", "ge", "lt", "le", "eq", "difference"]
    bridges: list[SourceBridge] = Field(default_factory=list)
    context_source_ids: list[NonEmpty] = Field(default_factory=list)
    difference_source_id: str = ""
    difference_token: str = ""
    difference_mode: Literal["absolute", "relative_percent", "percentage_points", "unresolved"] = "unresolved"

    @model_validator(mode="after")
    def exact_operand_selectors(self):
        for side in ("left", "right"):
            cells = [getattr(self, f"{side}_{key}") for key in ("cell_id", "label_cell_id")]
            text = [getattr(self, f"{side}_{key}") for key in ("source_id", "token", "value_context")]
            if not ((all(cells) and not any(text)) or (all(text) and not any(cells))):
                raise ValueError(
                    f"{side}: choose exactly one complete cell or prose source/token/context selector"
                )
        return self


class CatalogItemScope(Contract):
    item_index: int = Field(ge=0)
    condition_id: NonEmpty
    applicability: Literal["applicable", "not_applicable", "unverified"]
    grounds_source_ids: list[NonEmpty] = Field(min_length=1)
    rationale: NonEmpty
    full_support: bool = False
    qualifiers_complete: bool = False
    comparison_objects: Literal["matched", "unmatched", "unresolved", "not_comparative"] = "unresolved"
    unresolved_qualifiers: list[str] = Field(default_factory=list)
    nonblocking_notes: list[str] = Field(default_factory=list)
    comparisons: list[CatalogComparison] = Field(default_factory=list)


class CatalogScopeReview(Contract):
    schema_version: Literal["catalog-v1"]
    conditions: list[CatalogConditionScope]
    items: list[CatalogItemScope]


class CellOperand(Contract):
    kind: Literal["cell"]
    cell_id: NonEmpty
    label_cell_id: NonEmpty


class ProseOperand(Contract):
    kind: Literal["prose"]
    number_id: NonEmpty


class CatalogComparisonV2(Contract):
    case_id: NonEmpty
    left: CellOperand | ProseOperand = Field(discriminator="kind")
    right: CellOperand | ProseOperand = Field(discriminator="kind")
    relation: Literal["gt", "ge", "lt", "le", "eq", "difference"]
    bridges: list[SourceBridge] = Field(default_factory=list)
    context_source_ids: list[NonEmpty] = Field(default_factory=list)
    difference_number_id: str = ""
    difference_mode: Literal["absolute", "relative_percent", "percentage_points", "unresolved"] = "unresolved"


class CandidateSourceUse(Contract):
    source_id: NonEmpty
    roles: list[
        Literal[
            "result",
            "metric_definition",
            "setup_definition",
            "table_reference",
            "protocol",
            "other_qualifier",
        ]
    ] = Field(min_length=1)
    rationale: NonEmpty


class CatalogItemScopeV2(CatalogItemScope):
    comparisons: list[CatalogComparisonV2] = Field(default_factory=list)
    source_uses: list[CandidateSourceUse] = Field(
        default_factory=list,
        description=(
            "Only for joint candidates whose first-pass additional_sources is nonempty: exactly one "
            "entry per declared member_source_id, including the primary. For every index in "
            "single_source_candidate_indices (additional_sources=[]), this field MUST be []. "
            "Grounds or bridges on a single-source candidate do not permit nonempty source_uses."
        ),
    )


class CatalogScopeReviewV2(Contract):
    schema_version: Literal["catalog-v2"]
    conditions: list[CatalogConditionScope]
    items: list[CatalogItemScopeV2]
    plan_projection_reviews: list[ProjectionScopeDecision | SemanticProjectionScopeDecision] = Field(
        default_factory=list
    )


def _field(condition, path):
    if path in {"dataset", "metric"}:
        return getattr(condition, path)
    if path.startswith("settings.") and path[9:] in condition.settings:
        return condition.settings[path[9:]]
    raise ValueError(f"Unknown condition field: {path}")


def _condition_roles(condition, review):
    """Recover uniquely named structured roles, retaining ambiguous roles as unconfirmed."""
    keys = {
        "subject": ("model", "method", "subject", "ablation", "variant", "treatment"),
        "comparator": ("comparison", "comparator", "baseline", "reference", "control"),
    }
    bindings = {}
    for role, candidates in keys.items():
        present = [key for key in candidates if key in condition.settings]
        identities = {json.dumps(condition.settings[key], sort_keys=True) for key in present}
        if len(identities) > 1:
            raise ValueError(f"Multiple structured {role} fields have unresolved identities")
        bindings[f"{role}_setting"] = present[0] if present else ""
    subject_key = bindings["subject_setting"]
    if subject_key and not bindings["comparator_setting"]:
        value = condition.settings[subject_key]
        treatment = any(
            key in condition.settings and key != subject_key and role == "subject"
            for key, role in review.setting_scopes.items()
        )
        if (
            treatment
            and isinstance(value, str)
            and review.comparator
            and _has_label(value, review.comparator)
        ):
            bindings["comparator_setting"] = subject_key
    return bindings


def _catalog_passage(catalog, source_id, materials, condition_id):
    source = resolve_source(catalog, source_id, materials)
    if source["kind"] == "claim_text":
        raise ValueError("An extracted claim is not an original evidence passage")
    if source["covered"] and condition_id not in source["covered"]:
        raise ValueError("Catalog source belongs to another condition")
    _paper_pointer(materials, source["block_id"], source["quote"])
    return ScopePassage(block_id=source["block_id"], quote=source["quote"])


def _expected_number(claim, condition_id, token, materials, catalog=None):
    if not token:
        return None
    for block_id, quote in _claim_passages(claim, condition_id):
        if block_id is None:
            continue
        candidate = PaperNumber(block_id=block_id, quote=quote, token=token)
        try:
            _number(materials, candidate)
            record_assertion(catalog, materials, candidate, purpose="asserted_endpoint")
            return candidate
        except ValueError:
            continue
    raise ValueError("Expected endpoint is absent from this condition's original source")


def _asserted_endpoints(claim, condition, materials=None):
    """Read explicit endpoint assertions; table observations cannot supply their own targets."""
    token = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?"
    pair = re.compile(rf"(?<![\w.])({token})\s+vs\.?\s+(?:baseline\s+)?({token})(?![\w%]|\.\d)", re.I)
    direct = pair.findall(condition.description or "")
    transition = re.compile(rf"\bfrom\s+({token})\s+to\s+({token})(?![\w%]|\.\d)", re.I)
    matches = list(transition.finditer(claim.text))
    candidates = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(claim.text)
        local = claim.text[match.start() : end]
        if len(claim.conditions) == 1 or _has_label(condition.dataset, local):
            candidates.append((match.group(2), match.group(1)))
    if len(direct) > 1 or len(candidates) > 1:
        raise ValueError("Claim endpoint assertion is ambiguous for this condition")
    if direct and candidates and direct[0] != candidates[0]:
        raise ValueError("Condition description and claim text assert conflicting endpoints")
    explicit = (direct or candidates or [None])[0]
    named = transition_endpoints(claim, condition, materials) if materials is not None else None
    if explicit and named and explicit != named:
        raise ValueError("Claim asserts conflicting bare and named-setting endpoints")
    return explicit or named


def _occurrence_number(catalog, identifier, materials):
    record = resolve_number(catalog.get("numbers", {}), identifier, materials)
    require_passage(catalog, materials, record["block_id"], record["quote"], purpose="number_occurrence")
    if "joint_view" in catalog:
        trace = catalog["joint_view"].setdefault("selector_consumption", [])
        selected = {
            "kind": "number",
            "selected_id": identifier,
            "parent_number_id": identifier,
            **{
                key: record[key]
                for key in (
                    "block_id",
                    "start",
                    "end",
                    "sentence_start",
                    "sentence_end",
                    "token",
                    "unit_suffix",
                )
            },
        }
        if selected not in trace:
            trace.append(selected)
    return PaperNumber(**{key: record[key] for key in ("block_id", "quote", "token")})


def _assertion_number(catalog, identifier, materials):
    record = resolve_number(
        catalog.get("assertion_numbers", catalog.get("numbers", {})), identifier, materials
    )
    return PaperNumber(**{key: record[key] for key in ("block_id", "quote", "token")})


def _occurrence_units(catalog, identifier, materials, *, assertion=False):
    numbers = (
        catalog.get("assertion_numbers", catalog.get("numbers", {}))
        if assertion
        else catalog.get("numbers", {})
    )
    record = resolve_number(numbers, identifier, materials)
    suffix = re.sub(r"\s+", " ", record["unit_suffix"].casefold())
    if suffix == "percentage points":
        suffix = "pp"
    return {_canonical_unit(suffix) or suffix} if suffix else set()


def _catalog_operand(item, side, scope, case, condition, materials, catalog):
    if isinstance(item, CatalogComparisonV2):
        selector = getattr(item, side)
        if isinstance(selector, ProseOperand):
            number = _occurrence_number(catalog, selector.number_id, materials)
            role = "subject" if side == "left" else "comparator"
            setting = getattr(scope, f"{role}_setting")
            label = case["settings"].get(setting, getattr(scope, role))
            return number, label, None, ""
        cell_id, label_id = selector.cell_id, selector.label_cell_id
    else:
        cell_id, label_id = getattr(item, f"{side}_cell_id"), getattr(item, f"{side}_label_cell_id")
    if cell_id:
        cell = resolve_cell(catalog, cell_id, materials)
        label = resolve_cell(catalog, label_id, materials)
        if cell["cell_type"] != "number" or label["cell_type"] != "label":
            raise ValueError("Numerical and role label cell IDs have different required types")
        if cell["table_id"] != label["table_id"] or not (
            (cell["row"] == label["row"] and label["column"] < cell["column"])
            or (cell["column"] == label["column"] and label["row"] < cell["row"])
        ):
            raise ValueError("Role label cell does not belong to the selected numerical axis")
        return (
            PaperNumber(**{key: cell[key] for key in ("block_id", "quote", "token")}),
            label["token"],
            TableCell(**{key: cell[key] for key in ("table", "row", "column")}),
            cell["source_id"],
        )
    source_id = getattr(item, f"{side}_source_id")
    passage = _catalog_passage(catalog, source_id, materials, condition.id)
    # Numerical HTML evidence must retain its row/column binding. A prose
    # selector cannot flatten a table and bypass the table-axis guards.
    if re.search(r"<table\b", passage.quote, re.I):
        raise ValueError("Prose numerical selectors cannot replace HTML table coordinates")
    context = getattr(item, f"{side}_value_context")
    if context not in passage.quote:
        raise ValueError("Prose value_context must be an exact substring of its selected source")
    role = "subject" if side == "left" else "comparator"
    setting = getattr(scope, f"{role}_setting")
    label = case["settings"].get(setting, getattr(scope, role))
    return (
        PaperNumber(**passage.model_dump(), token=getattr(item, f"{side}_token"), value_context=context),
        label,
        None,
        source_id,
    )


def _catalog_item(row, claim, materials, catalog, scope):
    condition = next(c for c in claim.conditions if c.id == row.condition_id)
    comparisons = []
    endpoints = _asserted_endpoints(claim, condition, materials)
    for item in row.comparisons:
        case = resolve_case(catalog, condition.id, item.case_id)
        left = _catalog_operand(item, "left", scope, case, condition, materials, catalog)
        right = _catalog_operand(item, "right", scope, case, condition, materials, catalog)
        source_ids = list(
            dict.fromkeys(
                [
                    *item.context_source_ids,
                    *[source for bridge in item.bridges for source in bridge.source_ids],
                    left[3],
                    right[3],
                ]
            )
        )
        passages = [
            _catalog_passage(catalog, value, materials, condition.id) for value in source_ids if value
        ]
        for operand in (left, right):
            if not operand[3]:
                passages.append(ScopePassage(block_id=operand[0].block_id, quote=operand[0].quote))
        difference = None
        identifiers = {}
        if isinstance(item, CatalogComparisonV2):
            for side in ("left", "right"):
                selector = getattr(item, side)
                key = f"{side}_catalog_id" if isinstance(selector, CellOperand) else f"{side}_number_id"
                identifiers[key] = (
                    selector.cell_id if isinstance(selector, CellOperand) else selector.number_id
                )
            identifiers["difference_number_id"] = item.difference_number_id
            if item.difference_number_id:
                difference = _assertion_number(catalog, item.difference_number_id, materials)
        elif item.difference_source_id:
            source = _catalog_passage(catalog, item.difference_source_id, materials, condition.id)
            difference = PaperNumber(**source.model_dump(), token=item.difference_token)
        if not isinstance(item, CatalogComparisonV2):
            identifiers.update(left_catalog_id=item.left_cell_id, right_catalog_id=item.right_cell_id)
        comparisons.append(
            ScopeComparison(
                case=case["id"],
                settings=case["settings"],
                left=left[0],
                right=right[0],
                left_label=left[1],
                right_label=right[1],
                metric_label=condition.metric,
                relation=item.relation,
                difference=difference,
                difference_mode=item.difference_mode,
                left_cell=left[2],
                right_cell=right[2],
                context=passages,
                bridges=item.bridges,
                **identifiers,
                expected_left=_expected_number(
                    claim,
                    condition.id,
                    endpoints[0] if endpoints else "",
                    materials,
                    catalog,
                ),
                expected_right=_expected_number(
                    claim,
                    condition.id,
                    endpoints[1] if endpoints else "",
                    materials,
                    catalog,
                ),
            )
        )
    payload = row.model_dump(
        exclude={"grounds_source_ids", "comparisons", "nonblocking_notes", "source_uses"}
    )
    if row.nonblocking_notes:
        payload["rationale"] += "; nonblocking notes: " + "; ".join(row.nonblocking_notes)
    return ItemScope(
        **payload,
        grounds=[
            _catalog_passage(catalog, value, materials, condition.id) for value in row.grounds_source_ids
        ],
        comparisons=comparisons,
    )


def _own_table_reference(quote, table_name):
    """Accept bounded self-reference syntax; citations in other sentences are irrelevant."""
    boundaries = [0]
    for match in re.finditer(r"[.!?](?=\s|$)", quote):
        if not re.search(r"\bet\s+al\.$", quote[: match.end()], re.I):
            boundaries.append(match.end())
    boundaries.append(len(quote))
    target = re.escape(table_name) + r"(?![\w.]|\.\d)"
    self_reference = re.compile(
        rf"^\s*(?:(?:Our\s+)?{target}\s+(?:reports?|shows?|presents?|compares?|summarizes?|lists?|contains?|provides?)\b"
        rf"|As\s+(?:shown|reported|presented|summarized|listed|detailed)\s+in\s+{target})",
        re.I,
    )
    attributed = re.compile(
        r"\b(?:of|in|from)\s+[^.!?]{0,120}(?:\bet\s+al\b|\(\s*(?:19|20)\d{2}[a-z]?\s*\)|\[\s*\d+(?:\s*[,;\-]\s*\d+)*\s*\])",
        re.I,
    )
    for start, end in pairwise(boundaries):
        sentence = quote[start:end].strip()
        if self_reference.search(sentence) and not attributed.search(sentence):
            return True
    return False


def _prose_setting_scopes(settings, subject_key, comparator_key, transition):
    roles = {key: "shared" for key in settings}
    if transition:
        roles.update(from_setting="comparator", to_setting="subject")
    else:
        for role, key in (("subject", subject_key), ("comparator", comparator_key)):
            if key:
                roles[key] = role
    return roles


def _prose_bindings(claim, condition, scope, comparison, materials, catalog):
    if (
        catalog is None
        or not comparison.left_number_id
        or not comparison.right_number_id
        or comparison.left_cell is not None
        or comparison.right_cell is not None
        or comparison.left_catalog_id
        or comparison.right_catalog_id
        or comparison.bridges
    ):
        raise ValueError("Prose occurrence pairs require two original number IDs without table selectors")
    for side in ("left", "right"):
        identifier = getattr(comparison, f"{side}_number_id")
        if getattr(comparison, side) != _occurrence_number(catalog, identifier, materials):
            raise ValueError("Prose operand differs from its original numbered occurrence")
    transition = transition_endpoints(claim, condition, materials) is not None
    settings = {**condition.settings, **comparison.settings}
    if any(
        key in {"unit", "units"} and _canonical_unit(str(value)) is None for key, value in settings.items()
    ):
        raise ValueError("Unknown condition unit remains unresolved for prose number occurrences")
    bind_pair(
        materials,
        catalog.get("numbers", {}),
        comparison.left_number_id,
        comparison.right_number_id,
        dataset=condition.dataset,
        metric=condition.metric,
        settings=settings,
        left_label=comparison.left_label,
        right_label=comparison.right_label,
        transition=transition,
    )
    roles = _prose_setting_scopes(settings, scope.subject_setting, scope.comparator_setting, transition)
    if transition:
        if scope.subject_setting != "method" or scope.comparator_setting != "method":
            raise ValueError("Named-setting transition must retain the same original method")
        after, before = transition_endpoints(claim, condition, materials)
        direction = "decrease" if float(after.rstrip("%")) < float(before.rstrip("%")) else "increase"
        if scope.relation not in {"lt" if direction == "decrease" else "gt", "difference"}:
            raise ValueError("Scope relation weakens or changes the original transition direction")
        if scope.relation == "difference" and scope.difference_direction != direction:
            raise ValueError("Scope difference direction changes the original transition assertion")
    if any(key not in roles or roles[key] != role for key, role in scope.setting_scopes.items()):
        raise ValueError("Prose setting scope conflicts with its directly named operand")
    return {
        role: {
            **{key: str(settings[key]) for key, owner in roles.items() if owner in {role, "shared"}},
            "role": f"settings.{getattr(scope, f'{role}_setting')}",
        }
        for role in ("subject", "comparator")
    }


def _bridge_bindings(claim, condition, scope, comparison, materials, catalog):
    """Validate exact field/source/table edges before granting contextual bindings."""
    if comparison.left_number_id or comparison.right_number_id:
        return _prose_bindings(claim, condition, scope, comparison, materials, catalog)
    result = {"subject": {}, "comparator": {}}
    for role in result:
        key = getattr(scope, f"{role}_setting")
        label = comparison.left_label if role == "subject" else comparison.right_label
        cell_id = comparison.left_catalog_id if role == "subject" else comparison.right_catalog_id
        value = condition.settings.get(key)
        if key and cell_id and isinstance(value, str) and _has_label(value, label):
            # Catalog label cells already prove this exact axis name. A literal
            # structured role match needs no second model-authored alias edge.
            result[role]["role"] = f"settings.{key}"
        if key and str(condition.settings.get(key)) == label:
            # A role's exact field and its own selected axis already establish
            # its operand-specific scope without an external setup definition.
            result[role][key] = label
    if not comparison.bridges:
        missing = [
            {"field": f"settings.{key}", "role": role}
            for key, role in scope.setting_scopes.items()
            if role != "shared" and key not in result[role]
        ]
        if missing:
            raise BindingContractError(
                "Operand-specific settings require explicit source bindings",
                code="missing_operand_setting_bridge",
                fields=missing,
            )
        return result
    if catalog is None or not comparison.left_catalog_id or not comparison.right_catalog_id:
        raise ValueError("Source bridges require verified catalog cell IDs")
    for bridge in comparison.bridges:
        value = _field(condition, bridge.condition_field)
        sources = [
            _catalog_passage(catalog, source_id, materials, condition.id) for source_id in bridge.source_ids
        ]
        roles = ("subject", "comparator") if bridge.applies_to == "shared" else (bridge.applies_to,)
        if bridge.kind in {"subject", "comparator"} and roles != (bridge.kind,):
            raise ValueError("Role binding targets the wrong operand")
        if bridge.kind == "setting":
            if not bridge.condition_field.startswith("settings."):
                raise ValueError("Setup binding must identify an existing setting")
            key = bridge.condition_field[9:]
            if bridge.applies_to != scope.setting_scopes.get(key, "shared"):
                raise ValueError("Setup binding changes the condition's subject/comparator scope")
        for role in roles:
            cell_id = comparison.left_catalog_id if role == "subject" else comparison.right_catalog_id
            cell = resolve_cell(catalog, cell_id, materials)
            if cell["table_id"] != bridge.table_id:
                raise ValueError("Source bridge points to another table")
            label = comparison.left_label if role == "subject" else comparison.right_label
            if bridge.kind in {"subject", "comparator"}:
                grid = _TableGrid(cell["quote"]).tables[cell["table"]]
                if (
                    isinstance(value, str)
                    and any(_has_label(value, text) for text in grid.values())
                    and not _has_label(value, label)
                ):
                    raise ValueError("Role bridge cannot override an explicit alternative table row")
                if not isinstance(value, str) or not _has_label(value, label):
                    raise ValueError(
                        "Nonliteral role alias is unresolved; passage co-occurrence cannot establish identity"
                    )
                result[role]["role"] = bridge.condition_field
                continue
            if (
                bridge.kind == "setting"
                and bridge.condition_field == f"settings.{getattr(scope, f'{role}_setting')}"
                and str(value) == label == bridge.paper_label
            ):
                result[role][bridge.condition_field[9:]] = label
                continue
            # External definitions must connect to this numbered table through
            # a separately located reference; a prefix from another table cannot authorize them.
            table_names = re.findall(r"(?:^|\n)\s*(Table\s+\d+)\s*[:.]", cell["caption"], re.I)
            target_name = table_names[-1] if table_names else ""
            external = [source for source in sources if source.block_id != cell["block_id"]]
            if not target_name or not any(
                _own_table_reference(source.quote, target_name) for source in external
            ):
                raise ValueError(
                    "External definition has no verified self-reference to the selected table; unknown or externally attributed reference syntax remains unresolved"
                )
            if bridge.kind == "metric":
                if bridge.condition_field != "metric" or not bridge.paper_label:
                    raise ValueError(
                        "Metric binding must name the original condition metric and source label"
                    )
                if not _has_label(bridge.paper_label, str(value)):
                    raise ValueError("Metric alias does not identify the original quantity")
                if str(value) != bridge.paper_label and not _has_label(
                    _metric_name_format(str(value)), _metric_name_format(bridge.explanation)
                ):
                    raise ValueError("Composite metric needs an explicit explanation of its complete name")
                if not any(
                    _has_label(condition.dataset, source.quote)
                    and _has_label(bridge.paper_label, source.quote)
                    for source in external
                ):
                    raise ValueError("Metric definition does not bind this dataset and quantity together")
                result[role]["metric"] = bridge.paper_label
            elif bridge.kind == "setting":
                key = bridge.condition_field[9:]
                if bridge.applies_to != "shared":
                    if not bridge.paper_label or not (
                        _has_label(bridge.paper_label, cell["row_labels"])
                        or _has_label(bridge.paper_label, cell["column_labels"])
                    ):
                        raise ValueError("Operand-specific treatment is absent from the selected table axis")
                    if not any(_has_label(bridge.paper_label, source.quote) for source in external):
                        raise ValueError("Treatment label has no exact source definition")
                scoped = any(
                    source.block_id == block_id and (quote in source.quote or source.quote in quote)
                    for source in external
                    for block_id, quote in _claim_passages(claim, condition.id)
                )
                if not any(_has_label(str(value), source.quote) for source in external) and not scoped:
                    raise ValueError("Setup definition lacks a literal value or a condition-scoped source")
                result[role][key] = str(value)
    for key, role in scope.setting_scopes.items():
        if key not in condition.settings:
            raise ValueError("Setting scope refers to an unknown condition field")
        if role != "shared" and key not in result[role]:
            raise BindingContractError(
                "Operand-specific setting is missing its source binding",
                code="missing_operand_setting_bridge",
                fields=[
                    {"field": f"settings.{name}", "role": owner}
                    for name, owner in scope.setting_scopes.items()
                    if owner != "shared" and name not in result[owner]
                ],
            )
    return result


def _table_value(number, cell, label, metric, settings, *, metric_binding=False, dataset=None, caption=None):
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
    if caption is None:
        caption = number.quote[prior_end : spans[cell.table].start()]
    captions = list(re.finditer(r"(?:^|\n)\s*Table\s+\d+\s*[:.]", caption, re.I))
    if len(captions) > 1:
        caption = caption[captions[-1].start() :]
    if not _has_label(metric, axes):
        if any(_has_label(metric, value) for value in grid.values()):
            raise ValueError("Selected table coordinates use a different explicit metric header")
        opposite_axis = column_labels if label_in_row else row_labels
        if metric_binding and _has_label(dataset, axes):
            return axes, caption, tuple(grid.values())
        if not _has_label(metric, caption) or not any(
            _has_label(value, opposite_axis) for value in settings.values()
        ):
            raise ValueError("Table coordinates do not bind the metric or comparison case")
    return axes, caption, tuple(grid.values())


def _side_settings(
    condition,
    scope,
    selected_settings,
    binding,
    *,
    axes=None,
    table_labels=(),
    role="subject",
    bridges=None,
    setup_axes=(),
):
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
        setting_scope = scope.setting_scopes.get(key, "shared")
        if setting_scope != "shared" and setting_scope != role:
            continue
        selected = selected_settings[key] if isinstance(value, list) else str(value)
        if _known_unit_setting(key, selected):
            # Each operand's actual explicit scale is checked below after the
            # exact table/prose binding, including conflicting unit fields.
            continue
        if bridges and bridges[role].get(key) == selected:
            explicit_axes = setup_axes if setting_scope != "shared" else table_labels
            if (
                axes is not None
                and any(_has_label(selected, value) for value in explicit_axes)
                and not _has_label(selected, axes)
            ):
                raise ValueError(f"Source bridge cannot override explicit setting axis {key}")
            continue
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


def _numeric_scope_check(
    claim,
    condition,
    scope,
    decision,
    materials,
    *,
    catalog=None,
    mode: Literal["support", "concern"] = "support",
    require_comparison: bool = False,
    binding_issues: list | None = None,
) -> list[str]:
    """Bind every cited pair; support additionally proves all cases and the relation."""
    reasons = []
    if scope.relation == "none":
        if _asserted_endpoints(claim, condition, materials):
            reasons.append("An original endpoint assertion requires a grounded numerical comparison")
        if require_comparison or (
            mode == "concern"
            and (
                scope.comparator
                or decision.comparisons
                or decision.comparison_objects != "not_comparative"
                or any(key in condition.settings for key in ("reference", "control"))
            )
        ):
            reasons.append("This statistical concern requires an explicit numerical comparison relation")
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
    # Complete support covers every setting. A concern can identify one affected
    # setting, while every selected case must still belong to this condition.
    dimensions = {
        key: [str(value) for value in values]
        for key, values in condition.settings.items()
        if isinstance(values, list)
    }
    expected_settings = set(product(*dimensions.values())) if dimensions else set()
    observed_settings = []
    cases = [item.case for item in decision.comparisons]
    if mode == "support":
        if len(cases) != len(set(cases)) or set(cases) != expected_cases:
            reasons.append("Numerical checks must cover each required comparison case exactly once")
    elif not cases or len(cases) != len(set(cases)) or not set(cases).issubset(expected_cases):
        reasons.append("A statistical concern needs distinct, grounded comparison cases from this condition")
    for comparison_index, comparison in enumerate(decision.comparisons):
        try:
            bridges = _bridge_bindings(claim, condition, scope, comparison, materials, catalog)
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
                role_field = bridges[role].get("role")
                if not declared or (label != declared and not role_field):
                    raise ValueError(f"Numerical operand is not bound to the claim's {role}")
                for key in setting_keys:
                    value = condition.settings.get(key)
                    if value is not None and not isinstance(value, (dict, list)) and label != str(value):
                        if role_field != f"settings.{key}":
                            raise ValueError(f"Numerical operand changes condition {role} setting {key}")
            if comparison.relation != scope.relation:
                raise ValueError("Numerical comparison weakens or changes the claim's relation")
            captions = {"subject": None, "comparator": None}
            if catalog is not None and "joint_view" in catalog:
                for role, number, cell, identifier in (
                    ("subject", comparison.left, comparison.left_cell, comparison.left_catalog_id),
                    ("comparator", comparison.right, comparison.right_cell, comparison.right_catalog_id),
                ):
                    if identifier and cell is not None:
                        selected = resolve_cell(catalog, identifier, materials)
                        if (number.block_id, number.quote, number.token) != (
                            selected["block_id"],
                            selected["quote"],
                            selected["token"],
                        ) or (cell.table, cell.row, cell.column) != (
                            selected["table"],
                            selected["row"],
                            selected["column"],
                        ):
                            raise ValueError("Caption association differs from the selected numerical cell")
                        # resolve_cell revalidates this table's complete authorized
                        # caption and member artifact; keep the body quote intact.
                        captions[role] = selected["caption"]
            context = "\n".join(
                [*(p.quote for p in comparison.context), *(value for value in captions.values() if value)]
            )
            for passage in comparison.context:
                _paper_pointer(materials, passage.block_id, passage.quote)
            if condition.dataset and not _has_label(condition.dataset, context):
                raise ValueError("Numerical context does not identify the claimed dataset")
            if not condition.metric or (
                not _has_label(condition.metric, context)
                and not all("metric" in bridges[role] for role in bridges)
            ):
                raise ValueError("Numerical context does not identify the exact claimed metric")
            if comparison.metric_label != condition.metric:
                raise ValueError("Numerical comparison changes the claimed metric")
            for key, value in condition.settings.items():
                candidates = [settings[key]] if isinstance(value, list) else [str(value)]
                if all(_known_unit_setting(key, candidate) for candidate in candidates):
                    continue
                if not all(_has_label(candidate, context) for candidate in candidates) and not any(
                    key in value for value in bridges.values()
                ):
                    raise ValueError(f"Numerical context does not bind condition setting {key}")
            values = []
            for number, label, cell, role in (
                (comparison.left, comparison.left_label, comparison.left_cell, "subject"),
                (comparison.right, comparison.right_label, comparison.right_cell, "comparator"),
            ):
                value = _number(materials, number)
                if cell is not None:
                    axes, caption, table_labels = _table_value(
                        number,
                        cell,
                        label,
                        bridges[role].get("metric", comparison.metric_label),
                        settings,
                        metric_binding="metric" in bridges[role],
                        dataset=condition.dataset,
                        caption=captions[role],
                    )
                    # A named variant's setup may be defined separately. Other
                    # model-row names are not alternative setting headers.
                    grid = _TableGrid(number.quote).tables[cell.table]
                    row_label_columns = [
                        c
                        for (r, c), text in grid.items()
                        if r == cell.row and c < cell.column and _has_label(label, text)
                    ]
                    column_label_rows = [
                        r
                        for (r, c), text in grid.items()
                        if c == cell.column and r < cell.row and _has_label(label, text)
                    ]
                    setup_axes = tuple(
                        text
                        for (r, c), text in grid.items()
                        if (row_label_columns and c > max(row_label_columns))
                        or (not row_label_columns and column_label_rows and r > max(column_label_rows))
                    )
                    _side_settings(
                        condition,
                        scope,
                        settings,
                        axes + " " + caption,
                        axes=axes,
                        table_labels=table_labels,
                        role=role,
                        bridges=bridges,
                        setup_axes=setup_axes,
                    )
                    values.append(value)
                    continue
                local = number.value_context or number.quote
                if comparison.left_number_id and comparison.right_number_id:
                    # _prose_bindings already matched exact occurrence offsets to
                    # the directly scoped subject-predicate grammar on both sides.
                    values.append(value)
                    continue
                if (
                    local not in number.quote
                    or not _has_label(label, local)
                    or not _has_label(comparison.metric_label, local)
                ):
                    raise ValueError("A numerical side lacks its exact subject/comparator and metric context")
                _side_settings(condition, scope, settings, local, role=role, bridges=bridges)
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
                _occurrence_units(catalog, identifier, materials)
                if identifier
                else (
                    _table_units(
                        number,
                        cell,
                        label,
                        bridges[role].get("metric", comparison.metric_label),
                        condition.dataset,
                        metric_binding="metric" in bridges[role],
                        caption=captions[role],
                        caption_settings={
                            key: settings[key] if isinstance(value, list) else str(value)
                            for key, value in condition.settings.items()
                            if key
                            not in {
                                "unit",
                                "units",
                                "model",
                                "method",
                                "subject",
                                "comparison",
                                "comparator",
                                "baseline",
                                scope.subject_setting,
                                scope.comparator_setting,
                            }
                            and scope.setting_scopes.get(key, "shared") in {role, "shared"}
                        },
                    )
                    if cell is not None
                    else _comparison_units(number, comparison.metric_label)
                )
                for number, role, identifier, cell, label in (
                    (
                        comparison.left,
                        "subject",
                        comparison.left_number_id,
                        comparison.left_cell,
                        comparison.left_label,
                    ),
                    (
                        comparison.right,
                        "comparator",
                        comparison.right_number_id,
                        comparison.right_cell,
                        comparison.right_label,
                    ),
                )
            ]
            if any(len(unit) > 1 for unit in units) or units[0] != units[1]:
                raise ValueError("Comparison units/scales do not match")
            for key, value in condition.settings.items():
                expected = settings[key] if isinstance(value, list) else str(value)
                if _known_unit_setting(key, expected) and any(
                    actual != {_canonical_unit(expected)} for actual in units
                ):
                    raise ValueError(f"Explicit operand units do not match condition setting {key}")
            left, right = values
            asserted = _asserted_endpoints(claim, condition, materials)
            if asserted and values != [float(token.rstrip("%")) for token in asserted]:
                raise ValueError(
                    "Selected values do not match the current condition's asserted from/to endpoints"
                )
            if (
                scope.endpoint_required
                or comparison.expected_left is not None
                or comparison.expected_right is not None
            ):
                if asserted is None:
                    raise ValueError("The condition has no uniquely bound asserted endpoint pair")
                if comparison.expected_left is None or comparison.expected_right is None:
                    raise ValueError("A from/to assertion requires both grounded endpoints")
                expected_values = []
                for expected in (comparison.expected_left, comparison.expected_right):
                    if not any(
                        expected.block_id == block_id and expected.quote in quote
                        for block_id, quote in _claim_passages(claim, condition.id)
                    ):
                        raise ValueError("Expected endpoint belongs to another claim condition")
                    expected_values.append(_number(materials, expected))
                if values != expected_values:
                    raise ValueError("Selected values do not match the asserted subject/comparator endpoints")
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
                if comparison.difference_number_id:
                    if comparison.difference != _assertion_number(
                        catalog, comparison.difference_number_id, materials
                    ):
                        raise ValueError("Expected difference differs from its original numbered occurrence")
                    difference_units = _occurrence_units(
                        catalog, comparison.difference_number_id, materials, assertion=True
                    )
                else:
                    difference_units = _comparison_units(comparison.difference, comparison.metric_label)
                record_assertion(catalog, materials, comparison.difference, purpose="asserted_difference")
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
            if mode == "support" and not holds:
                raise ValueError(f"Numerical relation {relation} is false for {left} and {right}")
        except ValueError as exc:
            if binding_issues is not None and isinstance(exc, BindingContractError):
                binding_issues.append({**exc.record(), "comparison_index": comparison_index})
            reasons.append(f"{comparison.case or condition.id}: {exc}")
    if dimensions:
        observed = set(observed_settings)
        if mode == "support" and (len(observed_settings) != len(observed) or observed != expected_settings):
            reasons.append(
                "Numerical checks do not cover every joint combination of listed condition settings"
            )
        elif mode == "concern" and (
            len(observed_settings) != len(observed) or not observed.issubset(expected_settings)
        ):
            reasons.append("Statistical comparisons select duplicate or unknown joint condition settings")
    return reasons


def _decode_scope(response, claim, materials, output, catalog, joint_catalogs=None, *, diagnostics=None):
    """One invalid condition keeps its own evidence unconfirmed; other conditions survive."""
    version = response.get("schema_version")
    if "schema_version" in response and (
        not isinstance(version, str) or version not in {"catalog-v1", "catalog-v2"}
    ):
        raise ValueError("Unknown experiment scope schema_version")
    catalog_mode = version in {"catalog-v1", "catalog-v2"}
    if (
        set(response) - {"schema_version", "conditions", "items"}
        or not isinstance(response.get("conditions"), list)
        or not isinstance(response.get("items"), list)
    ):
        raise ValueError("Scope review must return conditions/items lists and the declared schema")
    allowed = {condition.id: condition for condition in claim.conditions}
    errors, invalid, conditions, decisions, locations = [], set(), {}, {}, []
    hard_pairs, typed_pairs = set(), set()

    def reject(identifier, message):
        errors.append(f"{identifier}: {message}")
        # NonEmpty normalizes surrounding whitespace. Use the same identity
        # even when another field makes the whole row fail validation, so a
        # malformed duplicate cannot preserve or later restore its first row.
        canonical = identifier.strip() if isinstance(identifier, str) else None
        if canonical in allowed:
            invalid.add(canonical)

    for raw in response["conditions"]:
        identifier = raw.get("condition_id") if isinstance(raw, dict) else None
        try:
            row = (CatalogConditionScope if catalog_mode else ConditionScope).model_validate(raw)
            if row.condition_id not in allowed or row.condition_id in conditions:
                raise ValueError("Scope condition ID is unknown or duplicated")
            if catalog_mode:
                row.claim_quote = row.claim_quote or claim.text
                row.required_cases = [case["id"] for case in catalog["conditions"][row.condition_id]["cases"]]
                if not row.required_cases:
                    raise ValueError("Condition has no finite catalog cases")
                condition = allowed[row.condition_id]
                roles = _condition_roles(condition, row)
                if transition_endpoints(claim, condition, materials) is not None:
                    if (
                        roles["comparator_setting"] not in {"", "method"}
                        or roles["subject_setting"] != "method"
                    ):
                        raise ValueError(
                            "Named-setting transition conflicts with the explicit comparison roles"
                        )
                    roles["comparator_setting"] = "method"
                row = ConditionScope(
                    **row.model_dump(),
                    **roles,
                    endpoint_required=bool(_asserted_endpoints(claim, condition, materials)),
                )
                for role in ("subject", "comparator"):
                    key = getattr(row, f"{role}_setting")
                    if key:
                        value = _field(allowed[row.condition_id], f"settings.{key}")
                        if not isinstance(value, list):
                            setattr(row, role, str(value))
            passages = [claim.text, *(quote for _, quote in _claim_passages(claim, row.condition_id))]
            if not any(row.claim_quote in quote for quote in passages):
                raise ValueError("Scope review claim quote is not an exact part of the current claim")
            if row.credited_component and not any(row.credited_component in quote for quote in passages):
                raise ValueError("Credited component is absent from the current claim")
            for role in ("subject", "comparator"):
                label, key = getattr(row, role), getattr(row, f"{role}_setting")
                if label and not key and not any(_has_label(label, quote) for quote in passages):
                    raise ValueError(f"Scope {role} is absent from the current claim source")
            conditions[row.condition_id] = row
        except (ValueError, TypeError) as exc:
            reject(identifier, str(exc))
    for identifier in allowed.keys() - conditions.keys():
        reject(identifier, "Scope review must cover each claim condition exactly once")
    expected = {(index, condition) for index, item in enumerate(output.items) for condition in item.covered}
    joint_catalogs = joint_catalogs or {}
    seen_pairs, invalid_pairs = set(), set()

    def reject_item(key, identifier, message, *, error=None, missing=False):
        if key in expected and key[0] in joint_catalogs:
            if isinstance(error, BindingContractError):
                typed_pairs.add(key)
            elif not (missing and key in typed_pairs):
                hard_pairs.add(key)
            invalid_pairs.add(key)
            errors.append(f"joint {key}: {message}")
            joint_catalogs[key[0]]["joint_view"]["errors"].append(message)
        else:
            reject(identifier, message)

    for position, raw in enumerate(response["items"]):
        identifier = raw.get("condition_id") if isinstance(raw, dict) else None
        index = raw.get("item_index") if isinstance(raw, dict) else None
        key = (index, identifier.strip()) if type(index) is int and isinstance(identifier, str) else None
        duplicate = key in seen_pairs
        if key in expected:
            seen_pairs.add(key)
        try:
            if duplicate:
                raise ValueError("Scope candidate/condition pair is duplicated and cannot be restored")
            item_schema = (
                CatalogItemScopeV2
                if version == "catalog-v2"
                else (CatalogItemScope if catalog_mode else ItemScope)
            )
            row = item_schema.model_validate(raw)
            parsed_key = (row.item_index, row.condition_id)
            if row.item_index in joint_catalogs and key != parsed_key:
                raise ValueError(
                    "Joint item_index must be an original non-boolean integer with a canonical covered condition"
                )
            key = parsed_key
            if key not in expected or key in decisions or key in invalid_pairs:
                raise ValueError("Scope candidate/condition pair is unknown or duplicated")
            item_catalog = joint_catalogs.get(row.item_index, catalog)
            if row.item_index in joint_catalogs:
                if version != "catalog-v2":
                    raise ValueError("Joint candidates require the catalog-v2 source_uses contract")
                if item_catalog["joint_view"]["errors"]:
                    raise ValueError(
                        "Joint source manifest is invalid: " + "; ".join(item_catalog["joint_view"]["errors"])
                    )
                validate_source_uses(
                    item_catalog,
                    row,
                    materials,
                    own_table_reference=_own_table_reference,
                    has_label=_has_label,
                    condition=allowed[row.condition_id],
                )
            elif isinstance(row, CatalogItemScopeV2) and row.source_uses:
                raise ValueError("Single-source candidates cannot claim joint source_uses")
            if catalog_mode:
                if row.condition_id not in conditions:
                    raise ValueError("Catalog operand has no validated condition scope")
                row = _catalog_item(row, claim, materials, item_catalog, conditions[row.condition_id])
            for passage in row.grounds:
                _paper_pointer(materials, passage.block_id, passage.quote)
            decisions[key] = row
            locations.append(
                {
                    "candidate_index": row.item_index,
                    "condition_id": row.condition_id,
                    "response_pointer": f"/response/items/{position}",
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            reject_item(key, identifier, str(exc), error=exc)
    for key in expected - decisions.keys():
        reject_item(
            key, key[1], "Scope review must cover every candidate/condition pair exactly once", missing=True
        )
    conditions = {key: row for key, row in conditions.items() if key not in invalid}
    decisions = {
        key: row for key, row in decisions.items() if key[1] not in invalid and key not in invalid_pairs
    }
    if diagnostics is not None:
        diagnostics.update(hard_condition_ids=set(invalid), hard_pairs=hard_pairs, typed_pairs=typed_pairs)
    return conditions, decisions, errors, locations


def _bound_prose_pair_choices(claim, materials, catalog):
    """Prepare strict structural input choices; never issue a support/applicability decision."""
    choices = {condition.id: [] for condition in claim.conditions}
    source_coverage = {}
    for source in catalog["sources"].values():
        if source["kind"] == "claim_source":
            key = (source["block_id"], source["start"], source["end"])
            source_coverage.setdefault(key, set()).update(source["covered"])
    for condition in claim.conditions:
        try:
            scope = ConditionScope(
                condition_id=condition.id,
                claim_quote=claim.text,
                assertion="unresolved",
                matched_controls_required=False,
                uncertainty_sensitive=False,
                relation="eq",
                subject="",
                comparator="",
                rationale="Structural selector validation only; semantic applicability is unassessed.",
                required_cases=[case["id"] for case in catalog["conditions"][condition.id]["cases"]],
            )
            roles = _condition_roles(condition, scope)
            transition = transition_endpoints(claim, condition, materials) is not None
            if transition:
                if roles["subject_setting"] != "method" or roles["comparator_setting"] not in {"", "method"}:
                    continue
                roles["comparator_setting"] = "method"
            if not all(roles.values()):
                continue
            endpoints = _asserted_endpoints(claim, condition, materials)
            for case in catalog["conditions"][condition.id]["cases"]:
                settings = {**condition.settings, **case["settings"]}
                subject = settings[roles["subject_setting"]]
                comparator = settings[roles["comparator_setting"]]
                if not isinstance(subject, str) or not isinstance(comparator, str):
                    continue
                canonical = _prose_setting_scopes(
                    settings, roles["subject_setting"], roles["comparator_setting"], transition
                )
                for pair in direct_pair_candidates(
                    materials,
                    catalog["numbers"],
                    dataset=condition.dataset,
                    metric=condition.metric,
                    settings=settings,
                    left_label=subject,
                    right_label=comparator,
                    transition=transition,
                ):
                    # Identical source ranges may explicitly cover several conditions.
                    # Distinct overlapping ranges retain their own coverage boundary.
                    if any(
                        condition.id not in covered
                        and block_id == record["block_id"]
                        and start < record["end"]
                        and end > record["start"]
                        for (block_id, start, end), covered in source_coverage.items()
                        for record in (pair["left"], pair["right"])
                    ):
                        continue
                    left, right = (float(pair[side]["token"].rstrip("%")) for side in ("left", "right"))
                    relation = "lt" if left < right else ("gt" if left > right else "eq")
                    binding_scope = scope.model_copy(
                        update={
                            **roles,
                            "subject": subject,
                            "comparator": comparator,
                            "setting_scopes": canonical,
                            "relation": relation,
                            "endpoint_required": bool(endpoints),
                        }
                    )
                    comparison = ScopeComparison(
                        case=case["id"],
                        settings=case["settings"],
                        left=_occurrence_number(catalog, pair["left"]["number_id"], materials),
                        right=_occurrence_number(catalog, pair["right"]["number_id"], materials),
                        left_label=subject,
                        right_label=comparator,
                        metric_label=condition.metric,
                        relation=relation,
                        left_number_id=pair["left"]["number_id"],
                        right_number_id=pair["right"]["number_id"],
                        context=[
                            ScopePassage(block_id=pair["left"]["block_id"], quote=pair["left"]["quote"])
                        ],
                        expected_left=_expected_number(
                            claim, condition.id, endpoints[0] if endpoints else "", materials
                        ),
                        expected_right=_expected_number(
                            claim, condition.id, endpoints[1] if endpoints else "", materials
                        ),
                    )
                    decision = ItemScope(
                        item_index=0,
                        condition_id=condition.id,
                        applicability="unverified",
                        grounds=comparison.context,
                        rationale=scope.rationale,
                        comparison_objects="matched",
                        comparisons=[comparison],
                    )
                    # The common numerical gate checks exact roles, every case
                    # setting, units, source locations and original endpoints.
                    # Concern mode permits one case of a multi-case condition;
                    # neither this unresolved scope nor decision is emitted.
                    if _numeric_scope_check(
                        claim, condition, binding_scope, decision, materials, catalog=catalog, mode="concern"
                    ):
                        continue
                    choices[condition.id].append(
                        {
                            "structural_only": True,
                            "case_id": case["id"],
                            "binding_type": pair["kind"],
                            "left": {"kind": "prose", "number_id": pair["left"]["number_id"]},
                            "right": {"kind": "prose", "number_id": pair["right"]["number_id"]},
                            "subject": {
                                "setting_key": roles["subject_setting"],
                                "value": subject,
                                **({"to_setting": settings["to_setting"]} if transition else {}),
                            },
                            "comparator": {
                                "setting_key": roles["comparator_setting"],
                                "value": comparator,
                                **({"from_setting": settings["from_setting"]} if transition else {}),
                            },
                            "canonical_setting_scopes": canonical,
                            "result_sentence_id": sentence_id(pair["left"]),
                            "condition_source_ids": [
                                key
                                for key, source in catalog["sources"].items()
                                if source["kind"] == "claim_source" and condition.id in source["covered"]
                            ],
                        }
                    )
        except (ValueError, TypeError, KeyError):
            # An unavailable input hint must not alter candidate judgments or
            # replace the original strict scope response validation.
            choices[condition.id] = []
    return choices


def _probe_joint_binding(claim, materials, raw, scope, bounded):
    """Revalidate one pristine pair without discarding any non-binding failure."""
    typed, blockers, decision = [], [], None
    try:
        row = CatalogItemScopeV2.model_validate(raw)
        condition = next(c for c in claim.conditions if c.id == row.condition_id)
        try:
            validate_source_uses(
                bounded,
                row,
                materials,
                own_table_reference=_own_table_reference,
                has_label=_has_label,
                condition=condition,
            )
        except BindingContractError as exc:
            typed.append(exc.record())
        decision = _catalog_item(row, claim, materials, bounded, scope)
        for passage in decision.grounds:
            _paper_pointer(materials, passage.block_id, passage.quote)
        numerical = []
        reasons = _numeric_scope_check(
            claim,
            condition,
            scope,
            decision,
            materials,
            catalog=bounded,
            binding_issues=numerical,
        )
        typed.extend(numerical)
        # Each typed numerical exception contributes exactly one reason. Any
        # additional reason remains a blocker, without classifying its text.
        if len(reasons) != len(numerical):
            blockers.extend(reasons)
        revalidate_members(bounded, materials)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        blockers.append(str(exc))
    return {"binding_issues": typed, "blocking_errors": blockers, "decision": decision}


def _scope_review(
    claim,
    materials,
    output,
    *,
    call,
    catalog,
    joint_catalogs=None,
    binding_repair_rounds=0,
    projection_reviews=None,
):
    """Independent semantic review; its grounded structural/numeric gates fail closed."""
    from common import run_stats

    joint_catalogs = joint_catalogs or {}
    payload = {
        "claim": claim.model_dump(mode="json"),
        "paper_blocks": [block.model_dump() for block in materials.blocks],
        "candidate_items": [item.model_dump() for item in output.items],
        "candidate_plans": [plan.model_dump(mode="json") for plan in output.plans],
        "single_source_candidate_indices": [
            index for index, item in enumerate(output.items) if not item.additional_sources
        ],
        "catalog": catalog_prompt(catalog),
        "joint_candidates": {
            str(index): {"manifest": bounded["joint_view"], "catalog": catalog_prompt(bounded)}
            for index, bounded in joint_catalogs.items()
        },
        "bound_prose_pair_choices": _bound_prose_pair_choices(claim, materials, catalog),
        "scope_field_contract": {
            "setting_scopes": "Use only bare keys from this condition.settings; dataset, metric and settings.KEY are not valid keys.",
            "allowed_setting_keys": {
                condition.id: list(condition.settings) for condition in claim.conditions
            },
            "bridge_condition_field": "SourceBridge.condition_field uses metric or settings.KEY, unlike setting_scopes.",
        },
        "output_schema": CatalogScopeReviewV2.model_json_schema(),
    }
    if any(target.projection is not None for plan in output.plans for target in plan.targets):
        from verification.execution_projection import projection_context

        try:
            payload["execution_projection_context"] = projection_context(claim, materials)
            from verification.execution_projection_semantics import semantic_request_context

            payload["execution_projection_semantics_context"] = semantic_request_context(claim, materials)
            from verification.execution_projection import digest

            payload["execution_projection_semantics_context"]["proposals"] = [
                {
                    "plan_index": index,
                    "condition_id": target.condition_id,
                    "proposal_sha256": digest(
                        target.projection.model_dump(mode="json")
                        if hasattr(target.projection, "model_dump")
                        else target.projection
                    ),
                }
                for index, plan in enumerate(output.plans)
                for target in plan.targets
                if target.projection is not None
            ]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            payload["execution_projection_context"] = {"unavailable": str(exc)}
    audit = {"claim_id": claim.id, "input": copy.deepcopy(payload)}
    stats = run_stats.stats_path()
    audit_path = stats.parent / "experiment_scope" / f"{uuid.uuid4().hex}.json" if stats is not None else None
    try:
        response = ask(
            "Independently audit experiment candidate applicability and ENTIRE condition support. "
            "Do not defer to candidate detail or fully_supported_conditions. Return output_schema JSON, "
            "one conditions entry per claim condition and exactly one items entry per candidate item_index/covered condition. "
            "Review each candidate's ENTIRE item.quote; its optional comparison subquotes do not narrow that primary quote. "
            "For an explicit additional_sources candidate, independently review the joint set of primary and additional exact sources for its ONE condition. Use only its joint_candidates[item_index].catalog IDs for grounds, cells, numbers, contexts and bridges. Global or another candidate's IDs are invalid. "
            "For joint difference_number_id use only that candidate's assertion_numbers, which are the current condition's original assertion occurrences. These assertion selectors do not authorize observations, context, bridges or source_uses. Expected endpoints remain program-derived from original condition-scoped assertions and need not be declared supporting members. "
            "Only for those joint candidates, return source_uses exactly once for each member_source_id, including the primary, with its roles and rationale. For every single_source_candidate_indices entry, additional_sources is empty and source_uses MUST be []; this also applies when that single-source review uses grounds or bridges. Use the global catalog for its normal grounds/comparisons; do not put its primary source in source_uses. result/metric_definition/setup_definition/table_reference roles must actually feed the corresponding number/cell/bridge; protocol and other_qualifier uses require grounded semantic review and do not prove numerical bindings. Do not combine or upgrade other partial candidates. Invalid manifests remain unverified. "
            "Use schema_version catalog-v2. Keep condition IDs exactly as supplied. Leave claim_quote empty: its existing identity is retained. "
            "Sources, cells, numbers and cases are program-generated choices. Source/cell/number arrays follow source_fields/cell_fields/number_fields in catalog; axes and original sentences are shared by ID. Number ordinal is its one-based position among numeric occurrences in that sentence. "
            "Distinguish descriptive reports (reported scores, hardware or duration) from controlled comparisons, "
            "causal component attribution and statistical generalization. A leaderboard score/gap alone does not "
            "assert matched training data/budget/tuning or causal superiority. Missing ablations apply only to "
            "an actually credited component; do not demand hardware-count ablations for a resource report. "
            "Missing variance affects the claim only when uncertainty could overturn an asserted comparison or "
            "generalization; keep large-gap and descriptive-report limitations as notes. "
            "Set full_support only when every qualifier, subject, exact comparator, dataset, metric, setting and "
            "quantifier is established. A tie cannot establish strict improvement/degradation. A lower-ranked "
            "baseline cannot establish a comparison against the top system. Record unresolved_qualifiers explicitly. "
            "For numerical comparisons choose case_id from the current condition, one comparison per supplied case; do not invent case names or settings. "
            "Each left/right operand is either {kind:cell, cell_id, label_cell_id} on its exact table axis or {kind:prose, number_id} from the original numeric occurrence catalog. Do not copy numbers, offsets, contexts or tables. HTML tables require cell selectors. "
            "For prose choose the directly named method's result occurrence, with a shared dataset/split prefix governing both predicates in that same sentence; ambiguous scope and contrastive role mentions remain unverified. "
            "Prefer bound_prose_pair_choices for the exact condition/case: copy both left/right selector objects from one row, and inspect its result_sentence_id plus condition_source_ids. Do not substitute repeated endpoint numbers from the following assertion sentence. These choices establish structural binding only; independently assess applicability and ENTIRE condition support without upgrading partial items. "
            "setting_scopes keys are BARE condition.settings keys (split, unit, method, from_setting, to_setting), never dataset, metric or settings.KEY. A structural pair choice supplies canonical_setting_scopes; preserve its exact keys and roles. SourceBridge.condition_field separately uses metric or settings.KEY. "
            "A named same-method transition requires original from_setting/to_setting fields and its exact from/to assertion: left selects the to-setting result, right the from-setting result. Preserve these setting scopes. "
            "left is the claim subject; right is its comparator. Structured role settings are derived from the condition by the program; do not emit subject_setting/comparator_setting. "
            "Leave subject/comparator empty when existing model/method/ablation and comparison/baseline fields identify them. Otherwise provide exact role phrases from the claim source. A training dataset or procedure is a separate setup qualifier. "
            "An extended table row name can bind to its condition field through a subject/comparator bridge with exact source IDs. "
            "Bridge condition_field uses metric or settings.KEY, and table_id must identify the selected cell's actual table. "
            "Metric bridges require the original quantity label, its dataset definition, and a separate passage referencing that specific table. "
            "paper_label for a metric bridge is the measured quantity in its definition (such as mIoU or top-1 accuracy), never the dataset header ADE20K/ImageNet or a joined axis label. "
            "Table reference_source_ids lists exact numbered-reference candidates to inspect and explicitly include when applicable; a candidate alone does not establish semantic support. "
            "For composite metric names explain the complete original name in explanation, preserving the quantity and dataset. Explicit metric/dataset axes always win. "
            "Setup bridges mark applies_to subject/comparator/shared to match condition setting_scopes; a treatment applied only to the subject must not be required of its baseline. "
            "Operand-specific setting bridges must give paper_label: the exact treatment label on that selected row/column and in its source definition. "
            "Use current-condition original sources or explicit definitions, plus the named table reference; mere unrelated co-occurrence cannot prove a bridge. "
            "For numerical from/to assertions use lt/gt as appropriate. Asserted endpoints are derived automatically from the claim and condition and checked against original sources; no expected endpoint or endpoint_required fields are accepted. "
            "A from/to statement does not assert a separately quoted difference: never invent a delta. Only explicit claimed gaps use relation=difference and difference_number_id selected from the current condition's original source assertion. "
            "difference_mode must explicitly distinguish absolute deltas, relative_percent and percentage_points. "
            "For a stated positive increase/decrease magnitude, set condition difference_direction and copy its exact "
            "source qualifier into difference_direction_quote; decrease uses comparator minus subject, with the comparator "
            "as the relative baseline. The default signed mode preserves subject minus comparator. Keep operand identities fixed. "
            "Percentage points require explicit percent-scaled inputs; do not infer units from metric names. Preserve expected difference units. "
            "Context passages must identify the exact condition dataset/metric/settings and common units. "
            "Multi-value tables without uniquely bound values remain partial. Non-numerical descriptive support "
            "does not require invented numerical comparisons. Explain applicability with exact grounds; uncertainty "
            "or missing review information must remain unverified, without creating a new flaw. Place limitations that do not block the stated result in nonblocking_notes; "
            "unresolved_qualifiers is reserved for actual missing claim requirements. A complete condition can combine its table and connected definition/setup passages. "
            "For each non-null plan target projection independently return one plan_projection_reviews decision, even when candidate_items is empty. "
            "execution_projection_context.request_choices is a versioned structural menu under each config. Its roles, source lexical hits and prose candidates do not establish applicability or full support. Review the entire original claim and all original fields; do not inherit a decision from a catalog hint. "
            "Keep these decisions separate from paper support. Review the unchanged whole claim and condition, every original field path, "
            "its selected exact paper sources and released repository definition. Confirm only an absolute fixed released-prediction "
            "exact-match fraction whose dataset/split, model, complete sample scope and every qualifier are retained. "
            "Inference/training provenance, comparisons, population generalizations, filtering, unknown runtime obligations or "
            "unsupported definitions must remain unresolved. Source IDs must identify each required original passage; field path "
            "coverage alone is insufficient. Configuration/code supplied for plans grants no additional paper-support/joint access. "
            "For released-predictions-v2, use the matching versioned plan review schema and the exact proposal_sha256 supplied in execution_projection_semantics_context.proposals. Independently review each unchanged atom, field binding and claim span; keep each atom's exact unique source ID set and every original mapping. The separate v2 context does not alter v1 decisions or grant paper/joint access. All original text must satisfy the finite recipe and retain its meaning; unknown predicates, inference, derived metrics and positive generalizations remain unresolved even if positions are covered. "
            "Plans without a projection need no projection decision. Do not append, rewrite or merge candidates.",
            payload,
            module="verification.experiments.scope",
            call=call,
        )
        audit["response"] = copy.deepcopy(response)
        pristine = copy.deepcopy(joint_catalogs) if binding_repair_rounds else {}
        diagnostics = {}
        paper_response = {key: value for key, value in response.items() if key != "plan_projection_reviews"}
        conditions, decisions, errors, locations = _decode_scope(
            paper_response, claim, materials, output, catalog, joint_catalogs, diagnostics=diagnostics
        )
        if projection_reviews is not None:
            from verification.execution_projection import decode_projection_reviews

            plan_decisions, plan_errors = decode_projection_reviews(response, output.plans)
            if response.get("schema_version") != "catalog-v2":
                plan_decisions, plan_errors = (
                    {},
                    ["Execution projections require the catalog-v2 independent review"],
                )
            projection_reviews.update(plan_decisions)
            audit["plan_projection_errors"] = plan_errors
        audit["validated"] = not errors
        audit["item_locations"] = locations
        audit["binding_errors"] = errors
        audit["resolved_items"] = [row.model_dump(mode="json") for row in decisions.values()]
        if binding_repair_rounds:
            try:
                accepted, repair_audit = repair_bindings(
                    claim=claim,
                    output=output,
                    response=response,
                    scopes=conditions,
                    pristine=pristine,
                    diagnostics=diagnostics,
                    call=call,
                    probe=lambda raw, scope, bounded: _probe_joint_binding(
                        claim, materials, raw, scope, bounded
                    ),
                )
            except Exception as exc:
                from llm.diagnostics import redact_provider_details

                accepted = {}
                repair_audit = {
                    "status": "failed",
                    "outcomes": [],
                    "error": redact_provider_details(f"{type(exc).__name__}: {exc}"),
                }
            audit["binding_repair"] = repair_audit
            for key, (decision, bounded, position) in accepted.items():
                decisions[key] = decision
                joint_catalogs[key[0]] = bounded
                locations[:] = [
                    row for row in locations if (row["candidate_index"], row["condition_id"]) != key
                ]
                locations.append(
                    {
                        "candidate_index": key[0],
                        "condition_id": key[1],
                        "response_pointer": f"/binding_repair/effective_response/items/{position}",
                    }
                )
            if repair_audit["status"] != "not_needed":
                audit["binding_repair"]["resolved_items"] = [
                    row.model_dump(mode="json") for row in decisions.values()
                ]
                # Preserve the original errors even when a new explicit patch
                # passes. They remain independently inspectable in the audit.
                repair_errors = [
                    message
                    for outcome in repair_audit["outcomes"]
                    for message in outcome.get("validation_errors", [])
                ]
                errors = [
                    *errors,
                    f"Binding repair {repair_audit['status']}: {len(accepted)} pair(s) accepted",
                    *repair_errors,
                ]
        prefix = (
            "Original scope diagnostics and binding repair history: "
            if binding_repair_rounds and audit["binding_repair"]["status"] != "not_needed"
            else "Experimental scope review unconfirmed: "
        )
        issue = prefix + "; ".join(errors) if errors else None
        return conditions, decisions, issue, audit_path
    except Exception as exc:
        if projection_reviews is not None:
            projection_reviews.clear()
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


def _plan(
    claim: Claim,
    materials: SharedMaterials,
    candidate: PlanCandidate,
    *,
    projection_reviews=None,
    scope_audit=None,
    plan_index=0,
) -> ExecutionPlan:
    conditions = {condition.id: condition for condition in claim.conditions}
    ids = _covered(claim, [target.condition_id for target in candidate.targets])
    values = {target.condition_id: _number(materials, target.reported) for target in candidate.targets}
    target_issues = []
    target_bindings = {}
    for target in candidate.targets:
        condition = conditions[target.condition_id]
        if target.projection is not None:
            from verification.execution_projection import bind_projection_target

            try:
                if len(candidate.targets) != 1:
                    raise ValueError(
                        "The first released-predictions recipe covers one original condition per plan"
                    )
                review = (projection_reviews or {}).get((plan_index, condition.id))
                if review is None:
                    raise ValueError("No healthy independent scope decision for this plan projection")
                target_bindings[target.condition_id] = bind_projection_target(
                    claim,
                    condition,
                    target.reported,
                    materials,
                    selector=target.selector,
                    proposal=target.projection,
                    entry_script=candidate.entry_script,
                    config_path=candidate.config,
                    review=review,
                    audit_path=scope_audit,
                )
            except (ValueError, OSError, TypeError, KeyError, IndexError, AttributeError) as exc:
                target_issues.append(f"Execution projection unavailable for {condition.id}: {exc}")
            continue
        # Exact quotes must identify the target metric/dataset, possibly across
        # a full table containing the header and dataset row.
        if not _has_label(condition.metric, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's metric")
        if condition.dataset and not _has_label(condition.dataset, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's dataset")
        try:
            target_bindings[target.condition_id] = bind_execution_target(
                claim, condition, target.reported, materials, selector=target.selector
            )
        except TargetBindingError as exc:
            ambiguity = _target_ambiguity(claim, target, values[target.condition_id])
            detail = f"{ambiguity}; {exc}" if ambiguity else str(exc)
            target_issues.append(f"Unresolved paper target {target.condition_id}: {detail}")
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
    released_predictions = (
        bool(target_bindings)
        and all(binding.version == 2 for binding in target_bindings.values())
        and len(target_bindings) == len(candidate.targets)
    )
    if candidate.run_mode == "evaluation" and not candidate.weight_paths and not released_predictions:
        blockers.append("Released weights are missing")
    if candidate.feasibility == "blocked" and not blockers:
        blockers.append("Plan is blocked; runtime requirements need clarification")
    # Training approval/budget is enforced by L3; preserve its candidate here.
    return ExecutionPlan(
        id=f"{claim.id}.plan",
        claim_id=claim.id,
        condition_ids=ids,
        target_conditions=[
            conditions[key].model_copy(deep=True) if released_predictions else conditions[key] for key in ids
        ],
        y_paper=values,
        target_bindings=target_bindings,
        task=ExecutionTask(
            entry_script=candidate.entry_script,
            config=candidate.config,
            command=["python", "-I", "-S", candidate.entry_script] if released_predictions else [],
        ),
        run_mode=candidate.run_mode,
        feasibility="blocked" if blockers else "ready",
        blocker="; ".join(blockers),
        priority=candidate.priority,
        estimated_cost=candidate.estimated_cost,
    )


def verify_experiments(
    claim: Claim,
    materials: SharedMaterials,
    *,
    call=None,
    scope_call=None,
    scope_binding_repair_rounds=None,
) -> BranchResult:
    if scope_binding_repair_rounds is None:
        from common.config import get_settings

        scope_binding_repair_rounds = get_settings().experiment_scope_binding_repair_rounds
    if type(scope_binding_repair_rounds) is not int or scope_binding_repair_rounds not in (0, 1):
        raise ValueError("scope_binding_repair_rounds must be 0 or 1")
    frozen_inputs = (
        copy.deepcopy((claim.model_dump(), materials.model_dump())) if scope_binding_repair_rounds else None
    )
    source_limit = joint_limit()
    from verification.execution_projection import projection_context, projection_snapshot

    target_catalog = build_catalog(claim, materials)
    try:
        projection_inputs = projection_context(claim, materials)
        from verification.execution_projection_semantics import semantic_request_context

        semantic_inputs = semantic_request_context(claim, materials)
        initial_projection_snapshot = projection_snapshot(claim, materials)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        projection_inputs = {"unavailable": str(exc)}
        semantic_inputs = {"unavailable": str(exc)}
        initial_projection_snapshot = None
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
            "When one located passage needs connected definitions/setup/results, you may propose ONE NEW paper_support item per condition with additional_sources (exact block_id/quote pairs). Such a joint candidate covers exactly one condition; its primary plus additional passages must together establish every qualifier before fully_supported_conditions can include it. Keep separate partial observations unchanged. Do not join disjoint passages into any quote or automatically merge partial flags. The configured joint_source_limit includes the primary source; do not exceed it. "
            "Declare the COMPLETE member set in that candidate now. When a selected table relies on metric/setup definitions outside the table, include both those definitions and the exact manuscript passage explicitly referencing that numbered table to connect them. Definitions and the exact manuscript table reference may be different members; a definition need not repeat the table number. The cited passages must connect the same task, dataset and applicable settings; topic overlap alone is insufficient. One exact member may serve multiple roles; do not duplicate it. A table-reference passage emitted only as another partial item grants this joint candidate no access; the independent scope review cannot append it. Preserve missing-reference or unresolved-qualifier limitations and leave fully_supported_conditions empty when the complete condition is not established. "
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
            "whole-table references remain blocked until the target is resolved. Include actual indexed data/weight paths. "
            "Execution targets require an absolute measurement for the original subject, dataset, metric and all settings. "
            "A gap, improvement, percentage-point change, ratio, comparator score or one case of a composite condition "
            "cannot replace that absolute target. Unknown target bindings remain blocked. Preserve the complete original "
            "sentence or native table; a narrowed value_context cannot remove its governing subject or scope. "
            "Keep plans with missing code, data, weights, or budget as blocked with a reason. Priority follows "
            "the link to the paper's core contribution. Emit no execution evidence or final verdict. "
            "For released prediction-file evaluation you may explicitly propose released-predictions-v1 projection using "
            "the provided original field inventory, paper source/number selectors and indexed resources. The first recipe "
            "only computes full-list exact-match accuracy as a fraction. Classify every original semantic leaf without "
            "editing it, choose exact source IDs for each role, and retain sample scope and conclusion boundaries. "
            "Use execution_projection_context.request_choices for exact repository-relative entry/config/data identifiers, number_id-only prose choices and per-config field roles. Set data_paths to exactly [projection.data_path], keeping config and entry in their separate fields. A field with no supported role remains unresolved. Explicitly select source_ids, using complete_source_ids or combined lexical hits only as syntax hints; never add omitted sources automatically. Every choice still requires independent semantic review and final binding. "
            "The original reported accuracy is a target, not a runtime setting. A combined dataset/split must have an "
            "exact identity in the released config. Unknown obligations stay unresolved. Projection requires the actual "
            "released evaluator/data/config, a prose number_id, one original condition and evaluation mode; omit weights "
            "only for this released-predictions resource mode. It does not prove model inference or training. "
            "The explicit released-predictions-v2 alternative can retain a qualified original metric/condition through source-grounded finite atoms. It preserves every original field and claim character span; select canonical identity/metric only from the actual config, full exact-match definition, complete fixed sample and explicit negative boundaries. Classify arbitrary field keys by their entire values; direct config keys retain runtime precedence. Use execution_projection_semantics_context as a separate version domain, without rewriting the original claim or condition. Each atom must have original condition-scoped paper source IDs; every field and claim fragment lists exactly its finite obligations. Unknown residue cannot be discharged by labeling it irrelevant. Do not convert null/old responses into a proposal or add a model stage. "
            "Use projection=null for ordinary model-inference/v1 targets. No support flag is granted by a proposal.",
            {
                "claim": claim.model_dump(mode="json"),
                "allowed_condition_ids": [condition.id for condition in claim.conditions],
                "joint_source_limit": source_limit,
                "paper_blocks": [b.model_dump() for b in materials.blocks],
                "repository_index": materials.repository.model_dump() if materials.repository else None,
                "execution_projection_context": projection_inputs,
                "execution_projection_semantics_context": semantic_inputs,
                "target_catalog": catalog_prompt(target_catalog),
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
        if not item.additional_sources:
            _paper_pointer(materials, item.block_id, item.quote)
        _fully_supported(_covered(claim, item.covered), item.fully_supported_conditions)
    has_projection = any(target.projection is not None for plan in output.plans for target in plan.targets)
    catalog = target_catalog if output.items or has_projection else None
    joint_catalogs = {
        index: prepare_joint_candidate(claim, materials, catalog, item, index, max_sources=source_limit)
        for index, item in enumerate(output.items)
        if item.additional_sources
    }
    by_condition = {}
    for index, bounded in joint_catalogs.items():
        by_condition.setdefault(bounded["joint_view"]["condition_id"], []).append(index)
    for indices in by_condition.values():
        if len(indices) > 1:
            for index in indices:
                joint_catalogs[index]["joint_view"]["errors"].append(
                    "Each condition permits only one explicit joint candidate"
                )
    frozen_output = copy.deepcopy(output.model_dump()) if scope_binding_repair_rounds else None
    projection_reviews = {}
    scopes, decisions, scope_issue, scope_audit = (
        _scope_review(
            claim,
            materials,
            output,
            call=scope_call or call,
            catalog=catalog,
            joint_catalogs=joint_catalogs,
            binding_repair_rounds=scope_binding_repair_rounds,
            projection_reviews=projection_reviews if has_projection else None,
        )
        if output.items or has_projection
        else ({}, {}, None, None)
    )
    if scope_issue:
        result.issues.append(scope_issue)
    if has_projection:
        try:
            if (
                initial_projection_snapshot is None
                or projection_snapshot(claim, materials) != initial_projection_snapshot
            ):
                raise ValueError("Original paper or released resources changed across projection review")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            projection_reviews.clear()
            result.issues.append(f"Execution projection unavailable: {exc}")
    if scope_binding_repair_rounds and (
        frozen_inputs != (claim.model_dump(), materials.model_dump()) or frozen_output != output.model_dump()
    ):
        result.issues.append("Binding repair input snapshot changed; no evidence or plans accepted")
        return result
    condition_map = {condition.id: condition for condition in claim.conditions}
    for index, item in enumerate(output.items):
        item_catalog = joint_catalogs.get(index, catalog)
        view = item_catalog.get("joint_view") if item_catalog else None
        additional_pointers = []
        if view is not None:
            result.issues.extend(f"Joint candidate {index}: {error}" for error in view["errors"])
            if view["primary_source_id"] is None:
                continue
            additional_pointers = [m["pointer"] for m in view["members"] if m["ordinal"] > 0]
        required_aspect = {
            "missing_ablation": "isolation",
            "missing_statistic": "stability",
            "small_gap_without_statistics": "stability",
            "large_gap_no_variance": "stability",
            "text_table_contradiction": "consistency",
        }.get(item.kind)
        if required_aspect and item.aspect != required_aspect:
            raise ValueError("Experimental concern is assigned to the wrong aspect")
        try:
            pointer = _paper_pointer(materials, item.block_id, item.quote)
        except ValueError as exc:
            if view is None:
                raise
            view["errors"].append(f"Primary source became unavailable: {exc}")
            result.issues.append(f"Joint candidate {index}: {view['errors'][-1]}")
            continue
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
                    if item.kind in {"missing_statistic", "small_gap_without_statistics"}:
                        if not scope.uncertainty_sensitive or scope.assertion == "descriptive":
                            reasons.append(
                                "Unreported uncertainty does not bear on this descriptive assertion"
                            )
                        reasons.extend(
                            _numeric_scope_check(
                                claim,
                                condition_map[condition_id],
                                scope,
                                decision,
                                materials,
                                catalog=item_catalog,
                                mode="concern",
                                require_comparison=item.kind == "small_gap_without_statistics",
                            )
                        )
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
                        _numeric_scope_check(
                            claim,
                            condition_map[condition_id],
                            scope,
                            decision,
                            materials,
                            catalog=item_catalog,
                        )
                    )
                scope_notes.append(f"{condition_id}: {scope.assertion}; {decision.rationale}")
                scope_notes.append(
                    f"{condition_id} scope_source_refs=[{_scope_sources(decision, materials)}]"
                )
            if view is not None:
                try:
                    revalidate_members(item_catalog, materials)
                except (ValueError, OSError) as exc:
                    reasons.append(str(exc))
                if view["errors"]:
                    reasons.extend(view["errors"])
                view.setdefault("final_checks", []).append(
                    {
                        "condition_id": condition_id,
                        "errors": list(reasons),
                        "first_pass_full": condition_id in item.fully_supported_conditions,
                    }
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
        if view is not None and view.get("binding_repair"):
            note += "; binding_repair=accepted (original rejection retained in scope audit)"
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
                    additional_pointers=additional_pointers,
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
    if scope_audit is not None and joint_catalogs:
        audit = json.loads(scope_audit.read_text(encoding="utf-8"))
        audit["joint_sources"] = {
            str(index): bounded["joint_view"] for index, bounded in joint_catalogs.items()
        }
        scope_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for plan_index, candidate in enumerate(output.plans):
        try:
            plan = _plan(
                claim,
                materials,
                candidate,
                projection_reviews=projection_reviews,
                scope_audit=scope_audit,
                plan_index=plan_index,
            )
            if (
                any(target.projection is not None for target in candidate.targets)
                and plan.feasibility == "blocked"
            ):
                raise ValueError(plan.blocker)
            result.plans.append(plan)
        except ValueError as exc:
            if any(target.projection is not None for target in candidate.targets):
                from schemas.limitations import VerificationLimitation

                result.issues.append(f"Execution projection plan rejected: {exc}")
                result.verification_limitations.append(
                    VerificationLimitation(
                        claim_id=claim.id,
                        condition_ids=[c.id for c in claim.conditions],
                        stage="Experiments",
                        kind="plan_rejected",
                        reason=f"Execution projection unavailable: {exc}",
                    )
                )
            else:
                # Preserve the original single-source plan rejection contract.
                raise RejectedPlan(str(exc), result) from exc
    return result
