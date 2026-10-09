"""Deterministic, source-grounded scalar execution targets and consumer revalidation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from schemas.claim import ExecutionTargetBinding, PaperTargetPassage, PaperTargetSelector
from screening.checks import grounded_paper_pointer
from verification.experiment_catalog import TableGrid, build_catalog, resolve_cell
from verification.prose_numbers import NUMBER, TOKEN, _sentences, resolve_number


class TargetBindingError(ValueError):
    """A target remains readable, but cannot authorize a deciding execution."""


_ROLES = {"method", "model", "subject", "ablation", "variant", "treatment"}
_COMPARATORS = {"comparison", "comparator", "baseline", "reference", "control"}
# Complete names with a supported absolute-measurement meaning. New quantities
# need an explicit definition contract; substring aliases cannot grant one.
_ABSOLUTE_METRICS = {
    "accuracy",
    "acc",
    "top-1 accuracy",
    "top-5 accuracy",
    "f1",
    "dev f1",
    "test f1",
    "precision",
    "recall",
    "auc",
    "map",
    "ndcg",
    "mrr",
    "mr",
    "miou",
    "iou",
    "bleu",
    "rouge-l",
    "rouge-1",
    "rouge-2",
    "latency",
    "training time",
}
_UNITS = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "us": "us",
    "µs": "us",
    "μs": "us",
    "microsecond": "us",
    "microseconds": "us",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hour": "h",
    "hours": "h",
}


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _claim_fingerprint(claim):
    return _fingerprint(
        claim.model_dump(
            mode="json",
            include={
                "id",
                "text",
                "loc",
                "conditions",
                "source_block_id",
                "source_quote",
                "source_refs",
            },
        )
    )


def _has(label, text):
    return bool(label and re.search(r"(?<!\w)" + re.escape(str(label)) + r"(?!\w)", text, re.I))


def _unit(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value.strip().casefold() not in _UNITS:
        raise TargetBindingError("Paper target has an unresolved unit or quantity scale")
    return _UNITS[value.strip().casefold()]


def _condition_parts(condition):
    if not condition.dataset or not condition.metric:
        raise TargetBindingError("Scalar target requires an explicit dataset and metric")
    if any(
        isinstance(v, (list, dict)) or v is None or isinstance(v, bool)
        for k, v in condition.settings.items()
        if k != "reported_variance"
    ):
        raise TargetBindingError("List/composite target settings require a separate complete-case contract")
    if _COMPARATORS & condition.settings.keys():
        raise TargetBindingError("Comparative conditions cannot be established by one scalar target")
    roles = [str(v) for k, v in condition.settings.items() if k in _ROLES]
    if len(set(roles)) > 1:
        raise TargetBindingError("Target subject has multiple unresolved structured identities")
    subject = roles[0] if roles else None
    units = {_unit(condition.settings[k]) for k in ("unit", "units") if k in condition.settings}
    if len(units) > 1:
        raise TargetBindingError("Condition unit fields conflict")
    metric = condition.metric
    match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", metric)
    if match:
        metric, metric_unit = match.group(1).strip(), _unit(match.group(2))
        units.add(metric_unit)
    if metric.casefold() not in _ABSOLUTE_METRICS and not re.fullmatch(
        r"(?:hits|ndcg)@[1-9]\d*", metric, re.I
    ):
        raise TargetBindingError(
            "Target quantity semantics are unresolved for this metric name; an absolute measurement definition is required"
        )
    if len(units) > 1:
        raise TargetBindingError("Metric and condition units conflict")
    return subject, metric, next(iter(units), None)


def _scalar_match(text, condition, *, claim_statement=False):
    """A finite whole-statement grammar binds the value slot, never bag-of-labels."""
    subject, metric, declared_unit = _condition_parts(condition)
    dataset = re.escape(condition.dataset)
    split = condition.settings.get("split")
    settings = [
        (k, v)
        for k, v in condition.settings.items()
        if k not in _ROLES | {"split", "unit", "units", "reported_variance"}
    ]
    # Scalar settings must name their key as well as their value. Preserve seed=42.
    attributes = "".join(
        rf"(?:\s+{re.escape(k)}\s*[:=]?\s*{re.escape(str(v))})" + ("?" if claim_statement else "")
        for k, v in settings
    )
    short_scope = dataset + (rf"\s+{re.escape(str(split))}" if split is not None else "")
    scope = short_scope + attributes
    suffix = r"(?:\s*(?P<unit>%|percent|percentage|milliseconds?|microseconds?|seconds?|sec|minutes?|hours?|ms|us|µs|μs|s|min|h))?"
    number = rf"(?P<value>{TOKEN}){suffix}"
    quantities = [re.escape(metric) + r"\s*(?:is\s+|=\s*)?" + number, number + r"\s+" + re.escape(metric)]
    patterns = []
    for quantity in quantities:
        if subject:
            actor = rf"(?:(?:method|model)\s+)?{re.escape(subject)}\s+(?:has|achieves|records|reports)\s+"
            patterns.append(rf"(?:On|For)\s+{scope},\s*{actor}{quantity}")
            if split is not None and not settings:
                patterns.append(
                    rf"(?:On|For)\s+the\s+{re.escape(str(split))}\s+split\s+of\s+(?:dataset\s+)?{dataset},\s*{actor}{quantity}"
                )
        else:
            patterns.append(scope + r"\s+" + quantity)
    for pattern in patterns:
        match = re.fullmatch(pattern + r"\s*[.!]?", text.strip(), re.I)
        if match:
            token = match.group("value")
            units = {_unit(match.group("unit"))} if match.group("unit") else set()
            if token.endswith("%"):
                units.add("%")
            if declared_unit:
                units.add(declared_unit)
            if len(units) > 1:
                raise TargetBindingError("Target occurrence and declared units conflict")
            # A declared unit must also appear in the actual measurement text.
            if declared_unit and not (match.group("unit") or token.endswith("%")):
                raise TargetBindingError("Target measurement does not state its declared unit")
            return match, float(token.rstrip("%")), next(iter(units), None), subject
    return None


def _claim_target(claim, condition, materials):
    statements = [claim.text[start:end] for start, end in _sentences(claim.text)]
    selected = []
    for statement in statements:
        matches = []
        for candidate in claim.conditions:
            try:
                match = _scalar_match(statement, candidate, claim_statement=True)
            except TargetBindingError:
                match = None
            if match is not None:
                matches.append((candidate.id, match))
        if not matches:
            raise TargetBindingError("Claim contains an unresolved non-scalar or comparative assertion")
        selected.extend(match for identifier, match in matches if identifier == condition.id)
    if len(selected) != 1:
        raise TargetBindingError(
            "Claim is not a completely bound absolute scalar measurement; gap/ratio/composite or unknown assertions remain blocked"
        )
    matched = selected[0]
    if condition.description and condition.description.strip() != claim.text.strip():
        description = _scalar_match(condition.description, condition, claim_statement=True)
        if description is None or description[1:3] != matched[1:3]:
            raise TargetBindingError("Condition description has unresolved or conflicting scalar qualifiers")
    primary = [
        ref
        for ref in claim.source_refs
        if (ref.source_block_id, ref.source_quote) == (claim.source_block_id, claim.source_quote)
    ]
    passages = [
        (ref.source_block_id, ref.source_quote) for ref in claim.source_refs if condition.id in ref.covered
    ]
    if not primary or any(condition.id in ref.covered for ref in primary):
        if claim.source_block_id:
            passages.append((claim.source_block_id, claim.source_quote))
    elif not passages:
        raise TargetBindingError("The original claim source excludes this target condition")
    anchored = not passages
    for block_id, quote in passages:
        block = next((b for b in materials.blocks if b.id == block_id), None)
        if block is None:
            raise TargetBindingError("Original claim target source is missing")
        grounded_paper_pointer(materials, block, quote)
        if re.search(r"<table\b", quote, re.I):
            # Native table observations are checked separately with their own axes.
            continue
        for start, end in _sentences(quote):
            source = _scalar_match(quote[start:end], condition, claim_statement=True)
            if source is not None:
                if source[1:3] != matched[1:3]:
                    raise TargetBindingError("Original claim source and extracted target disagree")
                anchored = True
    if not anchored and not any(re.search(r"<table\b", quote, re.I) for _, quote in passages):
        raise TargetBindingError(
            "Original condition-scoped claim sources do not identify this scalar assertion"
        )
    return matched[1:]


class _TargetTable(TableGrid):
    """Separate actual heading rows from data, while retaining native span expansion."""

    def __init__(self, text):
        self.tags = []
        self.current_tags = []
        super().__init__(text)
        self.header_rows = 0
        for index, tags in enumerate(self.tags):
            if tags and (set(tags) == {"th"} or index == 0):
                self.header_rows += 1
            else:
                break
        if len(self.tables) != 1 or self.header_rows >= len(self.tags):
            raise TargetBindingError("Table headings and data rows are unresolved")

    def handle_starttag(self, tag, attrs):
        names = [name for name, _ in attrs]
        if len(names) != len(set(names)):
            raise TargetBindingError("Repeated table attributes have unresolved target coordinates")
        if tag == "tr":
            self.current_tags = []
        elif tag in {"td", "th"}:
            self.current_tags.append(tag)
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "tr":
            self.tags.append(self.current_tags)
        super().handle_endtag(tag)

    def handle_comment(self, data):
        raise TargetBindingError("Table comments require an explicit target source interpretation")

    def handle_decl(self, decl):
        raise TargetBindingError("Table declarations have unresolved target interpretation")

    def unknown_decl(self, data):
        raise TargetBindingError("Table declarations have unresolved target interpretation")

    def handle_pi(self, data):
        raise TargetBindingError("Table processing instructions have unresolved target interpretation")


def _table_binding(cell, condition):
    subject, metric, declared_unit = _condition_parts(condition)
    table_html = list(re.finditer(r"<table\b.*?</table>", cell["quote"], re.I | re.S))[cell["table"]].group()
    table = _TargetTable(table_html)
    grid = table.tables[0]
    if cell["row"] < table.header_rows:
        raise TargetBindingError("Selected target belongs to a heading rather than a data cell")

    def headings(column):
        return list(dict.fromkeys(grid.get((row, column), "") for row in range(table.header_rows)))

    fields = {}
    aliases = {
        "models": "model",
        "methods": "method",
        "systems": "model",
        "system": "model",
        "datasets": "dataset",
    }
    for column in sorted({c for r, c in grid if r == cell["row"]} - {cell["column"]}):
        header = " ".join(headings(column)).strip().casefold()
        key = aliases.get(header, header)
        if key in fields and fields[key] != grid.get((cell["row"], column), ""):
            raise TargetBindingError("Selected table row repeats conflicting field columns")
        fields[key] = grid.get((cell["row"], column), "")
    selected_headings = headings(cell["column"])
    column_scope = " ".join(selected_headings)
    role_values = [value for key, value in fields.items() if key in _ROLES]
    if subject and not (
        (role_values and all(v.casefold() == subject.casefold() for v in role_values))
        or (not role_values and any(v.casefold() == subject.casefold() for v in selected_headings))
    ):
        raise TargetBindingError("Selected table row/column identifies a different subject")
    if not subject and role_values:
        raise TargetBindingError("Table contains named methods but the scalar claim has no bound subject")
    required = {
        "dataset": condition.dataset,
        **{
            k: v
            for k, v in condition.settings.items()
            if k not in _ROLES | {"unit", "units", "reported_variance"}
        },
    }
    for key, value in fields.items():
        if key in _ROLES | {"metric", "unit", "units"} or key in required:
            continue
        other_quantity = key
        for label in [condition.dataset, *condition.settings.values()]:
            other_quantity = re.sub(
                r"(?<!\w)" + re.escape(str(label)) + r"(?!\w)", " ", other_quantity, flags=re.I
            )
        other_quantity = re.sub(r"\s*\([^()]*\)\s*$", "", other_quantity)
        other_quantity = " ".join(other_quantity.split()).casefold()
        if NUMBER.fullmatch(value) is None or other_quantity not in _ABSOLUTE_METRICS:
            raise TargetBindingError(f"Selected table row has an undeclared or unresolved dimension: {key}")
    for key, expected in required.items():
        if key in fields:
            if fields[key].casefold() != str(expected).casefold():
                raise TargetBindingError(f"Selected table row has a different {key}")
        elif not _has(str(expected), column_scope):
            raise TargetBindingError(f"Selected table column does not bind {key}")
    # Consume exactly the known dimensional labels, leaving the named absolute
    # quantity. 'accuracy improvement' and an adjacent F1 column stay distinct.
    quantity = column_scope
    for expected in required.values():
        quantity = re.sub(r"(?<!\w)" + re.escape(str(expected)) + r"(?!\w)", " ", quantity, flags=re.I)
    if subject and not role_values:
        quantity = re.sub(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", " ", quantity, flags=re.I)
    quantity = " ".join(quantity.split())
    if "metric" in fields:
        if quantity:
            raise TargetBindingError("Transposed metric column has unresolved additional qualifiers")
        quantity = fields["metric"]
    metric_match = re.fullmatch(re.escape(metric) + r"(?:\s*\(([^)]+)\))?", quantity, re.I)
    if metric_match is None:
        raise TargetBindingError("Selected table quantity is not the exact absolute metric")
    units = {"%"} if cell["token"].endswith("%") else set()
    units.update(_unit(fields[key]) for key in ("unit", "units") if key in fields)
    if metric_match.group(1):
        units.add(_unit(metric_match.group(1)))
    if len(units) > 1 or (declared_unit and units != {declared_unit}):
        raise TargetBindingError("Selected table metric unit is missing or inconsistent")
    return next(iter(units), None), subject


def _bind(claim, condition, reported, materials, selector):
    reported = PaperTargetPassage.model_validate(
        reported.model_dump() if hasattr(reported, "model_dump") else reported
    )
    selector = (
        PaperTargetSelector.model_validate(
            selector.model_dump() if hasattr(selector, "model_dump") else selector
        )
        if selector
        else None
    )
    if next((c for c in claim.conditions if c.id == condition.id), None) != condition:
        raise TargetBindingError("Target condition differs from the original claim")
    block = next((b for b in materials.blocks if b.id == reported.block_id), None)
    if block is None:
        raise TargetBindingError("Target paper block is missing")
    pointer = grounded_paper_pointer(materials, block, reported.quote)
    if NUMBER.fullmatch(reported.token) is None:
        raise TargetBindingError("Target must retain one finite original numeric token")
    value = float(reported.token.rstrip("%"))
    if not math.isfinite(value):
        raise TargetBindingError("Target must be finite")
    context = reported.value_context or reported.quote
    if context not in reported.quote:
        raise TargetBindingError("value_context must be an exact substring of the quoted paper passage")
    if len(list(re.finditer(re.escape(reported.quote), block.text))) != 1:
        raise TargetBindingError("Target quote does not uniquely identify an original block range")
    quote_start = block.text.index(reported.quote)
    if reported.quote.count(context) != 1:
        raise TargetBindingError("Target value context is repeated within the original quote")
    start = quote_start + reported.quote.index(context)
    end = start + len(context)
    expected_value, expected_unit, _ = _claim_target(claim, condition, materials)
    catalog = build_catalog(claim, materials)
    if re.search(r"<table\b", block.text, re.I):
        if selector and selector.number_id:
            raise TargetBindingError("An HTML target requires an original table-cell selector")
        candidates = [
            identifier
            for identifier, row in catalog["cells"].items()
            if row["cell_type"] == "number"
            and row["token"] == reported.token
            and catalog["sources"][row["source_id"]]["block_id"] == block.id
        ]
        if selector:
            candidates = [selector.cell_id] if selector.cell_id in candidates else []
        accepted = []
        for identifier in candidates:
            cell = resolve_cell(catalog, identifier, materials)
            spans = list(re.finditer(r"<table\b.*?</table>", block.text, re.I | re.S))
            table = spans[cell["table"]]
            if not (quote_start <= table.start() and table.end() <= quote_start + len(reported.quote)):
                continue
            if reported.value_context and not (start <= table.start() and table.end() <= end):
                continue
            try:
                unit, subject = _table_binding(cell, condition)
            except TargetBindingError:
                continue
            accepted.append((PaperTargetSelector(cell_id=identifier), unit, subject))
        if len(accepted) != 1:
            raise TargetBindingError(
                "Exact table context does not uniquely bind the reported number to this condition"
            )
        selected, unit, subject = accepted[0]
    else:
        if selector and selector.cell_id:
            raise TargetBindingError("A prose target requires an original number-occurrence selector")
        candidates = [
            identifier
            for identifier, row in catalog["numbers"].items()
            if row["block_id"] == block.id
            and row["token"] == reported.token
            and start <= row["start"] < row["end"] <= end
        ]
        if selector:
            candidates = [selector.number_id] if selector.number_id in candidates else []
        if len(candidates) != 1:
            raise TargetBindingError(
                "Exact value context does not uniquely bind the reported number to this condition"
            )
        record = resolve_number(catalog["numbers"], candidates[0], materials)
        match = _scalar_match(record["quote"], condition)
        if match is None:
            raise TargetBindingError(
                "The complete original sentence does not bind this subject and absolute measurement"
            )
        parsed, observed_value, unit, subject = match
        leading = len(record["quote"]) - len(record["quote"].lstrip())
        if parsed.span("value") != (
            record["start"] - record["sentence_start"] - leading,
            record["end"] - record["sentence_start"] - leading,
        ):
            raise TargetBindingError("Selected original number belongs to a setting or different value role")
        if observed_value != value:
            raise TargetBindingError("Selected target differs from the bound measurement")
        selected = PaperTargetSelector(number_id=candidates[0])
    if value != expected_value or unit != expected_unit:
        raise TargetBindingError("Reported target differs from the original claim's scalar value or scale")
    return ExecutionTargetBinding(
        condition_id=condition.id,
        reported=reported,
        selector=selected,
        pointer=pointer,
        block_sha256=_hash(block.text),
        artifact_sha256=hashlib.sha256(Path(pointer.locator).read_bytes()).hexdigest(),
        claim_sha256=_claim_fingerprint(claim),
        condition_sha256=_fingerprint(condition.model_dump(mode="json")),
        value=value,
        subject=subject,
        unit=unit,
    )


def bind_execution_target(claim, condition, reported, materials, *, selector=None):
    """Return a fully reconstructed proof, or a conservative actionable blocker."""
    try:
        return _bind(claim, condition, reported, materials, selector)
    except TargetBindingError:
        raise
    except (ValueError, OSError, KeyError, TypeError, IndexError, AssertionError) as exc:
        raise TargetBindingError(f"Paper target binding unavailable: {exc}") from exc


def validate_plan_targets(plan, claim, materials):
    ids = [condition.id for condition in plan.target_conditions]
    if (
        plan.claim_id != claim.id
        or not ids
        or len(ids) != len(set(ids))
        or len(plan.condition_ids) != len(set(plan.condition_ids))
        or set(ids) != set(plan.condition_ids)
        or set(ids) != set(plan.y_paper)
    ):
        raise TargetBindingError("Paper target identities do not uniquely match the linked claim and plan")
    if set(plan.target_bindings) != set(plan.condition_ids):
        raise TargetBindingError(
            "Paper target bindings are missing; regenerate the plan from its original sources"
        )
    result = {}
    for condition in plan.target_conditions:
        binding = plan.target_bindings[condition.id]
        if binding.version == 2:
            from verification.execution_projection import bind_projection_target

            projection = binding.projection
            expected_paths = {str(materials.source_pdf), str(materials.markdown_path)}
            if set(projection.paper_hashes) != expected_paths or any(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha
                for path, sha in projection.paper_hashes.items()
            ):
                raise TargetBindingError("Projected paper artifacts changed after independent review")
            if (
                len(plan.condition_ids) != 1
                or plan.run_mode != "evaluation"
                or plan.task.entry_script != projection.entry_script
                or plan.task.config != projection.config_path
                or plan.task.command != ["python", "-I", "-S", projection.entry_script]
                or plan.task.workdir != "."
                or plan.task.metric_output is not None
            ):
                raise TargetBindingError(
                    "Released-predictions plan differs from its bound single evaluation task"
                )
            try:
                rebuilt = bind_projection_target(
                    claim,
                    condition,
                    binding.reported,
                    materials,
                    selector=binding.selector,
                    proposal=projection.proposal,
                    entry_script=projection.entry_script,
                    config_path=projection.config_path,
                    review=projection.scope_review,
                    audit_path=projection.scope_audit,
                )
            except (ValueError, OSError, TypeError, KeyError, IndexError) as exc:
                raise TargetBindingError(f"Execution projection unavailable: {exc}") from exc
        else:
            rebuilt = bind_execution_target(
                claim, condition, binding.reported, materials, selector=binding.selector
            )
        if rebuilt != binding or plan.y_paper.get(condition.id) != rebuilt.value:
            raise TargetBindingError(
                f"Paper target binding changed or conflicts with y_paper: {condition.id}"
            )
        result[condition.id] = rebuilt
    return result


def runtime_target_issue(binding, observation_settings, *, observation_unit=None):
    """Check observed units without copying a paper unit or converting the value."""
    if binding.version == 2:
        units = [observation_settings[k] for k in ("unit", "units") if k in observation_settings]
        if observation_unit != "fraction" or any(unit != "fraction" for unit in units):
            return "Released-predictions measurement requires its independently computed fraction scale"
        return ""
    try:
        actual = {_unit(observation_settings[k]) for k in ("unit", "units") if k in observation_settings}
        if observation_unit is not None:
            actual.add(_unit(observation_unit))
        if len(actual) > 1:
            return "Runtime unit fields conflict; comparison unavailable"
        unit = next(iter(actual), None)
        if unit != binding.unit:
            return "Runtime and paper units are missing or inconsistent; comparison unavailable"
        return ""
    except TargetBindingError as exc:
        return f"{exc}; comparison unavailable"
