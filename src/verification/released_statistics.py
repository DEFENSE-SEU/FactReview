"""Finite whole-file statistics with explicit original-field semantic review.

The registry grants resource eligibility only. Paper expectations come from
original source tokens; observed counts are produced separately by the host.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from schemas.claim import (
    Condition,
    Contract,
    ExecutionPlan,
    ExecutionTask,
    NonEmpty,
    ProjectedExecutionTargetBinding,
    ReleasedStatisticsProjectionRecord,
    ReleasedStatisticsProposal,
    ReleasedStatisticsReview,
    ReleasedStatisticsSelection,
)
from verification.execution_projection import (
    ProjectionError,
    _pointer,
    _words_count,
    digest,
    entry_recipe,
    field_inventory,
    file_hash,
    projection_snapshot,
    repository_context_files,
)
from verification.execution_projection_choices import (
    MAX_CHOICES,
    MAX_COMBINATION_CHECKS,
    MAX_RESOURCE_BYTES,
)
from verification.execution_projection_semantics import _metric, _same, _source_authorized, _source_supports
from verification.experiment_catalog import build_catalog, resolve_source

DOMAIN = "full-list-exact-match-statistics-v1"
WIRE = "experiments-execution-v5"
FRESH_WIRE = "experiments-execution-v6"
RECORD = "released-statistics-choice-v1"


class _StatisticsBudget(ProjectionError):
    pass


class StatisticsDecision(Contract):
    kind: Literal["statistics_choice"]
    selection: ReleasedStatisticsSelection


class StatisticsNone(Contract):
    kind: Literal["none"]
    rationale: NonEmpty


class StatisticsOriginalFragment(Contract):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    # Exact source characters include whitespace-only separators.
    original_text: str = Field(min_length=1, strict=True)
    atom_ids: list[NonEmpty]
    separator: bool = Field(strict=True)


class StatisticsOriginalField(Contract):
    path: str = Field(min_length=1, strict=True)
    original_value: JsonValue
    atom_ids: list[NonEmpty]
    original_empty: bool = Field(strict=True)
    fragments: list[StatisticsOriginalFragment]


class StatisticsOriginalProposal(ReleasedStatisticsProposal):
    version: Literal["released-statistics-originals-v1"]
    field_bindings: list[StatisticsOriginalField] = Field(min_length=1)
    claim_bindings: list[StatisticsOriginalFragment] = Field(min_length=1)


class StatisticsOriginalSelection(ReleasedStatisticsSelection):
    version: Literal["released-statistics-originals-choice-v1"]
    proposal: StatisticsOriginalProposal


class StatisticsOriginalDecision(Contract):
    kind: Literal["statistics_choice"]
    selection: StatisticsOriginalSelection


def original_output_schema():
    from verification.experiments import ExperimentItem

    class Output(Contract):
        schema_version: Literal["experiments-execution-v6"]
        execution_domain: Literal["full-list-exact-match-statistics-v1"]
        execution_context_digest: NonEmpty
        checked_aspects: list[str]
        items: list[ExperimentItem]
        issues: list[str] = Field(default_factory=list)
        execution: Annotated[StatisticsOriginalDecision | StatisticsNone, Field(discriminator="kind")]

    return Output


def output_schema():
    from verification.experiments import ExperimentItem

    class Output(Contract):
        schema_version: Literal["experiments-execution-v5"]
        execution_domain: Literal["full-list-exact-match-statistics-v1"]
        execution_context_digest: NonEmpty
        checked_aspects: list[str]
        items: list[ExperimentItem]
        issues: list[str] = Field(default_factory=list)
        execution: Annotated[StatisticsDecision | StatisticsNone, Field(discriminator="kind")]

    return Output


def scope_schema():
    from verification.experiments import CatalogScopeReviewV3

    class Scope(CatalogScopeReviewV3):
        execution_scope_domain: Literal["full-list-exact-match-statistics-v1"]
        released_statistics_reviews: list[ReleasedStatisticsReview]

    return Scope


def _sources(catalog, materials, condition):
    return {sid: source for sid in catalog["sources"]
            if _source_authorized(catalog, source := resolve_source(catalog, sid, materials), condition.id)
            and source["block_id"] is not None}


def _snapshot(claim, materials):
    return {"claim": claim.model_dump(mode="json"), "materials": materials.model_dump(mode="json"),
            "original_sha256": projection_snapshot(claim, materials),
            "implementations": {str(Path(p).resolve()): file_hash(p) for p in (
                __file__, Path(__file__).with_name("execution_projection.py"),
                Path(__file__).with_name("execution_projection_choices.py"),
                Path(__file__).with_name("execution_projection_semantics.py"),
                Path(__file__).with_name("experiment_source_authority.py"))}}


def build_statistics_registry(claim, materials, *, catalog, previous_registry, route_state):
    """Use the remainder of the completed original registry's one shared cap."""
    from verification.execution_projection_catalog import semantic_request_choices
    from verification.experiment_targets import _scalar_match
    from verification.prose_numbers import resolve_number

    if (len(claim.conditions) != 1 or previous_registry["candidates"]
            or not route_state.get("enumeration_complete")
            or not route_state.get("source_snapshot_available")
            or route_state.get("budget_exhausted") or route_state.get("candidate_bound_exceeded")):
        raise ProjectionError("Statistics route requires one complete, healthy, empty original enumeration")
    registry = {"version": "released-predictions-registry-v3", "domain": DOMAIN,
                "snapshot": _snapshot(claim, materials), "previous_registry": copy.deepcopy(previous_registry),
                "route_state": copy.deepcopy(route_state), "candidates": {}, "unavailable": [],
                "budget": copy.deepcopy(previous_registry["budget"])}
    budget = registry["budget"]
    if budget["checks_used"] >= MAX_COMBINATION_CHECKS or budget["resource_bytes"] >= MAX_RESOURCE_BYTES:
        raise ProjectionError("Shared statistics enumeration budget is exhausted")
    menu = semantic_request_choices(claim, materials, repository_files=repository_context_files(materials))
    condition = claim.conditions[0]
    sources = _sources(catalog, materials, condition)
    if not sources:
        raise ProjectionError("Original condition has no authorized manuscript sources")

    def check():
        if budget["checks_used"] >= MAX_COMBINATION_CHECKS:
            raise _StatisticsBudget("Shared statistics combination budget exhausted")
        budget["checks_used"] += 1

    try:
        for config, row in menu["conditions"].get(condition.id, {}).get("by_config", {}).items():
            if not row["finite_metric_scope_compatible"]:
                continue
            for entry in menu["entries"]:
                for data in menu["data_candidates"]:
                    check()
                    try:
                        root = Path(materials.repository.root).resolve()
                        paths = [(root / name).resolve() for name in (entry, config, data)]
                        if any(not p.is_relative_to(root) for p in paths):
                            raise ValueError("Statistics resource escapes the repository")
                        size = sum(p.stat().st_size for p in paths)
                        if budget["resource_bytes"] + size > MAX_RESOURCE_BYTES:
                            raise _StatisticsBudget("Shared statistics resource budget exhausted")
                        budget["resource_bytes"] += size
                        recipe = entry_recipe(materials, entry, config, data)
                    except (OSError, KeyError, SyntaxError, ValueError) as exc:
                        if isinstance(exc, _StatisticsBudget):
                            raise
                        registry["unavailable"].append(str(exc))
                        continue
                    cfg = recipe["configuration"]
                    if cfg["metric"] not in {"accuracy", "exact-match accuracy"} or _metric(condition.metric, cfg) is None:
                        continue
                    runtime = Condition(id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"])
                    for occurrence in row["prose_scalar_candidates"]:
                        check()
                        number = resolve_number(catalog["numbers"], occurrence["number_id"], materials)
                        scalar = _scalar_match(number["quote"], runtime)
                        leading = len(number["quote"]) - len(number["quote"].lstrip())
                        if (scalar is None or scalar[2] is not None or not 0 <= scalar[1] <= 1
                                or scalar[0].span("value") != (number["start"]-number["sentence_start"]-leading,
                                    number["end"]-number["sentence_start"]-leading)):
                            continue
                        block = next(b for b in materials.blocks if b.id == number["block_id"])
                        if block.text.count(number["quote"]) != 1:
                            continue
                        candidate = {"condition_id": condition.id, "condition": condition.model_dump(mode="json"),
                            "entry_script": entry, "config_path": config, "data_path": data,
                            "runtime_target": runtime.model_dump(mode="json"), "value": scalar[1], "subject": scalar[3],
                            "number_id": occurrence["number_id"],
                            "reported": {"block_id": block.id, "quote": number["quote"], "token": number["token"], "value_context": ""},
                            "pointer": _pointer(materials, block, number["quote"]).model_dump(mode="json"),
                            "field_inventory": field_inventory(condition), "sources": sources,
                            "recipe_facts": {k: copy.deepcopy(recipe[k]) for k in (
                                "configuration", "sample_count", "denominator", "label_key", "prediction_key", "repository_hashes")}}
                        cid = "statistics_" + digest([registry["snapshot"], candidate])
                        candidate["candidate_id"] = cid
                        registry["candidates"][cid] = candidate
                        if len(registry["candidates"]) > MAX_CHOICES:
                            raise _StatisticsBudget("Shared statistics candidate cap exceeded")
    except ProjectionError as exc:
        registry["candidates"].clear()
        registry["unavailable"].append(str(exc))
        registry["status"] = "unavailable"
    else:
        registry["status"] = "structural_choices_only"
    registry["context_digest"] = digest(registry)
    return registry


def revalidate_statistics(registry, claim, materials):
    from verification.execution_projection_choices import build_choice_registry

    state = {}
    previous = build_choice_registry(claim, materials, route_state=state)
    rebuilt = build_statistics_registry(claim, materials, catalog=build_catalog(claim, materials),
                                       previous_registry=previous, route_state=state)
    if rebuilt != registry:
        raise ProjectionError("Statistics complete source, recipe, route or budget changed")


def lower_v5(response, registry):
    parsed = output_schema().model_validate(response)
    if response.get("schema_version") != WIRE or parsed.execution_context_digest != registry["context_digest"]:
        raise ProjectionError("Statistics first response does not declare the exact requested context/version")
    normalized = {key: copy.deepcopy(response.get(key, [])) for key in ("checked_aspects", "items", "issues")}
    normalized.update(plans=[], execution_choices=[])
    selection = parsed.execution.selection if isinstance(parsed.execution, StatisticsDecision) else None
    if selection is not None:
        candidate = registry["candidates"].get(selection.candidate_id)
        if candidate is None or selection.condition_id != candidate["condition_id"]:
            raise ProjectionError("Statistics selection is not a current original candidate")
    return normalized, selection


def _same_original(value, original):
    if type(value) is not type(original):
        return False
    if isinstance(original, dict):
        return value.keys() == original.keys() and all(_same_original(value[k], v) for k, v in original.items())
    if isinstance(original, list):
        return len(value) == len(original) and all(_same_original(v, o) for v, o in zip(value, original, strict=True))
    return value == original


def _original_fragments(text, rows):
    end, result = 0, []
    for row in rows:
        fragment = text[row.start:row.end]
        if row.start != end or row.end > len(text) or not fragment or row.original_text != fragment:
            raise ProjectionError("Original fragment text/Unicode offsets are missing or stale")
        binding = row.model_dump(mode="json", exclude={"original_text"})
        binding["text_digest"] = digest(fragment)
        result.append(binding)
        end = row.end
    if end != len(text):
        raise ProjectionError("Original string is not completely bound")
    return result


def lower_v6(response, registry):
    """Bind exact original values only; independent scientific review is unchanged."""
    parsed = original_output_schema().model_validate(response)
    if parsed.execution_context_digest != registry["context_digest"]:
        raise ProjectionError("Statistics first response does not declare the exact requested context/version")
    normalized = {key: copy.deepcopy(response.get(key, [])) for key in ("checked_aspects", "items", "issues")}
    normalized.update(plans=[], execution_choices=[])
    if isinstance(parsed.execution, StatisticsNone):
        return normalized, None
    raw = parsed.execution.selection
    candidate = registry["candidates"].get(raw.candidate_id)
    if candidate is None or raw.condition_id != candidate["condition_id"]:
        raise ProjectionError("Statistics selection is not a current original candidate")
    inventory = candidate["field_inventory"]
    if [row.path for row in raw.proposal.field_bindings] != list(inventory):
        raise ProjectionError("Original semantic field denominator/order changed")
    fields = []
    for row in raw.proposal.field_bindings:
        original = inventory[row.path]
        if not _same_original(row.original_value, original):
            raise ProjectionError("Original full field value/type changed")
        binding = row.model_dump(mode="json", exclude={"original_value", "fragments"})
        binding["value_digest"] = digest(original)
        if isinstance(original, str):
            binding["fragments"] = _original_fragments(original, row.fragments)
        elif row.fragments:
            raise ProjectionError("Nonstring original field cannot acquire synthetic text fragments")
        else:
            binding["fragments"] = []
        fields.append(binding)
    canonical = raw.model_dump(mode="json")
    canonical["version"] = "released-predictions-choice-v2"
    canonical["proposal"].update(version="released-predictions-v3", field_bindings=fields,
        claim_bindings=_original_fragments(registry["snapshot"]["claim"]["text"], raw.proposal.claim_bindings))
    return normalized, ReleasedStatisticsSelection.model_validate(canonical)


def _fragments(text, rows, atoms):
    end = 0
    result = []
    for row in rows:
        fragment = text[row.start:row.end]
        if row.start != end or row.end > len(text) or not fragment or row.text_digest != digest(fragment):
            raise ProjectionError("Original fragment coverage/digest is missing or stale")
        if row.separator:
            if row.atom_ids or re.fullmatch(r"[\s.,;:!?()\[\]{}\-]+", fragment) is None:
                raise ProjectionError("A separator cannot discharge a scientific qualifier")
        elif not row.atom_ids or len(set(row.atom_ids)) != len(row.atom_ids) or not set(row.atom_ids) <= atoms:
            raise ProjectionError("Every original semantic fragment needs explicit known atoms")
        result.append(row.model_dump(mode="json"))
        end = row.end
    if end != len(text):
        raise ProjectionError("Original string is not completely reviewed")
    return result


def obligations(selection, candidate, claim, materials):
    """Mechanical binding only; independently confirmed meanings remain required."""
    from verification.prose_numbers import resolve_number

    proposal = selection.proposal
    if (proposal.candidate_id != candidate["candidate_id"] or proposal.data_path != candidate["data_path"]
            or proposal.reported_number_id != candidate["number_id"]):
        raise ProjectionError("Statistics proposal changed the selected original resources/result")
    atoms = {row.id: row for row in proposal.atoms}
    if len(atoms) != len(proposal.atoms):
        raise ProjectionError("Statistics atom identities are not unique")
    runtime = Condition.model_validate(candidate["runtime_target"])
    cfg = candidate["recipe_facts"]["configuration"]
    catalog = build_catalog(claim, materials)
    number = resolve_number(catalog["numbers"], candidate["number_id"], materials)
    kinds, runtime_keys, outcomes, boundaries = set(), set(), set(), set()
    result = []
    for atom in proposal.atoms:
        if len(set(atom.source_ids)) != len(atom.source_ids) or not set(atom.source_ids) <= set(candidate["sources"]):
            raise ProjectionError("Statistics atom has foreign/duplicate sources")
        if atom.kind == "unresolved":
            raise ProjectionError("An unresolved original meaning cannot authorize statistics")
        if atom.kind == "outcome_count":
            ref = atom.count_binding
            if ref.source_id not in atom.source_ids or atom.outcome in outcomes:
                raise ProjectionError("Expected count source/outcome is missing or duplicate")
            source = candidate["sources"][ref.source_id]["quote"]
            token = source[ref.start:ref.end]
            if ref.end > len(source) or token != ref.token or ref.start >= ref.end:
                raise ProjectionError("Expected count is not an exact original source token")
            before, after = source[max(0, ref.start-1):ref.start], source[ref.end:ref.end+1]
            if (any(re.fullmatch(r"[\w%+-]", char) for char in (before, after) if char)
                    or before == "."
                    or (before == "," and ref.start > 1 and source[ref.start-2].isdigit())
                    or (after in {".", ","} and ref.end+1 < len(source) and source[ref.end+1].isdigit())
                    or source[ref.end:].lstrip().startswith("%")):
                raise ProjectionError("Expected count must retain the complete original integer token")
            value = (int(token) if re.fullmatch(r"0|[1-9][0-9]*", token)
                     else {_words_count(n): n for n in range(21)}.get(token.casefold()))
            if value != atom.expected_count or value is None:
                raise ProjectionError("Expected count token is outside the finite strict-integer contract")
            outcomes.add(atom.outcome)
        else:
            if atom.kind == "dataset_identity" and (atom.dataset != cfg["dataset"] or atom.split != cfg["settings"].get("split")):
                raise ProjectionError("Statistics dataset/split differs from actual config")
            if atom.kind == "runtime_setting":
                if atom.key in runtime_keys or atom.key not in cfg["settings"] or not _same(atom.value, cfg["settings"][atom.key]):
                    raise ProjectionError("Statistics runtime setting is duplicated or changed")
                runtime_keys.add(atom.key)
            if atom.kind == "reported_value" and atom.number_id != candidate["number_id"]:
                raise ProjectionError("Statistics reported scalar changed")
            if atom.kind == "sample_scope" and atom.count != candidate["recipe_facts"]["sample_count"]:
                raise ProjectionError("Full released sample is not the original fixed sample")
            if atom.kind == "conclusion_boundary":
                boundaries.update(atom.excludes)
            if not any(_source_supports(atom, candidate["sources"][sid]["quote"], runtime,
                         candidate["value"], candidate["recipe_facts"]["sample_count"], number) for sid in atom.source_ids):
                raise ProjectionError(f"Original {atom.id} atom lacks one complete authorized definition/source")
        kinds.add(atom.kind)
        result.append({"kind": "atom", "binding": atom.model_dump(mode="json")})
    if not {"dataset_identity", "reported_value", "sample_scope", "measurement_definition"} <= kinds:
        raise ProjectionError("Statistics omitted a required original measurement obligation")
    if runtime_keys != set(cfg["settings"]) or boundaries != {
            "repeated_run_uncertainty", "population_performance", "cross_model_ranking"}:
        raise ProjectionError("Statistics omitted runtime or negative conclusion boundaries")
    counts = {atom.outcome: atom.expected_count for atom in proposal.atoms if atom.kind == "outcome_count"}
    total = candidate["recipe_facts"]["sample_count"]
    if any(value > total for value in counts.values()) or (
            set(counts) == {"correct", "incorrect"} and sum(counts.values()) != total):
        raise ProjectionError("Original counts do not form a complete released-sample expectation")
    inventory = candidate["field_inventory"]
    if [row.path for row in proposal.field_bindings] != list(inventory):
        raise ProjectionError("Original semantic field denominator/order changed")
    used = set()
    for row in proposal.field_bindings:
        value = inventory[row.path]
        empty = value is None or value == "" or (type(value) in {list, dict} and not value)
        if row.value_digest != digest(value) or row.original_empty is not empty:
            raise ProjectionError("Original full field value changed")
        if empty:
            if row.atom_ids or row.fragments:
                raise ProjectionError("An original empty value cannot manufacture scientific obligations")
        else:
            if not row.atom_ids or len(set(row.atom_ids)) != len(row.atom_ids) or not set(row.atom_ids) <= set(atoms):
                raise ProjectionError("Original field has missing/unknown semantic obligations")
            if isinstance(value, str):
                fragments = _fragments(value, row.fragments, set(atoms))
                if set(row.atom_ids) != {a for f in fragments for a in f["atom_ids"]}:
                    raise ProjectionError("Field fragments do not preserve the complete field mapping")
            elif row.fragments:
                raise ProjectionError("Nonstring original field cannot acquire synthetic text fragments")
            if type(value) in {int, float}:
                count_atoms = [atoms[a] for a in row.atom_ids if atoms[a].kind == "outcome_count"]
                if count_atoms and (type(value) is not int or any(a.expected_count != value for a in count_atoms)):
                    raise ProjectionError("Original numeric field and bound count are not exactly equal")
                if any(atoms[a].kind == "sample_scope" and (type(value) is not int or atoms[a].count != value)
                       for a in row.atom_ids):
                    raise ProjectionError("Original numeric sample field changed its complete population")
                if any(atoms[a].kind == "reported_value" and value != candidate["value"] for a in row.atom_ids):
                    raise ProjectionError("Original numeric result field changed its selected scalar")
            # Exact actual config keys always retain their runtime role.
            key = row.path.removeprefix("/settings/").replace("~1", "/").replace("~0", "~")
            if row.path.startswith("/settings/") and key in cfg["settings"]:
                if not any(atoms[a].kind == "runtime_setting" and atoms[a].key == key for a in row.atom_ids):
                    raise ProjectionError("A scientific count role cannot conceal a real runtime setting")
        used.update(row.atom_ids)
        result.append({"kind": "field", "original_value": copy.deepcopy(value), "binding": row.model_dump(mode="json")})
    fragments = _fragments(claim.text, proposal.claim_bindings, set(atoms))
    used.update(a for f in fragments for a in f["atom_ids"])
    result.extend({"kind": "claim_fragment", "original_text": claim.text[f["start"]:f["end"]], "binding": f} for f in fragments)
    if used != set(atoms):
        raise ProjectionError("Statistics has unconsumed semantic atoms")
    result.extend({"kind": "composition", "path": row.path, "original_value": inventory[row.path],
                   "binding": row.model_dump(mode="json")} for row in proposal.field_bindings)
    result.append({"kind": "composition", "path": "/text", "original_value": claim.text,
                   "binding": fragments})
    for row in result:
        row["obligation_id"] = "so_" + digest([selection.model_dump(mode="json"), row])
    return result


def scope_context(registry, selection, claim, materials):
    candidate = registry["candidates"][selection.candidate_id]
    return {"domain": DOMAIN, "context_digest": registry["context_digest"], "registry": registry,
            "selection": selection.model_dump(mode="json"),
            "proposal_sha256": digest(selection.proposal.model_dump(mode="json")),
            "obligations": obligations(selection, candidate, claim, materials)}


def bind_statistics_target(claim, materials, registry, selection, audit_path):
    from fact_generation.execution.resource_contract import _scientific_claim
    from schemas.claim import Claim
    from verification.experiment_source_authority import build_authority_context, decode_authority
    from verification.experiment_sources import prepare_joint_candidate
    from verification.experiment_targets import _claim_fingerprint, _fingerprint, _hash
    from verification.experiments import ExperimentItem

    original_claim = Claim.model_validate(registry["snapshot"]["claim"])
    if _scientific_claim(claim) != _scientific_claim(original_claim):
        raise ProjectionError("Statistics original scientific Claim was edited")
    claim = original_claim
    revalidate_statistics(registry, claim, materials)
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    data, response = audit["input"], audit["response"]
    original = audit["first_pass_response"]
    version = original.get("schema_version")
    if version == FRESH_WIRE:
        normalized, retained = lower_v6(original, registry)
    elif version == WIRE:
        normalized, retained = lower_v5(original, registry)
    else:
        raise ProjectionError("Statistics saved response has an unsupported wire version")
    context = scope_context(registry, selection, claim, materials)
    expected_wire = {"version": version, "domain": DOMAIN, "registry": registry, "errors": [],
        "raw_response_sha256": digest(original), "normalized_response_sha256": digest(normalized),
        "adapter_sha256": file_hash(__file__), "saved_response_sha256": digest(original),
        "saved_normalized_sha256": digest(normalized), "wire_representation": "original"}
    if version == FRESH_WIRE:
        canonical_selection = retained.model_dump(mode="json") if retained is not None else None
        expected_wire["canonical_selection_sha256"] = digest(canonical_selection)
        if audit.get("first_pass_canonical_selection") != canonical_selection:
            raise ProjectionError("Statistics canonical selection changed after original-value binding")
    if (retained != selection or audit.get("first_pass_normalized") != normalized
            or audit.get("execution_wire") != expected_wire
            or data.get("claim") != claim.model_dump(mode="json")
            or data.get("paper_blocks") != [b.model_dump(mode="json") for b in materials.blocks]
            or data.get("released_statistics_context") != context
            or data.get("output_schema") != scope_schema().model_json_schema()
            or data.get("paper_scope_format") != "catalog-v2"):
        raise ProjectionError("Statistics retained first/scope origin or full original context changed")
    scope_schema().model_validate(response)
    catalog = build_catalog(claim, materials)
    items = [ExperimentItem.model_validate(i) for i in data["candidate_items"]]
    joint = {i: prepare_joint_candidate(claim, materials, catalog, item, i,
                max_sources=data["joint_candidates"][str(i)]["manifest"]["max_sources"])
             for i, item in enumerate(items) if item.additional_sources}
    authority = build_authority_context(claim, materials, items, catalog=catalog, joint_catalogs=joint)
    if data.get("source_authority_context") != authority:
        raise ProjectionError("Statistics authority denominator/source changed")
    decode_authority(response, authority, materials, execution_scope_domain=DOMAIN)
    rows = response["released_statistics_reviews"]
    if len(rows) != 1:
        raise ProjectionError("Statistics needs exactly one independent original review")
    review = ReleasedStatisticsReview.model_validate(rows[0])
    if (review.condition_id != selection.condition_id or review.candidate_id != selection.candidate_id
            or review.context_digest != context["context_digest"] or review.proposal_sha256 != context["proposal_sha256"]
            or review.decision != "confirmed" or [r.obligation_id for r in review.reviews] != [o["obligation_id"] for o in context["obligations"]]
            or any(r.decision != "confirmed" for r in review.reviews)):
        raise ProjectionError("Statistics original atom/field/fragment/composition review is incomplete")
    candidate = registry["candidates"][selection.candidate_id]
    block = next(b for b in materials.blocks if b.id == candidate["reported"]["block_id"])
    from schemas.claim import EvidencePointer
    pointers = {sid: _pointer(materials, next(b for b in materials.blocks if b.id == src["block_id"]), src["quote"])
                for sid, src in candidate["sources"].items()}
    binding = ProjectedExecutionTargetBinding(
        condition_id=selection.condition_id, reported=candidate["reported"], selector={"number_id": candidate["number_id"]},
        pointer=EvidencePointer.model_validate(candidate["pointer"]), block_sha256=_hash(block.text),
        artifact_sha256=file_hash(candidate["pointer"]["locator"]), claim_sha256=_claim_fingerprint(claim),
        condition_sha256=_fingerprint(candidate["condition"]), value=candidate["value"], subject=candidate["subject"], unit="fraction",
        projection=ReleasedStatisticsProjectionRecord(record_version=RECORD, proposal=selection.proposal,
            runtime_target=candidate["runtime_target"], sample_count=candidate["recipe_facts"]["sample_count"],
            source_pointers=pointers, source_hashes={s: file_hash(p.locator) for s, p in pointers.items()},
            paper_hashes={str(p): file_hash(p) for p in (materials.source_pdf, materials.markdown_path)},
            repository_hashes=candidate["recipe_facts"]["repository_hashes"], repository_root=str(Path(materials.repository.root).resolve()),
            entry_script=candidate["entry_script"], config_path=candidate["config_path"],
            label_key=candidate["recipe_facts"]["label_key"], prediction_key=candidate["recipe_facts"]["prediction_key"],
            scope_review=review, scope_audit=str(audit_path.resolve()), scope_audit_sha256=file_hash(audit_path),
            recipe_sha256=file_hash(Path(__file__).with_name("execution_projection.py")), selection=selection,
            registry_snapshot=registry, registry_sha256=registry["context_digest"], first_response_sha256=digest(original),
            normalized_response_sha256=digest(normalized), selected_origin_sha256=digest(original["execution"]["selection"]),
            adapter_sha256=file_hash(__file__), statistics_builder_sha256=file_hash(__file__), obligations=context["obligations"]))
    revalidate_statistics(registry, claim, materials)
    return binding


def statistics_plan(claim, materials, registry, selection, audit_path):
    binding = bind_statistics_target(claim, materials, registry, selection, audit_path)
    candidate = registry["candidates"][selection.candidate_id]
    return ExecutionPlan(id=f"{claim.id}.plan", claim_id=claim.id, condition_ids=[binding.condition_id],
        target_conditions=[claim.conditions[0].model_copy(deep=True)], y_paper={binding.condition_id: binding.value},
        target_bindings={binding.condition_id: binding}, task=ExecutionTask(entry_script=candidate["entry_script"],
            config=candidate["config_path"], data_paths=[candidate["data_path"]],
            command=["python", "-I", "-S", candidate["entry_script"]]), run_mode="evaluation", feasibility="ready",
        priority="high" if claim.importance == "core" else "medium", estimated_cost="unknown")
