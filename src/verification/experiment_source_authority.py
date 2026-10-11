"""Closed paper-support authority; semantic classifications remain model judgments.

This module binds whole original values and actual manuscript sources. It does
not authorize external facts, execution, scientific roles or model inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

from fact_generation.execution.resource_contract import _scientific_claim
from schemas.claim import Contract, NonEmpty
from verification.experiment_catalog import build_catalog, resolve_source


class AuthorityRequirement(Contract):
    unit_id: NonEmpty
    requirement: Literal["manuscript_definition", "manuscript_reported_result",
        "external_artifact_or_implementation", "external_availability",
        "external_execution_or_training", "unresolved"]
    claim_source_ids: list[NonEmpty]
    rationale: NonEmpty


class AuthorityBasis(Contract):
    unit_id: NonEmpty
    basis: Literal["definition_or_protocol", "reported_measurement", "shown_derivation",
        "author_assertion", "unresolved"]
    evidence_source_ids: list[NonEmpty]
    rationale: NonEmpty


class AuthorityCandidate(Contract):
    pair_id: NonEmpty
    units: list[AuthorityBasis]


class SourceAuthorityReview(Contract):
    schema_version: Literal["source-authority-v1"]
    context_digest: NonEmpty
    requirements: list[AuthorityRequirement]
    candidates: list[AuthorityCandidate]


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _pointer_token(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def build_authority_context(claim, materials, candidate_items, *, catalog, joint_catalogs=None):
    """Build a complete deterministic denominator, without classifying its meaning."""
    scientific = _scientific_claim(claim)
    joint_catalogs = joint_catalogs or {}
    sources, units, pairs = {}, [], []
    for current in [catalog, *joint_catalogs.values()]:
        for key in current["sources"]:
            source = resolve_source(current, key, materials)
            if source["block_id"] is None:  # Synthetic Claim text is input context only.
                continue
            sources[key] = copy.deepcopy(source)
    for index, condition in enumerate(scientific["conditions"]):
        fields = [("/text", scientific["text"])]
        if condition["description"].strip():
            fields.append((f"/conditions/{index}/description", condition["description"]))
        for field in ("dataset", "metric"):
            if condition[field]:
                fields.append((f"/conditions/{index}/{field}", condition[field]))
        fields.extend((f"/conditions/{index}/settings/{_pointer_token(key)}", value)
            for key, value in condition["settings"].items())
        claim_sources = [key for key, source in sources.items()
            if source["kind"] == "claim_source" and condition["id"] in source["covered"]]
        for pointer, value in fields:
            row = {"condition_id": condition["id"], "field_pointer": pointer,
                "value": copy.deepcopy(value), "value_digest": _digest(value),
                "unit_scope": "condition_and_shared_qualifiers", "claim_source_ids": claim_sources}
            units.append({"unit_id": "au_" + _digest([scientific, row])[:24], **row})
    for index, candidate in enumerate(candidate_items):
        item = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else candidate
        block = next(b for b in materials.blocks if b.id == item["block_id"])
        quote = item["quote"]
        if not quote or block.text.count(quote) != 1 or block.loc is None:
            raise ValueError("Authority primary must identify one exact located original quote")
        start = block.text.index(quote)
        primary = {"block_id": block.id, "start": start, "end": start+len(quote),
            "quote": quote, "kind": "candidate_primary", "covered": item["covered"],
            "block_sha256": hashlib.sha256(block.text.encode("utf-8")).hexdigest(),
            "loc": block.loc.model_dump(mode="json", exclude_none=True)}
        key = "primary_" + _digest(primary)[:24]
        sources[key] = primary
        current = joint_catalogs.get(index, catalog)
        visible = [key, *[sid for sid in current["sources"] if sid in sources]]
        for identifier in item["covered"]:
            row = {"item_index": index, "condition_id": identifier,
                "unit_ids": [u["unit_id"] for u in units if u["condition_id"] == identifier],
                "primary_source_id": key, "visible_source_ids": list(dict.fromkeys(visible))}
            pairs.append({"pair_id": "ap_" + _digest([scientific, item, row])[:24], **row})
    context = {"schema_version": "source-authority-registry-v1", "units": units,
        "pairs": pairs, "sources": sources,
        "snapshot_digest": _digest({"scientific_claim": scientific, "candidate_items": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in candidate_items],
        "paper_snapshot": {"paper_key": materials.paper_key, "source_pdf": materials.source_pdf,
            "markdown_path": materials.markdown_path, "markdown": materials.markdown,
            "blocks": [b.model_dump(mode="json") for b in materials.blocks],
            "artifacts": {str(path): _file_hash(path) for path in
                (materials.source_pdf, materials.markdown_path) if Path(path).is_file()}}}),
        "implementation_sha256": _file_hash(__file__)}
    return {**context, "context_digest": _digest(context)}


def decode_authority(response, context, materials, *, execution_scope_domain=None):
    """Validate exact rows and source access; unknown semantics are a legal result."""
    if response.get("schema_version") != "catalog-v3":
        raise ValueError("Fresh paper-support authority unavailable: catalog-v3 required")
    allowed = {"schema_version", "paper_scope_format", "conditions", "items", "source_authority",
        "plan_projection_reviews", "execution_choice_reviews"}
    if execution_scope_domain is not None:
        if (execution_scope_domain != "full-list-exact-match-statistics-v1"
                or response.get("execution_scope_domain") != execution_scope_domain
                or response.get("paper_scope_format") != "catalog-v2"):
            raise ValueError("Unknown or mismatched explicit statistics authority domain")
        allowed |= {"execution_scope_domain", "released_statistics_reviews"}
    if set(response) - allowed or not all(isinstance(response.get(k), list) for k in ("conditions", "items")):
        raise ValueError("Source-authority envelope has undeclared fields or malformed original lists")
    if response.get("paper_scope_format") not in {"legacy", "catalog-v1", "catalog-v2"}:
        raise ValueError("Unknown actual authority inner format")
    if response["paper_scope_format"] != "catalog-v2" and any(
        response.get(key, []) != [] for key in ("plan_projection_reviews", "execution_choice_reviews")
    ):
        raise ValueError("Legacy/v1 inner cannot authorize execution review arrays")
    raw = response.get("source_authority")
    review = SourceAuthorityReview.model_validate(raw)
    if review.context_digest != context["context_digest"]:
        raise ValueError("Source-authority context changed")
    if [r.unit_id for r in review.requirements] != [u["unit_id"] for u in context["units"]]:
        raise ValueError("Authority requirements must retain the complete ordered original denominator")
    if [r.pair_id for r in review.candidates] != [p["pair_id"] for p in context["pairs"]]:
        raise ValueError("Authority candidate pairs must retain the complete ordered original denominator")
    for row, unit in zip(review.requirements, context["units"], strict=True):
        if len(row.claim_source_ids) != len(set(row.claim_source_ids)) or not set(row.claim_source_ids).issubset(unit["claim_source_ids"]):
            raise ValueError("Authority requirement needs its real condition-covered Claim body sources")
        if row.requirement != "unresolved" and not row.claim_source_ids:
            raise ValueError("Authority requirement has no actual original Claim source")
    for row, pair in zip(review.candidates, context["pairs"], strict=True):
        if [u.unit_id for u in row.units] != pair["unit_ids"]:
            raise ValueError("Authority candidate omitted, duplicated or changed an original unit")
        for unit in row.units:
            ids = unit.evidence_source_ids
            if len(ids) != len(set(ids)) or not set(ids).issubset(pair["visible_source_ids"]):
                raise ValueError("Authority candidate source is synthetic, foreign or outside its original view")
            if unit.basis != "unresolved" and not ids:
                raise ValueError("Authority basis has no actual manuscript source")
    # A model cannot attach new raw keys or rely on identity normalization.
    if raw != review.model_dump(mode="json"):
        raise ValueError("Authority response must retain exact typed identities")
    return review


def _passages(decision):
    passages = list(decision.grounds)
    for row in decision.comparisons:
        passages.extend([row.left, row.right, *row.context])
        if row.difference is not None:
            passages.append(row.difference)
    return passages


def paper_authority_errors(review, context, pair_id, *, decision):
    """Paper support uses only the primary or actually decoded candidate grounds."""
    pair = next(p for p in context["pairs"] if p["pair_id"] == pair_id)
    candidate = next(r for r in review.candidates if r.pair_id == pair_id)
    requirements = {r.unit_id: r for r in review.requirements}
    consumed = {(p.block_id, p.quote) for p in _passages(decision)}
    matrix = {"manuscript_definition": "definition_or_protocol",
        "manuscript_reported_result": "reported_measurement"}
    errors = []
    for row in candidate.units:
        requirement = requirements[row.unit_id]
        if matrix.get(requirement.requirement) != row.basis:
            errors.append(f"{row.unit_id}: paper authority {requirement.requirement}/{row.basis} cannot establish support")
        for sid in row.evidence_source_ids:
            source = context["sources"][sid]
            if sid != pair["primary_source_id"] and (source["block_id"], source["quote"]) not in consumed:
                errors.append(f"{row.unit_id}: authority source was not consumed by this candidate")
    return errors


def retained_envelope(audit, claim, materials):
    """Rebuild retained real v3 input; do not trust a version label or self-seal."""
    from verification.experiment_sources import prepare_joint_candidate
    from verification.experiments import CatalogScopeReviewV3, ExperimentItem

    data, response = audit["input"], audit.get("response", {})
    if data.get("paper_scope_format") != "catalog-v2" or response.get("paper_scope_format") != "catalog-v2":
        raise ValueError("Execution subcontracts require the requested catalog-v2 inner contract")
    if data.get("output_schema") != CatalogScopeReviewV3.model_json_schema():
        raise ValueError("Retained execution scope did not request the current exact v3 schema")
    catalog = build_catalog(claim, materials)
    items = [ExperimentItem.model_validate(row) for row in data["candidate_items"]]
    joints = {i: prepare_joint_candidate(claim, materials, catalog, item, i,
        max_sources=data["joint_candidates"][str(i)]["manifest"]["max_sources"])
        for i, item in enumerate(items) if item.additional_sources}
    expected = build_authority_context(claim, materials, items, catalog=catalog, joint_catalogs=joints)
    if data.get("source_authority_context") != expected:
        raise ValueError("Retained source-authority input differs from the current original claim/materials")
    decode_authority(response, expected, materials)
    return True
