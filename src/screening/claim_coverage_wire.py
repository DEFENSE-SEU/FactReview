"""Pure v5 catalog factorization and explicit atom-to-compatibility views.

No semantic choices are inferred here. Existing source/carrier/materiality guards
validate the generated views before coverage can authorize any change.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import Field, StrictStr, field_validator

from schemas.claim import Contract
from screening.claim_scope import Dimension, ScopeEffect, ScopeSpan, ScopeState, fingerprint

METADATA_KEYS = frozenset({
    "block_id", "availability", "unavailable_reason", "block_digest", "trusted_span",
    "block_local_whole_span",
})
WIRE_VERSION = "claim-coverage-scope-catalog-v5"


def pack_scope_payload(payload):
    """Factor only identical source metadata; retain all science and edge fields."""
    if "source_catalog" in payload or "scope_wire_version" in payload:
        raise ValueError("Scope payload is already packed or contains reserved wire fields")
    result = copy.deepcopy(payload)
    catalog = {}
    for context in result["scope_context"]:
        refs = []
        for source in context["sources"]:
            if not source.keys() >= METADATA_KEYS:
                raise ValueError("Incomplete producer source metadata")
            key = source["block_id"]
            metadata = {k: source[k] for k in sorted(METADATA_KEYS)}
            if key in catalog and fingerprint(catalog[key]) != fingerprint(metadata):
                raise ValueError("Same source has conflicting producer metadata")
            catalog.setdefault(key, metadata)
            refs.append({"source_id": key, **{k: v for k, v in source.items() if k not in METADATA_KEYS}})
        context["sources"] = refs
    result.update(source_catalog=catalog, scope_wire_version=WIRE_VERSION)
    if unpack_scope_payload(result) != payload:
        raise ValueError("Scope catalog factorization changed original data")
    return result


def unpack_scope_payload(payload):
    """Strictly expand a current catalog; historical uncompressed inputs stay readable."""
    if "source_catalog" not in payload and "scope_wire_version" not in payload:
        return copy.deepcopy(payload)
    if payload.get("scope_wire_version") != WIRE_VERSION or not isinstance(payload.get("source_catalog"), dict):
        raise ValueError("Invalid scope catalog version or shape")
    result = copy.deepcopy(payload)
    catalog = result.pop("source_catalog")
    result.pop("scope_wire_version")
    used, claim_ids = set(), set()
    for key, value in catalog.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, dict):
            raise ValueError("Invalid source catalog identity")
        if set(value) != METADATA_KEYS or value["block_id"] != key:
            raise ValueError("Catalog metadata must be exact and cannot change source identity")
    for context in result["scope_context"]:
        claim_id = context["claim_id"]
        if claim_id in claim_ids:
            raise ValueError("Duplicate scope context claim")
        claim_ids.add(claim_id)
        sources, seen = [], set()
        for ref in context["sources"]:
            key = ref.get("source_id")
            if key not in catalog or key in seen or METADATA_KEYS & ref.keys():
                raise ValueError("Foreign, duplicate or metadata-overriding source edge")
            seen.add(key)
            used.add(key)
            sources.append({**catalog[key], **{k: v for k, v in ref.items() if k != "source_id"}})
        context["sources"] = sources
    if used != set(catalog):
        raise ValueError("Source catalog has undeclared extra entries")
    return result


class ScopeCarrierV5(Contract):
    claim_path: StrictStr = Field(
        min_length=1,
        description="Exact original semantic scalar leaf under /conditions or whole /text. Source/ID metadata is forbidden; path presence alone proves no entailment.",
    )
    claim_value: Any = Field(description="Exact unchanged JSON type and value at this original claim path.")

    @field_validator("claim_path")
    @classmethod
    def semantic_path(cls, value):
        if value == "/text":
            return value
        parts = value.split("/")
        if (len(parts) < 4 or parts[0] or parts[1] != "conditions" or not parts[2].isdigit()
                or parts[3] not in {"dataset", "metric", "description", "settings"}):
            raise ValueError("V5 carrier path must locate original claim semantics, never metadata")
        return value


class ScopeAtomV5(Contract):
    id: StrictStr = Field(min_length=1)
    dimension: Dimension
    condition_ids: list[StrictStr] = Field(min_length=1)
    state: ScopeState
    restriction: StrictStr = Field(min_length=1, description="One authoritative governing restriction.")
    sources: list[ScopeSpan] = Field(min_length=1)
    carriers: list[ScopeCarrierV5] = Field(
        description="Explicit original carriers for preserved only; nonempty and covering every atom condition. Missing/unresolved have none.",
    )
    effect: ScopeEffect | None
    reason: StrictStr = Field(min_length=1)


class ScopeSourceGroupV5(Contract):
    source_ids: list[StrictStr] = Field(min_length=1)
    state: Literal["considered", "irrelevant", "unavailable"]
    condition_ids: list[StrictStr] = Field(min_length=1)
    reason: StrictStr = Field(min_length=1, description="Explicit same scoped judgment for every listed source; no default remainder.")


def expand_scope_row(raw):
    """Create indexed compatibility views only from explicit v5 atom choices."""
    result = copy.deepcopy(raw)
    sources = result.pop("source_review_groups")
    result["source_reviews"] = []
    seen = set()
    fields = []
    for index, group in enumerate(sources):
        for key in group["source_ids"]:
            if not key.strip() or key in seen:
                raise ValueError("V5 source partition contains empty or repeated IDs")
            seen.add(key)
            generated_index = len(result["source_reviews"])
            result["source_reviews"].append({"block_id": key, **{k: v for k, v in group.items() if k != "source_ids"}})
            fields.append({"raw_path": f"/source_review_groups/{index}", "generated_path": f"/source_reviews/{generated_index}"})
    result["preserved_qualifiers"] = []
    result["findings"] = result.pop("other_findings")
    fields.extend({"raw_path": f"/other_findings/{i}", "generated_path": f"/findings/{i}"}
                  for i in range(len(result["findings"])))
    for index, atom in enumerate(result["scope_atoms"]):
        carriers = atom.pop("carriers")
        atom.update(preserved_indices=[], finding_index=None)
        base = f"/scope_atoms/{index}"
        if len({fingerprint(c) for c in carriers}) != len(carriers):
            raise ValueError("V5 atom carriers must not repeat the same original path/value")
        if atom["state"] == "preserved":
            if not carriers or atom["effect"] is not None:
                raise ValueError("V5 preserved atom requires explicit carriers and no missing effect")
            for carrier in carriers:
                i = len(result["preserved_qualifiers"])
                atom["preserved_indices"].append(i)
                result["preserved_qualifiers"].append({
                    "restriction": atom["restriction"],
                    "source_block_ids": [s["block_id"] for s in atom["sources"]],
                    **carrier,
                })
                fields.append({"raw_path": base, "generated_path": f"/preserved_qualifiers/{i}"})
        elif atom["state"] == "missing":
            if carriers or atom["effect"] is None:
                raise ValueError("V5 missing atom requires its explicit effect and no preserved carriers")
            i = len(result["findings"])
            atom["finding_index"] = i
            result["findings"].append({
                "kind": "missing_qualifier_or_condition", "condition_ids": atom["condition_ids"],
                "source_block_ids": [s["block_id"] for s in atom["sources"]],
                "restriction": atom["restriction"], "material_effect": atom["effect"]["explanation"],
                "reason": atom["reason"],
            })
            fields.append({"raw_path": base, "generated_path": f"/findings/{i}"})
        elif atom["state"] != "unresolved" or carriers or atom["effect"] is not None:
            raise ValueError("V5 unresolved atom has no carriers/effect; non-governing has no atom")
    return result, {"raw_digest": fingerprint(raw), "generated_digest": fingerprint(result), "fields": fields}
