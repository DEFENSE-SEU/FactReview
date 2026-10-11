"""External model fixtures for the explicit experiment authority contract.

Callers choose the scientific requirement/basis from their fixture's original
meaning. This helper copies program selectors and order only; it never reads
production sufficiency/status or invokes a scientific validator.
"""

from __future__ import annotations

import copy


def _choice(profile, condition_id):
    return profile[condition_id] if isinstance(profile, dict) else profile


def _source_for_passage(payload, passage):
    context = payload["source_authority_context"]
    sources = context.get("sources", {})
    matched = [sid for sid, source in sources.items()
               if source.get("block_id") == passage.get("block_id")
               and source.get("quote") == passage.get("quote")]
    if matched:
        return matched[0]
    catalog = payload["catalog"]
    fields = catalog["source_fields"]
    blocks = {block["id"]: block["text"] for block in payload["paper_blocks"]}
    for sid, values in catalog["sources"].items():
        source = dict(zip(fields, values, strict=True))
        block = blocks.get(source.get("block_id"))
        if (block is not None and source.get("block_id") == passage.get("block_id")
                and block[source["start"]:source["end"]] == passage.get("quote")):
            return sid
    # Keep an original invalid passage invalid, with a visible unmapped selector.
    return "fixture_unmapped_original_passage"


def _declared_source_ids(payload, row):
    result = []
    def visit(value):
        if isinstance(value, dict):
            if set(value) >= {"block_id", "quote"}:
                result.append(_source_for_passage(payload, value))
            for key, child in value.items():
                if key in {"grounds_source_ids", "ground_source_ids", "source_ids", "context_source_ids"}:
                    result.extend(child)
                elif key == "source_id":
                    result.append(child)
                elif key == "number_id":
                    catalog = payload["catalog"]
                    values = catalog["numbers"].get(child)
                    if values is None:
                        result.append("fixture_unmapped_original_number")
                    else:
                        number = dict(zip(catalog["number_fields"], values, strict=True))
                        result.append(_source_for_passage(payload, {
                            "block_id": number["block_id"],
                            "quote": catalog["sentences"][number["sentence_id"]],
                        }))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(row)
    return list(dict.fromkeys(result))


def authority_scope_fixture(payload, response, *, requirement, basis, rationale):
    """Supply explicit semantic rows while retaining the original inner response.

    The requested format is explicit in the real payload. Only the default-v2
    empty-comparison fixtures need grounds selectors; nonempty legacy comparison
    fields must use a caller-requested legacy format and are never translated.
    """
    raw = copy.deepcopy(response)
    paper_format = payload["paper_scope_format"]
    if paper_format == "catalog-v2":
        for row in raw.get("items", []):
            if "grounds" in row:
                if row.get("comparisons"):
                    raise ValueError("Nonempty legacy comparisons need an explicit legacy fixture request")
                row["grounds_source_ids"] = [_source_for_passage(payload, p) for p in row.pop("grounds")]
    raw["schema_version"] = "catalog-v3"
    raw["paper_scope_format"] = paper_format
    registry = payload["source_authority_context"]
    units = {row["unit_id"]: row for row in registry["units"]}
    requirements = [
        {"unit_id": row["unit_id"], "requirement": _choice(requirement, row["condition_id"]),
         "claim_source_ids": copy.deepcopy(row["claim_source_ids"]), "rationale": rationale}
        for row in registry["units"]
    ]
    candidates = []
    for pair in registry["pairs"]:
        rows = [row for row in raw.get("items", [])
                if row.get("item_index") == pair["item_index"]
                and row.get("condition_id") == pair["condition_id"]]
        source_ids = [pair["primary_source_id"]]
        if len(rows) == 1:
            source_ids.extend(_declared_source_ids(payload, rows[0]))
        source_ids = list(dict.fromkeys(source_ids))
        candidates.append({"pair_id": pair["pair_id"], "units": [
            {"unit_id": uid, "basis": _choice(basis, units[uid]["condition_id"]),
             "evidence_source_ids": copy.deepcopy(source_ids), "rationale": rationale}
            for uid in pair["unit_ids"]
        ]})
    raw["source_authority"] = {"schema_version": "source-authority-v1",
        "context_digest": registry["context_digest"], "requirements": requirements,
        "candidates": candidates}
    return raw
