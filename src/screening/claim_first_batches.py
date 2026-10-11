"""Non-semantic partition of full first-review contexts and exact raw provenance."""

from __future__ import annotations

import copy
import hashlib
import json

from screening.claim_scope import fingerprint


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def plan_first_batches(parent, size):
    if type(size) is not int or size < 1:
        raise ValueError("First batch size must be an integer >= 1")
    required = parent["required_claim_checks"]
    registry = {r["claim_id"]: r for r in parent["current_claims"]}
    ids = [r["claim_id"] for r in required]
    contexts = {r["claim_id"]: r for r in parent["scope_context"]}
    if (len(registry) != len(parent["current_claims"]) or len(ids) != len(set(ids))
            or len(contexts) != len(parent["scope_context"]) or any(
                key not in registry or key not in contexts for key in ids)
            or any(r["digest"] != registry[r["claim_id"]]["digest"] for r in required)):
        raise ValueError("First parent lacks unique current required identities/source directories")
    parent_digest = digest(parent)
    discovery = {**copy.deepcopy(parent), "parent_digest": parent_digest}
    batches = []
    for start in range(0, len(required), size):
        # Parent is NOT packed. Keep full scientific context; only output duties change.
        child = {**copy.deepcopy(parent), "parent_digest": parent_digest,
                 "batch_id": f"first_batch_{len(batches) + 1:03d}",
                 "required_claim_checks": copy.deepcopy(required[start:start + size])}
        batches.append(child)
    return parent_digest, discovery, batches


def check_first_rows(payload, response):
    expected = [(r["claim_id"], r["digest"]) for r in payload["required_claim_checks"]]
    actual = [(r["claim_id"], r["claim_digest"]) for r in response["claim_reviews"]]
    if actual != expected:
        raise ValueError("First batch must return every exact required review once in order")


def discovery_remaining_blocks(parent, response):
    """Check raw reading declarations before any union can hide an invalid ID/order."""
    ids = [block["id"] for block in parent["blocks"]]
    reviewed = response["reviewed_block_ids"]
    selected = set(reviewed)
    if len(selected) != len(reviewed) or reviewed != [key for key in ids if key in selected]:
        raise ValueError("Discovery reviewed IDs must be a unique ordered subset of the original window")
    if parent["discovery_required_block_ids"] != ids:
        raise ValueError("Discovery duties must retain every original window block in order")
    return [key for key in ids if key not in selected]


def _source_acknowledgements(parent, row, result, batch_id, local):
    """Locate explicit source-reading judgments; no entailment or omission judgment is created."""
    context = next(c for c in parent["scope_context"] if c["claim_id"] == row["claim_id"])
    claim = next(c for c in parent["current_claims"] if c["claim_id"] == row["claim_id"])
    if context["claim_digest"] != row["claim_digest"] or claim["digest"] != row["claim_digest"]:
        raise ValueError("First source acknowledgement changed the original claim identity")
    bodies = {b["id"]: b for b in parent["blocks"]}
    window_ids = set(bodies)
    for supplied in parent.get("supplemental_sources", []):
        body = supplied["block"]
        if body["id"] in bodies and bodies[body["id"]] != body:
            raise ValueError("First source acknowledgement has conflicting supplied bodies")
        bodies[body["id"]] = body
    sources = {s["block_id"]: s for s in context["sources"]}
    conditions = {c["id"] for c in claim["conditions"]}
    seen, mappings = set(), []
    for index, group in enumerate(row["source_review_groups"]):
        covered = set(group["condition_ids"])
        if not covered or len(covered) != len(group["condition_ids"]) or not covered <= conditions:
            raise ValueError("First source acknowledgement contains repeated or foreign conditions")
        for source_index, key in enumerate(group["source_ids"]):
            if key in seen:
                raise ValueError("First source acknowledgement repeats a source")
            seen.add(key)
            source, body = sources.get(key), bodies.get(key)
            if source is None and key not in window_ids:
                raise ValueError("First source acknowledgement contains a foreign source")
            if source is not None and not set(source["condition_ids"]) <= covered:
                raise ValueError("First source acknowledgement omits affected source conditions")
            unavailable = source is not None and source["availability"] == "unavailable"
            if (group["state"] == "unavailable") != unavailable:
                raise ValueError("First source acknowledgement contradicts original source availability")
            if unavailable:
                continue
            if body is None or not body["text"].strip() or body.get("loc") is None:
                raise ValueError("First source acknowledgement lacks the supplied original body")
            body_digest = fingerprint(body)
            if source is not None and source["block_digest"] != body_digest:
                raise ValueError("First source acknowledgement changed the original body digest")
            path = f"/claim_reviews/{local}/source_review_groups/{index}"
            mappings.append({
                "batch_id": batch_id, "claim_id": row["claim_id"], "block_id": key,
                "state": group["state"], "condition_ids": copy.deepcopy(group["condition_ids"]),
                "raw_audit_path": result["raw_audit_path"], "raw_path": path,
                "raw_digest": digest(group), "source_path": f"{path}/source_ids/{source_index}",
                "block_digest": body_digest,
            })
    return mappings


def aggregate_first_batches(parent_digest, discovery, batches, results, *, parent):
    if digest(parent) != parent_digest:
        raise ValueError("First aggregation changed the complete parent identity")
    if len(results) > len(batches):
        raise ValueError("First aggregation has undeclared batches")
    receipts, rows, mappings, source_mappings = [], [], [], []
    for index, result in enumerate([discovery, *results]):
        if result is None:
            receipts.append(None)
            continue
        receipts.append({k: copy.deepcopy(result[k]) for k in (
            "context_id", "input_digest", "response_digest", "parsed_digest", "status", "raw_audit_path")})
        response = result["parsed_response"]
        if response is None:
            continue
        if response["parent_digest"] != parent_digest or digest(response) != result["parsed_digest"]:
            raise ValueError("First receipt parent/parsed identity changed")
        if index == 0:
            continue
        payload = batches[index - 1]
        if response["batch_id"] != payload["batch_id"]:
            raise ValueError("First receipt batch order/identity changed")
        check_first_rows(payload, response)
        for local, row in enumerate(response["claim_reviews"]):
            source_mappings.extend(_source_acknowledgements(
                parent, row, result, payload["batch_id"], local))
            mappings.append({"batch_id": payload["batch_id"], "raw_audit_path": result["raw_audit_path"],
                             "raw_path": f"/claim_reviews/{local}", "aggregate_path": f"/claim_reviews/{len(rows)}",
                             "raw_digest": digest(row)})
            rows.append(copy.deepcopy(row))
    discovery_response = discovery["parsed_response"] if discovery is not None else None
    if discovery_response is not None:
        discovery_remaining_blocks(parent, discovery_response)
    reviewed = set(discovery_response["reviewed_block_ids"]) if discovery_response else set()
    window_ids = [body["id"] for body in parent["blocks"]]
    reviewed.update(row["block_id"] for row in source_mappings if row["block_id"] in window_ids)
    reviewed_ids = [key for key in window_ids if key in reviewed]
    for row in source_mappings:
        row["aggregate_path"] = (f"/reviewed_block_ids/{reviewed_ids.index(row['block_id'])}"
                                 if row["block_id"] in reviewed_ids else None)
    findings = copy.deepcopy(discovery_response["new_findings"]) if discovery_response else []
    discovery_mappings = [{"raw_audit_path": discovery["raw_audit_path"],
                           "raw_path": f"/new_findings/{index}", "aggregate_path": f"/new_findings/{index}",
                           "raw_digest": digest(row)} for index, row in enumerate(findings)]
    context = digest({"parent_digest": parent_digest, "planned_batches": [digest(p) for p in batches],
                      "raw_receipts": receipts})
    return {"version": "program-first-review-adapter-v1", "context_id": context,
            "claim_reviews": rows, "new_findings": findings,
            "reviewed_block_ids": reviewed_ids, "reviewed_source_mappings": source_mappings,
            "explanation": discovery_response["explanation"] if discovery_response else "Discovery unavailable; original duties remain unreviewed.",
            "raw_receipts": receipts, "path_mappings": mappings, "discovery_mappings": discovery_mappings}
