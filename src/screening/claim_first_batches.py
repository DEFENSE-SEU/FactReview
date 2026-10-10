"""Non-semantic partition of full first-review contexts and exact raw provenance."""

from __future__ import annotations

import copy
import hashlib
import json


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


def aggregate_first_batches(parent_digest, discovery, batches, results):
    if len(results) > len(batches):
        raise ValueError("First aggregation has undeclared batches")
    receipts, rows, mappings = [], [], []
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
            mappings.append({"batch_id": payload["batch_id"], "raw_audit_path": result["raw_audit_path"],
                             "raw_path": f"/claim_reviews/{local}", "aggregate_path": f"/claim_reviews/{len(rows)}",
                             "raw_digest": digest(row)})
            rows.append(copy.deepcopy(row))
    discovery_response = discovery["parsed_response"] if discovery is not None else None
    findings = copy.deepcopy(discovery_response["new_findings"]) if discovery_response else []
    discovery_mappings = [{"raw_audit_path": discovery["raw_audit_path"],
                           "raw_path": f"/new_findings/{index}", "aggregate_path": f"/new_findings/{index}",
                           "raw_digest": digest(row)} for index, row in enumerate(findings)]
    context = digest({"parent_digest": parent_digest, "planned_batches": [digest(p) for p in batches],
                      "raw_receipts": receipts})
    return {"version": "program-first-review-adapter-v1", "context_id": context,
            "claim_reviews": rows, "new_findings": findings,
            "reviewed_block_ids": copy.deepcopy(discovery_response["reviewed_block_ids"]) if discovery_response else [],
            "explanation": discovery_response["explanation"] if discovery_response else "Discovery unavailable; original duties remain unreviewed.",
            "raw_receipts": receipts, "path_mappings": mappings, "discovery_mappings": discovery_mappings}
