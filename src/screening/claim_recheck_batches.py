"""Exact, non-semantic partition and aggregation of frozen original reviews."""

from __future__ import annotations

import copy
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def plan_original_batches(parent, size):
    if type(size) is not int or size < 1:
        raise ValueError("Original batch size must be an integer >= 1")
    required = parent["required_original_claim_reviews"]
    registry = {r["claim_id"]: r for r in parent["current_claims"]}
    contexts = {r["claim_id"]: r for r in parent["scope_context"]}
    ids = [r["claim_id"] for r in required]
    if (
        len(set(ids)) != len(ids)
        or len(registry) != len(parent["current_claims"])
        or len(contexts) != len(parent["scope_context"])
    ):
        raise ValueError("Original batch parent contains duplicate identities")
    if any(key not in registry or key not in contexts for key in ids):
        raise ValueError("Original batch parent lacks a required claim or source directory")
    if any(r["digest"] != registry[r["claim_id"]]["digest"] for r in required):
        raise ValueError("Original batch parent has a stale required claim")
    parent_digest = digest(parent)
    batches = []
    for start in range(0, len(required), size):
        rows = copy.deepcopy(required[start : start + size])
        batch_ids = [r["claim_id"] for r in rows]
        payload = {
            key: copy.deepcopy(parent[key])
            for key in ("window_id", "blocks", "observations", "supplemental_sources", "parent_provenance", "review_only_block_ids")
            if key in parent
        }
        payload.update(
            parent_digest=parent_digest,
            batch_id=f"original_batch_{len(batches) + 1:03d}",
            required_original_claim_reviews=rows,
            current_claims=[copy.deepcopy(registry[key]) for key in batch_ids],
            scope_context=[copy.deepcopy(contexts[key]) for key in batch_ids],
        )
        batches.append(payload)
    return parent_digest, batches


def check_batch_rows(payload, response):
    expected = [(r["claim_id"], r["digest"]) for r in payload["required_original_claim_reviews"]]
    actual = [(r["claim_id"], r["claim_digest"]) for r in response["original_claim_reviews"]]
    if expected != actual:
        raise ValueError("Original batch must return every exact required row once in order")


def aggregate_original_batches(parent_digest, batches, results):
    """Move only the original-review index prefix. Science and atom paths stay exact."""
    if len(results) > len(batches):
        raise ValueError("Original aggregation has undeclared batches")
    rows, links, mappings, receipts, errors = [], [], [], [], []
    for payload, result in zip(batches, results, strict=False):
        if result["batch_id"] != payload["batch_id"]:
            raise ValueError("Original aggregation batch order/identity changed")
        receipts.append({key: copy.deepcopy(result[key]) for key in ("batch_id", "response_digest", "error")})
        response = result["response"]
        if response is None:
            continue
        if (
            response["parent_digest"] != parent_digest
            or response["batch_id"] != payload["batch_id"]
            or digest(response) != result["response_digest"]
        ):
            raise ValueError("Original aggregation response identity or digest changed")
        check_batch_rows(payload, response)
        offset = len(rows)
        for index, row in enumerate(response["original_claim_reviews"]):
            rows.append(copy.deepcopy(row))
            mappings.append(
                {
                    "batch_id": payload["batch_id"],
                    "raw_path": f"/original_claim_reviews/{index}",
                    "aggregate_path": f"/original_claim_reviews/{offset + index}",
                    "raw_digest": digest(row),
                }
            )
        for link in response["observation_links"]:
            match = re.fullmatch(r"/original_claim_reviews/(0|[1-9][0-9]*)(/.*)", link["review_path"])
            if match is None or int(match[1]) >= len(response["original_claim_reviews"]):
                errors.append(f"{payload['batch_id']}: original link is outside its own review rows")
                continue
            copied = copy.deepcopy(link)
            copied["review_path"] = f"/original_claim_reviews/{offset + int(match[1])}{match[2]}"
            links.append(copied)
            mappings.append(
                {
                    "batch_id": payload["batch_id"],
                    "raw_path": link["review_path"],
                    "aggregate_path": copied["review_path"],
                    "raw_digest": digest(link),
                }
            )
    context = digest(
        {
            "parent_digest": parent_digest,
            "planned_batches": [digest(p) for p in batches],
            "raw_receipts": receipts,
        }
    )
    aggregate = {
        "version": "program-original-batch-adapter-v1",
        "context_id": context,
        "original_claim_reviews": rows,
        "observation_links": links,
        "raw_receipts": receipts,
        "path_mappings": mappings,
        "aggregation_errors": errors,
    }
    return aggregate, digest(aggregate)
