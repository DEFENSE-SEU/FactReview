"""Exact, non-semantic partition and aggregation of frozen original reviews."""

from __future__ import annotations

import copy
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def plan_original_batches(parent, size, *, preserve_full_context=False):
    if type(size) is not int or size < 1:
        raise ValueError("Original batch size must be an integer >= 1")
    if type(preserve_full_context) is not bool:
        raise ValueError("preserve_full_context must be a boolean")
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
            current_claims=copy.deepcopy(parent["current_claims"]) if preserve_full_context
            else [copy.deepcopy(registry[key]) for key in batch_ids],
            scope_context=copy.deepcopy(parent["scope_context"]) if preserve_full_context
            else [copy.deepcopy(contexts[key]) for key in batch_ids],
        )
        batches.append(payload)
    return parent_digest, batches


def check_batch_rows(payload, response):
    expected = [(r["claim_id"], r["digest"]) for r in payload["required_original_claim_reviews"]]
    actual = [(r["claim_id"], r["claim_digest"]) for r in response["original_claim_reviews"]]
    if expected != actual:
        raise ValueError("Original batch must return every exact required row once in order")


def plan_remaining_original_batches(parent, parent_digest, remaining_ids, size):
    """Narrow output duties once; retain the complete frozen scientific input."""
    verified_digest, _ = plan_original_batches(parent, size)
    required = parent["required_original_claim_reviews"]
    remaining = [row for row in required if row["claim_id"] in remaining_ids]
    if (verified_digest != parent_digest or not remaining_ids
            or remaining_ids != [row["claim_id"] for row in remaining]):
        raise ValueError("Remaining original duties differ from the frozen ordered parent")
    batches = []
    for start in range(0, len(remaining), size):
        payload = {key: copy.deepcopy(parent[key]) for key in
                   ("window_id", "blocks", "observations", "supplemental_sources", "parent_provenance",
                    "review_only_block_ids", "current_claims", "scope_context") if key in parent}
        payload.update(parent_digest=parent_digest, batch_id=f"original_retry_001_{len(batches) + 1:03d}",
                       required_original_claim_reviews=copy.deepcopy(remaining[start:start + size]))
        batches.append(payload)
    return batches


def select_original_rows(parent, parent_digest, batches, results, valid_original_ids,
                         retry_batches, retry_results):
    """Select whole independently validated raw rows; move only index prefixes."""
    if digest(parent) != parent_digest:
        raise ValueError("Original selection parent changed")
    all_batches, all_results = [*batches, *retry_batches], [*results, *retry_results]
    if len(results) != len(batches) or len(retry_results) > len(retry_batches):
        raise ValueError("Original selection lacks its initial batch receipts")
    if len({payload["batch_id"] for payload in all_batches}) != len(all_batches):
        raise ValueError("Original selection repeats batch identities")
    # The original exact identity/order/link geometry checks remain mandatory for every raw.
    validated, _ = aggregate_original_batches(parent_digest, all_batches, all_results)
    if validated["aggregation_errors"]:
        raise ValueError("Original selection has an invalid raw link")
    required = parent["required_original_claim_reviews"]
    required_ids = [row["claim_id"] for row in required]
    if not set(valid_original_ids) <= set(required_ids):
        raise ValueError("Original selection has foreign accepted duties")
    selected, receipts = {}, []
    for batch_index, (payload, result) in enumerate(zip(all_batches, all_results, strict=False)):
        receipt = {key: copy.deepcopy(result.get(key)) for key in
                   ("batch_id", "response_digest", "error", "request_context_id", "input_digest", "raw_audit_path")}
        receipts.append(receipt)
        response = result["response"]
        if response is None:
            continue
        if response["context_id"] != result["request_context_id"]:
            raise ValueError("Original selection lost its independent request context")
        for index, row in enumerate(response["original_claim_reviews"]):
            key = row["claim_id"]
            eligible = key in valid_original_ids if batch_index < len(batches) else key not in valid_original_ids
            if not eligible:
                continue
            if key in selected or key not in required_ids:
                raise ValueError("Original selection has duplicate or foreign rows")
            selected[key] = (payload, result, index, row)
    rows, links, mappings, selections, unselected_links = [], [], [], [], []
    for key in required_ids:
        if key not in selected:
            continue
        payload, result, index, row = selected[key]
        target_index = len(rows)
        rows.append(copy.deepcopy(row))
        mapping = {"batch_id": payload["batch_id"], "raw_path": f"/original_claim_reviews/{index}",
                   "aggregate_path": f"/original_claim_reviews/{target_index}", "raw_digest": digest(row)}
        mappings.append(mapping)
        selections.append({"claim_id": key, "claim_digest": row["claim_digest"], **mapping,
                           "response_digest": result["response_digest"], "request_context_id": result["request_context_id"],
                           "raw_audit_path": result["raw_audit_path"]})
    positions = {row["claim_id"]: index for index, row in enumerate(rows)}
    for payload, result in zip(all_batches, all_results, strict=False):
        response = result["response"]
        if response is None:
            continue
        for link in response["observation_links"]:
            match = re.fullmatch(r"/original_claim_reviews/(0|[1-9][0-9]*)(/.*)", link["review_path"])
            index = int(match[1])
            key = response["original_claim_reviews"][index]["claim_id"]
            mapping = {"batch_id": payload["batch_id"], "raw_path": link["review_path"], "raw_digest": digest(link)}
            if key not in selected or selected[key][0]["batch_id"] != payload["batch_id"]:
                unselected_links.append(mapping)
                continue
            copied = copy.deepcopy(link)
            copied["review_path"] = f"/original_claim_reviews/{positions[key]}{match[2]}"
            links.append(copied)
            mappings.append({**mapping, "aggregate_path": copied["review_path"]})
    context = digest({"parent_digest": parent_digest, "planned_batches": [digest(p) for p in all_batches],
                      "raw_receipts": receipts, "row_selections": selections, "unselected_raw_links": unselected_links})
    aggregate = {"version": "program-original-row-selection-adapter-v1", "context_id": context,
                 "original_claim_reviews": rows, "observation_links": links, "raw_receipts": receipts,
                 "path_mappings": mappings, "row_selections": selections, "unselected_raw_links": unselected_links,
                 "aggregation_errors": []}
    return aggregate, digest(aggregate)


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
