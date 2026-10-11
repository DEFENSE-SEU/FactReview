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


def plan_remaining_first_batches(parent, parent_digest, invalid_ids, size):
    """Keep the frozen full input; narrow only this wave's output obligations."""
    verified, _, _ = plan_first_batches(parent, size)
    required = [row for row in parent["required_claim_checks"] if row["claim_id"] in invalid_ids]
    if verified != parent_digest or not invalid_ids or invalid_ids != [row["claim_id"] for row in required]:
        raise ValueError("Remaining first duties differ from the frozen ordered parent")
    return [{**copy.deepcopy(parent), "parent_digest": parent_digest,
             "batch_id": f"first_retry_001_{index + 1:03d}",
             "required_claim_checks": copy.deepcopy(required[start:start + size])}
            for index, start in enumerate(range(0, len(required), size))]


def check_first_rows(payload, response):
    expected = [(r["claim_id"], r["digest"]) for r in payload["required_claim_checks"]]
    actual = [(r["claim_id"], r["claim_digest"]) for r in response["claim_reviews"]]
    if actual != expected:
        raise ValueError("First batch must return every exact required review once in order")


def first_failure_identity(payload, response):
    """Only a lossless JSON response with every exact closed request identity can retry."""
    def plain(value):
        if type(value) in (str, int, float, bool, type(None)):
            return True
        if type(value) is list:
            return all(plain(item) for item in value)
        return type(value) is dict and all(type(key) is str and plain(item) for key, item in value.items())

    if (type(response) is not dict or not plain(response)
            or set(response) != {"schema_version", "context_id", "window_id", "parent_digest", "batch_id", "claim_reviews"}
            or response["schema_version"] != "claim-coverage-first-batch-v1"
            or any(type(response[key]) is not str or response[key] != payload[key]
                   for key in ("context_id", "window_id", "parent_digest", "batch_id"))
            or type(response["claim_reviews"]) is not list):
        return False
    identities = {row["claim_id"]: row["digest"] for row in payload["required_claim_checks"]}
    supplied = {body["id"] for body in payload["blocks"]}
    supplied.update(row["block"]["id"] for row in payload.get("supplemental_sources", []))
    directory = set(payload.get("source_catalog", {})) | supplied
    contexts = {row["claim_id"]: row for row in payload["scope_context"]}
    claims = {row["claim_id"]: row for row in payload["current_claims"]}

    def selected(values, available):
        return type(values) is list and bool(values) and all(type(key) is str and key in available for key in values)

    for row in response["claim_reviews"]:
        if (type(row) is not dict or type(row.get("claim_id")) is not str
                or row["claim_id"] not in identities or row.get("claim_digest") != identities[row["claim_id"]]
                or not selected(row.get("source_block_ids"), supplied)):
            return False
        for field in ("assertion_groups", "other_findings"):
            if type(row.get(field)) is not list or any(type(group) is not dict
                    or not selected(group.get("source_block_ids"), supplied) for group in row[field]):
                return False
        if type(row.get("source_review_groups")) is not list or any(type(group) is not dict
                or not selected(group.get("source_ids"), directory) for group in row["source_review_groups"]):
            return False
        sources = {source.get("source_id", source.get("block_id")): source
                   for source in contexts[row["claim_id"]]["sources"]}
        condition_ids = {condition["id"] for condition in claims[row["claim_id"]]["conditions"]}
        acknowledgements = {}
        for group in row["source_review_groups"]:
            if (group.get("state") not in {"considered", "irrelevant", "unavailable"}
                    or not selected(group.get("condition_ids"), condition_ids)
                    or len(set(group["condition_ids"])) != len(group["condition_ids"])):
                return False
            for key in group["source_ids"]:
                if key in acknowledgements or (key not in sources and key not in {b["id"] for b in payload["blocks"]}):
                    return False
                source = sources.get(key)
                availability = payload.get("source_catalog", {}).get(key, source or {}).get("availability")
                if ((group["state"] == "unavailable") != (availability == "unavailable")
                        or (source is not None and not set(source["condition_ids"]) <= set(group["condition_ids"]))):
                    return False
                acknowledgements[key] = group["state"]
        if not set(sources) <= set(acknowledgements):
            return False
        if type(row.get("scope_atoms")) is not list or any(type(atom) is not dict
                or type(atom.get("sources")) is not list or not atom["sources"] or any(type(span) is not dict
                    or type(span.get("block_id")) is not str or span["block_id"] not in supplied
                    or acknowledgements.get(span["block_id"]) != "considered"
                    for span in atom["sources"]) for atom in row["scope_atoms"]):
            return False
    try:
        json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def check_first_failed_receipt(parent, payload, result):
    """A failed fixed slot can retain provenance, and supplies no selectable scientific row."""
    if (result is None or result["status"] != "failed" or result["parsed_response"] is not None
            or result.get("failure_stage") not in {"schema", "required_order"}
            or any(not result.get(key) for key in ("context_id", "input_digest", "response_digest", "raw_audit_path"))):
        raise ValueError("First failed slot lacks a bound mechanical failure receipt")
    expected = {**copy.deepcopy(parent), "parent_digest": digest(parent), "batch_id": payload["batch_id"],
                "required_claim_checks": payload["required_claim_checks"]}
    if digest(payload) != digest(expected):
        raise ValueError("First failed slot changed its complete frozen input")


def check_first_receipt(parent, payload, result):
    """Check saved typed/raw identities before any first row can be selected."""
    if (result is None or result["status"] != "returned" or result["parsed_response"] is None
            or not result["context_id"] or not result["input_digest"] or not result["raw_audit_path"]):
        raise ValueError("First selection lacks a complete independent receipt")
    response = result["parsed_response"]
    if (response["context_id"] != result["context_id"] or response["window_id"] != parent["window_id"]
            or response["parent_digest"] != digest(parent)
            or digest(response) != result["parsed_digest"]
            or result["parsed_digest"] != result["response_digest"]):
        raise ValueError("First selection changed request, raw or typed identity")
    expected = {**copy.deepcopy(parent), "parent_digest": digest(parent)}
    if "batch_id" in payload:
        expected.update(batch_id=payload["batch_id"], required_claim_checks=payload["required_claim_checks"])
        if response["batch_id"] != payload["batch_id"] or response["schema_version"] != "claim-coverage-first-batch-v1":
            raise ValueError("First selection changed batch protocol or identity")
        check_first_rows(payload, response)
    elif response["schema_version"] != "claim-coverage-discovery-v1":
        raise ValueError("First selection changed discovery protocol")
    if digest(payload) != digest(expected):
        raise ValueError("First selection changed the complete scientific parent input")


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


def select_first_rows(parent, parent_digest, discovery, batches, results, valid_ids, retry_batches, retry_results,
                      *, partial=False):
    """Select complete validated rows after an entirely successful recovery wave."""
    if type(partial) is not bool:
        raise ValueError("First partial selection must use a boolean")
    if digest(parent) != parent_digest or len(results) != len(batches) or len(retry_results) != len(retry_batches):
        raise ValueError("First selection lacks its complete frozen wave")
    all_batches, all_results = [*batches, *retry_batches], [*results, *retry_results]
    if not retry_batches or len({p["batch_id"] for p in all_batches}) != len(all_batches):
        raise ValueError("First selection has empty or repeated recovery batches")
    required = parent["required_claim_checks"]
    required_ids = [row["claim_id"] for row in required]
    if not set(valid_ids) <= set(required_ids):
        raise ValueError("First selection contains foreign accepted duties")
    if [row for p in batches for row in p["required_claim_checks"]] != required:
        raise ValueError("First selection changed initial obligations")
    invalid = [row for row in required if row["claim_id"] not in valid_ids]
    if [row for p in retry_batches for row in p["required_claim_checks"]] != invalid:
        raise ValueError("First selection changed the exact mechanical recovery duties")
    discovery_payload = {**copy.deepcopy(parent), "parent_digest": parent_digest}
    check_first_receipt(parent, discovery_payload, discovery)
    if discovery_remaining_blocks(parent, discovery["parsed_response"]):
        raise ValueError("First selection lacks full original discovery")
    for index, (payload, result) in enumerate(zip(all_batches, all_results, strict=True)):
        if partial and index < len(batches) and result["parsed_response"] is None:
            check_first_failed_receipt(parent, payload, result)
            continue
        check_first_receipt(parent, payload, result)
        if index >= len(batches):
            lowering = result["lowering"]
            expected = [(f"/claim_reviews/{i}", row["claim_id"])
                        for i, row in enumerate(payload["required_claim_checks"])]
            actual = [(row["raw_path"], row["claim_id"]) for row in lowering["mappings"]]
            if lowering["errors"] or actual != expected:
                raise ValueError("First selection lacks every validated fresh scientific row")
    initial = aggregate_first_batches(parent_digest, discovery, batches, results, parent=parent)
    aggregate_first_batches(parent_digest, discovery, retry_batches, retry_results, parent=parent)
    selected = {}
    for index, (payload, result) in enumerate(zip(all_batches, all_results, strict=True)):
        if partial and result["parsed_response"] is None:
            continue
        for local, row in enumerate(result["parsed_response"]["claim_reviews"]):
            key = row["claim_id"]
            if (key in valid_ids) != (index < len(batches)):
                continue
            if key in selected or key not in required_ids:
                raise ValueError("First selection contains repeated or foreign rows")
            selected[key] = (payload, result, local, row)
    if set(selected) != set(required_ids):
        raise ValueError("First selection is incomplete; successful subsets cannot be adopted")
    rows, mappings, selections, source_mappings = [], [], [], []
    for key in required_ids:
        payload, result, local, row = selected[key]
        mapping = {"batch_id": payload["batch_id"], "raw_audit_path": result["raw_audit_path"],
                   "raw_path": f"/claim_reviews/{local}", "aggregate_path": f"/claim_reviews/{len(rows)}",
                   "raw_digest": digest(row)}
        mappings.append(copy.deepcopy(mapping))
        selections.append({**mapping, "claim_id": key, "claim_digest": row["claim_digest"],
                           "request_context_id": result["context_id"], "input_digest": result["input_digest"],
                           "response_digest": result["response_digest"], "parsed_digest": result["parsed_digest"]})
        source_mappings.extend(_source_acknowledgements(parent, row, result, payload["batch_id"], local))
        rows.append(copy.deepcopy(row))
    reviewed_ids = initial["reviewed_block_ids"]
    for mapping in source_mappings:
        mapping["aggregate_path"] = (f"/reviewed_block_ids/{reviewed_ids.index(mapping['block_id'])}"
                                     if mapping["block_id"] in reviewed_ids else None)
    receipts = [{key: copy.deepcopy(result[key]) for key in
                 ("context_id", "input_digest", "response_digest", "parsed_digest", "status", "raw_audit_path")}
                for result in [discovery, *all_results]]
    context = digest({"parent_digest": parent_digest, "discovery_payload_digest": digest(discovery_payload),
                      "planned_batches": [digest(p) for p in all_batches], "raw_receipts": receipts,
                      "row_selections": selections})
    view = {**copy.deepcopy(initial), "version": "program-first-row-selection-adapter-v1", "context_id": context,
            "claim_reviews": rows, "path_mappings": mappings, "row_selections": selections,
            "reviewed_source_mappings": source_mappings, "raw_receipts": receipts}
    if partial:
        previous = {row["claim_id"]: row for row in initial["claim_reviews"]}
        differences = []
        for row, selection in zip(rows, selections, strict=True):
            old = previous.get(row["claim_id"])
            if old is not None and old["state"] == "unresolved" and row["claim_id"] not in valid_ids:
                origin = next(mapping for mapping in initial["path_mappings"]
                              if mapping["raw_digest"] == digest(old))
                differences.append({"claim_id": row["claim_id"], "previous_row": copy.deepcopy(old),
                                    "selected_row": copy.deepcopy(row), "previous_origin": copy.deepcopy(origin),
                                    "selected_origin": copy.deepcopy(selection), "changed": digest(old) != digest(row)})
        view["review_differences"] = differences
        view["context_id"] = digest({"selection_context": context, "review_differences": differences})
    return view
