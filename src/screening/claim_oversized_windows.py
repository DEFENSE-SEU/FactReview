"""Exact reading slices; original block identities and scientific sources stay whole."""

from __future__ import annotations

import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def plan_slices(block, limit):
    """Partition every original codepoint without manufacturing a MaterialBlock."""
    if (type(limit) is not int or limit < 1 or type(block) is not dict
            or type(block.get("id")) is not str or not block["id"].strip()
            or type(block.get("text")) is not str or not block["text"].strip()
            or type(block.get("loc")) is not dict or len(block["text"]) <= limit):
        raise ValueError("Oversized discovery requires a located original block and positive limit")
    text, slices, start = block["text"], [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + limit // 2, end)
            if boundary >= 0:
                end = boundary + 1
        body = text[start:end]
        slices.append({"slice_id": f"slice_{len(slices) + 1:03d}", "start": start,
                       "end": end, "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
                       "text": body})
        start = end
    directory = [{k: v for k, v in row.items() if k != "text"} for row in slices]
    return {"version": "oversized-original-slices-v1", "block_id": block["id"],
            "block_digest": digest(block), "source_loc": copy.deepcopy(block["loc"]),
            "characters": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "window_chars": limit, "slice_directory": directory,
            "slice_directory_digest": digest(directory), "slices": slices}


def slice_payloads(parent, plan):
    """Give each request the complete slice directory and unchanged semantic registry."""
    if (len(parent["blocks"]) != 1 or
            plan_slices(parent["blocks"][0], plan["window_chars"]) != plan):
        raise ValueError("Oversized slice plan differs from the complete original parent")
    return [{"window_id": parent["window_id"], "parent_digest": digest(parent),
             "paper_title": parent["paper_title"],
             "current_claims": copy.deepcopy(parent["current_claims"]),
             "previous_unresolved_observations": copy.deepcopy(parent["previous_unresolved_observations"]),
             "source_identity": {k: copy.deepcopy(plan[k]) for k in (
                 "block_id", "block_digest", "source_loc", "characters", "text_sha256")},
             "slice_directory": copy.deepcopy(plan["slice_directory"]),
             "slice_directory_digest": plan["slice_directory_digest"],
             "slice": copy.deepcopy(part)} for part in plan["slices"]]


def check_slice_response(payload, response):
    """Require this exact whole-directory/context ACK, never an implied full-block ACK."""
    part = payload["slice"]
    identity = payload["source_identity"]
    expected = {"schema_version": "claim-oversized-discovery-slice-v1",
                "context_id": payload["context_id"], "window_id": payload["window_id"],
                "parent_digest": payload["parent_digest"], "block_id": identity["block_id"],
                "block_digest": identity["block_digest"], "slice_id": part["slice_id"],
                "start": part["start"], "end": part["end"], "text_sha256": part["text_sha256"],
                "slice_directory_digest": payload["slice_directory_digest"], "reviewed": True}
    if (type(response) is not dict or set(response) != {*expected, "new_findings", "explanation"}
            or any(type(response[k]) is not type(v) or response[k] != v for k, v in expected.items())
            or type(response["new_findings"]) is not list):
        raise ValueError("Oversized discovery response does not acknowledge its exact assigned slice")
    for finding in response["new_findings"]:
        if (type(finding) is not dict or finding.get("source_block_ids") != [identity["block_id"]]):
            raise ValueError("Oversized discovery finding selects an unsupplied original source")


def aggregate_slices(parent, plan, receipts):
    """All slices must validate before an explicit program omission adapter exists."""
    payloads = slice_payloads(parent, plan)
    if len(receipts) != len(payloads):
        raise ValueError("Oversized discovery lacks its entire planned wave")
    findings, mappings, seen = [], [], set()
    for expected, receipt in zip(payloads, receipts, strict=True):
        payload = {**expected, "context_id": digest(expected)}
        if receipt["input"] != payload or receipt["status"] != "returned":
            raise ValueError("Oversized discovery receipt changed its full input or failed")
        raw = receipt["response"]
        check_slice_response(payload, raw)
        if receipt["response_digest"] != digest(raw) or receipt["parsed_digest"] != digest(raw):
            raise ValueError("Oversized discovery raw/typed identity changed")
        for local, finding in enumerate(raw["new_findings"]):
            key = digest(finding)
            if key in seen:
                raise ValueError("Oversized discovery repeats a finding across slices")
            seen.add(key)
            mappings.append({"slice_id": payload["slice"]["slice_id"],
                             "raw_audit_path": receipt["raw_audit_path"],
                             "raw_path": f"/new_findings/{local}",
                             "aggregate_path": f"/new_findings/{len(findings)}", "raw_digest": key})
            findings.append(copy.deepcopy(finding))
    context = digest({"parent_digest": digest(parent), "receipts": [
        {k: r[k] for k in ("context_id", "response_digest", "raw_audit_path")} for r in receipts]})
    return {"schema_version": "claim-coverage-discovery-v1", "context_id": context,
            "window_id": parent["window_id"], "parent_digest": digest(parent),
            "reviewed_block_ids": [plan["block_id"]], "new_findings": findings,
            "explanation": "Program aggregation of every exact original-block omission slice."}, mappings
