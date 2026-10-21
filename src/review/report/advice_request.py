"""Condensed model-facing advice input and explicit per-item basis requirements."""

from __future__ import annotations

import copy
import json

from assessment.rules import _decisive_flaw
from schemas.claim import ClaimStatus

VERSION = "advice-request-v2"
_CLAIM_FIELDS = ("id", "text", "status", "importance", "needs", "source_quote", "conditions")
_DROP = ("pointer", "locator", "sha", "digest", "audit", "schema_version", "claim_id", "item_index")


def _clip(value, limit):
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _quote(pointer):
    return {"quote": _clip(pointer.get("quote"), 300), "page": pointer.get("page")}


def _condense(content):
    """Keep a record's verdict-bearing fields; long text is clipped and paths dropped."""
    if isinstance(content, str):
        return _clip(content, 600)
    if not isinstance(content, dict):
        return _clip(json.dumps(content, ensure_ascii=False), 600)
    result = {}
    for key, value in content.items():
        if key in ("pointer", "source_pointer") and isinstance(value, dict):
            result["quote"] = _quote(value)
        elif key == "trace" and isinstance(value, dict):
            result["trace"] = {
                "outcome": value.get("outcome"),
                "completion_reason": _clip(value.get("completion_reason"), 300),
                "gaps": [_clip(gap.get("reason") if isinstance(gap, dict) else gap, 200) for gap in value.get("gaps", [])][:5],
            }
        elif any(part in key for part in _DROP) or key == "additional_pointers":
            continue
        elif isinstance(value, str):
            result[key] = _clip(value, 600)
        elif isinstance(value, list):
            result[key] = [_clip(v, 300) if isinstance(v, str) else v for v in value[:5]] if all(
                isinstance(v, (str, int, float, bool)) or v is None for v in value
            ) else _clip(json.dumps(value[:3], ensure_ascii=False), 600)
        elif isinstance(value, dict):
            result[key] = _clip(json.dumps(value, ensure_ascii=False), 400)
        else:
            result[key] = value
    return {key: value for key, value in result.items() if value is not None}


def project_input(data):
    """Condensed model-facing view: the claim's frozen state plus each basis record's
    verdict, gaps and a clipped quote. The full input stays the audit and validation contract."""
    known = {"claim", "basis", "ledger", "source_files", "input_version"}
    if set(data) - known:
        raise ValueError("Unknown advice input fields cannot be omitted from the request")
    claim = data["claim"]
    return {
        "request_version": VERSION,
        "input_version": data.get("input_version", "advice-v1"),
        "claim": {key: copy.deepcopy(claim[key]) for key in _CLAIM_FIELDS if key in claim},
        "basis": {
            ref: {"condition_ids": list(row["condition_ids"]), "content": _condense(row["content"])}
            for ref, row in data["basis"].items()
        },
    }


def requirements(claim, data):
    """Describe the existing validator's choices without authorizing new evidence."""
    from review.report.advice import _usable

    rows = []
    for condition in claim.conditions:
        cid = condition.id
        evidence = [
            (f"/evidence/{index}", item)
            for index, item in enumerate(claim.evidence)
            if _usable(item) and cid in item.covered
        ]
        support = [ref for ref, e in evidence if e.direction == "support" and e.sufficient]
        opposition = [ref for ref, e in evidence if e.direction == "flaw" and e.sufficient]
        required_refs, alternatives, action = [], [], None
        if claim.status == ClaimStatus.SUPPORTED:
            alternatives = [support]
        elif claim.status == ClaimStatus.FLAWED:
            alternatives = [[ref for ref, e in evidence if _decisive_flaw(e)]]
        elif claim.status == ClaimStatus.QUESTIONED:
            alternatives = (
                [support, opposition]
                if support and opposition
                else [[ref for ref, e in evidence if e.concern or (e.direction == "flaw" and e.sufficient)]]
            )
        else:
            gap = f"/coverage_gaps/{cid}"
            if gap not in data["basis"]:
                continue
            required_refs = [gap]
            limitations = [
                ref
                for ref, item in data["basis"].items()
                if ref.startswith("/verification_limitations/") and cid in item["condition_ids"]
            ]
            if limitations:
                action = "verification_followup"
                required_refs.extend(limitations)
        if any(not group for group in alternatives):
            continue
        rows.append(
            {
                "condition_id": cid,
                "required_basis_refs": required_refs,
                "one_or_more_from_each_group": alternatives,
                "required_action": action,
            }
        )
    required_conditions = (
        [condition.id for condition in claim.conditions]
        if claim.status == ClaimStatus.SUPPORTED
        else [row["condition_id"] for row in rows]
        if claim.status == ClaimStatus.UNVERIFIED
        else []
    )
    return {
        "scope": "Every returned item must satisfy these requirements for EACH condition it names.",
        "conditions": rows,
        "required_condition_ids": required_conditions,
        "reference_rule": "Use only existing basis refs; each ref must overlap the item's conditions.",
        "guidance": "Write one consolidated item per eligible condition, combining its context and limitations. Do not append a context-only item that lacks that condition's required bases. Conditions absent from this catalog have no supported advice basis under the recorded status.",
    }


def build_request(claim, data):
    request = project_input(data)
    request["item_requirements"] = requirements(claim, data)
    named = {
        ref
        for row in request["item_requirements"]["conditions"]
        for ref in [*row["required_basis_refs"], *(r for group in row["one_or_more_from_each_group"] for r in group)]
    }
    if named - set(request["basis"]):
        raise ValueError("Advice requirements reference a basis record absent from the request")
    return request
