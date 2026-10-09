"""Lossless model-facing advice input and explicit per-item basis requirements."""

from __future__ import annotations

import copy

from assessment.rules import _decisive_flaw
from schemas.claim import ClaimStatus

VERSION = "advice-request-v1"
_ARRAYS = ("evidence", "questions", "notes", "theory_derivations", "verification_limitations")


def project_input(data):
    """Remove only checked duplicate content; the full input remains the audit contract."""
    known = {"claim", "basis", "ledger", "source_files", "input_version"}
    if set(data) - known:
        raise ValueError("Unknown advice input fields cannot be omitted from the request")
    result = {
        "request_version": VERSION,
        "input_version": data.get("input_version", "advice-v1"),
        "claim": copy.deepcopy(data["claim"]),
        "basis": copy.deepcopy(data["basis"]),
        "catalog_arrays": {},
        "ledger": [],
    }
    for kind in _ARRAYS:
        if kind not in result["claim"]:
            continue
        rows = result["claim"].pop(kind)
        refs = [f"/{kind}/{index}" for index in range(len(rows))]
        if [result["basis"][ref]["content"] for ref in refs] != rows:
            raise ValueError("Advice request cannot remove unequal claim content")
        result["catalog_arrays"][kind] = refs
    for index, row in enumerate(data["ledger"]):
        ref = f"/ledger/{index}"
        if ref in result["basis"]:
            if result["basis"][ref]["content"] != row:
                raise ValueError("Advice request cannot remove unequal ledger content")
            result["ledger"].append({"basis_ref": ref})
        else:
            result["ledger"].append({"unreferenced_context": copy.deepcopy(row)})
    rebuilt_claim, rebuilt_ledger = reconstruct_input(result)
    if rebuilt_claim != data["claim"] or rebuilt_ledger != data["ledger"]:
        raise ValueError("Advice request must reconstruct the complete original scientific input")
    return result


def reconstruct_input(request):
    """Recover the complete claim and linked ledger for integrity checks."""
    claim = copy.deepcopy(request["claim"])
    for kind, refs in request["catalog_arrays"].items():
        claim[kind] = [copy.deepcopy(request["basis"][ref]["content"]) for ref in refs]
    ledger = [
        copy.deepcopy(
            row["unreferenced_context"]
            if "unreferenced_context" in row
            else request["basis"][row["basis_ref"]]["content"]
        )
        for row in request["ledger"]
    ]
    return claim, ledger


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
    return request
