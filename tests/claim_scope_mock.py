"""Explicit offline v4 judgments, selected only by a supplied v4 source directory."""

import copy

from screening.claim_coverage_wire import unpack_scope_payload
from screening.claim_scope import SCOPE_DIMENSIONS


def current_wire(raw, payload):
    """Migrate explicit happy fixtures only when the current request is v5 packed."""
    if "scope_wire_version" not in payload:
        return raw
    from schemas.materials import MaterialBlock
    from screening import claim_coverage as m

    expanded = unpack_scope_payload(payload)
    key = "claim_reviews" if "claim_reviews" in raw else "original_claim_reviews"
    claims = {c["claim_id"]: c for c in expanded["current_claims"]}
    contexts = {c["claim_id"]: c for c in expanded["scope_context"]}
    blocks = {b["id"]: MaterialBlock.model_validate(b) for b in expanded["blocks"]}
    blocks.update({r["block"]["id"]: MaterialBlock.model_validate(r["block"])
                   for r in expanded.get("supplemental_sources", [])})
    try:
        for row in raw[key]:
            claim = claims[row["claim_id"]]
            if row["claim_digest"] != claim["digest"]:
                return raw
            m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, contexts[row["claim_id"]], blocks)
    except (ValueError, KeyError, IndexError, TypeError):
        # Invalid historic/negative fixtures stay invalid; no declaration is filled.
        return raw
    result = copy.deepcopy(raw)
    maps = []
    for row in result[key]:
        views, findings, reviews = row.pop("preserved_qualifiers"), row.pop("findings"), row.pop("source_reviews")
        other, paths = [], {}
        for i, finding in enumerate(findings):
            if finding["kind"] != "missing_qualifier_or_condition":
                paths[i] = "/other_findings/" + str(len(other))
                other.append(finding)
        for i, atom in enumerate(row["scope_atoms"]):
            atom["carriers"] = [{"claim_path": views[j]["claim_path"], "claim_value": views[j]["claim_value"]}
                                for j in atom.pop("preserved_indices")]
            finding_index = atom.pop("finding_index")
            if finding_index is not None:
                paths[finding_index] = "/scope_atoms/" + str(i)
        row["other_findings"] = other
        row["source_review_groups"] = [
            {"source_ids": [r["block_id"]], **{k: v for k, v in r.items() if k != "block_id"}}
            for r in reviews
        ]
        maps.append(paths)
    for link in result.get("observation_links", []):
        parts = link["review_path"].split("/")
        if len(parts) == 5 and parts[3] == "findings":
            link["review_path"] = "/original_claim_reviews/" + parts[2] + maps[int(parts[2])][int(parts[4])]
    result["schema_version"] = "claim-coverage-v5" if key == "claim_reviews" else "claim-coverage-validation-v5"
    return result


def review_v4(payload, observations, legacy):
    """Upgrade explicit happy-path mocks; retain malformed legacy controls unchanged."""
    from screening import claim_coverage as m

    observations = copy.deepcopy(legacy["observations"])

    known = {c["claim_id"]: c for c in payload["current_claims"]}
    required_ids = {c["claim_id"] for c in payload["required_claim_checks"]}
    visible = {b["id"] for b in payload["blocks"]}
    if any(
        (o["target_claim_id"] is not None and o["target_claim_id"] not in known)
        or (o["target_claim_id"] is not None and o["target_claim_id"] not in required_ids)
        or any(s["block_id"] not in visible for s in o["sources"])
        for o in observations
    ):
        return legacy, {}
    raw = {
        "schema_version": "claim-coverage-v4",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "reviewed_block_ids": legacy["reviewed_block_ids"],
        "claim_reviews": [],
        "new_findings": [],
        "explanation": legacy["explanation"],
    }
    aliases = {}
    for required in payload["required_claim_checks"]:
        claim = known[required["claim_id"]]
        own = [o for o in observations if o["target_claim_id"] == claim["claim_id"]]
        check = next(c for c in legacy["claim_checks"] if c["claim_id"] == claim["claim_id"])
        source_ids = list(
            dict.fromkeys(
                [*required["source_block_ids"], *(s["block_id"] for o in own for s in o["sources"])]
            )
        )
        row = {
            "claim_id": claim["claim_id"],
            "claim_digest": claim["digest"],
            "state": "unresolved" if any(o["kind"] == "uncertain" for o in own) else "resolved",
            "assertion_groups": [{**g, "source_block_ids": source_ids} for g in check["assertion_groups"]],
            "findings": [],
            "preserved_qualifiers": [],
            "source_block_ids": source_ids,
            "reason": "Explicit original offline semantic judgment.",
        }
        positions = {}
        for o in own:
            if o["kind"] == "merged_conclusions":
                positions[o["id"]] = None
                continue
            finding = {
                "kind": o["kind"],
                "condition_ids": [c["id"] for c in claim["conditions"]],
                "source_block_ids": [s["block_id"] for s in o["sources"]],
                "reason": o["reason"],
            }
            if o["kind"] == "missing_qualifier_or_condition":
                finding.update(
                    restriction=o["reason"],
                    material_effect="Mock declares a materially different verification setting.",
                )
            elif o["kind"] == "missing_needs":
                finding["needs"] = [
                    n for n in ("Literature", "Theory", "Code", "Experiments") if n not in claim["needs"]
                ][:1]
            positions[o["id"]] = len(row["findings"])
            row["findings"].append(finding)
        row = with_scope(row, claim, payload)
        path = "/claim_reviews/" + str(len(raw["claim_reviews"]))
        for o in own:
            index = positions[o["id"]]
            origin, body = (
                (path, row)
                if index is None or row["state"] == "unresolved"
                else (path + "/findings/" + str(index), row["findings"][index])
            )
            kind = "uncertain" if row["state"] == "unresolved" else o["kind"]
            aliases[o["id"]] = "v3_" + m._digest([payload["context_id"], origin, body, kind])[:24]
        raw["claim_reviews"].append(row)
    for o in observations:
        if o["target_claim_id"] is not None:
            continue
        row = {
            "kind": o["kind"],
            "source_block_ids": [s["block_id"] for s in o["sources"]],
            "reason": o["reason"],
        }
        path = "/new_findings/" + str(len(raw["new_findings"]))
        aliases[o["id"]] = "v3_" + m._digest([payload["context_id"], path, row, row["kind"]])[:24]
        raw["new_findings"].append(row)
    return current_wire(raw, payload), aliases


def with_scope(row, claim, payload):
    payload = unpack_scope_payload(payload)
    context = next(c for c in payload["scope_context"] if c["claim_id"] == claim["claim_id"])
    ids = [c["id"] for c in claim["conditions"]]
    blocks = {b["id"]: b for b in payload["blocks"]}
    blocks.update({r["block"]["id"]: r["block"] for r in payload.get("supplemental_sources", [])})
    atoms = []
    considered = set()
    for index, finding in enumerate(row["findings"]):
        if finding["kind"] != "missing_qualifier_or_condition":
            continue
        considered.update(finding["source_block_ids"])
        atoms.append(
            {
                "id": "mock_atom_" + str(index),
                "dimension": "other_boundary",
                "condition_ids": finding["condition_ids"],
                "state": "missing",
                "restriction": finding["restriction"],
                "sources": [
                    {"block_id": key, "start": 0, "end": len(blocks[key]["text"])}
                    for key in finding["source_block_ids"]
                ],
                "preserved_indices": [],
                "finding_index": index,
                "effect": {
                    "kind": "verification_setting",
                    "source_setting": "mock restricted setting",
                    "alternative_setting": "mock materially different setting",
                    "original_permits_alternative": True,
                    "assertion_path": "/text",
                    "assertion_value": claim["text"],
                    "explanation": finding["material_effect"],
                },
                "reason": finding["reason"],
            }
        )
    groups = []
    for condition in ids:
        dimensions = {}
        for dimension in SCOPE_DIMENSIONS:
            relevant = [
                a["id"] for a in atoms if a["dimension"] == dimension and condition in a["condition_ids"]
            ]
            dimensions[dimension] = {
                "state": "unresolved"
                if row["state"] == "unresolved"
                else "missing"
                if relevant
                else "not_governing",
                "atom_ids": relevant,
                "reason": "Injected per-dimension scope decision.",
            }
        existing = next((g for g in groups if g["dimensions"] == dimensions), None)
        if existing is None:
            groups.append({"condition_ids": [condition], "dimensions": dimensions})
        else:
            existing["condition_ids"].append(condition)
    row = copy.deepcopy(row)
    source_directory = list(context["sources"])
    for key in sorted(considered - {s["block_id"] for s in source_directory}):
        source_directory.append({"block_id": key, "availability": "in_window"})
    row.update(
        scope_groups=groups,
        scope_atoms=atoms,
        source_reviews=[
            {
                "block_id": s["block_id"],
                "condition_ids": ids,
                "state": "unavailable"
                if s["availability"] == "unavailable"
                else "considered"
                if s["block_id"] in considered
                else "irrelevant",
                "reason": "Injected source relationship and actual visibility.",
            }
            for s in source_directory
        ],
    )
    return row
