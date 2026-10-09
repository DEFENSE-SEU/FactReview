"""Source-bound qualification of optional related-work and baseline findings."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import StrictStr, model_validator

from schemas.claim import ClaimLocation, Contract, Evidence, Finding
from screening.checks import grounded_paper_pointer
from screening.claims import _location


class OmissionAssessment(Contract):
    version: Literal["omission-v1"]
    decision: Literal["important_missing", "candidate_only", "not_applicable", "unresolved"]
    basis: Literal["method_positioning", "same_problem_alternative", "evaluation_baseline", "none"]
    target_source_id: StrictStr
    target_quote: StrictStr
    external_role: Literal[
        "scientific_contribution", "evaluation_result", "background_or_bibliography", "unresolved"
    ]
    reason: StrictStr

    @model_validator(mode="after")
    def qualified(self):
        if not self.reason.strip():
            raise ValueError("An omission assessment requires a concrete reason")
        if self.decision == "important_missing" and (
            self.basis == "none"
            or self.external_role not in {"scientific_contribution", "evaluation_result"}
            or not self.target_source_id.strip()
            or not self.target_quote.strip()
        ):
            raise ValueError("Important omissions require a scientific comparison and manuscript target")
        return self


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _files(materials):
    result = {}
    for name in (materials.markdown_path, materials.source_pdf):
        path = Path(name).resolve()
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


class OmissionContext:
    """A local source domain; global targets carry no claim-support authority."""

    def __init__(self, materials, excerpts, *, global_review=False):
        self.materials = materials
        self.global_review = global_review
        self.targets = {}
        self.unavailable = []
        self.decisions = []
        self._blocks = {}
        self._state = (materials.markdown_path, materials.source_pdf, materials.markdown)
        if not excerpts:
            self.unavailable.append(
                {"input_index": None, "reason": "No located manuscript target was supplied"}
            )
        try:
            self._files = _files(materials)
        except OSError:
            self._files = None
        bibliography = {block.id for block in materials.bibliography}
        for index, excerpt in enumerate(excerpts):
            try:
                block_id, quote = excerpt["source_block_id"], excerpt["source_quote"]
                blocks = [block for block in materials.blocks if block.id == block_id]
                if len(blocks) != 1 or not isinstance(quote, str) or not quote.strip():
                    raise ValueError("Unknown, ambiguous, or empty manuscript source")
                block = blocks[0]
                if block.id in bibliography or block.kind in {"bibliography", "reference", "references"}:
                    raise ValueError("A bibliography entry cannot supply the manuscript comparison target")
                loc = ClaimLocation.model_validate(excerpt["loc"])
                if _location(block, quote, materials.markdown) != loc:
                    raise ValueError("Manuscript target differs from its original location")
                covered = excerpt.get("covered", [])
                if not isinstance(covered, list) or any(
                    not isinstance(cid, str) or not cid.strip() for cid in covered
                ):
                    raise ValueError("Invalid original manuscript target condition scope")
                pointer = grounded_paper_pointer(materials, block, quote)
                identity = {
                    "source_block_id": block.id,
                    "source_quote": quote,
                    "loc": loc.model_dump(mode="json"),
                }
                source_id = "manuscript:" + _digest(identity)
                if source_id not in self.targets:
                    self.targets[source_id] = {
                        "source_id": source_id,
                        **identity,
                        "pointer": pointer.model_dump(mode="json"),
                        "origins": [],
                        "block_sha256": _digest(block.model_dump(mode="json")),
                    }
                    self._blocks[block.id] = block.model_dump(mode="json")
                origin = {"claim_id": excerpt.get("claim_id"), "covered": list(covered)}
                if origin not in self.targets[source_id]["origins"]:
                    self.targets[source_id]["origins"].append(origin)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.unavailable.append({"input_index": index, "reason": str(exc)})

    def payload(self):
        return copy.deepcopy(list(self.targets.values()))

    def resolve(self, source_id, quote, covered):
        target = self.targets.get(source_id)
        if target is None or not quote.strip() or target["source_quote"].count(quote) != 1:
            raise ValueError("Unknown manuscript target or nonunique quote outside its declared range")
        materials = self.materials
        if self._state != (materials.markdown_path, materials.source_pdf, materials.markdown):
            raise ValueError("Manuscript target material changed during comparison")
        if self._files is None or _files(materials) != self._files:
            raise ValueError("Manuscript target artifact changed during comparison")
        blocks = [block for block in materials.blocks if block.id == target["source_block_id"]]
        if len(blocks) != 1 or blocks[0].model_dump(mode="json") != self._blocks[target["source_block_id"]]:
            raise ValueError("Manuscript target block changed during comparison")
        if not self.global_review:
            allowed = {cid for origin in target["origins"] for cid in origin["covered"]}
            if not set(covered).issubset(allowed):
                raise ValueError("Manuscript target is outside the comparison condition scope")
        block = blocks[0]
        return _location(block, quote, materials.markdown), grounded_paper_pointer(materials, block, quote)

    @staticmethod
    def duplicate_indices(comparisons):
        keys = {}
        for index, row in enumerate(comparisons):
            raw_purpose = row.get("purpose")
            purpose = raw_purpose.strip() if isinstance(raw_purpose, str) else None
            if purpose not in {"related_work", "baseline"}:
                continue
            assessment = row.get("omission_assessment")
            target = assessment.get("target_source_id") if isinstance(assessment, dict) else None
            if isinstance(target, str):
                keys[index] = (str(row.get("paper_id", "")).strip(), purpose, target.strip())
        counts = Counter(keys.values())
        return {index for index, key in keys.items() if counts[key] > 1}

    def finding(self, comparison, index, *, eligible, pointer, covered, duplicate=False):
        record = {
            "comparison_index": index,
            "paper_id": comparison.get("paper_id"),
            "purpose": comparison.get("purpose"),
            "decision": "unresolved",
        }
        self.decisions.append(record)
        try:
            if duplicate:
                raise ValueError("Duplicate omission assessment for the same paper, purpose, and target")
            if comparison.get("omission_assessment") is None:
                raise ValueError("Omission qualification was not supplied; candidate retained in raw audit")
            assessment = OmissionAssessment.model_validate(comparison["omission_assessment"])
            record["assessment"] = assessment.model_dump(mode="json")
            if assessment.decision != "important_missing":
                record["decision"] = assessment.decision
                return None
            if not eligible:
                raise ValueError(
                    "Relevance, prior-work, bibliography, or reader identity does not permit an omission"
                )
            if not all(
                isinstance(comparison.get(key), str) and comparison[key].strip()
                for key in ("mechanism", "setting", "protocol")
            ):
                raise ValueError("Omission qualification requires all three concrete comparison dimensions")
            purpose = comparison["purpose"]
            allowed = (
                {"evaluation_baseline"}
                if purpose == "baseline"
                else {"method_positioning", "same_problem_alternative"}
            )
            if assessment.basis not in allowed:
                raise ValueError("Omission basis does not match the proposed finding kind")
            loc, target_pointer = self.resolve(assessment.target_source_id, assessment.target_quote, covered)
            finding = Finding(
                kind=purpose,
                loc=loc,
                level="missing",
                text=assessment.reason,
                evidence=[
                    Evidence(
                        source="literature",
                        pointer=pointer,
                        covered=covered,
                        direction="support",
                        sufficient=False,
                        concern=False,
                        affects_claim=False,
                        note=assessment.reason,
                    ),
                    Evidence(
                        source="paper_internal",
                        pointer=target_pointer,
                        covered=covered,
                        direction="support",
                        sufficient=False,
                        concern=False,
                        affects_claim=False,
                        note="Original manuscript target for this qualified literature comparison.",
                    ),
                ],
            )
            record["decision"] = "important_missing"
            record["target_pointer"] = target_pointer.model_dump(mode="json")
            return finding
        except (ValueError, TypeError, KeyError, OSError) as exc:
            record["reason"] = str(exc)
            return None
