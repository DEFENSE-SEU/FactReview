"""Ground and retain model-authored derivation traces without proving mathematics."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from common import run_stats
from llm.diagnostics import redact_provider_details, sanitized_endpoint
from schemas.claim import Contract, NonEmpty, TheoryDerivationRecord, TheorySource, TheoryTrace
from screening import checks

VERSION = "theory-derivation-v1"
ExactString = Annotated[str, Field(min_length=1, strict=True)]


class SourceQuote(Contract):
    block_id: ExactString
    quote: ExactString = Field(description="One exact, unique substring of a supplied allowed theory block.")


class Assumption(Contract):
    id: ExactString
    text: NonEmpty
    status: Literal["paper_explicit", "required_unstated"]
    sources: list[SourceQuote]


class Step(Contract):
    id: ExactString
    statement: NonEmpty
    reason: NonEmpty
    assumption_ids: list[ExactString]
    previous_step_ids: list[ExactString]
    sources: list[SourceQuote]


class Gap(Contract):
    at: ExactString
    reason: NonEmpty
    needed: NonEmpty
    sources: list[SourceQuote]


class Derivation(Contract):
    goal: NonEmpty
    assumptions: list[Assumption]
    steps: list[Step]
    gaps: list[Gap]
    outcome: Literal["completed", "partial", "unable"]
    completion_reason: NonEmpty


class IndexedDerivation(Contract):
    item_index: int = Field(ge=0, strict=True)
    trace: Derivation


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _file_hash(path):
    candidate = Path(path)
    return hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None


def _available_image_hash(path):
    # A missing/unreadable unused page must not block a text-only derivation.
    try:
        return _file_hash(path)
    except (OSError, ValueError):
        return None


@dataclass
class SourceContext:
    blocks: dict
    markdown: str
    paths: dict
    claim: dict
    pages: list[dict]
    image_hashes: dict[str, str | None]
    consumed_pages: set[int] = field(default_factory=set)

    @classmethod
    def capture(cls, claim, materials, blocks):
        if len({block.id for block in materials.blocks}) != len(materials.blocks):
            raise ValueError("Theory material block IDs must be unique")
        return cls(
            {block.id: block.model_dump(mode="json") for block in blocks},
            materials.markdown,
            {str(path): _file_hash(path) for path in (materials.markdown_path, materials.source_pdf)},
            claim.model_dump(mode="json"),
            [page.model_dump(mode="json") for page in materials.pages],
            {page.path: _available_image_hash(page.path) for page in materials.pages},
        )

    @property
    def hashes(self):
        return {path: digest for path, digest in self.paths.items() if digest is not None}

    def check(self, claim, materials):
        current = {block.id: block for block in materials.blocks}
        if len(current) != len(materials.blocks) or any(
            identifier not in current or current[identifier].model_dump(mode="json") != snapshot
            for identifier, snapshot in self.blocks.items()
        ):
            raise ValueError("Theory source block text/location changed during the model call")
        if materials.markdown != self.markdown or set(self.paths) != {
            str(materials.markdown_path),
            str(materials.source_pdf),
        }:
            raise ValueError("Theory source context changed during the model call")
        if any(_file_hash(path) != digest for path, digest in self.paths.items()):
            raise ValueError("Theory source artifact changed during the model call")
        if claim.model_dump(mode="json") != self.claim:
            raise ValueError("Theory target claim changed during the model call")
        for page_number in self.consumed_pages:
            self.check_page(page_number, materials)

    def check_page(self, page_number, materials, *, consume=False):
        """Only a consumed notation page is required to retain its initial pixels."""
        original = [page for page in self.pages if page["page"] == page_number]
        current = [page.model_dump(mode="json") for page in materials.pages if page.page == page_number]
        if len(original) != 1 or current != original:
            raise ValueError("Theory notation page identity changed or is ambiguous")
        path = original[0]["path"]
        expected = self.image_hashes[path]
        if expected is None or _available_image_hash(path) != expected:
            raise ValueError("Theory notation page pixels changed or were unavailable before verification")
        if consume:
            self.consumed_pages.add(page_number)
        return {path: expected}

    def source(self, quote, claim, materials):
        self.check(claim, materials)
        if quote.block_id not in self.blocks:
            raise ValueError("Theory trace source is outside this pass's supplied blocks")
        block = next(block for block in materials.blocks if block.id == quote.block_id)
        if block.text.count(quote.quote) != 1:
            raise ValueError("Theory trace quote must be an exact, unique source substring")
        pointer = checks.grounded_paper_pointer(materials, block, quote.quote)
        digest = _file_hash(pointer.locator)
        if not digest or self.paths.get(pointer.locator) != digest:
            # Paths can be relative in test/legacy materials, while pointers are absolute.
            if not digest or not any(
                Path(path).resolve() == Path(pointer.locator).resolve() and expected == digest
                for path, expected in self.paths.items()
            ):
                raise ValueError("Theory trace pointer does not refer to the frozen source artifact")
        return TheorySource(
            block_id=block.id,
            pointer=pointer,
            block_sha256=hashlib.sha256(block.text.encode()).hexdigest(),
            artifact_sha256=digest,
        )


def request(system, payload, *, module, call, context, output_dir=None):
    """Preserve both successful and failed original model boundaries, without new calls."""
    cfg = checks.resolve_llm_config()
    stats = run_stats.stats_path()
    directory = (
        Path(output_dir) if output_dir is not None else stats.parent / "theory_derivations" if stats else None
    )
    path = directory / f"{uuid.uuid4().hex}.json" if directory else None
    audit = {
        "module": module,
        "provider": cfg.provider,
        "model": cfg.model,
        "endpoint": sanitized_endpoint(cfg.base_url),
        "transport": "injected" if call is not None else "live",
        "system": system,
        "request": payload,
        "request_sha256": _digest(payload),
        "source_hashes": context.hashes,
        "status": "started",
    }

    def save():
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(redact_provider_details(audit, cfg), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    save()
    try:
        response = checks.ask(system, payload, module=module, call=call)
        audit.update(status="returned", response=response)
        return response, str(path) if path else None, cfg
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


def _ids(values, *, label):
    if any(identifier != identifier.strip() or not identifier for identifier in values) or len(values) != len(
        set(values)
    ):
        raise ValueError(f"Theory {label} must contain distinct exact identifiers")
    return set(values)


def safe_text(value, cfg):
    return redact_provider_details(value, cfg)


def _trace(raw, context, claim, materials, cfg):
    trace = Derivation.model_validate(raw)
    return validate_trace_structure(trace, context, claim, materials, cfg)


def validate_trace_structure(trace, context, claim, materials, cfg, *, output_model=TheoryTrace):
    """Shared dependency/completeness checks; each version retains its own source resolver."""
    assumption_ids = _ids([a.id for a in trace.assumptions], label="assumptions")
    step_ids = _ids([step.id for step in trace.steps], label="steps")
    if assumption_ids & step_ids:
        raise ValueError("Theory assumption and step IDs must be distinct")
    if any(safe_text(identifier, cfg) != identifier for identifier in assumption_ids | step_ids):
        raise ValueError("Theory dependency identifiers contain provider credentials")
    grounded_assumptions = set()
    assumptions = []
    for assumption in trace.assumptions:
        sources = [context.source(s, claim, materials) for s in assumption.sources]
        if assumption.status == "paper_explicit" and not sources:
            raise ValueError("Explicit Theory assumptions require manuscript sources")
        if sources:
            grounded_assumptions.add(assumption.id)
        assumptions.append(
            {**assumption.model_dump(), "text": safe_text(assumption.text, cfg), "sources": sources}
        )
    previous = set()
    steps = []
    for step in trace.steps:
        used = _ids(step.assumption_ids, label="assumption dependencies")
        earlier = _ids(step.previous_step_ids, label="step dependencies")
        if not used.issubset(assumption_ids) or not earlier.issubset(previous):
            raise ValueError("Theory step dependencies must name existing assumptions and earlier steps")
        sources = [context.source(s, claim, materials) for s in step.sources]
        if not sources and not earlier and not (used & grounded_assumptions):
            raise ValueError("Theory generated step lacks a grounded source/dependency chain")
        steps.append(
            {
                **step.model_dump(),
                "statement": safe_text(step.statement, cfg),
                "reason": safe_text(step.reason, cfg),
                "sources": sources,
            }
        )
        previous.add(step.id)
    gaps = []
    for gap in trace.gaps:
        if gap.at not in {"goal", *assumption_ids, *step_ids}:
            raise ValueError("Theory gap must name the goal, an assumption, or a step")
        gaps.append(
            {
                **gap.model_dump(),
                "reason": safe_text(gap.reason, cfg),
                "needed": safe_text(gap.needed, cfg),
                "sources": [context.source(s, claim, materials) for s in gap.sources],
            }
        )
    if trace.outcome == "completed" and (
        not steps or gaps or any(a.status == "required_unstated" for a in trace.assumptions)
    ):
        raise ValueError("Completed Theory traces require steps and no unresolved gaps/assumptions")
    if trace.outcome != "completed" and not gaps:
        raise ValueError("Incomplete Theory traces require explicit gaps")
    context.check(claim, materials)
    return output_model(
        **{
            **trace.model_dump(),
            "goal": safe_text(trace.goal, cfg),
            "completion_reason": safe_text(trace.completion_reason, cfg),
            "assumptions": assumptions,
            "steps": steps,
            "gaps": gaps,
        }
    )


def records(raw, items, *, claim, materials, context, phase, adopted, audit_pointer, cfg, injected):
    version = raw.get("schema_version")
    if version is not None and version != VERSION:
        raise ValueError("Unknown Theory derivation schema_version")
    if version is None and "derivations" in raw:
        raise ValueError("Versionless Theory response cannot include new derivations")
    indexed, invalid = {}, {}
    global_issue = ""
    if version == VERSION:
        values = raw.get("derivations")
        if not isinstance(values, list):
            global_issue = "Versioned Theory response requires a derivations list"
        else:
            for value in values:
                identifier = value.get("item_index") if isinstance(value, dict) else None
                if type(identifier) is not int or not 0 <= identifier < len(items):
                    global_issue = "Theory derivation item_index must be a known strict integer"
                    continue
                if identifier in indexed or identifier in invalid:
                    indexed.pop(identifier, None)
                    invalid[identifier] = "Duplicate Theory derivation item_index"
                    continue
                if set(value) != {"item_index", "trace"}:
                    invalid[identifier] = "Unknown or missing indexed Theory derivation fields"
                else:
                    indexed[identifier] = value["trace"]
    if global_issue and not items:
        raise ValueError(global_issue)
    result = []
    for index, item in enumerate(items):
        record = TheoryDerivationRecord(
            schema_version=version,
            claim_id=claim.id,
            item_index=index,
            phase=phase,
            adopted=adopted,
            covered=item.covered,
            state="legacy_unavailable" if version is None else "invalid",
            audit_pointer=audit_pointer,
            source_hashes=context.hashes,
            provider=cfg.provider,
            model=cfg.model,
            transport="injected" if injected else "live",
        )
        try:
            block = next(b for b in materials.blocks if b.id == item.block_id)
            record.source_pointer = checks.grounded_paper_pointer(materials, block, item.quote)
        except (ValueError, StopIteration):
            pass  # The unchanged evidence path separately rejects an invalid primary pointer.
        if version is None:
            record.issues.append("legacy_trace_unavailable: response has no versioned derivation record")
        else:
            try:
                if global_issue or index in invalid or index not in indexed:
                    raise ValueError(
                        global_issue or invalid.get(index, "Missing Theory derivation for this item")
                    )
                trace = _trace(indexed[index], context, claim, materials, cfg)
                if item.kind == "no_proof" and trace.outcome != "unable":
                    raise ValueError("A no_proof observation requires an unable derivation trace")
                record.trace = trace
                record.state = "validated"
            except (ValueError, OSError, TypeError) as exc:
                record.issues.append(redact_provider_details(str(exc), cfg))
        result.append(record)
    return result


def positive_trace_issue(record):
    if record.schema_version is None:
        return ""  # Legacy evidence retains its historical contract, with trace unavailable.
    if record.state != "validated":
        return "; ".join(record.issues) or "Theory derivation trace is invalid"
    if record.trace.outcome != "completed":
        return "Theory derivation is incomplete: " + record.trace.completion_reason
    return ""
