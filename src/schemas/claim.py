"""V2 claim, evidence, and handoff contracts (method specification §2 and §6)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_serializer, model_validator

from schemas.code_joint import CodeJointBinding
from schemas.limitations import VerificationLimitation
from schemas.reference import ReferenceCorrection
from schemas.theory_concern import TheoryConcernReview

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    FLAWED = "flawed"
    QUESTIONED = "questioned"
    UNVERIFIED = "unverified"


class EvidenceNeed(StrEnum):
    LITERATURE = "Literature"
    THEORY = "Theory"
    CODE = "Code"
    EXPERIMENTS = "Experiments"


class ClaimLocation(Contract):
    """One-based page, section anchor, and/or offsets into the parsed paper."""

    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def localizable(self) -> Self:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.char_start is not None and self.char_end <= self.char_start:
            raise ValueError("char_end must follow char_start")
        if self.page is None and not (self.section or "").strip() and self.char_start is None:
            raise ValueError("a claim location requires a page, section, or character span")
        return self


class Condition(Contract):
    """A stable coverage unit; settings can include a task, split, model, or seed."""

    id: NonEmpty
    dataset: NonEmpty | None = None
    metric: NonEmpty | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="after")
    def specified(self) -> Self:
        if not (self.dataset or self.metric or self.settings or self.description.strip()):
            raise ValueError("a condition requires dataset, metric, settings, or a description")
        return self


class EvidencePointer(Contract):
    """Checkable artifact locator and passage/line/output key.

    Examples: paper.pdf + page/quote; arXiv/DOI URL + quote; config.yaml +
    line; runs/eval/metrics.json + key. The producing branch checks existence;
    this schema checks the information needed to locate the evidence.
    """

    locator: NonEmpty
    quote: str = ""
    page: int | None = Field(default=None, ge=1)
    line: int | None = Field(default=None, ge=1)
    key: str | None = None


class ExecutionProvenance(Contract):
    """Identify recomputation from the authors' released data or logs."""

    released_artifact: bool = False
    artifact_kind: Literal["data", "logs", "other"] = "other"
    environment_explanation_possible: bool = True
    run_id: str | None = None
    command: list[str] = Field(default_factory=list)
    runtime_conditions: list[Condition] = Field(default_factory=list)
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    repository: str | None = None
    recomputation_pointer: str | None = None


class Evidence(Contract):
    source: Literal["paper_internal", "literature", "theory", "code", "execution"]
    pointer: EvidencePointer
    additional_pointers: list[EvidencePointer] = Field(default_factory=list)
    covered: list[NonEmpty] = Field(default_factory=list)
    direction: Literal["support", "flaw"]
    sufficient: bool = False
    note: str = ""
    concern: bool = False
    affects_claim: bool = True
    overturnable: bool = True
    aligned: bool | None = None
    provenance: ExecutionProvenance | None = None
    code_joint_binding: CodeJointBinding | None = None

    @model_serializer(mode="wrap")
    def serialize_joint(self, handler):
        result = handler(self)
        if self.code_joint_binding is None:
            result.pop("code_joint_binding", None)
        return result

    @model_validator(mode="after")
    def check_pointer_and_alignment(self) -> Self:
        pointer = self.pointer
        joint = self.code_joint_binding
        if joint is not None:
            if (
                self.source != "code"
                or self.direction != "support"
                or self.covered != [joint.condition_id]
                or not self.additional_pointers
                or len(joint.members) != 1 + len(self.additional_pointers)
            ):
                raise ValueError("Code joint binding requires one-condition multi-source code support")
            import json
            from hashlib import sha256

            for member, source in zip(joint.members, [pointer, *self.additional_pointers], strict=True):
                fingerprint = sha256(
                    json.dumps(
                        source.model_dump(mode="json"),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                if not source.quote.strip() or source.line is None or member.pointer_sha256 != fingerprint:
                    raise ValueError("Code joint pointer identity is incomplete or mismatched")
        if self.additional_pointers:
            if joint is None and (self.source != "paper_internal" or self.direction != "support"):
                raise ValueError("additional pointers require paper_internal support evidence")
            seen = {(pointer.locator, pointer.page, pointer.line, pointer.key, pointer.quote)}
            for additional in self.additional_pointers:
                if joint is None and (
                    not additional.quote.strip() or not (additional.page or (additional.key or "").strip())
                ):
                    raise ValueError("additional paper pointers require an exact quote and page or key")
                identity = (
                    additional.locator,
                    additional.page,
                    additional.line,
                    additional.key,
                    additional.quote,
                )
                if identity in seen:
                    raise ValueError("evidence pointers must identify distinct exact sources")
                seen.add(identity)
        if self.source in {"paper_internal", "theory"}:
            if not pointer.quote.strip() or not (pointer.page or (pointer.key or "").strip()):
                raise ValueError("paper/theory evidence requires a quote and page or section key")
        elif self.source == "literature":
            url = urlsplit(pointer.locator)
            valid_url = url.scheme in {"https", "http"} and bool(url.netloc)
            valid_identifier = re.fullmatch(
                r"(?i)(?:doi:)?10\.\d{4,9}/\S+|arxiv:(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?",
                pointer.locator,
            )
            search_audit = pointer.locator.lower().endswith(".json") and pointer.key == "search_scope"
            if not (valid_url or valid_identifier or search_audit):
                raise ValueError(
                    "literature evidence requires a DOI, arXiv id, URL, or saved search-scope audit"
                )
            if not pointer.quote.strip():
                raise ValueError("literature evidence requires the retrieved passage")
        elif self.source == "code" and pointer.line is None:
            raise ValueError("code evidence requires a source/config file and line")
        elif self.source == "execution":
            if not (pointer.key or "").strip():
                raise ValueError("execution evidence requires a log/metric path and key")
            if self.sufficient and self.aligned is not True:
                raise ValueError("sufficient execution evidence must be aligned")
        if len(self.covered) != len(set(self.covered)):
            raise ValueError("evidence coverage ids must be unique")
        if self.sufficient and not self.covered:
            raise ValueError("sufficient evidence must identify covered conditions")
        return self


class AuthorQuestion(Contract):
    text: NonEmpty
    claim_id: str | None = None
    reason: str = ""


class AdviceItem(Contract):
    """Reviewer-facing wording bound to this claim's frozen report inputs."""

    text: NonEmpty
    action: Literal["reviewer_guidance", "author_question", "verification_followup"] = "reviewer_guidance"
    condition_ids: list[str] = Field(min_length=1)
    basis_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_references(self) -> Self:
        for values in (self.condition_ids, self.basis_refs):
            if len(values) != len(set(values)) or any(not v or v != v.strip() for v in values):
                raise ValueError("advice references must be distinct exact identifiers")
        return self


class ClaimAdvice(Contract):
    """An optional report-stage result; absent in historical and upstream records."""

    state: Literal["generated", "unavailable"]
    input_version: Literal["advice-v1", "advice-v2"] = "advice-v1"
    items: list[AdviceItem] = Field(default_factory=list)
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audit_pointer: str | None = None
    failure_reason: str = ""
    provider: str = ""
    model: str = ""

    @model_validator(mode="after")
    def generation_result(self) -> Self:
        if self.state == "generated" and (not self.items or self.failure_reason):
            raise ValueError("generated advice requires items and no failure reason")
        if self.state == "unavailable" and (self.items or not self.failure_reason.strip()):
            raise ValueError("unavailable advice requires a reason and no generated items")
        return self


class ClaimSourceRef(Contract):
    """An original manuscript passage for specified conditions, not verification evidence."""

    source_block_id: NonEmpty
    source_quote: str = Field(min_length=1)
    loc: ClaimLocation
    covered: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if not self.source_quote.strip():
            raise ValueError("source_quote must contain original manuscript text")
        if len(self.covered) != len(set(self.covered)):
            raise ValueError("source reference coverage ids must be unique")
        return self


class TheorySource(Contract):
    """A program-grounded manuscript source for a generated derivation record."""

    block_id: NonEmpty
    pointer: EvidencePointer
    block_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TheoryAssumption(Contract):
    id: NonEmpty
    text: NonEmpty
    status: Literal["paper_explicit", "required_unstated"]
    sources: list[TheorySource] = Field(default_factory=list)


class TheoryStep(Contract):
    id: NonEmpty
    statement: NonEmpty
    reason: NonEmpty
    assumption_ids: list[str] = Field(default_factory=list)
    previous_step_ids: list[str] = Field(default_factory=list)
    sources: list[TheorySource] = Field(default_factory=list)


class TheoryGap(Contract):
    at: NonEmpty
    reason: NonEmpty
    needed: NonEmpty
    sources: list[TheorySource] = Field(default_factory=list)


class TheoryTrace(Contract):
    """Model-authored mathematical reasoning, without formal-proof guarantees."""

    goal: NonEmpty
    assumptions: list[TheoryAssumption]
    steps: list[TheoryStep]
    gaps: list[TheoryGap]
    outcome: Literal["completed", "partial", "unable"]
    completion_reason: NonEmpty


class TheoryDerivationRecord(Contract):
    schema_version: Literal["theory-derivation-v1"] | None = None
    claim_id: NonEmpty
    item_index: int | None = Field(default=None, ge=0, strict=True)
    phase: Literal["main", "appendix"]
    adopted: bool
    covered: list[str] = Field(default_factory=list)
    source_pointer: EvidencePointer | None = None
    state: Literal["validated", "invalid", "legacy_unavailable"]
    validation_scope: Literal["structure_and_source_only"] = "structure_and_source_only"
    trace: TheoryTrace | None = None
    issues: list[str] = Field(default_factory=list)
    audit_pointer: str | None = None
    source_hashes: dict[str, str] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""
    transport: Literal["live", "injected"] = "injected"
    concern_reviews: list[TheoryConcernReview] = Field(default_factory=list)

    @model_validator(mode="after")
    def trace_state(self) -> Self:
        if any(not identifier or identifier != identifier.strip() for identifier in self.covered):
            raise ValueError("Theory record condition IDs must be exact")
        if len(self.covered) != len(set(self.covered)):
            raise ValueError("Theory record condition IDs must be unique")
        if self.state == "validated" and (self.schema_version is None or self.trace is None):
            raise ValueError("Validated Theory records require a versioned trace")
        if self.state == "legacy_unavailable" and (self.schema_version is not None or self.trace is not None):
            raise ValueError("Legacy Theory records must explicitly lack a versioned trace")
        identities = [(review.item_index, review.condition_id) for review in self.concern_reviews]
        if len(identities) != len(set(identities)) or any(
            index != self.item_index or condition not in self.covered for index, condition in identities
        ):
            raise ValueError(
                "Theory concern reviews must uniquely match this derivation and its covered conditions"
            )
        return self


class TheoryVisualSource(Contract):
    source_kind: Literal["visual"] = "visual"
    origin: Literal["model_transcribed_from_original_pixels"] = "model_transcribed_from_original_pixels"
    visual_source_id: NonEmpty
    target_id: NonEmpty
    page_id: NonEmpty
    anchor_block_id: NonEmpty
    printed_anchor: str
    pointer: EvidencePointer
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_path: NonEmpty
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    anchor_block_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audit_pointer: NonEmpty


class TheoryVisualAssumption(TheoryAssumption):
    sources: list[TheorySource | TheoryVisualSource] = Field(default_factory=list)


class TheoryVisualStep(TheoryStep):
    sources: list[TheorySource | TheoryVisualSource] = Field(default_factory=list)


class TheoryVisualGap(TheoryGap):
    sources: list[TheorySource | TheoryVisualSource] = Field(default_factory=list)


class TheoryVisualTrace(TheoryTrace):
    assumptions: list[TheoryVisualAssumption]
    steps: list[TheoryVisualStep]
    gaps: list[TheoryVisualGap]


class TheoryVisualRecord(TheoryDerivationRecord):
    schema_version: Literal["theory-visual-derivation-v1"]
    phase: Literal["visual_recheck"]
    state: Literal["validated", "invalid"]
    validation_scope: Literal["structure_source_identity_and_model_visual_reading"] = (
        "structure_source_identity_and_model_visual_reading"
    )
    trace: TheoryVisualTrace | None = None
    target_id: NonEmpty
    original_record_index: int = Field(ge=0, strict=True)
    response_item_index: int | None = Field(default=None, ge=0, strict=True)
    audit_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def visual_audit_required(self) -> Self:
        if self.state == "validated" and (not self.audit_pointer or not self.audit_sha256):
            raise ValueError("Validated visual Theory records require their audit path and hash")
        return self


TheoryRecord = TheoryDerivationRecord | TheoryVisualRecord


class Claim(Contract):
    """The same record is enriched from extraction through final assessment."""

    id: NonEmpty
    text: NonEmpty
    loc: ClaimLocation
    # Optional for previously saved records; extraction retains these together.
    source_block_id: NonEmpty | None = None
    source_quote: str | None = None
    source_refs: list[ClaimSourceRef] = Field(default_factory=list)
    conditions: list[Condition] = Field(min_length=1)
    needs: list[EvidenceNeed]
    importance: Literal["core", "secondary"] = "secondary"
    questions: list[AuthorQuestion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    notes: list[str] = Field(default_factory=list)
    advice: ClaimAdvice | None = None
    theory_derivations: list[TheoryRecord] = Field(default_factory=list)
    verification_limitations: list[VerificationLimitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        if (self.source_block_id is None) != (self.source_quote is None):
            raise ValueError("source_block_id and source_quote must be supplied together")
        if self.source_quote is not None and not self.source_quote.strip():
            raise ValueError("source_quote must contain original manuscript text")
        ids = [condition.id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition ids must be unique within a claim")
        if len(self.needs) != len(set(self.needs)):
            raise ValueError("needs must not contain duplicate branches")
        source_keys = []
        for ref in self.source_refs:
            if not set(ref.covered).issubset(ids):
                raise ValueError("source reference coverage must refer to this claim's condition ids")
            source_keys.append((ref.source_block_id, ref.source_quote, tuple(sorted(ref.covered))))
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source references must not be duplicated")
        for item in self.evidence:
            if not set(item.covered).issubset(ids):
                raise ValueError("evidence coverage must refer to this claim's condition ids")
        for question in self.questions:
            if question.claim_id is not None and question.claim_id != self.id:
                raise ValueError("an author question must refer to its enclosing claim")
        if self.advice:
            for item in self.advice.items:
                if not set(item.condition_ids).issubset(ids):
                    raise ValueError("advice must refer to its enclosing claim's condition ids")
        for record in self.theory_derivations:
            if record.claim_id != self.id or not set(record.covered).issubset(ids):
                raise ValueError("Theory derivations must refer to the enclosing claim and its condition ids")
        for limitation in self.verification_limitations:
            if limitation.claim_id != self.id or not set(limitation.condition_ids).issubset(ids):
                raise ValueError(
                    "Verification limitations must refer to the enclosing claim and its condition ids"
                )
        return self


class ExecutionTask(Contract):
    entry_script: str | None = None
    config: str | None = None
    command: list[str] = Field(default_factory=list)
    workdir: str = "."
    metric_output: str | None = None


class PaperTargetPassage(Contract):
    """The original reported target; no inferred or normalized source text."""

    block_id: NonEmpty
    quote: NonEmpty
    token: NonEmpty
    value_context: str = ""


class PaperTargetSelector(Contract):
    number_id: NonEmpty | None = None
    cell_id: NonEmpty | None = None

    @model_validator(mode="after")
    def one_original_location(self) -> Self:
        if (self.number_id is None) == (self.cell_id is None):
            raise ValueError("A paper target selects exactly one original number or table cell")
        return self


class ExecutionTargetBinding(Contract):
    """Reconstructable target proof. Consumers must revalidate it against the paper."""

    version: Literal[1] = 1
    condition_id: NonEmpty
    reported: PaperTargetPassage
    selector: PaperTargetSelector
    pointer: EvidencePointer
    block_sha256: NonEmpty
    artifact_sha256: NonEmpty
    claim_sha256: NonEmpty
    condition_sha256: NonEmpty
    value: FiniteNumber
    quantity_kind: Literal["absolute_measurement"] = "absolute_measurement"
    subject: str | None = None
    unit: str | None = None


class ProjectionFieldRole(Contract):
    path: NonEmpty
    role: Literal[
        "dataset_identity",
        "metric",
        "runtime_setting",
        "reported_value",
        "sample_scope",
        "measurement_definition",
        "conclusion_boundary",
        "unresolved",
    ]
    source_ids: list[NonEmpty] = Field(min_length=1)


class PredictionProjection(Contract):
    """A proposed interpretation of an unchanged original execution condition."""

    version: Literal["released-predictions-v1"]
    recipe: Literal["exact_match_accuracy"]
    data_path: NonEmpty
    field_roles: list[ProjectionFieldRole] = Field(min_length=1)


class ProjectionScopeDecision(Contract):
    plan_index: int = Field(ge=0, strict=True)
    condition_id: NonEmpty
    classification: Literal["absolute_fixed_predictions", "comparative", "unresolved"]
    dataset_identity_confirmed: bool = Field(strict=True)
    measurement_definition_confirmed: bool = Field(strict=True)
    all_original_qualifiers_preserved: bool = Field(strict=True)
    confirmed_field_paths: list[NonEmpty]
    source_ids: list[NonEmpty] = Field(min_length=1)
    unresolved: list[str]
    rationale: NonEmpty


class ProjectionAtom(Contract):
    id: NonEmpty
    source_ids: list[NonEmpty] = Field(min_length=1)


class DatasetProjectionAtom(ProjectionAtom):
    kind: Literal["dataset_identity"]
    dataset: NonEmpty
    split: str | None


class RuntimeProjectionAtom(ProjectionAtom):
    kind: Literal["runtime_setting"]
    key: NonEmpty
    value: (
        Annotated[str, Field(strict=True)]
        | Annotated[int, Field(strict=True)]
        | Annotated[float, Field(strict=True, allow_inf_nan=False)]
        | Annotated[bool, Field(strict=True)]
        | None
    )


class ValueProjectionAtom(ProjectionAtom):
    kind: Literal["reported_value"]
    number_id: NonEmpty


class SampleProjectionAtom(ProjectionAtom):
    kind: Literal["sample_scope"]
    count: int = Field(gt=0, strict=True)
    population: Literal["fixed_released_predictions"]
    split: str | None
    all_records: Literal[True]

    @model_validator(mode="before")
    @classmethod
    def strict_full_population(cls, value):
        if isinstance(value, dict) and value.get("all_records") is not True:
            raise ValueError("Full-population declaration must be the boolean true")
        return value


class DefinitionProjectionAtom(ProjectionAtom):
    kind: Literal["measurement_definition"]
    measure: Literal["exact_match_accuracy"]
    predicate: Literal["prediction_equals_label"]
    aggregation: Literal["fraction_of_all_records"]
    unit: Literal["fraction"]


class BoundaryProjectionAtom(ProjectionAtom):
    kind: Literal["conclusion_boundary"]
    excludes: list[Literal["repeated_run_uncertainty", "population_performance", "cross_model_ranking"]] = (
        Field(min_length=1)
    )


class UnresolvedProjectionAtom(ProjectionAtom):
    kind: Literal["unresolved"]
    reason: NonEmpty


SemanticProjectionAtom = Annotated[
    DatasetProjectionAtom
    | RuntimeProjectionAtom
    | ValueProjectionAtom
    | SampleProjectionAtom
    | DefinitionProjectionAtom
    | BoundaryProjectionAtom
    | UnresolvedProjectionAtom,
    Field(discriminator="kind"),
]


class ProjectionFieldBinding(Contract):
    path: NonEmpty
    atom_ids: list[NonEmpty] = Field(min_length=1)


class ProjectionClaimBinding(Contract):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    atom_ids: list[NonEmpty] = Field(min_length=1)


class SemanticPredictionProjection(Contract):
    version: Literal["released-predictions-v2"]
    recipe: Literal["exact_match_accuracy"]
    data_path: NonEmpty
    atoms: list[SemanticProjectionAtom] = Field(min_length=1)
    field_bindings: list[ProjectionFieldBinding] = Field(min_length=1)
    claim_bindings: list[ProjectionClaimBinding] = Field(min_length=1)


class ProjectionAtomReview(Contract):
    atom_id: NonEmpty
    decision: Literal["confirmed", "unresolved"]
    source_ids: list[NonEmpty] = Field(min_length=1)
    rationale: NonEmpty


class ProjectionFieldReview(ProjectionFieldBinding):
    decision: Literal["confirmed", "unresolved"]
    rationale: NonEmpty


class ProjectionClaimReview(ProjectionClaimBinding):
    decision: Literal["confirmed", "unresolved"]
    rationale: NonEmpty


class SemanticProjectionScopeDecision(ProjectionScopeDecision):
    version: Literal["released-predictions-v2"]
    proposal_sha256: NonEmpty
    atom_reviews: list[ProjectionAtomReview]
    field_reviews: list[ProjectionFieldReview]
    claim_reviews: list[ProjectionClaimReview]


class ProjectionRecord(Contract):
    proposal: PredictionProjection | SemanticPredictionProjection
    runtime_target: Condition
    sample_count: int = Field(gt=0, strict=True)
    source_pointers: dict[str, EvidencePointer]
    source_hashes: dict[str, NonEmpty]
    paper_hashes: dict[str, NonEmpty]
    repository_hashes: dict[str, NonEmpty]
    repository_root: NonEmpty
    entry_script: NonEmpty
    config_path: NonEmpty
    label_key: NonEmpty
    prediction_key: NonEmpty
    scope_review: ProjectionScopeDecision | SemanticProjectionScopeDecision
    scope_audit: NonEmpty
    scope_audit_sha256: NonEmpty
    recipe_sha256: NonEmpty


class SemanticProjectionRecord(ProjectionRecord):
    proposal: SemanticPredictionProjection
    scope_review: SemanticProjectionScopeDecision
    field_consumption: dict[str, Any]
    claim_consumption: list[dict[str, Any]]
    semantics_sha256: NonEmpty


ChoiceIdentity = Annotated[str, Field(strict=True, min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$")]


class ExecutionChoiceSelection(Contract):
    version: Literal["released-predictions-choice-v1"]
    condition_id: ChoiceIdentity
    decision: Literal["select"]
    candidate_id: ChoiceIdentity
    rationale: NonEmpty


class ExecutionChoiceUnresolved(Contract):
    version: Literal["released-predictions-choice-v1"]
    condition_id: ChoiceIdentity
    decision: Literal["unresolved"]
    rationale: NonEmpty


ExecutionChoice = Annotated[
    ExecutionChoiceSelection | ExecutionChoiceUnresolved, Field(discriminator="decision")
]


class ExecutionChoiceObligationReview(Contract):
    obligation_id: ChoiceIdentity
    decision: Literal["confirmed", "unresolved"]
    rationale: NonEmpty


class ExecutionChoiceReview(Contract):
    version: Literal["released-predictions-choice-v1"]
    condition_id: ChoiceIdentity
    candidate_id: ChoiceIdentity
    classification: Literal["absolute_fixed_predictions", "unresolved"]
    decision: Literal["confirmed", "unresolved"]
    reviews: list[ExecutionChoiceObligationReview]
    rationale: NonEmpty


class ChoiceProjectionRecord(SemanticProjectionRecord):
    # REQUIRED discriminator, no default added to old records. proposal and
    # scope_review are explicitly program-normalized v2 structures; these fields
    # retain the actual new wire decision and its audit list positions.
    record_version: Literal["released-predictions-choice-v1"]
    selection: ExecutionChoiceSelection
    choice_review: ExecutionChoiceReview
    selection_index: int = Field(ge=0, strict=True)
    review_index: int = Field(ge=0, strict=True)
    candidate_id: ChoiceIdentity
    registry_sha256: NonEmpty
    registry_snapshot: dict[str, Any]
    choice_builder_sha256: NonEmpty


class ProjectedExecutionTargetBinding(ExecutionTargetBinding):
    version: Literal[2] = 2
    projection: ChoiceProjectionRecord | ProjectionRecord | SemanticProjectionRecord


class ExecutionPlan(Contract):
    id: NonEmpty
    claim_id: NonEmpty
    condition_ids: list[NonEmpty] = Field(min_length=1)
    target_conditions: list[Condition] = Field(min_length=1)
    task: ExecutionTask = Field(default_factory=ExecutionTask)
    run_mode: Literal["evaluation", "analysis", "training"]
    # Condition-id keys preserve two datasets reporting the same metric.
    y_paper: dict[str, FiniteNumber] = Field(min_length=1)
    # Empty is the readable historical form; it never grants execution trust.
    target_bindings: dict[str, ExecutionTargetBinding | ProjectedExecutionTargetBinding] = Field(
        default_factory=dict
    )
    feasibility: Literal["ready", "blocked"]
    blocker: str = ""
    priority: Literal["high", "medium", "low"]
    estimated_cost: str = "unknown"

    @model_validator(mode="after")
    def check_feasibility_and_conditions(self) -> Self:
        if self.feasibility == "blocked" and not self.blocker.strip():
            raise ValueError("blocked plans require a blocker reason")
        if self.feasibility == "ready" and self.blocker.strip():
            raise ValueError("ready plans cannot have an unresolved blocker")
        if self.feasibility == "ready" and not (self.task.entry_script or self.task.command):
            raise ValueError("ready plans require a candidate script or command")
        ids = [condition.id for condition in self.target_conditions]
        if len(self.condition_ids) != len(set(self.condition_ids)) or len(ids) != len(set(ids)):
            raise ValueError("plan condition ids must be unique")
        if set(self.condition_ids) != set(ids):
            raise ValueError("condition_ids must match target_conditions")
        if set(self.y_paper) != set(ids):
            raise ValueError("y_paper keys must match condition_ids")
        if not set(self.target_bindings).issubset(ids) or any(
            key != binding.condition_id for key, binding in self.target_bindings.items()
        ):
            raise ValueError("target bindings must identify their original plan conditions")
        if any(not condition.metric for condition in self.target_conditions):
            raise ValueError("execution target conditions require a metric")
        return self


class Finding(Contract):
    kind: Literal["writing", "figure", "reference", "related_work", "baseline", "table"]
    loc: ClaimLocation
    evidence: list[Evidence] = Field(min_length=1)
    level: NonEmpty
    text: NonEmpty
    reference_correction: ReferenceCorrection | None = None

    @model_validator(mode="after")
    def correction_kind(self) -> Self:
        if self.reference_correction is not None and self.kind != "reference":
            raise ValueError("Reference corrections may only be attached to reference findings")
        return self
