"""Local reconstruction of joint Code sources and their independent scope audit."""

import hashlib
import json
from pathlib import Path

from schemas.code_joint import CodeJointBinding, CodeJointMember


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def target_record(claim):
    value = claim.model_dump(mode="json") if hasattr(claim, "model_dump") else claim
    return {
        key: value[key]
        for key in (
            "id",
            "text",
            "conditions",
            "loc",
            "source_block_id",
            "source_quote",
            "source_refs",
            "needs",
            "importance",
        )
    }


def make_binding(claim, condition_id, index, members, audit):
    if audit is None:
        raise ValueError("Joint Code scope audit is unavailable")
    return CodeJointBinding(
        claim_id=claim.id,
        condition_id=condition_id,
        candidate_index=index,
        target_sha256=digest(target_record(claim)),
        scope_audit_pointer=str(Path(audit).resolve()),
        scope_audit_sha256=hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
        members=[
            CodeJointMember(pointer_sha256=digest(row["pointer"]), artifact_sha256=row["artifact_sha256"])
            for row in members
        ],
    )


def checked_joint_sources(evidence, claim):
    """Check persisted pointers without changing historical evidence or assessment."""
    binding = evidence.code_joint_binding
    if binding is None:
        return {}
    # Revalidate deserialized/model_copy records before trusting their typed marker.
    from schemas.claim import Evidence
    from verification.code import CodeItem
    from verification.code_scope import CodeConditionScope, CodeScopeDecision

    Evidence.model_validate(evidence.model_dump(mode="json"))
    if (
        binding.claim_id != claim.id
        or binding.condition_id not in {c.id for c in claim.conditions}
        or binding.target_sha256 != digest(target_record(claim))
    ):
        raise ValueError("Joint Code evidence belongs to a different claim/condition")
    path = Path(binding.scope_audit_pointer)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != binding.scope_audit_sha256:
        raise ValueError("Joint Code scope audit changed or is unavailable")
    audit = json.loads(path.read_text("utf-8"))
    request = audit["request"]
    if target_record(request["claim"]) != target_record(claim):
        raise ValueError("Joint Code scope target is inconsistent")
    indices = request["candidate_indices"]
    if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Joint Code original candidate indices are invalid")
    candidate = CodeItem.model_validate(request["candidate_items"][indices.index(binding.candidate_index)])
    original = CodeItem.model_validate(audit["first_response"]["items"][binding.candidate_index])
    if (
        candidate != original
        or candidate.covered != [binding.condition_id]
        or candidate.direction != "support"
    ):
        raise ValueError("Joint Code original candidate identity changed")
    members = request["joint_members"][str(binding.candidate_index)]
    pointers = [evidence.pointer, *evidence.additional_pointers]
    if len(members) != len(pointers) or len(pointers) != len(binding.members):
        raise ValueError("Joint Code member count changed")
    result = {str(path.resolve()): binding.scope_audit_sha256}
    for name, expected in audit["paper_source_hashes"].items():
        paper = Path(name)
        if not paper.is_file() or hashlib.sha256(paper.read_bytes()).hexdigest() != expected:
            raise ValueError("Joint Code original manuscript source changed")
        result[str(paper.resolve())] = expected
    root = Path(audit["repository_root"]).resolve(strict=True)
    spans = [candidate, *candidate.additional_sources]
    for ordinal, (member, pointer, identity, span) in enumerate(
        zip(members, pointers, binding.members, spans, strict=True)
    ):
        file = root / member["file"]
        if not file.is_file():
            raise ValueError("Joint Code source artifact is unavailable")
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(root):
            raise ValueError("Joint Code file escapes its recorded repository")
        if (
            member["source_index"] != ordinal
            or pointer.model_dump(mode="json") != member["pointer"]
            or digest(member["pointer"]) != identity.pointer_sha256
            or str(file.resolve()) != pointer.locator
        ):
            raise ValueError("Joint Code member pointer changed")
        if (span.file, span.line, span.quote) != (member["file"], pointer.line, pointer.quote):
            raise ValueError("Joint Code member was not selected by the original candidate")
        content = file.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != identity.artifact_sha256
            or member["artifact_sha256"] != identity.artifact_sha256
        ):
            raise ValueError("Joint Code source artifact changed after verification")
        lines, quote = content.decode("utf-8").splitlines(), pointer.quote.splitlines()
        if not pointer.quote.strip() or lines[pointer.line - 1 : pointer.line - 1 + len(quote)] != quote:
            raise ValueError("Joint Code exact source lines changed")
        result[str(file.resolve())] = identity.artifact_sha256
    pair = [
        row
        for row in audit["validated_items"]
        if row["item_index"] == binding.candidate_index and row["condition_id"] == binding.condition_id
    ]
    if len(pair) != 1 or sorted(x["source_index"] for x in pair[0]["source_uses"]) != list(
        range(len(members))
    ):
        raise ValueError("Joint Code independent member consumption is unavailable")
    raw_pairs = [
        row
        for row in audit["response"]["items"]
        if isinstance(row, dict)
        and type(row.get("item_index")) is int
        and row["item_index"] == binding.candidate_index
        and isinstance(row.get("condition_id"), str)
        and row["condition_id"].strip() == binding.condition_id
    ]
    raw_scopes = [
        row
        for row in audit["response"]["conditions"]
        if isinstance(row, dict)
        and isinstance(row.get("condition_id"), str)
        and row["condition_id"].strip() == binding.condition_id
    ]
    scope = audit["validated_conditions"][binding.condition_id]
    if (
        len(raw_pairs) != 1
        or len(raw_scopes) != 1
        or CodeScopeDecision.model_validate(raw_pairs[0]).model_dump(mode="json") != pair[0]
        or CodeConditionScope.model_validate(raw_scopes[0]).model_dump(mode="json") != scope
    ):
        raise ValueError("Joint Code scope no longer matches its unique original response")
    claim_sources = {row["source_id"]: row for row in request["claim_sources"]}
    if not scope["claim_source_ids"] or any(
        binding.condition_id not in claim_sources[name]["covered"] for name in scope["claim_source_ids"]
    ):
        raise ValueError("Joint Code scope source is outside its condition")
    if evidence.affects_claim != evidence.sufficient or evidence.concern:
        raise ValueError("Joint Code support effects do not match its recorded sufficiency")
    if evidence.sufficient:
        if not (
            set(scope["required_facets"]) <= {"implementation", "repository_contents"}
            and scope["required_facets"]
            and pair[0]["relation"] == "supports_implementation"
            and pair[0]["full_condition"] is True
            and pair[0]["basis"] == "direct_source"
            and not pair[0]["missing_qualifiers"]
            and binding.condition_id in candidate.fully_supported_conditions
        ):
            raise ValueError("Joint Code sufficient evidence lacks its original independent full review")
    return result
