"""Explicit multi-file Code evidence stays one condition and one assessed observation."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from common import run_stats
from llm.client import LLMConfig
from preprocessing.materials import index_repository
from review.report.advice import advice_input, generate_advice
from review.report.v2 import write_review
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidencePointer
from schemas.materials import MaterialBlock, SharedMaterials
from schemas.review import FinalReview
from screening import checks
from verification.code import verify_code
from verification.code_joint import checked_joint_sources


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("CODE_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "joint-code", None, None))
    monkeypatch.setattr(
        "review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "joint-code", None, None)
    )
    monkeypatch.setattr("review.report.advice.llm_json", lambda **kw: pytest.fail("Unmocked advice"))
    for target in (
        "screening.checks.llm_json",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "requests.sessions.Session.request",
        "subprocess.run",
    ):
        monkeypatch.setattr(target, lambda *a, **kw: pytest.fail("Unmocked external boundary"))


def fixture(tmp_path):
    text = "The supplied evaluator loads its configuration and four prediction rows from this repository."
    paper = tmp_path / "paper.md"
    paper.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    root = tmp_path / "repo"
    root.mkdir()
    files = {
        "evaluate.py": 'import json\nfrom pathlib import Path\nconfig = json.loads(Path("config.json").read_text())\nrows = json.loads(Path("rows.json").read_text())\n',
        "config.json": '{"split": "test"}\n',
        "rows.json": '[{"p":0},{"p":1},{"p":0},{"p":1}]\n',
    }
    for name, value in files.items():
        (root / name).write_text(value, encoding="utf-8")
    materials = SharedMaterials(
        paper_key="joint-code",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(paper),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
        repository=index_repository(root),
    )
    claim = Claim(
        id="c",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[Condition(id="c1", description=text)],
        needs=["Code"],
    )
    item = {
        "file": "evaluate.py",
        "line": 1,
        "quote": files["evaluate.py"].rstrip("\n"),
        "additional_sources": [
            {"file": n, "line": 1, "quote": files[n].rstrip("\n")} for n in ["config.json", "rows.json"]
        ],
        "paper_block_id": "b1",
        "paper_quote": text,
        "covered": ["c1"],
        "fully_supported_conditions": ["c1"],
        "direction": "support",
        "aspect": "evaluation",
        "detail": "The three explicitly selected members jointly establish the supplied repository contents.",
    }
    scope = {
        "conditions": [
            {
                "condition_id": "c1",
                "required_facets": ["implementation", "repository_contents"],
                "claim_source_ids": ["primary"],
                "rationale": "Only contents of this given repository are asserted.",
            }
        ],
        "items": [
            {
                "item_index": 0,
                "condition_id": "c1",
                "relation": "supports_implementation",
                "full_condition": True,
                "basis": "direct_source",
                "bridge_quotes": [],
                "missing_qualifiers": [],
                "rationale": "The evaluator links the exact configuration and rows; the supplied JSON has four rows.",
                "source_uses": [
                    {
                        "source_index": i,
                        "role": role,
                        "rationale": "This selected member establishes its part of the original condition.",
                    }
                    for i, role in enumerate(["link", "configuration", "artifact_contents"])
                ],
            }
        ],
    }
    return claim, materials, item, scope


def run(tmp_path, claim, materials, items, scope, hook=None):
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        if hook:
            hook(kwargs)
        return copy.deepcopy(scope if kwargs["module"].endswith(".scope") else {"items": items})

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        branch = verify_code(claim, materials, call=model)
    assessed = claim.model_copy(deep=True)
    assessed.evidence.extend(branch.evidence)
    assessed.verification_limitations.extend(branch.verification_limitations)
    return branch, assess_claim(assessed), calls


def test_explicit_joint_has_one_evidence_and_all_sources(tmp_path):
    c, m, item, scope = fixture(tmp_path)
    result, assessed, calls = run(tmp_path, c, m, [item], scope)
    assert [x["module"] for x in calls] == ["verification.code", "verification.code.scope"]
    assert assessed.status == "supported" and len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.sufficient and evidence.covered == ["c1"]
    assert len(evidence.additional_pointers) == 2 and evidence.code_joint_binding is not None
    assert Evidence.model_validate(evidence.model_dump()).model_dump() == evidence.model_dump()
    request = json.loads(calls[1]["prompt"])
    assert request["source_context"] == {}
    assert [row["pointer"]["quote"] for row in request["joint_members"]["0"]] == [
        item["quote"],
        *[s["quote"] for s in item["additional_sources"]],
    ]


def test_availability_does_not_become_local_contents(tmp_path):
    c, m, item, scope = fixture(tmp_path)
    scope["conditions"][0]["required_facets"].append("availability")
    result, assessed, _ = run(tmp_path, c, m, [item], scope)
    assert assessed.status == "unverified" and result.evidence and not result.evidence[0].sufficient


def test_old_partial_flags_never_union(tmp_path):
    c, m, item, scope = fixture(tmp_path)
    item["fully_supported_conditions"] = []
    result, assessed, _ = run(tmp_path, c, m, [item], scope)
    assert assessed.status == "unverified" and not result.evidence[0].sufficient


def test_bare_code_additional_pointers_stays_rejected():
    p = EvidencePointer(locator="a.py", line=1, quote="x=1")
    with pytest.raises(ValueError):
        Evidence(
            source="code",
            pointer=p,
            additional_pointers=[EvidencePointer(locator="b.py", line=1, quote="y=2")],
            covered=["c1"],
            direction="support",
        )


def neighbor(c, item, scope):
    c.conditions.append(Condition(id="c2", description="The evaluator loads the configuration."))
    single = copy.deepcopy(item)
    single.update(additional_sources=[], covered=["c2"], fully_supported_conditions=["c2"])
    scope["conditions"].append(
        {**copy.deepcopy(scope["conditions"][0]), "condition_id": "c2", "required_facets": ["implementation"]}
    )
    scope["items"].append(
        {**copy.deepcopy(scope["items"][0]), "condition_id": "c2", "item_index": 1, "source_uses": []}
    )
    return single


@pytest.mark.parametrize(
    "bad", ["line", "quote", "foreign_file", "duplicate", "unknown_field", "two_conditions", "flaw"]
)
def test_invalid_joint_is_local_and_original_neighbor_index_survives(tmp_path, bad):
    c, m, item, scope = fixture(tmp_path)
    single = neighbor(c, item, scope)
    if bad == "line":
        item["additional_sources"][0]["line"] = 2
    elif bad == "quote":
        item["additional_sources"][0]["quote"] = "wrong"
    elif bad == "foreign_file":
        item["additional_sources"][0]["file"] = "unselected.json"
    elif bad == "duplicate":
        item["additional_sources"].append(copy.deepcopy(item["additional_sources"][0]))
    elif bad == "unknown_field":
        item["additional_sources"][0]["other"] = True
    elif bad == "two_conditions":
        item["covered"].append("c2")
    else:
        item["direction"] = "flaw"
    result, assessed, calls = run(tmp_path, c, m, [item, single], scope)
    assert len(calls) == 2
    assert (
        len(result.evidence) == 1 and result.evidence[0].covered == ["c2"] and result.evidence[0].sufficient
    )
    assert assessed.status == "unverified" and result.verification_limitations
    assert result.verification_limitations[0].responsibility == "system"
    audit = json.loads(next((tmp_path / "code_scope_reviews").glob("*.json")).read_text("utf-8"))
    assert audit["request"]["candidate_indices"] == [1]
    assert audit["first_response"] == {"items": [item, single]}


@pytest.mark.parametrize(
    "bad",
    [
        "missing_member",
        "duplicate_member",
        "foreign_member",
        "foreign_condition",
        "padded_condition_duplicate",
        "pair_tombstone",
        "missing_scope",
        "foreign_claim_source",
    ],
)
def test_scope_rejection_cannot_become_joint_support_and_keeps_neighbor(tmp_path, bad):
    c, m, item, scope = fixture(tmp_path)
    single = neighbor(c, item, scope)
    pair = scope["items"][0]
    if bad == "missing_member":
        pair["source_uses"].pop()
    elif bad == "duplicate_member":
        pair["source_uses"].append(copy.deepcopy(pair["source_uses"][0]))
    elif bad == "foreign_member":
        pair["source_uses"][0]["source_index"] = 99
    elif bad == "foreign_condition":
        pair["condition_id"] = "unknown"
    elif bad == "padded_condition_duplicate":
        scope["conditions"] += [
            {**copy.deepcopy(scope["conditions"][0]), "condition_id": " c1 "},
            copy.deepcopy(scope["conditions"][0]),
        ]
    elif bad == "pair_tombstone":
        scope["items"] += [{**copy.deepcopy(pair), "condition_id": " c1 "}, copy.deepcopy(pair)]
    elif bad == "missing_scope":
        scope["conditions"].pop(0)
    else:
        scope["conditions"][0]["claim_source_ids"] = ["foreign"]
    result, _, calls = run(tmp_path, c, m, [item, single], scope)
    assert len(calls) == 2 and not any(e.sufficient and "c1" in e.covered for e in result.evidence)
    assert any(e.sufficient and e.covered == ["c2"] for e in result.evidence)


@pytest.mark.parametrize("facet", ["empirical_outcome", "novelty", "availability", "uncertain"])
def test_unproven_facets_preserve_partial_joint(tmp_path, facet):
    c, m, item, scope = fixture(tmp_path)
    scope["conditions"][0]["required_facets"].append(facet)
    result, assessed, _ = run(tmp_path, c, m, [item], scope)
    assert assessed.status == "unverified" and len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.evidence[0].code_joint_binding


@pytest.mark.parametrize("phase", ["verification.code", "verification.code.scope"])
@pytest.mark.parametrize("target", ["primary", "secondary", "claim", "materials"])
def test_callback_source_mutation_cannot_rebase_joint(tmp_path, phase, target):
    c, m, item, scope = fixture(tmp_path)

    def hook(kwargs):
        if kwargs["module"] != phase:
            return
        if target == "claim":
            c.conditions[0].description = "changed"
        elif target == "materials":
            m.blocks[0].text = "changed"
        else:
            (Path(m.repository.root) / ("evaluate.py" if target == "primary" else "rows.json")).write_text(
                "changed", encoding="utf-8"
            )

    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, c, m, [item], scope, hook)


@pytest.mark.parametrize("invalid_joint", [False, True])
def test_joint_audit_failure_keeps_healthy_single(tmp_path, monkeypatch, invalid_joint):
    c, m, item, scope = fixture(tmp_path)
    single = neighbor(c, item, scope)
    if invalid_joint:
        item["additional_sources"][0]["quote"] = "not its original source"
    original = Path.write_text

    def fail(path, *args, **kwargs):
        if path.parent.name == "code_scope_reviews":
            raise OSError("audit unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    result, _, calls = run(tmp_path, c, m, [item, single], scope)
    assert len(calls) == 2 and len(result.evidence) == 1 and result.evidence[0].covered == ["c2"]
    assert result.evidence[0].sufficient and result.verification_limitations
    assert any("audit" in issue.lower() for issue in result.issues)


def test_all_invalid_joint_preserves_original_first_response(tmp_path):
    c, m, item, scope = fixture(tmp_path)
    item["additional_sources"][0]["line"] = False
    # The second selected range cannot match line zero after non-strict legacy coercion.
    result, _, calls = run(tmp_path, c, m, [item], scope)
    assert len(calls) == 1 and not result.evidence
    audit = json.loads(next((tmp_path / "code_scope_reviews").glob("*.json")).read_text("utf-8"))
    assert audit["first_response"] == {"items": [item]} and audit["state"] == "no_grounded_candidate"


def good_advice(claim):
    return {
        "status": "ok",
        "claim_id": claim.id,
        "items": [
            {
                "condition_ids": ["c1"],
                "basis_refs": ["/evidence/0"],
                "action": "reviewer_guidance",
                "text": "The declared repository files support this local implementation claim.",
            }
        ],
    }


@pytest.mark.parametrize("target", ["secondary", "audit", "paper"])
@pytest.mark.parametrize("mutation", ["bytes", "delete"])
def test_joint_artifact_change_during_advice_cannot_publish_stale_text(tmp_path, target, mutation):
    c, m, item, scope = fixture(tmp_path)
    _, assessed, _ = run(tmp_path, c, m, [item], scope)
    before = assessed.model_dump(mode="json")
    evidence = assessed.evidence[0]
    path = Path(
        evidence.additional_pointers[-1].locator
        if target == "secondary"
        else evidence.code_joint_binding.scope_audit_pointer
        if target == "audit"
        else m.markdown_path
    )

    def model(**kwargs):
        if mutation == "bytes":
            path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
        else:
            path.unlink()
        return good_advice(assessed)

    result = generate_advice(
        FinalReview(paper_key="joint", run_id="offline", claims=[assessed]), tmp_path / "advice", call=model
    )
    assert result.review.claims[0].advice.state == "unavailable"
    assert any(
        word in result.review.claims[0].advice.failure_reason.lower() for word in ("changed", "unavailable")
    )
    assert result.review.claims[0].model_dump(mode="json", exclude={"advice"}) == {
        k: v for k, v in before.items() if k != "advice"
    }


@pytest.mark.parametrize("presentation", ["full", "layered"])
def test_delivery_keeps_all_pointers_and_stale_science_with_visible_integrity(tmp_path, presentation):
    c, m, item, scope = fixture(tmp_path)
    _, assessed, _ = run(tmp_path, c, m, [item], scope)
    review = FinalReview(paper_key="joint", run_id="offline", claims=[assessed])
    generated = generate_advice(review, tmp_path / "advice", call=lambda **kw: good_advice(assessed))
    assert generated.counts == {"generated": 1, "unavailable": 0}
    evidence = generated.review.claims[0].evidence[0]
    files = advice_input(assessed, [])["source_files"]
    assert evidence.code_joint_binding.scope_audit_pointer in files
    assert all(p.locator in files for p in [evidence.pointer, *evidence.additional_pointers])
    result = write_review(generated.review, tmp_path / "healthy", render_pdf=False, presentation=presentation)
    healthy = json.loads(Path(result["json"]).read_text("utf-8"))
    assert healthy["claims"][0] == generated.review.claims[0].model_dump(mode="json")
    Path(evidence.additional_pointers[0].locator).write_text("changed", encoding="utf-8")
    result = write_review(generated.review, tmp_path / "stale", render_pdf=False, presentation=presentation)
    saved = json.loads(Path(result["json"]).read_text("utf-8"))["claims"][0]
    assert saved["status"] == "supported" and saved["evidence"] == healthy["claims"][0]["evidence"]
    assert saved["advice"]["state"] == "unavailable"
    main = Path(result["markdown"]).read_text("utf-8")
    assert "Current Code source integrity is unavailable" in main
    all_text = "\n".join(Path(p).read_text("utf-8") for p in result.values() if str(p).endswith(".md"))
    for name in ("evaluate.py", "config.json", "rows.json"):
        assert name.replace(".", "\\.") in all_text


@pytest.mark.parametrize(
    "mutation", ["pointer", "claim", "missing_binding_member", "candidate_index", "audit_response"]
)
def test_persisted_joint_reconstruction_rejects_tampering(tmp_path, mutation):
    c, m, item, scope = fixture(tmp_path)
    _, assessed, _ = run(tmp_path, c, m, [item], scope)
    evidence = assessed.evidence[0]
    if mutation == "pointer":
        evidence.additional_pointers[0].quote = "wrong"
    elif mutation == "claim":
        assessed.conditions[0].description = "different"
    elif mutation == "missing_binding_member":
        evidence.code_joint_binding.members.pop()
    elif mutation == "candidate_index":
        evidence.code_joint_binding.candidate_index = 1
    else:
        path = Path(evidence.code_joint_binding.scope_audit_pointer)
        audit = json.loads(path.read_text("utf-8"))
        audit["response"]["items"].append(copy.deepcopy(audit["response"]["items"][0]))
        path.write_text(json.dumps(audit), encoding="utf-8")
        evidence.code_joint_binding.scope_audit_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, KeyError, IndexError)):
        checked_joint_sources(evidence, assessed)


def test_legacy_single_source_serialization_does_not_add_null_binding():
    evidence = Evidence(
        source="code",
        pointer=EvidencePointer(locator="a.py", line=1, quote="x"),
        covered=["c1"],
        direction="support",
    )
    assert "code_joint_binding" not in evidence.model_dump(mode="json")
    assert Evidence.model_validate(evidence.model_dump()).model_dump() == evidence.model_dump()


@pytest.mark.parametrize("presentation", ["full", "layered"])
def test_joint_pdf_exposes_all_exact_sources_and_live_destinations(tmp_path, presentation):
    from pypdf import PdfReader

    c, m, item, scope = fixture(tmp_path)
    _, assessed, _ = run(tmp_path, c, m, [item], scope)
    result = write_review(
        FinalReview(paper_key="joint", run_id="offline", claims=[assessed]),
        tmp_path / "pdf",
        presentation=presentation,
    )
    assert not any(key.endswith("_error") for key in result)
    paths = [Path(value) for key, value in result.items() if key.endswith("pdf")]
    assert paths
    readers = [PdfReader(path) for path in paths]
    text = "\n".join(page.extract_text() for reader in readers for page in reader.pages)
    for value in ("evaluate.py", "config.json", "rows.json", '"split": "test"'):
        assert value in text
    for reader in readers:
        pages = {page.indirect_reference.idnum for page in reader.pages}
        for page in reader.pages:
            for raw in page.get("/Annots", []):
                annotation = raw.get_object()
                dest = annotation.get("/Dest")
                if isinstance(dest, list):
                    assert dest[0].idnum in pages
