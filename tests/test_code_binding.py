"""Code agreement must establish the implementation asserted by this claim."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from common import run_stats
from llm.client import LLMConfig
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from screening import checks
from verification.code import verify_code


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("CODE_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "scope", None, None))
    monkeypatch.setattr(checks, "llm_json", lambda **kw: pytest.fail("Unmocked model"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Process"))


@pytest.fixture
def inputs(tmp_path):
    def make(text="Our method uses Adam.", *, optimizer="Adam"):
        methods = "The optimizer for our method is Adam."
        markdown = text + "\n\n" + methods
        path = tmp_path / "paper.md"
        path.write_text(markdown, encoding="utf-8")
        blocks = [
            MaterialBlock(
                id=identifier,
                text=value,
                loc=ClaimLocation(
                    page=1,
                    section=identifier,
                    char_start=markdown.index(value),
                    char_end=markdown.index(value) + len(value),
                ),
            )
            for identifier, value in [("abstract", text), ("methods", methods)]
        ]
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        code = f"optimizer = '{optimizer}'"
        (repo / "config.py").write_text(code + "\n", encoding="utf-8")
        materials = SharedMaterials(
            paper_key="code-scope",
            source_pdf=str(tmp_path / "paper.pdf"),
            markdown=markdown,
            markdown_path=str(path),
            content_list_path="",
            provider="fixture",
            blocks=blocks,
            repository=index_repository(repo),
        )
        claim = Claim(
            id="claim",
            text=text,
            loc=blocks[0].loc,
            source_block_id="abstract",
            source_quote=text,
            conditions=[Condition(id="c1", description=text)],
            needs=["Code"],
        )
        item = {
            "file": "config.py",
            "line": 1,
            "quote": code,
            "paper_block_id": "methods",
            "paper_quote": methods,
            "covered": ["c1"],
            "fully_supported_conditions": ["c1"],
            "direction": "support",
            "aspect": "optimizer",
            "detail": "First pass claims complete agreement; independently check applicability.",
        }
        return claim, materials, item

    return make


def review(claim, items, *, facets=None, relation="supports_implementation"):
    return {
        "conditions": [
            {
                "condition_id": condition.id,
                "required_facets": facets or ["implementation"],
                "claim_source_ids": ["primary"],
                "rationale": "The exact source specifies the requirement for this condition.",
            }
            for condition in claim.conditions
        ],
        "items": [
            {
                "item_index": index,
                "condition_id": condition,
                "relation": relation,
                "full_condition": relation == "supports_implementation",
                "basis": "direct_source",
                "bridge_quotes": [],
                "missing_qualifiers": [],
                "rationale": "The target source, method description and actual source line correspond.",
            }
            for index, item in enumerate(items)
            for condition in item["covered"]
        ],
    }


def run(claim, materials, items, decision):
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.code.scope":
            if isinstance(decision, Exception):
                raise decision
            return decision
        assert kwargs["module"] == "verification.code"
        return {"items": items}

    result = verify_code(claim, materials, call=model)
    claim.evidence = result.evidence
    return result, assess_claim(claim), calls


def test_exact_empirical_claim_cannot_be_supported_by_same_paper_adam_configuration(inputs):
    claim, materials, item = inputs("A test MRR is 0.4.")
    claim.conditions[0] = Condition(id="c1", dataset="A", metric="MRR", settings={"split": "test"})
    result, assessed, calls = run(
        claim, materials, [item], review(claim, [item], facets=["empirical_outcome"])
    )
    assert calls == ["verification.code", "verification.code.scope"]
    assert result.evidence and not result.evidence[0].sufficient
    assert assessed.status == "unverified"


def test_grounded_abstract_to_methods_correspondence_can_support(inputs):
    claim, materials, item = inputs()
    assert claim.source_quote not in item["paper_quote"]
    before = hashlib.sha256(Path(materials.repository.root, "config.py").read_bytes()).hexdigest()
    result, assessed, calls = run(claim, materials, [item], review(claim, [item]))
    assert calls == ["verification.code", "verification.code.scope"]
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    assert assessed.status == "supported"
    assert hashlib.sha256(Path(materials.repository.root, "config.py").read_bytes()).hexdigest() == before


@pytest.mark.parametrize("facet", ["empirical_outcome", "novelty", "availability", "uncertain"])
def test_joint_requirements_cannot_be_fully_established_by_an_implementation_line(inputs, facet):
    claim, materials, item = inputs()
    result, assessed, _ = run(
        claim, materials, [item], review(claim, [item], facets=["implementation", facet])
    )
    assert not any(e.sufficient for e in result.evidence)
    assert assessed.status == "unverified"


@pytest.mark.parametrize("relation", ["partial", "irrelevant", "uncertain"])
def test_unconfirmed_flaw_does_not_question_the_claim(inputs, relation):
    claim, materials, item = inputs("A test MRR is 0.4.", optimizer="SGD")
    item["direction"] = "flaw"
    result, assessed, _ = run(
        claim, materials, [item], review(claim, [item], facets=["empirical_outcome"], relation=relation)
    )
    assert result.evidence and all(not e.concern and not e.sufficient for e in result.evidence)
    assert assessed.status == "unverified"


def test_confirmed_specific_implementation_mismatch_remains_questioned(inputs):
    claim, materials, item = inputs(optimizer="SGD")
    item["direction"] = "flaw"
    result, assessed, _ = run(
        claim, materials, [item], review(claim, [item], relation="contradicts_implementation")
    )
    assert result.evidence[0].sufficient and result.evidence[0].concern
    assert assessed.status == "questioned"


def test_failed_scope_retains_exact_candidates_without_deciding_flaw(inputs, tmp_path):
    claim, materials, item = inputs(optimizer="SGD")
    item["direction"] = "flaw"
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result, assessed, _ = run(claim, materials, [item], RuntimeError("simulated timeout"))
    assert result.evidence and result.evidence[0].pointer.quote == item["quote"]
    assert assessed.status == "unverified" and result.issues
    assert list((tmp_path / "code_scope_reviews").glob("*.json"))


def test_bad_first_pass_pointer_is_rejected_before_scope_call(inputs):
    claim, materials, item = inputs()
    item["quote"] = "fabricated code"
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        return {"items": [item]}

    with pytest.raises(ValueError, match="indexed source lines"):
        verify_code(claim, materials, call=model)
    assert calls == ["verification.code"]


def test_invalid_bridge_quote_does_not_create_support(inputs):
    claim, materials, item = inputs()
    decision = review(claim, [item])
    decision["items"][0]["bridge_quotes"] = [{"block_id": "methods", "quote": "invented bridge"}]
    result, assessed, _ = run(claim, materials, [item], decision)
    assert result.issues and assessed.status == "unverified"


def test_invalid_one_condition_keeps_other_verified_observation(inputs):
    claim, materials, item = inputs()
    claim.conditions.append(Condition(id="c2", description="A second optimizer setting"))
    second = {**copy.deepcopy(item), "covered": ["c2"], "fully_supported_conditions": ["c2"]}
    decision = review(claim, [item, second])
    decision["conditions"][1]["claim_source_ids"] = ["foreign"]
    result, assessed, _ = run(claim, materials, [item, second], decision)
    assert {c for e in result.evidence if e.sufficient for c in e.covered} == {"c1"}
    assert assessed.status == "unverified" and result.issues


def test_narrow_source_reference_cannot_be_expanded_to_other_condition(inputs):
    claim, materials, item = inputs()
    claim.conditions.append(Condition(id="c2", description="Another setting"))
    claim.source_refs = [
        ClaimSourceRef(
            source_block_id=claim.source_block_id,
            source_quote=claim.source_quote,
            loc=claim.loc,
            covered=["c1"],
        )
    ]
    item["covered"] = item["fully_supported_conditions"] = ["c2"]
    result, assessed, _ = run(claim, materials, [item], review(claim, [item]))
    assert not any(e.sufficient for e in result.evidence)
    assert assessed.status == "unverified"


@pytest.mark.parametrize("collection", ["conditions", "items"])
@pytest.mark.parametrize("malformed", [False, True])
def test_duplicate_scope_rows_cannot_keep_the_first_positive_decision(inputs, collection, malformed):
    claim, materials, item = inputs()
    decision = review(claim, [item])
    duplicate = copy.deepcopy(decision[collection][0])
    if malformed:
        duplicate["rationale"] = ""
    decision[collection].append(duplicate)
    result, assessed, _ = run(claim, materials, [item], decision)
    assert not any(e.sufficient for e in result.evidence) and result.issues
    assert assessed.status == "unverified"


def test_one_multi_condition_candidate_retains_only_the_independently_confirmed_subset(inputs):
    claim, materials, item = inputs()
    claim.conditions.append(Condition(id="c2", description="Measured outcome"))
    item["covered"] = item["fully_supported_conditions"] = ["c1", "c2"]
    decision = review(claim, [item])
    decision["conditions"][1]["required_facets"] = ["empirical_outcome"]
    result, assessed, _ = run(claim, materials, [item], decision)
    assert result.evidence[0].covered == ["c1", "c2"] and not result.evidence[0].sufficient
    assert [e.covered for e in result.evidence if e.sufficient] == [["c1"]]
    assert assessed.status == "unverified"


@pytest.mark.parametrize("basis", ["absence", "uncertain"])
def test_incomplete_source_cannot_establish_absence_or_an_uncertain_mismatch(inputs, basis):
    claim, materials, item = inputs(optimizer="SGD")
    item["direction"] = "flaw"
    decision = review(claim, [item], relation="contradicts_implementation")
    decision["items"][0]["basis"] = basis
    result, assessed, _ = run(claim, materials, [item], decision)
    assert all(not e.concern and not e.sufficient for e in result.evidence)
    assert assessed.status == "unverified"


def test_wrong_or_unresolved_runtime_setting_cannot_decide_implementation_flaw(inputs):
    claim, materials, item = inputs(optimizer="SGD")
    item["direction"] = "flaw"
    decision = review(claim, [item], relation="contradicts_implementation")
    decision["items"][0]["missing_qualifiers"] = ["The selected setting may use a different config branch."]
    result, assessed, _ = run(claim, materials, [item], decision)
    assert not result.evidence[0].concern and assessed.status == "unverified"


def test_exact_additional_paper_bridge_is_kept_in_evidence_and_audit(inputs, tmp_path):
    claim, materials, item = inputs()
    decision = review(claim, [item])
    decision["items"][0]["bridge_quotes"] = [{"block_id": "methods", "quote": item["paper_quote"]}]
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result, assessed, _ = run(claim, materials, [item], decision)
    assert assessed.status == "supported" and "bridge " in result.evidence[0].note
    audit = json.loads(next((tmp_path / "code_scope_reviews").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["request"]["claim_sources"][0]["quote"] == claim.source_quote
    assert audit["response"] == decision and audit["validated_items"][0]["bridge_quotes"]
    assert audit["provider"] == "mock" and audit["model"] == "scope"


def test_scope_validation_echoes_are_redacted_in_audit_and_report(inputs, tmp_path, monkeypatch):
    claim, materials, item = inputs()
    cfg = LLMConfig(
        "mock", "scope", "https://u:p@example.test:8443/v1?key=secret-query", "fake-provider-secret"
    )
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    decision = review(claim, [item])
    decision["items"][0]["relation"] = cfg.api_key + " " + cfg.base_url
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result, assessed, _ = run(claim, materials, [item], decision)
    saved = next((tmp_path / "code_scope_reviews").glob("*.json")).read_text(encoding="utf-8")
    report = result.model_dump_json()
    for secret in ["fake-provider-secret", "secret-query", "u:p@"]:
        assert secret not in saved and secret not in report
    assert "example.test:8443/v1" in saved and assessed.status == "unverified"


def test_source_mutated_during_scope_cannot_leave_a_stale_sufficient_pointer(inputs):
    claim, materials, item = inputs()

    def model(**kwargs):
        if kwargs["module"] == "verification.code.scope":
            Path(materials.repository.root, "config.py").write_text("optimizer = 'SGD'\n", encoding="utf-8")
            return review(claim, [item])
        return {"items": [item]}

    with pytest.raises(ValueError, match="changed during verification"):
        verify_code(claim, materials, call=model)


@pytest.mark.parametrize("collection", ["conditions", "items"])
@pytest.mark.parametrize("padding", [" ", "\t"])
@pytest.mark.parametrize("later_positive", [False, True])
def test_padded_duplicate_scope_id_revokes_original_and_preserves_other_condition(
    inputs, collection, padding, later_positive
):
    claim, materials, item = inputs()
    claim.conditions.append(Condition(id="c2", description="A second optimizer setting"))
    second = {**copy.deepcopy(item), "covered": ["c2"], "fully_supported_conditions": ["c2"]}
    decision = review(claim, [item, second])
    duplicate = copy.deepcopy(decision[collection][0])
    duplicate["condition_id"] = padding + "c1" + padding
    if collection == "conditions":
        duplicate["required_facets"] = ["uncertain"]
    else:
        decision[collection][0]["relation"] = "uncertain"
        decision[collection][0]["full_condition"] = False
    decision[collection].append(duplicate)
    if later_positive:
        # A later exact positive must not restore a pair already invalidated by duplication.
        decision[collection].append(copy.deepcopy(review(claim, [item, second])[collection][0]))
    result, assessed, _ = run(claim, materials, [item, second], decision)
    assert {cid for evidence in result.evidence if evidence.sufficient for cid in evidence.covered} == {"c2"}
    assert assessed.status == "unverified" and result.issues


@pytest.mark.parametrize("field", ["scope_rationale", "detail", "issues"])
def test_successful_response_credentials_are_redacted_in_report(inputs, tmp_path, monkeypatch, field):
    claim, materials, item = inputs()
    cfg = LLMConfig(
        "mock", "scope", "https://u:p@example.test:8443/v1?key=secret-query", "fake-provider-secret"
    )
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    decision = review(claim, [item])
    diagnostics = f" Diagnostics: {cfg.api_key}; {cfg.base_url}"
    response = {"items": [item]}
    if field == "scope_rationale":
        decision["items"][0]["rationale"] += diagnostics
    elif field == "detail":
        item["detail"] += diagnostics
    else:
        response["issues"] = [diagnostics]

    def model(**kwargs):
        return decision if kwargs["module"] == "verification.code.scope" else response

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result = verify_code(claim, materials, call=model)
    claim.evidence = result.evidence
    saved = next((tmp_path / "code_scope_reviews").glob("*.json")).read_text(encoding="utf-8")
    report = result.model_dump_json()
    assert assess_claim(claim).status == "supported"
    for secret in ["fake-provider-secret", "secret-query", "u:p@"]:
        assert secret not in saved and secret not in report
    assert "example.test:8443/v1" in report
    assert result.evidence[0].pointer.quote == item["quote"]
    assert item["paper_quote"] in result.evidence[0].note


@pytest.mark.parametrize("field", ["condition_id", "item_condition_id", "claim_source_id", "bridge_block_id"])
def test_scope_identifiers_must_match_supplied_identifiers_exactly(inputs, field):
    claim, materials, item = inputs()
    decision = review(claim, [item])
    if field == "condition_id":
        decision["conditions"][0]["condition_id"] = " c1 "
    elif field == "item_condition_id":
        decision["items"][0]["condition_id"] = " c1 "
    elif field == "claim_source_id":
        decision["conditions"][0]["claim_source_ids"] = [" primary "]
    else:
        decision["items"][0]["bridge_quotes"] = [{"block_id": " methods ", "quote": item["paper_quote"]}]
    result, assessed, _ = run(claim, materials, [item], decision)
    assert result.issues and not any(evidence.sufficient for evidence in result.evidence)
    assert assessed.status == "unverified"


def test_first_pass_schema_errors_do_not_echo_provider_credentials(inputs, monkeypatch):
    claim, materials, item = inputs()
    cfg = LLMConfig("mock", "scope", "https://u:p@example.test/v1?key=secret-query", "fake-provider-secret")
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    item["aspect"] = cfg.api_key + " " + cfg.base_url
    with pytest.raises(ValueError) as raised:
        verify_code(claim, materials, call=lambda **kwargs: {"items": [item]})
    for secret in ["fake-provider-secret", "secret-query", "u:p@"]:
        assert secret not in str(raised.value)
