"""Offline branch contracts keep evidence tied to actual manuscript/repository bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, PageImage, RepositoryFile, RepositoryIndex, SharedMaterials
from screening import checks
from verification.code import verify_code
from verification.experiments import verify_experiments
from verification.theory import verify_theory

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


@pytest.fixture(autouse=True)
def mock_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr(checks, "llm_json", lambda **kwargs: pytest.fail("Unmocked LLM call"))


@pytest.fixture
def claim() -> Claim:
    return Claim(
        id="c1",
        text="A test MRR is 0.4.",
        loc=ClaimLocation(page=2),
        conditions=[Condition(id="a", dataset="A", metric="MRR", settings={"split": "test"})],
        needs=["Theory", "Code", "Experiments"],
        importance="core",
    )


@pytest.fixture
def materials(tmp_path: Path) -> SharedMaterials:
    main = "Theorem 1: the method converges. A test MRR is 0.4. We use Adam."
    appendix = "Proof of Theorem 1. By the stated assumption, x = y; therefore convergence follows."
    (tmp_path / "paper.md").write_text(main + appendix, encoding="utf-8")
    result = SharedMaterials(
        paper_key="tiny",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=main + appendix,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="content.json",
        provider="fixture",
        blocks=[
            MaterialBlock(id="main", text=main, loc=ClaimLocation(page=2, section="Results")),
            MaterialBlock(id="appendix", text=appendix, loc=ClaimLocation(page=8, section="Appendix A")),
        ],
    )
    root = tmp_path / "repo"
    root.mkdir()
    files = {
        "eval.py": ("entry", "optimizer = 'Adam'\nprint('MRR')\n"),
        "config.json": ("config", '{"dataset": "A"}'),
        "data.csv": ("asset", "1,2\n"),
        "weights.pt": ("asset", "weights"),
    }
    rows = []
    for name, (kind, content) in files.items():
        data = content.encode()
        (root / name).write_bytes(data)
        rows.append(RepositoryFile(path=name, kind=kind, sha256=hashlib.sha256(data).hexdigest()))
    result.repository = RepositoryIndex(
        root=str(root), files=rows, entry_scripts=["eval.py"], configs=["config.json"]
    )
    return result


@pytest.fixture
def theory_claim(materials: SharedMaterials) -> Claim:
    return Claim(
        id="theorem_1",
        text="The method converges under the stated assumption.",
        loc=materials.blocks[0].loc,
        source_block_id="main",
        source_quote="Theorem 1: the method converges.",
        conditions=[Condition(id="a", description="Convergence under the stated assumption")],
        needs=["Theory"],
        importance="core",
    )


@pytest.fixture
def code_claim(materials: SharedMaterials) -> Claim:
    return Claim(
        id="optimizer_claim",
        text="The method uses Adam.",
        loc=materials.blocks[0].loc,
        source_block_id="main",
        source_quote="We use Adam.",
        conditions=[Condition(id="a", description="The optimizer is Adam")],
        needs=["Code"],
        importance="core",
    )


def code_scope_response(kwargs):
    data = json.loads(kwargs["prompt"])
    return {
        "conditions": [
            {
                "condition_id": condition["id"],
                "required_facets": ["implementation"]
                if not condition.get("metric")
                else ["empirical_outcome"],
                "claim_source_ids": ["primary"],
                "rationale": "The fixture's exact source identifies the implementation requirement.",
            }
            for condition in data["claim"]["conditions"]
        ],
        "items": [
            {
                "item_index": index,
                "condition_id": cid,
                "relation": "supports_implementation",
                "full_condition": cid in item["fully_supported_conditions"],
                "basis": "direct_source",
                "bridge_quotes": [],
                "missing_qualifiers": [],
                "rationale": "The fixture's actual implementation agrees with its stated optimizer.",
            }
            for index, item in enumerate(data["candidate_items"])
            for cid in item["covered"]
        ],
    }


def scope_response(kwargs):
    data = json.loads(kwargs["prompt"])
    # This shared fixture describes an absolute reported result. The dedicated
    # isolation fixture explicitly attributes that result to a component.
    causal = data["claim"]["text"] == "The attention component causes the improvement in A test MRR."
    return {
        "conditions": [
            {
                "condition_id": condition["id"],
                "claim_quote": data["claim"]["text"],
                "assertion": "causal_attribution" if causal else "descriptive",
                "matched_controls_required": causal,
                "credited_component": "attention component" if causal else "",
                "uncertainty_sensitive": causal,
                "relation": "gt" if causal else "none",
                "subject": "",
                "comparator": "",
                "rationale": "The fixture states component attribution."
                if causal
                else "The fixture reports an absolute value.",
            }
            for condition in data["claim"]["conditions"]
        ],
        "items": [
            {
                "item_index": index,
                "condition_id": condition,
                "applicability": "applicable",
                "grounds": [{"block_id": item["block_id"], "quote": item["quote"]}],
                "rationale": "The exact fixture passage covers this stated condition.",
                "full_support": True,
                "qualifiers_complete": True,
                "comparison_objects": "not_comparative",
            }
            for index, item in enumerate(data["candidate_items"])
            for condition in item["covered"]
        ],
    }


def mock_response(payload: dict):
    return lambda **kwargs: (
        scope_response(kwargs)
        if kwargs["module"] == "verification.experiments.scope"
        else code_scope_response(kwargs)
        if kwargs["module"] == "verification.code.scope"
        else payload
    )


def theory_item(**changes) -> dict:
    return {
        "block_id": "appendix",
        "quote": "By the stated assumption, x = y; therefore convergence follows.",
        "covered": ["a"],
        "fully_supported_conditions": ["a"],
        "kind": "derivation",
        "direction": "support",
        "detail": "The derivation uses the assumption.",
        "step_quote": "x = y",
        "main_block_id": "main",
        **changes,
    }


def code_item(**changes) -> dict:
    return {
        "file": "eval.py",
        "line": 1,
        "quote": "optimizer = 'Adam'",
        "paper_block_id": "main",
        "paper_quote": "We use Adam.",
        "covered": ["a"],
        "fully_supported_conditions": ["a"],
        "direction": "support",
        "aspect": "optimizer",
        "detail": "The configured optimizer matches the paper.",
        **changes,
    }


def experiment_item(**changes) -> dict:
    return {
        "aspect": "correspondence",
        "kind": "paper_support",
        "block_id": "main",
        "quote": "A test MRR is 0.4.",
        "covered": ["a"],
        "fully_supported_conditions": ["a"],
        "detail": "The reported experiment covers A.",
        **changes,
    }


def plan_candidate(**changes) -> dict:
    return {
        "targets": [
            {
                "condition_id": "a",
                "reported": {"block_id": "main", "quote": "A test MRR is 0.4.", "token": "0.4"},
            }
        ],
        "entry_script": "eval.py",
        "config": "config.json",
        "run_mode": "evaluation",
        "feasibility": "ready",
        "priority": "high",
        "data_paths": ["data.csv"],
        "weight_paths": ["weights.pt"],
        **changes,
    }


def experiments_response(**changes) -> dict:
    return {"checked_aspects": ASPECTS, "items": [], "plans": [], **changes}


def test_theory_reads_main_before_requesting_relevant_appendix(
    theory_claim: Claim, materials: SharedMaterials
) -> None:
    seen = []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        seen.append(payload)
        if len(seen) == 1:
            assert [b["id"] for b in payload["main_text"]] == ["main"]
            assert "appendix_proofs" not in payload
            assert "output_schema" in payload
            return {"items": [], "appendix_block_ids": ["appendix"]}
        assert [b["id"] for b in payload["appendix_proofs"]] == ["appendix"]
        return {"items": [theory_item()]}

    result = verify_theory(theory_claim, materials, call=model)
    assert len(seen) == 2 and len(result.evidence) == 1
    assert result.evidence[0].source == "theory" and result.evidence[0].sufficient
    assert result.evidence[0].pointer.page == 8
    assert result.plans == []


def test_theory_missing_proof_yields_question_without_support(
    theory_claim: Claim, materials: SharedMaterials
) -> None:
    result = verify_theory(
        theory_claim,
        materials,
        call=mock_response(
            {
                "items": [
                    theory_item(
                        block_id="main",
                        quote="Theorem 1: the method converges.",
                        kind="no_proof",
                        direction="flaw",
                        step_quote="",
                        main_block_id=None,
                        detail="The main assertion supplies no proof.",
                    )
                ]
            }
        ),
    )
    assert result.evidence == [] and not result.questions and result.issues
    assert len(result.verification_limitations) == 1
    assert result.verification_limitations[0].kind == "source_context_unavailable"
    assert result.verification_limitations[0].condition_ids == ["a"]
    assert result.verification_limitations[0].responsibility == "system"


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [theory_item()]},
        {"items": [], "appendix_block_ids": ["nonexistent"]},
        {"items": [theory_item(block_id="main", quote="Fabricated derivation")]},
        {"items": [theory_item(block_id="main", quote="We use Adam.", step_quote="not in quote")]},
    ],
)
def test_theory_rejects_fabricated_or_unloaded_references(
    theory_claim: Claim, materials: SharedMaterials, payload: dict
) -> None:
    with pytest.raises(ValueError):
        verify_theory(theory_claim, materials, call=mock_response(payload))


def test_theory_theorem_assertion_alone_cannot_be_support(
    theory_claim: Claim, materials: SharedMaterials
) -> None:
    quote = "Theorem 1: the method converges."
    result = verify_theory(
        theory_claim,
        materials,
        call=mock_response(
            {"items": [theory_item(block_id="main", quote=quote, step_quote=quote, main_block_id=None)]}
        ),
    )
    assert result.evidence == [] and result.issues


def test_code_verifies_index_hash_and_exact_line_plus_paper_quote(
    code_claim: Claim, materials: SharedMaterials
) -> None:
    result = verify_code(code_claim, materials, call=mock_response({"items": [code_item()]}))
    evidence = result.evidence[0]
    assert evidence.source == "code" and evidence.pointer.locator == str(
        Path(materials.repository.root) / "eval.py"
    )
    assert evidence.pointer.line == 1 and evidence.sufficient
    assert "We use Adam." in evidence.note
    assert result.plans == []


@pytest.mark.parametrize(
    "changes",
    [
        {"file": "../outside.py"},
        {"line": 2},
        {"quote": "optimizer = 'SGD'"},
        {"paper_quote": "We use SGD."},
        {"covered": ["other"]},
        {"sufficient": True},
    ],
)
def test_code_rejects_unverified_model_evidence(
    claim: Claim, materials: SharedMaterials, changes: dict
) -> None:
    with pytest.raises(ValueError):
        verify_code(claim, materials, call=mock_response({"items": [code_item(**changes)]}))


def test_code_rejects_modified_indexed_files_before_llm(claim: Claim, materials: SharedMaterials) -> None:
    (Path(materials.repository.root) / "eval.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        verify_code(claim, materials, call=lambda **kwargs: pytest.fail("Should fail before LLM"))


def test_code_missing_repo_preserves_blocker_without_evidence(
    claim: Claim, materials: SharedMaterials
) -> None:
    materials.repository = None
    result = verify_code(claim, materials)
    assert result.evidence == [] and result.issues and result.questions


def test_code_preserves_indentation_in_exact_source_quote(
    code_claim: Claim, materials: SharedMaterials
) -> None:
    text = "def train():\n    optimizer = 'Adam'\n"
    path = Path(materials.repository.root) / "eval.py"
    path.write_bytes(text.encode())
    materials.repository.files[0].sha256 = hashlib.sha256(text.encode()).hexdigest()
    result = verify_code(
        code_claim,
        materials,
        call=mock_response({"items": [code_item(line=2, quote="    optimizer = 'Adam'")]}),
    )
    assert result.evidence[0].pointer.quote.startswith("    ")


def test_experiments_paper_support_and_claim_linked_plan(claim: Claim, materials: SharedMaterials) -> None:
    def model(**kwargs):
        if kwargs["module"] == "verification.experiments.scope":
            return scope_response(kwargs)
        assert all(aspect in kwargs["system"] for aspect in ASPECTS)
        assert "output_schema" in json.loads(kwargs["prompt"])
        return experiments_response(items=[experiment_item()], plans=[plan_candidate()])

    result = verify_experiments(claim, materials, call=model)
    assert result.evidence[0].source == "paper_internal" and result.evidence[0].sufficient
    plan = result.plans[0]
    assert plan.claim_id == claim.id and plan.priority == "high" and plan.feasibility == "ready"
    assert plan.target_conditions == claim.conditions and plan.y_paper == {"a": 0.4}
    assert plan.task.command == []


def test_experiments_large_gap_without_variance_is_non_decisive_note(
    claim: Claim, materials: SharedMaterials
) -> None:
    result = verify_experiments(
        claim,
        materials,
        call=mock_response(
            experiments_response(
                items=[
                    experiment_item(
                        aspect="stability",
                        kind="large_gap_no_variance",
                        detail="Variance not reported for a large gap.",
                    )
                ]
            )
        ),
    )
    evidence = result.evidence[0]
    assert not evidence.affects_claim and not evidence.concern and not evidence.sufficient
    assert result.questions == []


def test_experiments_missing_ablation_is_concrete_author_question(
    claim: Claim, materials: SharedMaterials
) -> None:
    claim.text = "The attention component causes the improvement in A test MRR."
    materials.blocks[0].text += " " + claim.text
    materials.markdown = materials.blocks[0].text + materials.blocks[1].text
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    result = verify_experiments(
        claim,
        materials,
        call=mock_response(
            experiments_response(
                items=[
                    experiment_item(
                        aspect="isolation",
                        kind="missing_ablation",
                        quote=claim.text,
                        detail="No isolated component experiment is reported.",
                    )
                ]
            )
        ),
    )
    assert result.evidence[0].concern and result.evidence[0].overturnable
    assert result.questions[0].claim_id == claim.id


def contradiction_response(materials, left, right, left_token, right_token, *, support=False):
    for block in materials.blocks:
        start = materials.markdown.index(block.text)
        block.loc.char_start = start
        block.loc.char_end = start + len(block.text)
    for block_id, text in (("comparison_left", left), ("comparison_right", right)):
        start = len(materials.markdown) + 1
        materials.blocks.append(
            MaterialBlock(
                id=block_id,
                text=text,
                loc=ClaimLocation(page=3, char_start=start, char_end=start + len(text)),
            )
        )
        materials.markdown += "\n" + text
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    return experiments_response(
        items=([experiment_item()] if support else [])
        + [
            experiment_item(
                aspect="consistency",
                kind="text_table_contradiction",
                block_id="comparison_left",
                quote=left,
                fully_supported_conditions=[],
                detail="The reported values differ.",
                comparison=[
                    {"block_id": "comparison_left", "quote": left, "token": left_token},
                    {"block_id": "comparison_right", "quote": right, "token": right_token},
                ],
            )
        ]
    )


@pytest.mark.parametrize(
    ("left", "right", "left_token", "right_token", "metric", "settings"),
    [
        ("A test MRR is 0.4.", "A dev MRR is 0.5.", "0.4", "0.5", "MRR", {"split": "test"}),
        (
            "A test seed=1 MRR is 0.4.",
            "A test seed=2 MRR is 0.5.",
            "0.4",
            "0.5",
            "MRR",
            {"split": "test", "seed": 1},
        ),
        (
            "A test MRR is 0.4, baseline MRR is 0.5.",
            "A test MRR is 0.5.",
            "0.4",
            "0.5",
            "MRR",
            {"split": "test"},
        ),
        ("A test MRR is 40%.", "A test MRR is 0.4.", "40%", "0.4", "MRR", {"split": "test"}),
        ("A test MRR (%) is 40.", "A test MRR is 0.4.", "40", "0.4", "MRR", {"split": "test"}),
        (
            "A test latency is 100 ms.",
            "A test latency is 0.1 seconds.",
            "100",
            "0.1",
            "latency",
            {"split": "test"},
        ),
    ],
)
def test_unbound_paper_comparisons_cannot_question_claim(
    claim, materials, left, right, left_token, right_token, metric, settings
):
    from assessment import assess_claim

    claim.conditions[0].metric = metric
    claim.conditions[0].settings = settings
    response = contradiction_response(materials, left, right, left_token, right_token)
    result = verify_experiments(claim, materials, call=mock_response(response))
    assert result.evidence == [] and result.questions == []
    assert any("contradiction unconfirmed" in issue for issue in result.issues)
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "unverified"


def test_invalid_comparison_preserves_independent_paper_support(claim, materials):
    from assessment import assess_claim

    response = contradiction_response(
        materials, "A test MRR is 0.4.", "A dev MRR is 0.5.", "0.4", "0.5", support=True
    )
    result = verify_experiments(claim, materials, call=mock_response(response))
    assert len(result.evidence) == 1 and result.evidence[0].direction == "support"
    assert result.issues and not result.questions
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "supported"


@pytest.mark.parametrize(
    ("left", "right", "left_token", "right_token", "metric", "settings"),
    [
        ("A test MRR is 0.4.", "A test MRR is 0.5.", "0.4", "0.5", "MRR", {"split": "test"}),
        (
            "A test seed=1 MRR is 0.4.",
            "A test seed=1 MRR is 0.5.",
            "0.4",
            "0.5",
            "MRR",
            {"split": "test", "seed": 1},
        ),
        ("A test MRR is 40%.", "A test MRR is 50%.", "40%", "50%", "MRR", {"split": "test"}),
        (
            "A test latency is 100 seconds.",
            "A test latency is 200 s.",
            "100",
            "200",
            "latency",
            {"split": "test"},
        ),
    ],
)
def test_same_setting_and_scale_paper_disagreement_remains_questioned(
    claim, materials, left, right, left_token, right_token, metric, settings
):
    from assessment import assess_claim

    claim.conditions[0].metric = metric
    claim.conditions[0].settings = settings
    response = contradiction_response(materials, left, right, left_token, right_token)
    result = verify_experiments(claim, materials, call=mock_response(response))
    assert len(result.evidence) == 1 and not result.issues
    evidence = result.evidence[0]
    assert evidence.sufficient and evidence.concern and evidence.overturnable
    assert left in evidence.note and right in evidence.note
    assert result.questions
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "questioned"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Latency (ms)\nA test latency is 100.", "Latency (s)\nA test latency is 0.1."),
        ("A test latency is 100 ms.", "A test latency is 0.1 seconds."),
    ],
)
def test_comparison_value_context_cannot_discard_units(claim, materials, left, right):
    claim.conditions[0].metric = "latency"
    response = contradiction_response(materials, left, right, "100", "0.1")
    comparison = response["items"][0]["comparison"]
    comparison[0]["value_context"] = "A test latency is 100"
    comparison[1]["value_context"] = "A test latency is 0.1"
    result = verify_experiments(claim, materials, call=mock_response(response))
    assert result.evidence == [] and result.questions == []
    assert any("units/scales" in issue for issue in result.issues)


def test_explicit_value_context_disambiguates_real_paper_disagreement(claim, materials):
    left = "A test MRR is 0.4. B test MRR is 0.8."
    right = "A test MRR is 0.5. B test MRR is 0.9."
    response = contradiction_response(materials, left, right, "0.4", "0.5")
    comparison = response["items"][0]["comparison"]
    comparison[0]["value_context"] = "A test MRR is 0.4."
    comparison[1]["value_context"] = "A test MRR is 0.5."
    result = verify_experiments(claim, materials, call=mock_response(response))
    assert len(result.evidence) == 1 and result.evidence[0].concern
    assert result.questions and not result.issues


@pytest.mark.parametrize(
    "changes",
    [
        {"entry_script": "unknown.py"},
        {"config": "unknown.yml"},
        {"data_paths": ["unknown.csv"]},
        {
            "targets": [
                {
                    "condition_id": "other",
                    "reported": {"block_id": "main", "quote": "A test MRR is 0.4.", "token": "0.4"},
                }
            ]
        },
        {
            "targets": [
                {
                    "condition_id": "a",
                    "reported": {"block_id": "main", "quote": "A test MRR is 0.4.", "token": "0.9"},
                }
            ]
        },
        {
            "targets": [
                {
                    "condition_id": "a",
                    "reported": {"block_id": "main", "quote": "A test MRR is 0.4.", "token": "nan"},
                }
            ]
        },
    ],
)
def test_plan_rejects_fabricated_repository_or_paper_targets(
    claim: Claim, materials: SharedMaterials, changes: dict
) -> None:
    with pytest.raises(ValueError):
        verify_experiments(
            claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(**changes)]))
        )


def test_missing_resources_keep_a_blocked_claim_linked_plan(claim: Claim, materials: SharedMaterials) -> None:
    materials.repository = None
    result = verify_experiments(
        claim,
        materials,
        call=mock_response(
            experiments_response(
                plans=[
                    plan_candidate(
                        entry_script=None,
                        config=None,
                        data_paths=[],
                        weight_paths=[],
                    )
                ]
            )
        ),
    )
    plan = result.plans[0]
    assert plan.feasibility == "blocked" and plan.claim_id == claim.id
    assert "code" in plan.blocker and "data" in plan.blocker and "weights" in plan.blocker


def test_experiments_reject_incomplete_five_aspect_check(claim: Claim, materials: SharedMaterials) -> None:
    with pytest.raises(ValueError, match="all five"):
        verify_experiments(
            claim, materials, call=mock_response(experiments_response(checked_aspects=["consistency"]))
        )


def test_plan_cannot_match_dataset_as_substring_of_another_word(
    claim: Claim, materials: SharedMaterials
) -> None:
    quote = "Data: B MRR is 0.4."
    materials.blocks[0].text += " " + quote
    materials.markdown += " " + quote
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    target = {"condition_id": "a", "reported": {"block_id": "main", "quote": quote, "token": "0.4"}}
    with pytest.raises(ValueError, match="dataset"):
        verify_experiments(
            claim,
            materials,
            call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])])),
        )


def test_parser_only_quote_uses_existing_pdf_page(claim: Claim, materials: SharedMaterials) -> None:
    # MinerU content-list text can be absent from its generated Markdown.
    materials.markdown = "Markdown omitted this parsed table."
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    with fitz.open() as pdf:
        pdf.new_page()
        page = pdf.new_page()
        page.insert_text((40, 50), materials.blocks[0].text)
        pdf.save(materials.source_pdf)
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(items=[experiment_item()]))
    )
    pointer = result.evidence[0].pointer
    assert pointer.locator == str(Path(materials.source_pdf).resolve())
    assert pointer.page == 2 and pointer.key is None
    with fitz.open(pointer.locator) as pdf:
        assert pointer.quote in pdf[pointer.page - 1].get_text()


@pytest.mark.parametrize("artifact_state", ["missing", "changed", "pdf_page_out_of_range"])
def test_missing_or_mismatched_paper_artifacts_cannot_mint_evidence(
    claim: Claim,
    materials: SharedMaterials,
    artifact_state: str,
) -> None:
    if artifact_state == "missing":
        Path(materials.markdown_path).unlink()
    else:
        Path(materials.markdown_path).write_text("The expected quote is absent.", encoding="utf-8")
        if artifact_state == "pdf_page_out_of_range":
            with fitz.open() as pdf:
                pdf.new_page()
                pdf.save(materials.source_pdf)
    with pytest.raises(ValueError, match="existing artifact"):
        verify_experiments(
            claim, materials, call=mock_response(experiments_response(items=[experiment_item()]))
        )


def test_multivalue_table_cannot_assign_another_dataset_number(
    claim: Claim, materials: SharedMaterials
) -> None:
    quote = "Dataset test MRR\nA 0.4\nB 0.6"
    materials.blocks[0].text = quote
    materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    target = {"condition_id": "a", "reported": {"block_id": "main", "quote": quote, "token": "0.6"}}
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])]))
    )
    assert result.plans[0].feasibility == "blocked"
    assert "Unresolved paper target a" in result.plans[0].blocker
    assert "uniquely bind" in result.plans[0].blocker


def test_exact_single_value_context_can_disambiguate_multivalue_quote(
    claim: Claim, materials: SharedMaterials
) -> None:
    quote = "A test MRR is 0.4. B test MRR is 0.6."
    materials.blocks[0].text = quote
    materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    target = {
        "condition_id": "a",
        "reported": {
            "block_id": "main",
            "quote": quote,
            "token": "0.4",
            "value_context": "A test MRR is 0.4.",
        },
    }
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])]))
    )
    assert result.plans[0].feasibility == "ready"
    assert result.plans[0].y_paper == {"a": 0.4}
    target["reported"]["token"] = "0.6"
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])]))
    )
    assert result.plans[0].feasibility == "blocked"


def test_target_context_must_identify_claimed_setting(claim: Claim, materials: SharedMaterials) -> None:
    quote = "A train MRR is 0.4."
    materials.blocks[0].text = quote
    materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    target = {"condition_id": "a", "reported": {"block_id": "main", "quote": quote, "token": "0.4"}}
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])]))
    )
    assert result.plans[0].feasibility == "blocked"
    assert "split=test" in result.plans[0].blocker


def test_numeric_setting_does_not_make_a_correct_target_ambiguous(claim, materials):
    quote = "A test seed 42 MRR 0.4"
    claim.conditions[0].settings["seed"] = 42
    materials.blocks[0].text = materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    target = {"condition_id": "a", "reported": {"block_id": "main", "quote": quote, "token": "0.4"}}
    result = verify_experiments(
        claim, materials, call=mock_response(experiments_response(plans=[plan_candidate(targets=[target])]))
    )
    assert result.plans[0].feasibility == "ready"
    assert result.plans[0].y_paper == {"a": 0.4}


def test_repeated_quote_uses_current_block_span_then_actual_pdf_page(materials: SharedMaterials) -> None:
    quote = "Same formula x = y."
    text = quote + "\n" + quote
    materials.markdown = text
    Path(materials.markdown_path).write_text(text, encoding="utf-8")
    block = MaterialBlock(
        id="second",
        text=quote,
        loc=ClaimLocation(
            page=2,
            char_start=len(quote) + 1,
            char_end=len(text),
        ),
    )
    pointer = checks.grounded_paper_pointer(materials, block, quote)
    assert pointer.key == f"chars:{len(quote) + 1}-{len(text)}"
    assert pointer.page == 2
    block.loc = ClaimLocation(page=2)
    with fitz.open() as pdf:
        pdf.new_page().insert_text((40, 50), quote)
        pdf.new_page().insert_text((40, 50), quote)
        pdf.save(materials.source_pdf)
    pointer = checks.grounded_paper_pointer(materials, block, quote)
    assert pointer.locator == str(Path(materials.source_pdf).resolve())
    assert pointer.page == 2 and pointer.key is None


@pytest.mark.parametrize("branch", [verify_theory, verify_code, verify_experiments])
def test_branch_llm_failure_is_visible(branch, claim: Claim, materials: SharedMaterials) -> None:
    with pytest.raises(RuntimeError, match="model request failed"):
        branch(claim, materials, call=mock_response({"status": "error", "error": "offline"}))


@pytest.mark.parametrize(
    ("branch", "item_schema", "response"),
    [
        (verify_code, "CodeItem", {"items": [], "issues": ["No condition-grounded evidence."]}),
        (verify_theory, "TheoryItem", {"items": []}),
        (
            verify_experiments,
            "ExperimentItem",
            experiments_response(issues=["No condition-grounded evidence."]),
        ),
    ],
)
def test_branch_request_exposes_exact_condition_ids_and_quote_contract(
    branch, item_schema, response, claim, materials
):
    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        assert payload["allowed_condition_ids"] == [condition.id for condition in claim.conditions] == ["a"]
        assert claim.id not in payload["allowed_condition_ids"]
        covered = payload["output_schema"]["$defs"][item_schema]["properties"]["covered"]
        assert covered["minItems"] == 1 and covered["uniqueItems"] is True
        assert "allowed_condition_ids" in covered["description"]
        assert "allowed_condition_ids" in kwargs["system"]
        assert "verbatim" in kwargs["system"] and "whitespace" in kwargs["system"]
        if branch is verify_theory:
            assert "from within quote" in kwargs["system"]
        return response

    result = branch(claim, materials, call=model)
    assert result.evidence == [] and result.plans == []
    assert result.issues == response.get("issues", [])


@pytest.mark.parametrize("branch", [verify_code, verify_theory, verify_experiments])
@pytest.mark.parametrize("covered", [[], ["a", "a"], ["c1"], ["A"], ["other"]])
def test_all_branches_reject_empty_duplicate_and_invented_condition_ids(branch, covered, claim, materials):
    if branch is verify_code:
        response = {"items": [code_item(covered=covered)]}
    elif branch is verify_theory:
        response = {
            "items": [theory_item(block_id="main", quote="Theorem 1: the method converges.", covered=covered)]
        }
    else:
        response = experiments_response(items=[experiment_item(covered=covered)])
    with pytest.raises(ValueError) as error:
        branch(claim, materials, call=mock_response(response))
    if covered:
        assert f"received={covered!r}" in str(error.value)
        assert "allowed_condition_ids=['a']" in str(error.value)


@pytest.mark.parametrize("changed_field", [None, "quote", "step_quote"])
def test_theory_requires_verbatim_math_and_internal_whitespace(changed_field, claim, materials):
    quote = r"Proof. $x = \frac{ a }{ b }$; hence convergence."
    step = r"x = \frac{ a }{ b }"
    materials.blocks[0].text = quote
    materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    item = theory_item(block_id="main", quote=quote, step_quote=step, main_block_id=None)
    if changed_field:
        item[changed_field] = item[changed_field].replace(r"\frac{ a }{ b }", r"\frac{a}{b}")
        with pytest.raises(ValueError):
            verify_theory(claim, materials, call=mock_response({"items": [item]}))
    else:
        result = verify_theory(claim, materials, call=mock_response({"items": [item]}))
        assert result.evidence[0].pointer.quote == quote


@pytest.fixture
def notation_materials(materials, tmp_path, request):
    quote = "The direction selector is lambda(r) = dim(r)."
    materials.blocks[0].text = materials.markdown = quote
    Path(materials.markdown_path).write_text(quote, encoding="utf-8")
    image_path = tmp_path / "original_page_2.png"
    with fitz.open() as pdf:
        pdf.new_page()
        page = pdf.new_page()
        selector = getattr(request, "param", "dir")
        page.insert_text((40, 50), f"The direction selector is lambda(r) = {selector}(r).")
        pdf.save(materials.source_pdf)
        page.get_pixmap().save(image_path)
        materials.pages = [
            PageImage(
                page=2, path=str(image_path), width_points=page.rect.width, height_points=page.rect.height
            )
        ]
    return materials


@pytest.mark.parametrize(
    ("classification", "notation_materials"),
    [("parser_artifact", "dir"), ("uncertain", "?"), ("manuscript_issue", "dim")],
    indirect=["notation_materials"],
)
def test_notation_flaws_require_original_pdf_visual_confirmation(classification, claim, notation_materials):
    materials = notation_materials
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "items": [
                    theory_item(
                        block_id="main",
                        quote=materials.blocks[0].text,
                        kind="notation",
                        direction="flaw",
                        step_quote="",
                        main_block_id=None,
                        detail="The direction selector appears as dim rather than dir.",
                    )
                ]
            }
        payload = json.loads(kwargs["prompt"])
        assert kwargs["module"] == "verification.theory.notation"
        assert kwargs["images"] == [materials.pages[0].path]
        assert payload["page"] == 2 and payload["parsed_quote"] == materials.blocks[0].text
        return {"classification": classification, "explanation": "Observed the printed selector on page 2."}

    result = verify_theory(claim, materials, call=model)
    assert len(calls) == 2
    if classification == "manuscript_issue":
        assert len(result.evidence) == 1 and result.evidence[0].direction == "flaw"
        assert (
            not result.evidence[0].sufficient and "original PDF page 2 confirmed" in result.evidence[0].note
        )
        assert not result.evidence[0].affects_claim and not result.evidence[0].concern
    else:
        assert result.evidence == []
        assert any(
            classification in issue and "Observed the printed selector" in issue for issue in result.issues
        )
    assert materials.blocks[0].text == "The direction selector is lambda(r) = dim(r)."


@pytest.mark.parametrize("missing", ["no_matching_page", "missing_file"])
def test_notation_without_original_pdf_image_stays_explicitly_unconfirmed(missing, claim, notation_materials):
    materials = notation_materials
    if missing == "no_matching_page":
        materials.pages[0].page = 1
    else:
        Path(materials.pages[0].path).unlink()
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        assert len(calls) == 1, "A missing original page must not be verified against parsed text alone"
        return {
            "items": [
                theory_item(
                    block_id="main", quote=materials.blocks[0].text, kind="notation", direction="flaw"
                )
            ]
        }

    result = verify_theory(claim, materials, call=model)
    assert result.evidence == []
    assert any("original PDF page image unavailable" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "response",
    [
        {"classification": "yes", "explanation": "Unvalidated verdict."},
        {"status": "error", "error": "offline"},
    ],
)
def test_notation_visual_failure_cannot_produce_deciding_flaw(response, claim, notation_materials):
    def model(**kwargs):
        if kwargs["module"] == "verification.theory.notation":
            return response
        return {
            "items": [
                theory_item(
                    block_id="main",
                    quote=notation_materials.blocks[0].text,
                    kind="notation",
                    direction="flaw",
                )
            ]
        }

    result = verify_theory(claim, notation_materials, call=model)
    assert result.evidence == []
    assert any("original PDF check failed" in issue for issue in result.issues)


async def test_rejected_plan_preserves_only_validated_observations(claim, materials, tmp_path):
    from schemas.claim import EvidenceNeed
    from verification.contracts import RejectedPlan
    from verification.dispatch import verify_claims

    candidate = plan_candidate()
    candidate["targets"][0]["reported"]["token"] = "0.999"
    response = experiments_response(items=[experiment_item()], plans=[candidate])
    with pytest.raises(RejectedPlan, match="token is absent") as caught:
        verify_experiments(claim, materials, call=mock_response(response))
    assert len(caught.value.observations.evidence) == 1
    assert caught.value.observations.plans == []

    claim.needs = [EvidenceNeed.EXPERIMENTS]
    result = await verify_claims(
        [claim],
        materials,
        tmp_path / "dispatch",
        branches={
            EvidenceNeed.EXPERIMENTS: lambda c, m: verify_experiments(c, m, call=mock_response(response))
        },
    )
    assert result.plans == []
    assert result.claims[0].evidence == caught.value.observations.evidence
    assert any("Execution plan rejected" in issue and "token is absent" in issue for issue in result.issues)


def support_response(branch, **changes):
    if branch is verify_code:
        return {"items": [code_item(**changes)]}
    if branch is verify_theory:
        return {
            "items": [
                theory_item(
                    block_id="main",
                    quote="A test MRR is 0.4.",
                    step_quote="MRR is 0.4",
                    main_block_id=None,
                    **changes,
                )
            ]
        }
    return experiments_response(items=[experiment_item(**changes)])


@pytest.mark.parametrize("branch", [verify_code, verify_theory, verify_experiments])
@pytest.mark.parametrize("full", [None, [], ["a"]])
def test_positive_evidence_requires_explicit_full_condition_support(
    branch, full, claim, theory_claim, code_claim, materials
):
    from assessment.rules import assess_claim

    if branch is verify_code:
        claim = code_claim
    if branch is verify_theory:
        claim = theory_claim
        materials.blocks[0].text = materials.markdown = (
            "Theorem 1: the method converges.\n"
            "Proof of Theorem 1. By the stated assumption, x = y; therefore convergence follows."
        )
        Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    response = support_response(branch, fully_supported_conditions=full or [])
    if branch is verify_theory:
        response["items"][0].update(
            quote="Proof of Theorem 1. By the stated assumption, x = y; therefore convergence follows.",
            step_quote="x = y",
        )
    if full is None:
        del response["items"][0]["fully_supported_conditions"]

    def model(**kwargs):
        assert "fully_supported_conditions" in kwargs["system"] and "ENTIRE" in kwargs["system"]
        return (
            scope_response(kwargs)
            if kwargs["module"] == "verification.experiments.scope"
            else code_scope_response(kwargs)
            if kwargs["module"] == "verification.code.scope"
            else response
        )

    result = branch(claim, materials, call=model)
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.pointer.quote and evidence.direction == "support"
    assert evidence.sufficient is bool(full)
    assert f"fully_supported_conditions={full or []!r}" in evidence.note
    claim.evidence = result.evidence
    assert assess_claim(claim).status == ("supported" if full else "unverified")


@pytest.mark.parametrize("branch", [verify_code, verify_theory, verify_experiments])
@pytest.mark.parametrize("full", [["foreign"], ["a", "a"], "a", None])
def test_fully_supported_conditions_must_be_a_distinct_covered_subset(branch, full, claim, materials):
    response = support_response(branch, fully_supported_conditions=full)
    with pytest.raises(ValueError, match="fully_supported_conditions"):
        branch(claim, materials, call=mock_response(response))


@pytest.mark.parametrize("branch", [verify_code, verify_theory, verify_experiments])
def test_full_support_for_only_one_covered_condition_is_insufficient(branch, claim, materials):
    claim.conditions.append(Condition(id="b", dataset="B", metric="MRR"))
    response = support_response(branch, covered=["a", "b"], fully_supported_conditions=["a"])
    if branch is verify_theory:
        materials.blocks[0].text = materials.markdown = "Therefore A test MRR is 0.4."
        Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
        response["items"][0]["quote"] = materials.markdown
    result = branch(claim, materials, call=mock_response(response))
    assert result.evidence[0].covered == ["a", "b"]
    assert not result.evidence[0].sufficient


def test_source_presence_does_not_establish_release_url_and_all_data_availability(claim, materials):
    from assessment.rules import assess_claim

    claim.text = "The source code and all datasets are available at the stated GitHub URL."
    claim.conditions = [
        Condition(
            id="a", settings={"artifact": "source code and all datasets", "url": "https://example.org/repo"}
        )
    ]
    paper_quote = claim.text
    materials.blocks[0].text = materials.markdown = paper_quote
    Path(materials.markdown_path).write_text(paper_quote, encoding="utf-8")
    code_quote = "load_dataset('A')"
    source = Path(materials.repository.root) / "eval.py"
    source.write_text(code_quote + "\n", encoding="utf-8")
    next(f for f in materials.repository.files if f.path == "eval.py").sha256 = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    detail = (
        "The source is present, but its presence does not verify the URL or availability of all datasets."
    )
    result = verify_code(
        claim,
        materials,
        call=mock_response(
            {
                "items": [
                    code_item(
                        quote=code_quote,
                        paper_quote=paper_quote,
                        fully_supported_conditions=[],
                        detail=detail,
                    )
                ]
            }
        ),
    )
    assert not result.evidence[0].sufficient and detail in result.evidence[0].note
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "unverified"


@pytest.mark.parametrize("branch", [verify_code, verify_theory])
def test_architecture_evidence_does_not_establish_historical_novelty(branch, claim, materials):
    from assessment.rules import assess_claim

    claim.text = "COMPGCN is a novel framework using entity-relation composition."
    claim.conditions = [
        Condition(id="a", description="Historical novelty and entity-relation composition architecture.")
    ]
    paper_quote = "By composition, message = phi(node, relation)."
    materials.blocks[0].text = materials.markdown = paper_quote
    Path(materials.markdown_path).write_text(paper_quote, encoding="utf-8")
    response = support_response(branch, fully_supported_conditions=[])
    if branch is verify_theory:
        response["items"][0].update(quote=paper_quote, step_quote="message = phi(node, relation)")
    else:
        code_quote = "message = compose(node, relation)"
        source = Path(materials.repository.root) / "eval.py"
        source.write_text(code_quote + "\n", encoding="utf-8")
        next(f for f in materials.repository.files if f.path == "eval.py").sha256 = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        response["items"][0].update(quote=code_quote, paper_quote=paper_quote, aspect="architecture")

    def model(**kwargs):
        assert "historical novelty" in kwargs["system"]
        return response

    result = branch(claim, materials, call=model)
    assert result.evidence and not result.evidence[0].sufficient
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "unverified"
