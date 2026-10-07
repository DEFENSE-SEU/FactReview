"""Offline branch contracts keep evidence tied to actual manuscript/repository bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, RepositoryFile, RepositoryIndex, SharedMaterials
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
        text="The method improves MRR on A.",
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


def mock_response(payload: dict):
    return lambda **kwargs: payload


def theory_item(**changes) -> dict:
    return {
        "block_id": "appendix",
        "quote": "By the stated assumption, x = y; therefore convergence follows.",
        "covered": ["a"],
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
    claim: Claim, materials: SharedMaterials
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

    result = verify_theory(claim, materials, call=model)
    assert len(seen) == 2 and len(result.evidence) == 1
    assert result.evidence[0].source == "theory" and result.evidence[0].sufficient
    assert result.evidence[0].pointer.page == 8
    assert result.plans == []


def test_theory_missing_proof_yields_question_without_support(
    claim: Claim, materials: SharedMaterials
) -> None:
    result = verify_theory(
        claim,
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
    assert result.evidence == [] and result.questions and result.issues


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
    claim: Claim, materials: SharedMaterials, payload: dict
) -> None:
    with pytest.raises(ValueError):
        verify_theory(claim, materials, call=mock_response(payload))


def test_theory_theorem_assertion_alone_cannot_be_support(claim: Claim, materials: SharedMaterials) -> None:
    quote = "Theorem 1: the method converges."
    result = verify_theory(
        claim,
        materials,
        call=mock_response(
            {"items": [theory_item(block_id="main", quote=quote, step_quote=quote, main_block_id=None)]}
        ),
    )
    assert result.evidence == [] and result.issues


def test_code_verifies_index_hash_and_exact_line_plus_paper_quote(
    claim: Claim, materials: SharedMaterials
) -> None:
    result = verify_code(claim, materials, call=mock_response({"items": [code_item()]}))
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


def test_code_preserves_indentation_in_exact_source_quote(claim: Claim, materials: SharedMaterials) -> None:
    text = "def train():\n    optimizer = 'Adam'\n"
    path = Path(materials.repository.root) / "eval.py"
    path.write_bytes(text.encode())
    materials.repository.files[0].sha256 = hashlib.sha256(text.encode()).hexdigest()
    result = verify_code(
        claim, materials, call=mock_response({"items": [code_item(line=2, quote="    optimizer = 'Adam'")]})
    )
    assert result.evidence[0].pointer.quote.startswith("    ")


def test_experiments_paper_support_and_claim_linked_plan(claim: Claim, materials: SharedMaterials) -> None:
    def model(**kwargs):
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
    result = verify_experiments(
        claim,
        materials,
        call=mock_response(
            experiments_response(
                items=[
                    experiment_item(
                        aspect="isolation",
                        kind="missing_ablation",
                        detail="No isolated component experiment is reported.",
                    )
                ]
            )
        ),
    )
    assert result.evidence[0].concern and result.evidence[0].overturnable
    assert result.questions[0].claim_id == claim.id


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
