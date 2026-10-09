"""Repository inputs remain bounded, read-only and honest about omitted source."""

import hashlib
import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.code import verify_code


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("CODE_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    root = tmp_path / "repository"
    root.mkdir()
    paper = "The optimizer uses Adam."
    path = tmp_path / "paper.md"
    path.write_text(paper, encoding="utf-8")
    materials = SharedMaterials(
        paper_key="scope",
        source_pdf="paper.pdf",
        markdown=paper,
        markdown_path=str(path),
        content_list_path="content.json",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=paper, loc=ClaimLocation(page=1))],
    )
    claim = Claim(
        id="c1",
        text=paper,
        loc=ClaimLocation(page=1),
        source_block_id="b1",
        source_quote=paper,
        conditions=[Condition(id="optimizer", description="optimizer is Adam")],
        needs=["Code"],
        importance="core",
    )
    return root, materials, claim


@pytest.mark.parametrize("suffix", [".R", ".jl", ".java", ".go", ".rs", ".cu", ".m", ".f90"])
def test_scientific_language_sources_reach_code_branch_with_exact_bytes(inputs, suffix):
    root, materials, claim = inputs
    path = root / ("optimizer" + suffix)
    content = "optimizer = Adam\n"
    path.write_text(content, encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    materials.repository = index_repository(root)

    def model(**request):
        data = json.loads(request["prompt"])
        if request["module"] == "verification.code.scope":
            assert data["source_context"][path.name] == [{"line": 1, "text": "optimizer = Adam"}]
            return {
                "conditions": [
                    {
                        "condition_id": "optimizer",
                        "required_facets": ["implementation"],
                        "claim_source_ids": ["primary"],
                        "rationale": "The paper claims the Adam optimizer.",
                    }
                ],
                "items": [
                    {
                        "item_index": 0,
                        "condition_id": "optimizer",
                        "relation": "supports_implementation",
                        "full_condition": True,
                        "basis": "direct_source",
                        "bridge_quotes": [],
                        "missing_qualifiers": [],
                        "rationale": "The indexed language file directly assigns Adam.",
                    }
                ],
            }
        assert data["files"][path.name] == [{"line": 1, "text": "optimizer = Adam"}]
        return {
            "items": [
                {
                    "file": path.name,
                    "line": 1,
                    "quote": "optimizer = Adam",
                    "paper_block_id": "b1",
                    "paper_quote": claim.text,
                    "covered": ["optimizer"],
                    "fully_supported_conditions": ["optimizer"],
                    "direction": "support",
                    "aspect": "optimizer",
                    "detail": "Exact optimizer assignment.",
                }
            ]
        }

    result = verify_code(claim, materials, call=model)
    assert result.evidence[0].pointer.locator == str(path.resolve())
    assert result.evidence[0].pointer.line == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_large_source_excluded_and_scope_visible_while_small_file_checked(inputs, monkeypatch):
    root, materials, claim = inputs
    (root / "model.py").write_text("# lots of model source\n" * 20000, encoding="utf-8")
    (root / "optimizer.json").write_text('{"optimizer": "Adam"}', encoding="utf-8")
    materials.repository = index_repository(root)
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", "1000")

    def model(**request):
        data = json.loads(request["prompt"])
        assert set(data["files"]) == {"optimizer.json"}
        assert len(json.dumps(data["files"], ensure_ascii=False).encode()) <= 1000
        assert data["source_scope"]["omitted_sample"] == [{"path": "model.py", "reason": "source_budget"}]
        assert data["source_scope"]["omitted_count"] == 1
        return {"items": []}

    result = verify_code(claim, materials, call=model)
    assert "coverage is partial" in result.issues[0] and "model.py" in result.issues[0]
    assert result.evidence == []


def test_budget_counts_line_metadata_and_cannot_ground_unseen_file(inputs, monkeypatch):
    root, materials, claim = inputs
    (root / "optimizer.json").write_text('{"optimizer": "Adam"}', encoding="utf-8")
    # Raw bytes fit, but serialized line numbers and JSON metadata exceed the budget.
    (root / "blank.py").write_text("\n" * 40, encoding="utf-8")
    materials.repository = index_repository(root)
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", "200")

    def model(**request):
        data = json.loads(request["prompt"])
        assert "blank.py" not in data["files"]
        return {
            "items": [
                {
                    "file": "blank.py",
                    "line": 1,
                    "quote": "invented",
                    "paper_block_id": "b1",
                    "paper_quote": claim.text,
                    "covered": ["optimizer"],
                    "direction": "flaw",
                    "aspect": "optimizer",
                    "detail": "Attempt to cite an omitted file.",
                }
            ]
        }

    with pytest.raises(ValueError, match="outside the repository index"):
        verify_code(claim, materials, call=model)


def test_no_source_fits_returns_system_limitation_without_model_call(inputs, monkeypatch):
    root, materials, claim = inputs
    (root / "model.py").write_text("source" * 1000, encoding="utf-8")
    materials.repository = index_repository(root)
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", "200")
    result = verify_code(claim, materials, call=lambda **_: pytest.fail("No inspectable input"))
    assert result.evidence == [] and result.questions == []
    assert len(result.verification_limitations) == 1
    limitation = result.verification_limitations[0]
    assert limitation.claim_id == claim.id
    assert limitation.condition_ids == [condition.id for condition in claim.conditions]
    assert limitation.stage == "Code" and limitation.kind == "source_context_unavailable"
    assert limitation.responsibility == "system" and "source_budget" in limitation.reason
    assert any("source_budget" in issue for issue in result.issues)


@pytest.mark.parametrize("budget", ["0", "-1", "wrong"])
def test_invalid_source_budget_fails_visibly(inputs, monkeypatch, budget):
    _, materials, claim = inputs
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", budget)
    with pytest.raises(ValueError):
        verify_code(claim, materials, call=lambda **_: pytest.fail("Bad configuration"))


def test_many_omitted_files_have_bounded_prompt_and_report_with_full_local_manifest(
    inputs, monkeypatch, tmp_path
):
    root, materials, claim = inputs
    (root / "optimizer.json").write_text('{"optimizer": "Adam"}', encoding="utf-8")
    for index in range(1000):
        (root / (f"large_{index:04d}_" + "x" * 70 + ".py")).write_text("#" * 1100, encoding="utf-8")
    materials.repository = index_repository(root)
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", "1000")

    def model(**request):
        data = json.loads(request["prompt"])
        assert len(json.dumps(data["files"]).encode()) <= 1000
        scope = data["source_scope"]
        assert scope["omitted_count"] == 1000 and len(scope["omitted_sample"]) == 10
        assert len(json.dumps(scope).encode()) < 4000
        manifest = json.loads(Path(scope["manifest"]).read_text(encoding="utf-8"))
        assert len(manifest["omitted_files"]) == 1000
        return {"items": []}

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result = verify_code(claim, materials, call=model)
    assert len(result.issues[0].encode()) < 4000
