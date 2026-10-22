"""Code support cannot outlive the exact target and paper inspected by its models."""

import asyncio
import copy
import hashlib

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition, EvidenceNeed
from schemas.materials import MaterialBlock, RepositoryFile, RepositoryIndex, SharedMaterials
from verification.code import verify_code
from verification.dispatch import verify_claims


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Code lifecycle controls must not contact services or execute a repository")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def inputs(tmp_path):
    text = "Our optimizer is Adam."
    markdown = tmp_path / "paper.md"
    markdown.write_text(text, encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"unused PDF source snapshot")
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    repository = tmp_path / "repo"
    repository.mkdir()
    files = []
    for name, code in {"train.py": "optimizer = 'Adam'\n", "config.py": "batch_size = 64\n"}.items():
        (repository / name).write_text(code, encoding="utf-8")
        files.append(
            RepositoryFile(
                path=name, kind="source", sha256=hashlib.sha256((repository / name).read_bytes()).hexdigest()
            )
        )
    materials = SharedMaterials(
        paper_key="lifecycle",
        source_pdf=str(pdf),
        markdown_path=str(markdown),
        content_list_path="",
        markdown=text,
        provider="fixture",
        blocks=[MaterialBlock(id="main", text=text, loc=loc)],
        repository=RepositoryIndex(root=str(repository), files=files),
    )
    claim = Claim(
        id="target",
        text=text,
        source_block_id="main",
        source_quote=text,
        loc=loc,
        conditions=[Condition(id="c1", description=text)],
        needs=["Code"],
    )
    first = {
        "items": [
            {
                "file": "train.py",
                "line": 1,
                "quote": "optimizer = 'Adam'",
                "paper_block_id": "main",
                "paper_quote": text,
                "covered": ["c1"],
                "fully_supported_conditions": ["c1"],
                "direction": "support",
                "aspect": "optimizer",
                "detail": "The direct optimizer matches.",
            }
        ]
    }
    second = {
        "conditions": [
            {
                "condition_id": "c1",
                "required_facets": ["implementation"],
                "claim_source_ids": ["primary"],
                "rationale": "Exact optimizer claim.",
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
                "rationale": "Exact implementation correspondence.",
            }
        ],
    }
    return claim, materials, first, second


def change_context(kind, claim, materials, tmp_path):
    from pathlib import Path

    if kind == "claim_text":
        claim.text = "The model achieves 99.99 percent accuracy."
    elif kind == "condition":
        claim.conditions[0].metric = "test accuracy"
    elif kind == "block_text":
        materials.blocks[0].text = "Our optimizer is SGD."
    elif kind == "block_replacement":
        materials.blocks = [materials.blocks[0].model_copy(update={"loc": ClaimLocation(page=9)})]
    elif kind == "markdown_value":
        materials.markdown += " A different context."
    elif kind == "markdown_bytes":
        Path(materials.markdown_path).write_text("A different paper.", encoding="utf-8")
    elif kind == "pdf_bytes":
        Path(materials.source_pdf).write_bytes(b"a changed PDF")
    elif kind == "delete_pdf":
        Path(materials.source_pdf).unlink()
    elif kind == "paper_path":
        copied = tmp_path / "copied.md"
        copied.write_text(materials.markdown, encoding="utf-8")
        materials.markdown_path = str(copied)
    elif kind == "repository_index":
        materials.repository.files[0].sha256 = "a" * 64
    elif kind == "unused_selected_code":
        Path(materials.repository.root, "config.py").write_text("batch_size = 1\n", encoding="utf-8")
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("phase", ["verification.code", "verification.code.scope"])
@pytest.mark.parametrize(
    "kind",
    [
        "claim_text",
        "condition",
        "block_text",
        "block_replacement",
        "markdown_value",
        "markdown_bytes",
        "pdf_bytes",
        "delete_pdf",
        "paper_path",
        "repository_index",
        "unused_selected_code",
    ],
)
def test_changed_context_revokes_all_candidates_at_each_model_boundary(tmp_path, phase, kind):
    claim, materials, first, second = inputs(tmp_path)
    before = copy.deepcopy((first, second))
    seen = []

    def call(**kwargs):
        seen.append(kwargs["module"])
        if kwargs["module"] == phase:
            change_context(kind, claim, materials, tmp_path)
        return copy.deepcopy(first if kwargs["module"] == "verification.code" else second)

    with pytest.raises(ValueError, match="Code"):
        verify_code(claim, materials, call=call)
    assert len(seen) == (1 if phase == "verification.code" else 2)
    assert (first, second) == before


def test_healthy_unavailable_unused_pdf_preserves_original_source_contract(tmp_path):
    from pathlib import Path

    claim, materials, first, second = inputs(tmp_path)
    Path(materials.source_pdf).unlink()
    seen = []

    def call(**kwargs):
        seen.append(kwargs["module"])
        return copy.deepcopy(first if kwargs["module"] == "verification.code" else second)

    result = verify_code(claim, materials, call=call)
    claim.evidence = result.evidence
    assert len(seen) == 2 and result.evidence[0].sufficient
    assert assess_claim(claim).status == "supported"


def test_changed_source_cannot_be_hidden_by_empty_or_failed_scope(tmp_path):
    claim, materials, first, _second = inputs(tmp_path)

    def call(**kwargs):
        if kwargs["module"] == "verification.code":
            return copy.deepcopy(first)
        change_context("markdown_bytes", claim, materials, tmp_path)
        return {"status": "error", "error": "Mock provider failure after source changed"}

    with pytest.raises(ValueError, match="source artifact changed"):
        verify_code(claim, materials, call=call)


def test_empty_first_pass_still_rechecks_claim(tmp_path):
    claim, materials, _first, _second = inputs(tmp_path)

    def call(**kwargs):
        claim.conditions[0].metric = "accuracy"
        return {"items": []}

    with pytest.raises(ValueError, match="target claim"):
        verify_code(claim, materials, call=call)


def test_final_check_covers_mutation_during_evidence_assembly(tmp_path, monkeypatch):
    import verification.code as module

    claim, materials, first, second = inputs(tmp_path)
    original = module.Evidence

    def evidence(**kwargs):
        value = original(**kwargs)
        change_context("markdown_bytes", claim, materials, tmp_path)
        return value

    monkeypatch.setattr(module, "Evidence", evidence)
    with pytest.raises(ValueError, match="source artifact changed"):
        verify_code(
            claim,
            materials,
            call=lambda **kw: copy.deepcopy(first if kw["module"] == "verification.code" else second),
        )


def test_dispatch_retains_system_failure_and_healthy_neighbor(tmp_path):
    claim, materials, first, second = inputs(tmp_path)
    healthy = claim.model_copy(deep=True, update={"id": "healthy"})
    original = claim.model_dump(mode="json")

    def branch(c, m):
        def call(**kwargs):
            if kwargs["module"] == "verification.code.scope" and c.id == "target":
                c.conditions[0].metric = "accuracy"
            return copy.deepcopy(first if kwargs["module"] == "verification.code" else second)

        return verify_code(c, m, call=call)

    result = asyncio.run(
        verify_claims(
            [claim, healthy], materials, tmp_path / "verification", branches={EvidenceNeed.CODE: branch}
        )
    )
    bad, good = result.claims
    assert not bad.evidence and not bad.questions
    assert bad.verification_limitations[0].kind == "branch_failed"
    assert bad.verification_limitations[0].responsibility == "system"
    assert good.evidence[0].sufficient and not good.verification_limitations
    assert claim.model_dump(mode="json") == original
