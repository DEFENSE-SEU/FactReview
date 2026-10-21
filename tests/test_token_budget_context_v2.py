"""Per-call context trimming keeps source binding and records per-site usage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from llm.codex_client import _coerce_usage
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.claim_context import (
    condition_groups,
    local_catalog,
    scope_block_ids,
)
from verification.experiment_catalog import build_catalog, resolve_source
from verification.experiments import verify_experiments

TABLE = (
    "Table 1: Accuracy on D.\n\n<table><tr><td>Model</td><td>Acc</td></tr>"
    + "".join(f"<tr><td>M{i}</td><td>{50 + i}</td></tr>" for i in range(70))
    + "<tr><td>Ours</td><td>99</td></tr></table>"
)


def _paper(tmp_path):
    texts = {
        "intro": "Unrelated introduction about history.",
        "claim": "Ours reaches the best accuracy on D (Table 1).",
        "next": "We train every model for ten epochs.",
        "table": TABLE,
        "ref": "As Table 1 shows, the margin is large.",
        "far": "Far away discussion of D accuracy definitions.",
        "other": "Table 2: Speed.\n\n<table><tr><td>Model</td><td>Time</td></tr><tr><td>Ours</td><td>3</td></tr></table>",
        "bib": "[1] Some cited paper about Table 1.",
    }
    materials = SharedMaterials(
        paper_key="fixture", source_pdf="fixture.pdf", markdown="",
        markdown_path=str(tmp_path / "paper.md"), content_list_path="c.json", provider="mock", blocks=[],
    )
    for block_id, text in texts.items():
        start = len(materials.markdown)
        materials.markdown += text + "\n"
        materials.blocks.append(MaterialBlock(
            id=block_id, text=text, kind="ref_text" if block_id == "bib" else "text",
            loc=ClaimLocation(page=1, char_start=start, char_end=start + len(text)),
        ))
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    claim = Claim(
        id="c", text=texts["claim"], loc=ClaimLocation(page=1), source_block_id="claim",
        source_quote=texts["claim"], needs=["Experiments"],
        conditions=[Condition(id="c1", dataset="D", metric="accuracy", settings={"method": "Ours"})],
    )
    return claim, materials


def test_scope_blocks_follow_the_claim_its_table_and_definitions(tmp_path):
    claim, materials = _paper(tmp_path)
    ids = scope_block_ids(claim, materials)
    assert {"claim", "intro", "next", "table", "ref", "far"} <= ids
    assert "bib" not in ids and "other" not in ids


def test_trimmed_sources_and_cells_cannot_be_resolved(tmp_path):
    claim, materials = _paper(tmp_path)
    catalog = build_catalog(claim, materials)
    ids = scope_block_ids(claim, materials)
    local = local_catalog(catalog, ids, condition_groups(claim))
    dropped = [k for k, s in catalog["sources"].items() if s.get("block_id") == "other"]
    assert dropped and not set(dropped) & set(local["sources"])
    with pytest.raises(ValueError, match="Unknown catalog source ID"):
        resolve_source(local, dropped[0], materials)
    kept_source = next(k for k, s in local["sources"].items() if s.get("block_id") == "table")
    assert resolve_source(local, kept_source, materials)["block_id"] == "table"
    # The 71-row table keeps the row naming the condition's method plus all labels.
    numbers = [c for c in local["cells"].values() if c["cell_type"] == "number"]
    assert [c["token"] for c in numbers] == ["99"]
    assert sum(c["cell_type"] == "label" for c in local["cells"].values()) == 73
    assert all(c["table_id"] in local["tables"] for c in local["cells"].values())


def test_disabled_execution_sends_no_execution_context(tmp_path, monkeypatch):
    claim, materials = _paper(tmp_path)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    calls = []

    def model(**kw):
        calls.append((kw["module"], kw["system"], json.loads(kw["prompt"])))
        return {
            "schema_version": "experiments-execution-v3",
            "checked_aspects": ["correspondence", "fairness", "isolation", "stability", "consistency"],
            "items": [],
            "execution": {"kind": "none", "rationale": "execution verification not requested"},
        }

    result = verify_experiments(claim, materials, call=model, execution_requested=False)
    assert [module for module, _, _ in calls] == ["verification.experiments"]
    _, system, payload = calls[0]
    for key in ("repository_index", "execution_projection_context", "execution_choice_context", "target_catalog"):
        assert key not in payload
    assert "bib" not in {block["id"] for block in payload["paper_blocks"]}
    assert '"legacy"' not in json.dumps(payload["output_schema"]) and "not requested" in system
    assert result.plans == []


def test_usage_is_recorded_per_call_site_with_cached_tokens(tmp_path):
    usage = _coerce_usage({
        "input_tokens": 100, "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 60}, "output_tokens_details": {"reasoning_tokens": 7},
    })
    assert usage["cached_input_tokens"] == 60 and usage["reasoning_output_tokens"] == 7
    with run_stats.run_scope(tmp_path / "stats.json"):
        run_stats.record_llm_call(module="verification.experiments.scope", usage=usage)
        run_stats.record_llm_call(module="report_generation", usage=usage)
        row = run_stats.read()["modules"]
    scope = row["analysis"]["submodules"]["verification.experiments.scope"]
    assert scope == {
        "requests": 1, "input_tokens": 100, "output_tokens": 20,
        "cached_input_tokens": 60, "reasoning_output_tokens": 7, "reasoning_effort": {"default": 1},
    }
    assert row["analysis"]["token_usage"]["input_tokens"] == 100
    assert list(row["report_generation"]["submodules"]) == [f"report_generation:{__name__}"]


def test_code_check_sends_and_accepts_only_claim_local_blocks(tmp_path, monkeypatch):
    from preprocessing.materials import index_repository
    from screening import checks
    from verification.code import verify_code

    monkeypatch.delenv("CODE_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    claim, materials = _paper(tmp_path)
    claim = claim.model_copy(update={"needs": ["Code"]})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 0.1\n", encoding="utf-8")
    materials = materials.model_copy(update={"repository": index_repository(repo)})
    sent = []

    def model(**kw):
        sent.append(json.loads(kw["prompt"]))
        return {"items": [{
            "file": "train.py", "line": 1, "quote": "lr = 0.1", "paper_block_id": "other",
            "paper_quote": "Table 2: Speed.", "covered": ["c1"], "direction": "support",
            "aspect": "hyperparameters", "detail": "Cites a block that was not sent.",
        }]}

    with pytest.raises(ValueError, match="block"):
        verify_code(claim, materials, call=model)
    ids = {block["id"] for block in sent[0]["paper_blocks"]}
    assert "claim" in ids and "other" not in ids and "bib" not in ids


def test_unanchored_claim_keeps_every_evidence_block(tmp_path):
    claim, materials = _paper(tmp_path)
    claim = claim.model_copy(update={"source_block_id": None, "source_quote": None})
    assert scope_block_ids(claim, materials) == {b.id for b in materials.blocks if b.id != "bib"}


def test_low_reasoning_effort_applies_only_to_configured_modules(tmp_path, monkeypatch):
    from llm import client

    sent = []

    def provider(**kwargs):
        sent.append(kwargs.get("reasoning_effort"))
        return '{"status":"ok"}', {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    monkeypatch.setattr(client, "get_codex_auth", lambda **kw: object())
    monkeypatch.setattr(client, "invoke_codex", provider)
    cfg = client.LLMConfig(provider="openai-codex", model="m", base_url="https://offline.invalid", api_key=None)
    with run_stats.run_scope(tmp_path / "stats.json"):
        for module in ("screening.claims.coverage", "verification.experiments"):
            client.llm_json("{}", "s", cfg, module=module)
        with client.effort_label("report_generation.advice"):
            client.llm_json("{}", "s", cfg, module="report_generation")
        modules = run_stats.read()["modules"]
    assert sent == ["low", None, "low"]
    assert modules["analysis"]["submodules"]["screening.claims.coverage"]["reasoning_effort"] == {"low": 1}
    assert modules["analysis"]["submodules"]["verification.experiments"]["reasoning_effort"] == {"default": 1}
    advice = modules["report_generation"]["submodules"]["report_generation:tests.test_token_budget_context_v2"]
    assert advice["reasoning_effort"] == {"low": 1}


def test_writing_index_keeps_only_labels_named_in_the_section():
    from screening.writing import _section_index

    index = {"complete": False, "entries": [
        {"kind": "table", "label": "1"}, {"kind": "figure", "label": "12"}, {"kind": "section", "label": "2.1"},
    ]}
    blocks = [MaterialBlock(id="b", text="As Table 1 and Section 2.1 show; see Figure 1.2.")]
    assert [e["label"] for e in _section_index(index, blocks)["entries"]] == ["1", "2.1"]


def test_coverage_registry_is_full_only_for_referenced_claims():
    from screening.claim_coverage import _registry_view

    registry = [
        {"claim_id": "claim_001", "text": "A.", "digest": "d1", "conditions": [{"id": "c1"}]},
        {"claim_id": "claim_002", "text": "B.", "digest": "d2", "conditions": [{"id": "c1"}]},
    ]
    payload = {"current_claims": registry, "required_claim_checks": [{"claim_id": "claim_002"}]}
    assert _registry_view(registry, payload) == [{"claim_id": "claim_001", "text": "A."}, registry[1]]


def test_first_pass_keeps_experiment_sections_abstract_and_contributions(tmp_path):
    from verification.claim_context import experiment_block_ids

    claim, materials = _paper(tmp_path)
    sections = {
        "intro": "section_1: ABSTRACT", "claim": "section_2: 1 INTRODUCTION", "next": "section_2: 1 INTRODUCTION",
        "table": "section_4: 4 EXPERIMENTS", "ref": "section_5: 4.2 Speed", "far": "section_3: 3 METHOD",
        "other": "section_6: 5 RELATED WORK", "bib": "section_7: REFERENCES",
    }
    for block in materials.blocks:
        block.loc.section = sections[block.id]
    materials.blocks.append(MaterialBlock(
        id="contrib", text="Our contributions are threefold.", loc=ClaimLocation(page=1, section="section_2: 1 INTRODUCTION"),
    ))
    materials.blocks.append(MaterialBlock(
        id="method", text="We define the loss.", loc=ClaimLocation(page=1, section="section_3: 3 METHOD"),
    ))
    ids = experiment_block_ids(claim, materials)
    assert {"intro", "claim", "table", "ref", "contrib"} <= ids
    assert "method" not in ids and "bib" not in ids


def test_code_claims_sharing_files_use_one_batched_request(tmp_path, monkeypatch):
    from preprocessing.materials import index_repository
    from screening import checks
    from verification.code import batch_code_responses, verify_code

    monkeypatch.delenv("CODE_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    claim, materials = _paper(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 0.1\n", encoding="utf-8")
    materials = materials.model_copy(update={"repository": index_repository(repo)})
    first = claim.model_copy(update={"id": "c_a", "needs": ["Code"]})
    second = claim.model_copy(update={"id": "c_b", "needs": ["Code"]})
    third = claim.model_copy(update={"id": "c_c", "needs": ["Code"]})
    modules = []
    item = {
        "file": "train.py", "line": 1, "quote": "lr = 0.1", "paper_block_id": "next",
        "paper_quote": "We train every model for ten epochs.", "covered": ["c1"], "direction": "support",
        "aspect": "hyperparameters", "detail": "Training setup matches.",
    }

    def model(**kw):
        modules.append(kw["module"])
        if kw["module"] == "verification.code.batch":
            payload = json.loads(kw["prompt"])
            assert [row["claim"]["id"] for row in payload["claims"]] == ["c_a", "c_b", "c_c"]
            return {"results": [{"claim_id": "c_a", "items": [item]}, {"claim_id": "c_b", "items": []}]}
        return {"items": []}

    batched = batch_code_responses([first, second, third], materials, call=model)
    assert set(batched) == {"c_a", "c_b"}  # c_c was omitted and falls back
    result = verify_code(first, materials, call=model, scope_call=lambda **kw: {}, batched_response=batched["c_a"])
    verify_code(third, materials, call=model, batched_response=batched.get("c_c"))
    assert modules.count("verification.code.batch") == 1 and modules.count("verification.code") == 1
    assert result.evidence and result.evidence[0].pointer.quote == "lr = 0.1"
