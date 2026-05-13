"""Claim-extract stage tests.

Tests the public ``extract_facts`` entry point that the stage_runner ends up
delegating to. The cases pin down the contract that downstream stages rely
on: heuristic mode remains available for explicit local use, LLM mode parses
LLM JSON into ``Claim`` objects, and auto mode uses the same strict LLM path
without heuristic fallback.
"""

from __future__ import annotations

import pytest

from preprocessing.claim_extract.extractor import ClaimExtractionError, extract_facts
from preprocessing.claim_extract.paper_loader import paper_from_parse_payload
from schemas.claim import ClaimType
from schemas.config import ClaimExtractCfg, LLMCfg


def test_heuristic_mode_extracts_all_claim_types(tiny_paper) -> None:
    result = extract_facts(tiny_paper, cfg=ClaimExtractCfg(mode="heuristic"))

    assert result.backend == "heuristic"
    assert result.claims, "heuristic must surface at least one claim from tiny_paper"
    assert result.core_claims == result.claims[:3]
    types = {c.type for c in result.claims}
    assert ClaimType.METHODOLOGICAL in types
    assert ClaimType.EMPIRICAL in types
    assert ClaimType.THEORETICAL in types

    # Empirical claims should carry the dataset hits the heuristic recognises;
    # this is the load-bearing piece downstream evidence-targeting reads.
    empirical = [c for c in result.claims if c.type == ClaimType.EMPIRICAL]
    assert empirical
    assert {"FB15k-237", "WN18RR"}.issubset(set(empirical[0].datasets))


def test_llm_mode_parses_llm_response(tiny_paper, monkeypatch) -> None:
    # Stub the LLM call so we exercise the JSON-parsing path without touching
    # the real client. The exact LLM JSON shape — claims under "claims" with
    # type / datasets / location — is what the prompt template asks for.
    from preprocessing.claim_extract import extractor as ext_mod

    captured: dict[str, object] = {}

    def fake_call(paper, reported, claim_cfg, llm_cfg):  # type: ignore[no-untyped-def]
        captured["paper_key"] = paper.metadata.paper_key
        claims = ext_mod._parse_llm_claims(
            [
                {
                    "id": "claim_01",
                    "text": "TinyMethod beats baselines on FB15k-237.",
                    "type": "empirical",
                    "scope": "local",
                    "datasets": ["FB15k-237"],
                    "metrics": ["MRR"],
                    "location": {"section_id": "sec_1"},
                },
                {
                    "id": "claim_02",
                    "text": "We prove TinyMethod generalizes prior work.",
                    "type": "theoretical",
                    "location": {"section_id": "sec_1"},
                },
            ]
        )
        return claims[:1], claims

    monkeypatch.setattr(ext_mod, "_call_llm_for_claims", fake_call)

    result = extract_facts(
        tiny_paper, cfg=ClaimExtractCfg(mode="llm", decompose_broad_claims=False), llm_cfg=LLMCfg()
    )

    assert result.backend == "llm"
    assert captured["paper_key"] == "tiny"
    assert [c.id for c in result.core_claims] == ["claim_01"]
    assert [c.id for c in result.claims] == ["claim_01", "claim_02"]
    assert result.claims[0].type is ClaimType.EMPIRICAL
    assert result.claims[1].type is ClaimType.THEORETICAL


def test_auto_mode_raises_when_llm_fails(tiny_paper, monkeypatch) -> None:
    from preprocessing.claim_extract import extractor as ext_mod

    def boom(paper, reported, claim_cfg, llm_cfg):  # type: ignore[no-untyped-def]
        raise ClaimExtractionError("simulated llm failure")

    monkeypatch.setattr(ext_mod, "_call_llm_for_claims", boom)

    with pytest.raises(ClaimExtractionError, match="simulated llm failure"):
        extract_facts(
            tiny_paper,
            cfg=ClaimExtractCfg(mode="auto", decompose_broad_claims=False),
            llm_cfg=LLMCfg(),
        )


def test_auto_mode_does_not_run_heuristic_alongside_llm(tiny_paper, monkeypatch) -> None:
    from preprocessing.claim_extract import extractor as ext_mod

    def fake_call(paper, reported, claim_cfg, llm_cfg):  # type: ignore[no-untyped-def]
        claims = ext_mod._parse_llm_claims(
            [
                {
                    "id": "claim_01",
                    "text": "TinyMethod is positioned as the first retrieval-style skill acquisition method.",
                    "type": "methodological",
                    "location": {"section_id": "sec_1"},
                }
            ]
        )
        return claims, claims

    def should_not_run(paper):  # type: ignore[no-untyped-def]
        raise AssertionError("auto mode must not call heuristic extraction")

    monkeypatch.setattr(ext_mod, "_call_llm_for_claims", fake_call)
    monkeypatch.setattr(ext_mod, "extract_claims_heuristic", should_not_run)

    result = extract_facts(
        tiny_paper,
        cfg=ClaimExtractCfg(mode="auto", decompose_broad_claims=False),
        llm_cfg=LLMCfg(),
    )

    assert result.backend == "auto:llm"
    assert [c.text for c in result.core_claims] == [
        "TinyMethod is positioned as the first retrieval-style skill acquisition method."
    ]
    assert [c.text for c in result.claims] == [
        "TinyMethod is positioned as the first retrieval-style skill acquisition method."
    ]


def test_stage_runner_builds_paper_from_parse_markdown(tmp_path) -> None:
    md_path = tmp_path / "paper.md"
    pdf_path = tmp_path / "paper.pdf"
    md_path.write_text("# Tiny Title\n\nIntro text.\n\n# 1 Method\n\nWe propose TinyMethod.", encoding="utf-8")
    pdf_path.write_bytes(b"%PDF-1.4\n")

    paper = paper_from_parse_payload(
        repo_root=tmp_path,
        paper_key="tiny",
        parse_payload={
            "source_pdf": str(pdf_path),
            "mineru_markdown_path": str(md_path),
            "markdown_provider": "mineru_v4",
        },
    )

    assert paper.metadata.title == "Tiny Title"
    assert paper.metadata.paper_key == "tiny"
    assert paper.markdown_path == md_path
    assert [section.title for section in paper.sections] == ["Tiny Title", "1 Method"]
