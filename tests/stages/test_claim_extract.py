"""Claim-extract stage tests.

Tests the public ``extract_facts`` entry point that the stage_runner ends up
delegating to. The three cases pin down the contract that downstream stages
rely on: heuristic mode produces the four claim types from the body text,
LLM mode parses LLM JSON into ``Claim`` objects, and auto mode falls back to
heuristics when the LLM call fails.
"""

from __future__ import annotations

from preprocessing.claim_extract.extractor import extract_facts
from schemas.claim import ClaimType
from schemas.config import ClaimExtractCfg, LLMCfg


def test_heuristic_mode_extracts_all_claim_types(tiny_paper) -> None:
    result = extract_facts(tiny_paper, cfg=ClaimExtractCfg(mode="heuristic"))

    assert result.backend == "heuristic"
    assert result.claims, "heuristic must surface at least one claim from tiny_paper"
    types = {c.type for c in result.claims}
    # tiny_paper's intro embeds one trigger sentence per claim type so the
    # heuristic's classification stays observable from the public API.
    assert ClaimType.METHODOLOGICAL in types
    assert ClaimType.EMPIRICAL in types
    assert ClaimType.THEORETICAL in types
    assert ClaimType.REPRODUCIBILITY in types

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

    def fake_call(paper, reported, llm_cfg):  # type: ignore[no-untyped-def]
        captured["paper_key"] = paper.metadata.paper_key
        return ext_mod._parse_llm_claims(
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
                    "text": "Code is open-source.",
                    "type": "reproducibility",
                    "location": {"section_id": "sec_1"},
                },
            ]
        )

    monkeypatch.setattr(ext_mod, "_call_llm_for_claims", fake_call)

    result = extract_facts(
        tiny_paper, cfg=ClaimExtractCfg(mode="llm", decompose_broad_claims=False), llm_cfg=LLMCfg()
    )

    assert result.backend == "llm"
    assert captured["paper_key"] == "tiny"
    assert [c.id for c in result.claims] == ["claim_01", "claim_02"]
    assert result.claims[0].type is ClaimType.EMPIRICAL
    assert result.claims[1].type is ClaimType.REPRODUCIBILITY


def test_auto_mode_falls_back_to_heuristic_when_llm_fails(tiny_paper, monkeypatch) -> None:
    from preprocessing.claim_extract import extractor as ext_mod

    def boom(paper, reported, llm_cfg):  # type: ignore[no-untyped-def]
        return None  # simulates llm.client raising / returning bad shape

    monkeypatch.setattr(ext_mod, "_call_llm_for_claims", boom)

    result = extract_facts(
        tiny_paper,
        cfg=ClaimExtractCfg(mode="auto", decompose_broad_claims=False),
        llm_cfg=LLMCfg(),
    )

    assert result.backend == "auto:heuristic-fallback"
    assert result.claims, "fallback must still surface the heuristic claims"
