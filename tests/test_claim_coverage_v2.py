"""Offline coverage followup of exact FixMatch excerpts and essential failure boundaries.

These controls inject semantic decisions; they do not measure model recall or correctness.
"""

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening import claim_coverage as m
from screening.claims import ClaimExtractionOutput, ExtractedClaim, _ground_claims

FIXTURE = Path(__file__).parent / "fixtures" / "claim_coverage_fixmatch.json"
CFG = LLMConfig(
    provider="mock", model="coverage", api_key="coverage-secret-123", base_url="https://provider.invalid/v1"
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("Unexpected external request"))


def projection(blocks, raw_claims=()):
    blocks = [MaterialBlock.model_validate(copy.deepcopy(b)) for b in blocks]
    markdown, offset = "\n\n".join(b.text for b in blocks), 0
    for b in blocks:
        b.loc = ClaimLocation(
            page=b.loc.page if b.loc else 1, char_start=offset, char_end=offset + len(b.text)
        )
        offset += len(b.text) + 2
    paper = SharedMaterials(
        paper_key="coverage",
        source_pdf="",
        markdown=markdown,
        markdown_path="",
        content_list_path="",
        provider="offline",
        blocks=blocks,
    )
    claims = []
    for raw in raw_claims:
        candidate = extracted(raw)
        grounded, invalid = _ground_claims(
            ClaimExtractionOutput(status="ok", claims=[candidate]), {b.id: b for b in blocks}, markdown
        )
        assert not invalid
        grounded[0].id = raw["id"]
        claims.extend(grounded)
    return paper, claims


def extracted(raw):
    result = {key: copy.deepcopy(raw[key]) for key in ExtractedClaim.model_fields}
    for ref in result["source_refs"]:
        ref.pop("loc", None)
    return result


def fixture():
    f = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return projection(f["blocks"], f["claims"])


def candidate(block, text=None, **extra):
    return {
        "text": text or block.text,
        "source_block_id": block.id,
        "source_quote": block.text,
        "source_refs": [],
        "conditions": [{"id": "c1", "description": text or block.text}],
        "needs": ["Code"],
        "importance": "secondary",
        **extra,
    }


def small():
    blocks = [
        {"id": "b1", "text": "The method uses batch size 16 and learning rate 0.1.", "loc": {"page": 1}},
        {"id": "b2", "text": "These settings apply to one 250-label split.", "loc": {"page": 2}},
    ]
    b = MaterialBlock.model_validate(blocks[0])
    return projection(blocks, [{**candidate(b), "id": "claim_007"}])


def observation(block, kind="missing_qualifier_or_condition", target="claim_007", ident="o1"):
    return {
        "id": ident,
        "kind": kind,
        "target_claim_id": target,
        "sources": [{"block_id": block.id, "quote": block.text}],
        "reason": "Predeclared source-grounded gap.",
    }


def review(payload, observations):
    return {
        "schema_version": "claim-coverage-v1",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "reviewed_block_ids": [b["id"] for b in payload["blocks"]],
        "observations": observations,
        "explanation": "Offline source-by-source judgment.",
    }


def action(payload, obs, kind, proposals):
    target = obs[0]["target_claim_id"]
    original = next((r for r in payload["current_claims"] if r["claim_id"] == target), None)
    return {
        "observation_ids": [o["id"] for o in obs],
        "action": kind,
        "original_claim_id": target,
        "original_index": original["index"] if original else None,
        "original_digest": original["digest"] if original else None,
        "claims": proposals,
        "reason": "Explicitly adopted complete source-grounded correction.",
    }


def followup(payload, actions):
    return {
        "schema_version": "claim-coverage-followup-v1",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "actions": actions,
    }


def caller(observations, kind, proposals, mutate=None):
    calls = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        calls.append((kw["module"], p))
        if kw["module"] == "screening.claims.coverage":
            return review(p, copy.deepcopy(observations))
        response = followup(p, [action(p, observations, kind, copy.deepcopy(proposals))])
        if mutate:
            mutate(response, p)
        return response

    return call, calls


@pytest.mark.parametrize("case", ["footnote", "qualifier", "independent_settings", "epoch_definition"])
def test_real_fixmatch_followup_types_preserve_sources_ids_and_originals(tmp_path, case):
    paper, claims = fixture()
    by_id, originals = {b.id: b for b in paper.blocks}, [c.model_dump(mode="json") for c in claims]
    before = paper.model_dump(mode="json")
    if case == "footnote":
        obs = [observation(by_id["block_34"], "missing_conclusion", None)]
        proposals = [candidate(by_id["block_34"], "All labeled data is included in U without its labels.")]
        kind, target = "append", None
    elif case == "qualifier":
        obs = [observation(by_id["block_84"], target="claim_037")]
        old = next(c for c in claims if c.id == "claim_037")
        fixed = extracted(old.model_dump(mode="json"))
        for condition in fixed["conditions"]:
            condition["settings"].update(
                labeled_examples=250, split_scope="single split", augmentation="CTAugment"
            )
        fixed["source_refs"] = [
            {
                "source_block_id": "block_84",
                "source_quote": by_id["block_84"].text,
                "covered": [c["id"] for c in fixed["conditions"]],
            }
        ]
        proposals, kind, target = [fixed], "revise", "claim_037"
    elif case == "independent_settings":
        obs = [observation(by_id["block_42"], "merged_conclusions", "claim_013")]
        old = next(c for c in claims if c.id == "claim_013")
        texts = [
            "Regularization is particularly important for FixMatch.",
            "All models and experiments use weight decay.",
            "Adam performed worse than SGD with momentum.",
            "Standard and Nesterov momentum did not differ substantially.",
            "FixMatch uses cosine learning-rate decay.",
            "Final performance uses an exponential moving average of parameters.",
        ]
        proposals = [
            candidate(by_id["block_42"], t, conditions=[c.model_dump(mode="json")], needs=old.needs)
            for t, c in zip(texts, old.conditions, strict=True)
        ]
        kind, target = "split", "claim_013"
    else:
        obs = [observation(by_id["block_227"], target="claim_050")]
        old = next(c for c in claims if c.id == "claim_050")
        fixed = extracted(old.model_dump(mode="json"))
        fixed["conditions"][0]["settings"].update(
            epoch_unlabeled_examples=1200000, labeled_passes_for_10_percent_task=10
        )
        fixed["source_refs"].append(
            {"source_block_id": "block_227", "source_quote": by_id["block_227"].text, "covered": ["c1"]}
        )
        proposals, kind, target = [fixed], "revise", "claim_050"
    call, calls = caller(obs, kind, proposals)
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.coverage["status"] == "complete", result.issues
    assert result.blocked_claim_ids == [] and len(calls) == 2
    assert paper.model_dump(mode="json") == before
    assert [c.model_dump(mode="json") for c in claims] == originals
    assert result.claims is not claims
    assert {c.id for c in claims} <= {c.id for c in result.claims}
    assert all(
        c.model_dump(mode="json") in originals
        for c in result.claims
        if c.id in {x.id for x in claims} - {target}
    )
    changed = [c for c in result.claims if c.id == target or c.id not in {x.id for x in claims}]
    assert len(changed) == len(proposals)
    assert all(int(c.id.removeprefix("claim_")) > 50 for c in changed if c.id != target)
    assert [extracted(c.model_dump(mode="json")) for c in changed] == [
        ExtractedClaim.model_validate(p).model_dump(mode="json") for p in proposals
    ]
    audit = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert len(audit["attempts"]) == 2 and audit["changes"][0]["before"] == [
        c for c in originals if c["id"] == target
    ]
    assert "attempts" not in result.coverage and "response" not in m.coverage_summary(result.coverage)


@pytest.mark.parametrize(
    "mode", ["unresolved", "stale_digest", "strict_index", "wrong_quote", "unchanged", "empty_actions"]
)
def test_confirmed_unfixed_claim_is_blocked_without_losing_originals(tmp_path, mode):
    paper, claims = small()
    obs = [observation(paper.blocks[1])]
    fixed = candidate(
        paper.blocks[0],
        "Batch size 16 and learning rate 0.1 apply to one 250-label split.",
        source_refs=[{"source_block_id": "b2", "source_quote": paper.blocks[1].text, "covered": ["c1"]}],
    )

    def mutate(raw, _):
        a = raw["actions"][0]
        if mode == "stale_digest":
            a["original_digest"] = "stale"
        elif mode == "strict_index":
            a["original_index"] = True
        elif mode == "wrong_quote":
            a["claims"][0]["source_quote"] = "Invented quote."
        elif mode == "unchanged":
            a["claims"] = [candidate(paper.blocks[0])]
        elif mode == "empty_actions":
            raw["actions"] = []

    kind = "unresolved" if mode == "unresolved" else "revise"
    call, _ = caller(obs, kind, [] if mode == "unresolved" else [fixed], mutate)
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.coverage["status"] == "partial"
    assert result.blocked_claim_ids == ["claim_007"]
    assert result.claims == claims


@pytest.mark.parametrize(
    "mode", ["uncertain", "review_error", "no_service", "review_budget", "followup_budget"]
)
def test_uncertain_service_and_budget_boundaries_are_visible_and_do_not_block_all(
    tmp_path, monkeypatch, mode
):
    paper, claims = small()
    obs = [
        observation(paper.blocks[1], "uncertain" if mode == "uncertain" else "missing_qualifier_or_condition")
    ]
    call, _ = caller(obs, "unresolved", [])
    options = {}
    if mode == "no_service":
        monkeypatch.setattr(
            m, "resolve_llm_config", lambda: (_ for _ in ()).throw(RuntimeError("No service"))
        )
    elif mode == "review_error":

        def call(**_):
            raise RuntimeError("Service unavailable")
    elif mode == "review_budget":
        options["max_review_calls"] = 0
    elif mode == "followup_budget":
        options["max_followup_calls"] = 0
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path, **options)
    assert result.claims == claims
    assert result.coverage["status"] != "complete"
    assert result.blocked_claim_ids == (["claim_007"] if mode == "followup_budget" else [])
    assert result.coverage["windows_total"] == 1
    assert result.coverage["windows_reviewed"] + result.coverage["windows_unreviewed"] == 1


def test_later_window_sees_adopted_claims_and_split_inherits_earlier_unresolved_problem(tmp_path):
    paper, claims = small()
    seen = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        seen.append(p)
        first = p["window_id"] == "window_001"
        if not first:
            assert p["previous_unresolved_observations"][0]["target_claim_id"] == "claim_007"
        obs = [
            observation(
                paper.blocks[0 if first else 1],
                "missing_qualifier_or_condition" if first else "merged_conclusions",
            )
        ]
        if kw["module"] == "screening.claims.coverage":
            return review(p, obs)
        if first:
            return followup(p, [action(p, obs, "unresolved", [])])
        parts = [candidate(paper.blocks[0], text) for text in ["Batch size is 16.", "Learning rate is 0.1."]]
        return followup(p, [action(p, obs, "split", parts)])

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path, window_chars=65)
    assert result.coverage["status"] == "partial" and len(result.claims) == 2
    assert result.blocked_claim_ids == ["claim_007", "claim_008"]
    assert result.coverage["windows"][0]["observations"][0]["blocked_claim_ids"] == result.blocked_claim_ids


def test_new_claims_are_visible_to_next_review_and_followup(tmp_path):
    paper, claims = small()
    modules = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kw["module"])
        if p["window_id"] == "window_001":
            obs = [observation(paper.blocks[0], "missing_conclusion", None)]
            if kw["module"] == "screening.claims.coverage":
                return review(p, obs)
            return followup(
                p, [action(p, obs, "append", [candidate(paper.blocks[0], "A separate extracted assertion.")])]
            )
        assert [r["claim_id"] for r in p["current_claims"]] == ["claim_007", "claim_008"]
        obs = [observation(paper.blocks[1], "uncertain", "claim_008")]
        return (
            review(p, obs)
            if kw["module"] == "screening.claims.coverage"
            else followup(p, [action(p, obs, "unresolved", [])])
        )

    result = m.review_claim_coverage(paper, claims, call=call, window_chars=65, output_dir=tmp_path)
    assert len(modules) == 4 and len(result.claims) == 2 and not result.blocked_claim_ids


def test_source_change_rolls_back_and_private_diagnostics_do_not_escape(tmp_path):
    paper, claims = small()
    raw = {"error": CFG.api_key, CFG.api_key: "diagnostic"}
    result = m.review_claim_coverage(paper, claims, call=lambda **_: raw, output_dir=tmp_path)
    assert result.claims == claims and result.coverage["status"] == "failed"
    assert CFG.api_key not in (tmp_path / "coverage.json").read_text(encoding="utf-8")
    assert CFG.api_key not in str(result.issues) and raw["error"] == CFG.api_key

    paper, claims = small()
    original = [c.model_dump(mode="json") for c in claims]
    obs = [observation(paper.blocks[1])]
    call, _ = caller(
        obs,
        "revise",
        [candidate(paper.blocks[0], "New scope")],
        lambda *_: setattr(paper.blocks[0], "text", "Changed source"),
    )
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path / "changed")
    assert result.coverage["status"] == "failed" and result.coverage["adoption_rolled_back"]
    assert [c.model_dump(mode="json") for c in result.claims] == original
    assert result.blocked_claim_ids == ["claim_007"]


def test_windows_pack_headings_keep_block_only_footnotes_and_markdown_only_spans(tmp_path):
    paper, _ = small()
    for b in paper.blocks:
        b.kind = "heading"
    footnote = MaterialBlock(
        id="footnote", text="Footnote absent from markdown.", kind="page_footnote", loc=ClaimLocation(page=2)
    )
    paper.blocks.append(footnote)
    paper.markdown += "\n\nOriginal markdown-only assertion."
    observed = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        observed.extend(b["text"] for b in p["blocks"])
        return review(p, [])

    result = m.review_claim_coverage(paper, [], call=call, output_dir=tmp_path)
    assert result.coverage["status"] == "complete" and result.coverage["windows_total"] == 1
    assert footnote.text in observed and any("Original markdown-only assertion." in x for x in observed)
    assert result.coverage["review_only_block_ids"]


def test_invalid_source_action_preserves_healthy_revision_and_full_original_diff(tmp_path):
    paper, claims = small()
    healthy = claims[0].model_copy(deep=True)
    healthy.id, healthy.text = "claim_009", "A second existing implementation assertion."
    claims.append(healthy)
    obs = [observation(paper.blocks[1]), observation(paper.blocks[0], "missing_needs", "claim_009", "o2")]

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if kw["module"] == "screening.claims.coverage":
            return review(p, obs)
        invalid = candidate(paper.blocks[0], "Invalid source correction.", source_quote="Absent source")
        valid = extracted(healthy.model_dump(mode="json"))
        valid["needs"] = ["Code", "Experiments"]
        return followup(p, [action(p, [obs[0]], "revise", [invalid]), action(p, [obs[1]], "revise", [valid])])

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.blocked_claim_ids == ["claim_007"] and result.claims[0] == claims[0]
    assert result.claims[1].needs == ["Code", "Experiments"]
    assert claims[1].needs == ["Code"]
    audit = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert audit["changes"][0]["before"] == [healthy.model_dump(mode="json")]


def test_short_unlocated_page_number_cannot_cut_markdown_occurrences():
    paper, _ = small()
    paper.blocks = [MaterialBlock(id="page", kind="page_number", text="1", loc=ClaimLocation(page=1))]
    paper.markdown = "The 101 examples use 1 common configuration.\n\n<table><tr><td>11</td></tr></table>"
    extra = m._review_blocks(paper)[1:]
    assert [b.text for b in extra] == [paper.markdown]
    assert all(paper.markdown[b.loc.char_start : b.loc.char_end] == b.text for b in extra)
