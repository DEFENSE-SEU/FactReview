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


def claim_check(claim, observations=()):
    conditions = [c["id"] for c in claim["conditions"]]
    merged = any(o["kind"] == "merged_conclusions" for o in observations)
    groups = [conditions[:1], conditions[1:] or conditions[:1]] if merged else [conditions]
    return {
        "atomicity": "independent_conclusions"
        if merged
        else "shared_settings"
        if len(conditions) > 1
        else "single_conclusion",
        "assertion_groups": [
            {"proposition": "Explicit offline proposition judgment.", "condition_ids": ids} for ids in groups
        ],
        "governing_qualifiers": "missing"
        if any(o["kind"] == "missing_qualifier_or_condition" for o in observations)
        else "preserved",
        "observation_ids": [o["id"] for o in observations],
        "reason": "Explicit offline atomicity and qualifier judgment.",
    }


def review(payload, observations, *, legacy=False):
    response = {
        "schema_version": "claim-coverage-v1" if legacy else "claim-coverage-v2",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "reviewed_block_ids": [b["id"] for b in payload["blocks"]],
        "observations": copy.deepcopy(observations),
        "explanation": "Offline source-by-source judgment.",
    }
    if not legacy:
        for row in response["observations"]:
            row["sources"] = [
                {"block_id": key} for key in dict.fromkeys(s["block_id"] for s in row["sources"])
            ]
        registry = {r["claim_id"]: r for r in payload["current_claims"]}
        response["claim_checks"] = [
            {
                "claim_id": r["claim_id"],
                **claim_check(
                    registry[r["claim_id"]],
                    [o for o in observations if o["target_claim_id"] == r["claim_id"]],
                ),
            }
            for r in payload["required_claim_checks"]
        ]
    return response


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


def followup(payload, actions, *, legacy=False):
    actions = copy.deepcopy(actions)
    if not legacy:
        for row in actions:
            for claim in row["claims"]:
                claim.pop("source_quote", None)
                for ref in claim["source_refs"]:
                    ref.pop("source_quote", None)
    return {
        "schema_version": "claim-coverage-followup-v1" if legacy else "claim-coverage-followup-v2",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "actions": actions,
    }


def validation(payload, *, original_problems=True):
    """Explicit offline semantic oracle; this helper makes no model-quality claim."""
    source_ids = [b["id"] for b in payload["blocks"]]
    raw = {
        "schema_version": "claim-coverage-validation-v3",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "change_decisions": [
            {
                "candidate_id": row["candidate_id"],
                "candidate_digest": row["candidate_digest"],
                "verdict": "accept_change",
                "source_block_ids": source_ids,
                "new_claim_checks": [
                    {
                        "new_claim_index": i,
                        **claim_check(c),
                        "observation_ids": row["action"]["observation_ids"],
                    }
                    for i, c in enumerate(row["new_claims"], 1)
                ],
                "reason": "Independent mock confirms this fixed change.",
            }
            for row in payload["candidates"]
        ],
        "observation_decisions": [
            {
                "observation_id": row["observation_id"],
                "observation_digest": row["observation_digest"],
                "verdict": "confirmed",
                "source_block_ids": source_ids,
                "reason": "Independent mock confirms the original observation.",
            }
            for row in payload["observations"]
        ],
    }

    # Independent semantic judgments are injected explicitly, alongside the unchanged
    # candidate/observation verdicts. These mocks do not establish model correctness.
    raw["original_claim_reviews"], raw["observation_links"] = [], []
    required = {r["claim_id"] for r in payload["required_original_claim_reviews"]}
    for claim in payload["current_claims"]:
        if claim["claim_id"] not in required:
            continue
        index = len(raw["original_claim_reviews"])
        ids = [c["id"] for c in claim["conditions"]]
        own = (
            [r for r in payload["observations"] if r["observation"]["target_claim_id"] == claim["claim_id"]]
            if original_problems
            else []
        )
        merged = any(r["observation"]["kind"] == "merged_conclusions" for r in own)
        uncertain = any(r["observation"]["kind"] == "uncertain" for r in own)
        groups = [ids[:1], ids[1:] or ids[:1]] if merged else [ids]
        row = {
            "claim_id": claim["claim_id"],
            "claim_digest": claim["digest"],
            "state": "unresolved" if uncertain else "resolved",
            "assertion_groups": [
                {
                    "proposition": "Explicit mock assertion " + str(i),
                    "condition_ids": partition,
                    "source_block_ids": source_ids,
                }
                for i, partition in enumerate(groups)
            ],
            "findings": [],
            "preserved_qualifiers": [],
            "source_block_ids": source_ids,
            "reason": "Explicit independent original-claim semantic oracle.",
        }
        for existing in own:
            o = existing["observation"]
            if o["kind"] == "merged_conclusions":
                suffix = "/assertion_groups"
            elif o["kind"] == "uncertain":
                suffix = "/state"
            else:
                suffix = f"/findings/{len(row['findings'])}"
                finding = {
                    "kind": o["kind"],
                    "condition_ids": ids,
                    "source_block_ids": [b["block_id"] for b in o["sources"]],
                    "reason": o["reason"],
                }
                if o["kind"] == "missing_qualifier_or_condition":
                    finding.update(
                        restriction=o["reason"],
                        material_effect="Mock declares a materially different verification setting.",
                    )
                else:
                    finding["needs"] = [
                        n for n in ("Literature", "Theory", "Code", "Experiments") if n not in claim["needs"]
                    ][:1]
                row["findings"].append(finding)
            raw["observation_links"].append(
                {
                    "review_path": f"/original_claim_reviews/{index}" + suffix,
                    "observation_id": existing["observation_id"],
                    "observation_digest": existing["observation_digest"],
                    "reason": "Mock explicitly identifies the same scientific gap.",
                }
            )
        raw["original_claim_reviews"].append(row)
    return raw


def caller(observations, kind, proposals, mutate=None, *, legacy_followup=False):
    calls = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        calls.append((kw["module"], p))
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        if kw["module"] == "screening.claims.coverage":
            return review(p, copy.deepcopy(observations))
        response = followup(
            p, [action(p, observations, kind, copy.deepcopy(proposals))], legacy=legacy_followup
        )
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
    assert result.blocked_claim_ids == [] and len(calls) == 3
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
    assert len(audit["attempts"]) == 3 and audit["changes"][0]["before"] == [
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
    call, _ = caller(
        obs,
        kind,
        [] if mode == "unresolved" else [fixed],
        mutate,
        legacy_followup=mode in {"wrong_quote", "unchanged"},
    )
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
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
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
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
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
    assert len(modules) == 6 and len(result.claims) == 2 and not result.blocked_claim_ids


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
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        raw = review(p, [])
        return {k: raw[k] for k in ("context_id", "window_id", "reviewed_block_ids", "explanation")} | {
            "schema_version": "claim-coverage-v3",
            "claim_reviews": [],
            "new_findings": [],
        }

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
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        if kw["module"] == "screening.claims.coverage":
            return review(p, obs)
        invalid = candidate(paper.blocks[0], "Invalid source correction.", source_quote="Absent source")
        valid = extracted(healthy.model_dump(mode="json"))
        valid["needs"] = ["Code", "Experiments"]
        # Saved quote protocol remains strict; bad text is never promoted to a selection.
        return followup(
            p, [action(p, [obs[0]], "revise", [invalid]), action(p, [obs[1]], "revise", [valid])], legacy=True
        )

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


def test_bad_observation_isolated_with_complete_semantic_registry(tmp_path):
    paper, claims = small()
    bad = observation(paper.blocks[0])
    bad["sources"][0]["quote"] = paper.blocks[1].text
    good = observation(paper.blocks[1], "missing_conclusion", None, "good")
    original = claims[0].model_dump(mode="json")
    outputs = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        row = p["current_claims"][0]
        assert row["digest"] == m._digest(original)
        assert {
            k: row[k] for k in ("text", "conditions", "needs", "importance", "loc", "source_block_id")
        } == {k: original[k] for k in ("text", "conditions", "needs", "importance", "loc", "source_block_id")}
        assert "claim" not in row and "source_quote" not in row
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        if kw["module"] == "screening.claims.coverage":
            raw = review(p, [bad, good], legacy=True)
        else:
            assert p["observations"] == [good]
            raw = followup(p, [action(p, [good], "append", [candidate(paper.blocks[1])])])
        outputs.append(copy.deepcopy(raw))
        return raw

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.coverage["status"] == "partial" and result.blocked_claim_ids == []
    assert result.coverage["windows_partial"] == 1
    assert result.coverage["windows_reviewed"] == result.coverage["windows_unreviewed"] == 0
    assert result.coverage["unresolved_observations"] == 1 and len(result.claims) == 2
    assert result.coverage["windows"][0]["observations"][0]["state"] == "invalid"
    audit = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert audit["attempts"][0]["response"] == outputs[0]
    assert outputs[0]["observations"][0] == bad


def test_action_target_conflicts_reject_all_affected_actions_and_keep_neighbor(tmp_path):
    paper, claims = small()
    other = claims[0].model_copy(deep=True)
    other.id, other.text = "claim_009", "Another original assertion."
    claims.append(other)
    observations = [
        observation(paper.blocks[0]),
        observation(paper.blocks[0], ident="same_target"),
        observation(paper.blocks[1], "missing_needs", "claim_009", "healthy"),
    ]

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        if kw["module"] == "screening.claims.coverage":
            return review(p, observations)
        proposals = [candidate(paper.blocks[0], "Independent new wording.") for _ in range(3)]
        proposals[2] = extracted(other.model_dump(mode="json"))
        proposals[2]["needs"] = ["Code", "Experiments"]
        return followup(
            p,
            [
                action(p, [o], "revise", [proposal])
                for o, proposal in zip(observations, proposals, strict=True)
            ],
        )

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.blocked_claim_ids == ["claim_007"] and result.claims[0] == claims[0]
    assert result.claims[1].needs == ["Code", "Experiments"]
    assert len(result.coverage["windows"][0]["rejected_actions"]) == 2
    assert result.coverage["status"] == "partial"


def test_partial_review_keeps_exact_unread_ranges_and_rejects_source_borrowing(tmp_path):
    paper, claims = small()
    bad = observation(paper.blocks[1])
    good = observation(paper.blocks[0], "missing_conclusion", None, "good")

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        if kw["module"] == "screening.claims.coverage":
            raw = review(p, [bad, good])
            raw["reviewed_block_ids"] = ["b1"]
            return raw
        assert p["observations"] == [good]
        return followup(
            p, [action(p, [good], "append", [candidate(paper.blocks[0], "A different assertion.")])]
        )

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    window = result.coverage["windows"][0]
    assert window["reviewed_block_ids"] == ["b1"] and window["unreviewed_block_ids"] == ["b2"]
    assert window["status"] == "partially_reviewed" and result.coverage["status"] == "partial"
    assert result.blocked_claim_ids == [] and len(result.claims) == 2


@pytest.mark.parametrize(
    "case,mode",
    [
        ("duplicate", "decided"),
        ("duplicate", "budget"),
        ("duplicate", "service_failure"),
        ("dismiss", "decided"),
        ("dismiss", "invalid_source"),
    ],
)
def test_actual_semantic_regressions_require_independent_bound_decisions(tmp_path, case, mode):
    fixture_path = FIXTURE.with_name("claim_coverage_semantic_fixmatch.json")
    original = json.loads(fixture_path.read_text(encoding="utf-8"))[case]
    frozen = copy.deepcopy(original)
    paper, claims = projection(original["blocks"], original["claims"])
    initial = [c.model_dump(mode="json") for c in claims]
    modules = []
    validation_inputs = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kw["module"])
        if kw["module"] == "screening.claims.coverage":
            return review(p, copy.deepcopy(original["observations"]))
        if kw["module"] == "screening.claims.coverage_followup":
            # The portable projection changes target index/loc digest only.
            rows = copy.deepcopy(original["actions"])
            for row in rows:
                if row["original_claim_id"]:
                    target = next(c for c in p["current_claims"] if c["claim_id"] == row["original_claim_id"])
                    row["original_index"], row["original_digest"] = target["index"], target["digest"]
            return followup(p, rows)
        assert kw["module"] == "screening.claims.coverage_validation"
        validation_inputs.append(copy.deepcopy(p))
        assert [c.model_dump(mode="json") for c in claims] == initial
        if mode == "service_failure":
            raise RuntimeError("Independent validation service unavailable")
        raw = validation(p, original_problems=case != "dismiss")
        if case == "duplicate":
            assert len(p["candidates"]) == 2  # Both same-batch changes are visible before adoption.
            for candidate_row, decision in zip(p["candidates"], raw["change_decisions"], strict=True):
                if candidate_row["action"]["action"] == "append":
                    decision.update(
                        verdict="reject_change",
                        reason="Duplicates the prototypicality result and omits its selection and median qualifiers.",
                    )
                else:
                    assert candidate_row["old_claims"] == [extracted(initial[0])]
                    assert "all labeled data" in candidate_row["new_claims"][0]["text"]
            raw["observation_decisions"][0].update(
                verdict="dismiss_observation",
                reason="The headline is already within the existing and revised scoped result.",
            )
        else:
            assert p["current_claims"][0]["conditions"] == initial[0]["conditions"]
            for decision in raw["change_decisions"]:
                decision.update(
                    verdict="reject_change",
                    reason="The scoped table comparison need not enumerate every baseline cell.",
                )
            raw["observation_decisions"][0].update(
                verdict="dismiss_observation",
                reason="The original comparison preserves datasets, augmentation, five folds, common codebase and its table source.",
            )
        if mode == "invalid_source":
            raw["observation_decisions"][0]["source_block_ids"] = ["unprovided_block"]
        return raw

    result = m.review_claim_coverage(
        paper,
        claims,
        call=call,
        output_dir=tmp_path,
        max_validation_calls=0 if mode == "budget" else 12,
    )
    assert original == frozen and [c.model_dump(mode="json") for c in claims] == initial
    assert len(result.claims) == 1
    assert result.coverage["budget"]["validation_calls"] == (0 if mode == "budget" else 1)
    assert modules.count("screening.claims.coverage_validation") == (0 if mode == "budget" else 1)
    if mode == "decided":
        assert result.coverage["status"] == "complete", result.issues
        assert not result.blocked_claim_ids
        if case == "duplicate":
            expected = copy.deepcopy(
                next(a["claims"][0] for a in original["actions"] if a["action"] == "revise")
            )
            # Explicit v2 chooses complete original blocks. Semantic fields, IDs and
            # covered relations are unchanged; the saved v1 exact subquote remains intact.
            by_id = {b.id: b for b in paper.blocks}
            for binding in [expected, *expected["source_refs"]]:
                whole = by_id[binding["source_block_id"]].text
                assert binding["source_quote"] in whole
                binding["source_quote"] = whole
            assert extracted(result.claims[0].model_dump(mode="json")) == ExtractedClaim.model_validate(
                expected
            ).model_dump(mode="json")
        else:
            assert result.claims == claims
        audit = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
        assert audit["attempts"][-1]["input"] == validation_inputs[0]
        assert audit["attempts"][-1]["selected_source_bindings"]
    else:
        assert result.claims == claims and result.coverage["status"] == "partial"
        assert result.blocked_claim_ids == [claims[0].id]


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_whole_block_selection_preserves_math_and_legacy_bad_quote(tmp_path, legacy):
    block = MaterialBlock(
        id="math",
        text="The limit is $T \\to 0 ,$ for fixed $N$.\nThe rate stays $1/N$.",
        loc=ClaimLocation(page=1),
    )
    paper, claims = projection([block.model_dump(mode="json")])
    obs = [observation(paper.blocks[0], "missing_conclusion", None)]
    proposal = candidate(paper.blocks[0], "The stated limit is conditional on fixed N.")
    bad = "The limit is $T \\to 0 ,$ ... The rate stays $1/N$."
    seen = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        seen.append(kw["module"])
        if kw["module"] == "screening.claims.coverage":
            return review(p, obs)
        if kw["module"] == "screening.claims.coverage_followup":
            # The bad legacy raw is retained. Only a distinct explicit v2 response
            # selects a source; it contains no quote to silently normalize.
            raw = copy.deepcopy(proposal)
            raw["source_quote"] = bad
            return followup(p, [action(p, obs, "append", [raw])], legacy=legacy)
        assert all("claims" not in c["action"] for c in p["candidates"])
        return validation(p)

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    audit = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert len(seen) == 3 and not result.blocked_claim_ids
    if legacy:
        assert result.claims == [] and result.coverage["status"] == "partial"
        assert audit["attempts"][1]["response"]["actions"][0]["claims"][0]["source_quote"] == bad
    else:
        assert result.coverage["status"] == "complete" and len(result.claims) == 1
        assert result.claims[0].source_quote == block.text
        loc = result.claims[0].loc
        assert paper.markdown[loc.char_start : loc.char_end] == block.text
        assert "source_quote" not in audit["attempts"][1]["response"]["actions"][0]["claims"][0]
        bound = audit["attempts"][-1]["selected_new_claim_checks"][0]
        assert bound["new_claim_digest"] == m._digest(
            audit["attempts"][-1]["input"]["candidates"][0]["new_claims"][0]
        )


@pytest.mark.parametrize(
    "mode", ["missing", "independent_without_observation", "qualifier_without_observation"]
)
def test_required_current_claim_check_gap_is_partial_without_invented_block(tmp_path, mode):
    paper, claims = small()

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        raw = review(p, [])
        if mode == "missing":
            raw["claim_checks"] = []
        elif mode == "independent_without_observation":
            raw["claim_checks"][0]["atomicity"] = "independent_conclusions"
            raw["claim_checks"][0]["assertion_groups"] *= 2  # Reusing c1 across groups is allowed.
        else:
            raw["claim_checks"][0]["governing_qualifiers"] = "missing"
        return raw

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.claims == claims and not result.blocked_claim_ids
    assert result.coverage["status"] == "partial" and result.coverage["windows_partial"] == 1
    assert result.coverage["claim_checks_required"] == result.coverage["claim_checks_unreviewed"] == 1
    assert result.coverage["claim_checks_completed"] == result.coverage["unresolved_observations"] == 0


@pytest.mark.parametrize(
    "mode", ["joint_configuration", "legacy_missing_checks", "independent", "missing_qualifier"]
)
def test_candidate_atomicity_and_qualifiers_are_required_before_acceptance(tmp_path, mode):
    paper, claims = small()
    obs = [observation(paper.blocks[1])]
    fixed = candidate(
        paper.blocks[0],
        "The joint batch-size and learning-rate configuration applies to one 250-label split.",
        source_refs=[{"source_block_id": "b2", "source_quote": paper.blocks[1].text, "covered": ["c1"]}],
    )

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if kw["module"] == "screening.claims.coverage":
            return review(p, obs)
        if kw["module"] == "screening.claims.coverage_followup":
            return followup(p, [action(p, obs, "revise", [fixed])])
        raw = validation(p)
        row = raw["change_decisions"][0]
        if mode == "legacy_missing_checks":
            raw["schema_version"] = "claim-coverage-validation-v1"
            raw.pop("original_claim_reviews")
            raw.pop("observation_links")
            row.pop("new_claim_checks")
        elif mode == "joint_configuration":
            row["new_claim_checks"][0]["atomicity"] = "shared_settings"
        elif mode == "independent":
            row["new_claim_checks"][0]["atomicity"] = "independent_conclusions"
            row["new_claim_checks"][0]["assertion_groups"] *= 2
        else:
            row["new_claim_checks"][0]["governing_qualifiers"] = "missing"
        return raw

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.coverage["candidate_claim_checks_required"] == 1
    assert result.coverage["budget"]["validation_calls"] == 1
    if mode == "joint_configuration":
        assert result.coverage["status"] == "complete" and not result.blocked_claim_ids
        assert result.claims[0].text == fixed["text"] and len(result.claims) == 1
        assert result.coverage["candidate_claim_checks_passed"] == 1
    else:
        assert result.claims == claims and result.blocked_claim_ids == ["claim_007"]
        assert (
            result.coverage["status"] == "partial" and result.coverage["candidate_claim_checks_passed"] == 0
        )
