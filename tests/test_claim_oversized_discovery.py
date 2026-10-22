"""One public-path combined control; scientific choices and every external call are mocked."""

import copy
import json
from itertools import pairwise

import pytest

from schemas.materials import MaterialBlock
from screening import claim_coverage as m
from tests.test_claim_coverage import CFG, action, candidate, followup, projection, validation
from tests.test_claim_recheck_batches import oracle, saved


def test_exact_oversized_discovery_whole_wave_sources_adoption_and_default_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External LLM forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Retrieval forbidden"))
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: pytest.fail("HTTP forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))
    body = ("The released model uses batch size 16.\n"
            + "完整原文 Unicode 👩🏽‍💻 context without an independent conclusion.\n" * 6
            + "The released model uses learning rate 0.1.")
    raw_block = {"id": "original", "text": body, "loc": {"page": 7}}
    source = MaterialBlock.model_validate(raw_block)
    original = {**candidate(source, text="The released model uses batch size 16."), "id": "claim_001"}
    missing = "The released model uses learning rate 0.1."
    window_chars = 140

    def run(mode="ok", **options):
        paper, claims = projection([raw_block], [original])
        if mode == "review_only":
            # The complete text remains visible only as a synthetic Markdown gap.
            paper.blocks = [MaterialBlock(id="other", text="Short located original.", loc={"page": 7})]
            claims = []
        elif mode == "missing_location":
            paper.blocks[0].loc = None
            claims = []
        elif mode == "bad_original_span":
            paper.blocks[0].loc.char_end -= 1
            claims = []
        elif mode == "later_window":
            paper, claims = projection([raw_block, {"id": "later", "text": "A short later passage.",
                                                    "loc": {"page": 8}}], [original])
        before_paper = paper.model_dump(mode="json")
        before_claims = [c.model_dump(mode="json") for c in claims]
        calls = []

        def call(**kw):
            p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
            calls.append((kw["module"], copy.deepcopy(p), copy.deepcopy(kw["system"])))
            module = kw["module"]
            if module.endswith("coverage_oversized_slice"):
                part, identity = p["slice"], p["source_identity"]
                assert len(part["text"]) <= window_chars
                assert part["text"] == body[part["start"]:part["end"]]
                assert identity["block_id"] == "original" and identity["source_loc"] == before_paper["blocks"][0]["loc"]
                assert p["current_claims"][0]["text"] == before_claims[0]["text"]
                assert "blocks" not in p and "scope_context" not in p and "source_catalog" not in p
                assert m._digest({k: v for k, v in p.items() if k != "context_id"}) == p["context_id"]
                assert p["slice_directory_digest"] == m._digest(p["slice_directory"])
                last = part["end"] == len(body)
                r = {"schema_version": "claim-oversized-discovery-slice-v1", "context_id": p["context_id"],
                     "window_id": p["window_id"], "parent_digest": p["parent_digest"],
                     "block_id": identity["block_id"], "block_digest": identity["block_digest"],
                     **{k: part[k] for k in ("slice_id", "start", "end", "text_sha256")},
                     "slice_directory_digest": p["slice_directory_digest"], "reviewed": True,
                     "new_findings": ([{"kind": "missing_conclusion", "source_block_ids": ["original"],
                                       "reason": "Explicit mock observes the last-slice learning-rate assertion."}]
                                      if last or mode in {"partial_wave", "duplicate"} else []),
                     "explanation": "Explicit injected slice reading and omission judgment."}
                if len([c for c in calls if c[0] == module]) == 2:
                    if mode == "partial_wave":
                        raise RuntimeError("Explicit second-slice transport failure")
                    if mode in {"context", "parent", "block", "directory", "slice", "offset", "unread", "foreign", "version", "bool_offset", "extra"}:
                        key, value = {
                            "context": ("context_id", "stale"), "parent": ("parent_digest", "stale"),
                            "block": ("block_digest", "stale"), "directory": ("slice_directory_digest", "stale"),
                            "slice": ("slice_id", "slice_001"), "offset": ("start", part["start"] - 1),
                            "bool_offset": ("start", True), "unread": ("reviewed", False),
                            "foreign": ("new_findings", [{"kind": "missing_conclusion", "source_block_ids": ["foreign"], "reason": "Foreign"}]),
                            "version": ("schema_version", "old"), "extra": ("claim_reviews", []),
                        }[mode]
                        r[key] = value
                    elif mode == "mutation":
                        paper.blocks[0].text += "changed"
                return r
            assert all(b["id"] != "slice_001" for b in p["blocks"])
            if module.endswith("coverage_first_batch"):
                assert p["blocks"][0] == before_paper["blocks"][0]
                assert p["current_claims"][0]["conditions"] == before_claims[0]["conditions"]
                raw = oracle(p)
                rows = raw["original_claim_reviews"]
                if mode == "unresolved_scope":
                    row = rows[0]
                    row.update(state="unresolved", other_findings=[], scope_atoms=[])
                    for group in row["scope_groups"]:
                        for dim in group["dimensions"].values():
                            dim.update(state="unresolved", atom_ids=[])
                return {"schema_version": "claim-coverage-first-batch-v1", "context_id": p["context_id"],
                        "window_id": p["window_id"], "parent_digest": p["parent_digest"],
                        "batch_id": p["batch_id"], "claim_reviews": rows}
            if module.endswith("coverage_followup"):
                assert p["blocks"][0] == before_paper["blocks"][0]
                assert p["observations"][0]["target_claim_id"] is None
                proposed = candidate(paper.blocks[0], text=missing)
                if mode == "bad_claim_source":
                    proposed["source_block_id"] = "foreign"
                return followup(p, [action(p, p["observations"], "append", [proposed])])
            if module.endswith("coverage_validation"):
                assert p["blocks"][0] == before_paper["blocks"][0]
                raw = validation(p)
                if mode == "reject_adoption":
                    raw["change_decisions"][0]["verdict"] = "reject_change"
                return raw
            if module.endswith("coverage_discovery"):
                return {"schema_version": "claim-coverage-discovery-v1", "context_id": p["context_id"],
                        "window_id": p["window_id"], "parent_digest": p["parent_digest"],
                        "reviewed_block_ids": [b["id"] for b in p["blocks"]], "new_findings": [],
                        "explanation": "Explicit healthy later-window judgment."}
            assert module == "screening.claims.coverage"
            raw = oracle(p)
            return {"schema_version": "claim-coverage-v5", "context_id": p["context_id"],
                    "window_id": p["window_id"], "reviewed_block_ids": [b["id"] for b in p["blocks"]],
                    "claim_reviews": raw["original_claim_reviews"], "new_findings": [],
                    "explanation": "Explicit healthy ordinary-window judgment."}

        result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path / mode,
                                       window_chars=window_chars, **options)
        audit = saved(tmp_path / mode)
        assert [c.model_dump(mode="json") for c in claims] == before_claims
        if mode != "mutation":
            assert paper.model_dump(mode="json") == before_paper
        assert audit["original_claims"] == before_claims
        assert result.coverage["budget"]["review_calls"] <= options.get("max_review_calls", 12)
        assert result.coverage["budget"]["validation_calls"] <= options.get("max_validation_calls", 12)
        return result, calls, audit, before_paper, before_claims

    result, calls, audit, paper, claims = run()
    assert result.coverage["status"] == "complete"
    assert len(result.claims) == 2 and len(result.coverage["revisions"]) == 1
    added = result.claims[-1]
    assert added.text == missing and added.source_block_id == "original" and added.source_quote == body
    assert added.loc.model_dump(mode="json") == paper["blocks"][0]["loc"]
    assert result.claims[0].model_dump(mode="json") == claims[0]
    w = audit["coverage"]["windows"][0]
    record = w["oversized_discovery"]
    assert record["complete"] is True and record["status"] == "complete"
    assert record["adoption_authorized"] is False and record["original_claim_scope_authorized"] is False
    assert record["parent_manifest"]["blocks"] == paper["blocks"]
    slices = [p for mod, p, _ in calls if mod.endswith("coverage_oversized_slice")]
    assert "".join(p["slice"]["text"] for p in slices) == body
    assert slices[0]["slice"]["start"] == 0 and slices[-1]["slice"]["end"] == len(body)
    assert all(a["slice"]["end"] == b["slice"]["start"] for a, b in pairwise(slices))
    assert all(p["slice_directory"] == slices[0]["slice_directory"] for p in slices)
    assert all(p["parent_digest"] == record["parent_digest"] for p in slices)
    assert len(record["receipts"]) == len(slices)
    assert record["path_mappings"][0]["raw_audit_path"] == record["receipts"][-1]["raw_audit_path"]
    assert record["adapter_kind"] == "program_omission_adapter_not_model_response"
    assert w["first_batches"]["raw_receipts"] if "raw_receipts" in w["first_batches"] else w["first_batches"]["aggregate"]["raw_receipts"]
    assert result.coverage["claim_checks_completed"] == result.coverage["original_claim_reviews_completed"] == 1
    assert result.coverage["candidate_claim_checks_passed"] == 1
    assert result.coverage["budget"]["review_calls"] == len(slices) + 1
    assert result.coverage["budget"]["followup_calls"] == result.coverage["budget"]["validation_calls"] == 1
    assert all(mod not in {"screening.claims.coverage", "screening.claims.coverage_discovery"} for mod, _, _ in calls)

    legacy, legacy_calls, legacy_audit, _, _ = run(
        "legacy", first_review_batch_size=None, original_review_batch_size=None)
    assert legacy.coverage["status"] == "complete" and len(legacy.claims) == 2
    assert legacy.coverage["budget"]["planned_first_claim_batches"] == 1
    assert len([mod for mod, _, _ in legacy_calls if mod.endswith("coverage_first_batch")]) == 1
    assert legacy_audit["coverage"]["windows"][0]["oversized_discovery"]["complete"] is True

    for mode in ("partial_wave", "context", "parent", "block", "directory", "slice", "offset", "bool_offset",
                 "unread", "foreign", "version", "extra", "duplicate", "mutation"):
        failed, attempted, stored, _, old = run(mode)
        assert failed.coverage["status"] != "complete" and not failed.coverage["revisions"]
        assert [c.model_dump(mode="json") for c in failed.claims] == old
        assert not any(mod.endswith(("coverage_first_batch", "coverage_followup", "coverage_validation")) for mod, _, _ in attempted)
        rw = stored["coverage"]["windows"][0]["oversized_discovery"]
        assert rw["complete"] is False and "adapter" not in rw
        assert rw["status"] == "failed" and rw["errors"]
        assert len(attempted) >= 2
        assert all(a["input"]["source_identity"]["block_id"] == "original" for a in stored["attempts"])

    for mode in ("unresolved_scope", "bad_claim_source", "reject_adoption"):
        failed, attempted, stored, _, old = run(mode)
        assert failed.coverage["status"] == "partial" and not failed.coverage["revisions"]
        assert [c.model_dump(mode="json") for c in failed.claims] == old
        assert stored["coverage"]["windows"][0]["oversized_discovery"]["complete"] is True
        if mode == "unresolved_scope":
            assert not any(mod.endswith(("coverage_followup", "coverage_validation")) for mod, _, _ in attempted)

    for mode, options in (("review_budget", {"max_review_calls": len(slices)}),
                          ("validation_budget", {"max_validation_calls": 0}),
                          ("followup_budget", {"max_followup_calls": 0}),
                          ("later_window", {"max_review_calls": len(slices) + 1})):
        limited, attempted, stored, _, _ = run(mode, **options)
        assert not any(mod.endswith("coverage_oversized_slice") for mod, _, _ in attempted)
        assert stored["coverage"]["windows"][0]["oversized_discovery"]["status"] == "not_run_budget"
        assert not limited.coverage["revisions"] and limited.coverage["status"] != "complete"

    for mode in ("review_only", "missing_location", "bad_original_span"):
        unavailable, attempted, stored, _, _ = run(mode)
        assert not any(mod.endswith("coverage_oversized_slice") for mod, _, _ in attempted)
        assert not unavailable.coverage["revisions"] and unavailable.coverage["status"] != "complete"

    healthy_paper, healthy_claims = projection(
        [{"id": "ordinary", "text": "A short located assertion.", "loc": {"page": 1}}], [])
    healthy_calls = []

    def healthy(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        healthy_calls.append((kw["module"], copy.deepcopy(p), set(kw)))
        if kw["module"].endswith("coverage_validation"):
            return validation(p)
        return {"schema_version": "claim-coverage-v5", "context_id": p["context_id"], "window_id": p["window_id"],
                "reviewed_block_ids": ["ordinary"], "claim_reviews": [], "new_findings": [],
                "explanation": "Explicit ordinary empty-omission judgment."}

    healthy_result = m.review_claim_coverage(healthy_paper, healthy_claims, call=healthy,
                                           output_dir=tmp_path / "healthy")
    assert healthy_result.coverage["status"] == "complete"
    assert [mod for mod, _, _ in healthy_calls] == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    assert all(keys == {"prompt", "system", "cfg", "module"} for _, _, keys in healthy_calls)
    assert all("oversized_discovery" not in p for _, p, _ in healthy_calls)
