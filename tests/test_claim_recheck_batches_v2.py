"""Three public-path groups. All scientific judgments and external calls are mocked."""

import copy
import json

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG, action, extracted, followup, validation
from tests.test_claim_coverage_v3 import case


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External LLM forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))


def inputs(count):
    paper, originals = case()
    claims = [originals[0].model_copy(deep=True, update={"id": f"claim_{i + 1:03d}"}) for i in range(count)]
    return paper, claims


def oracle(payload, merge=None):
    p = copy.deepcopy(payload)
    if "required_original_claim_reviews" not in p:
        p["required_original_claim_reviews"] = p["required_claim_checks"]
    p.setdefault("candidates", [])
    p.setdefault("observations", [])
    raw = validation(p)
    if merge is not None:
        row = next(r for r in raw["original_claim_reviews"] if r["claim_id"] == merge)
        groups = row["assertion_groups"]
        row["assertion_groups"] = [
            {**copy.deepcopy(groups[0]), "condition_ids": ["c1"]},
            {**copy.deepcopy(groups[0]), "condition_ids": ["c2"]},
        ]
    return raw


def callback(calls, *, merge=None, corrupt=None):
    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        calls.append((kw["module"], copy.deepcopy(p)))
        if kw["module"] == "screening.claims.coverage":
            raw = oracle(p, merge=merge)
            return {
                "schema_version": "claim-coverage-v5",
                "context_id": p["context_id"],
                "window_id": p["window_id"],
                "reviewed_block_ids": [b["id"] for b in p["blocks"]],
                "claim_reviews": raw["original_claim_reviews"],
                "new_findings": [],
                "explanation": "Explicit injected full-source scientific choices.",
            }
        if kw["module"] == "screening.claims.coverage_followup":
            original = next(c for c in p["current_claims"] if c["claim_id"] == merge)
            # Use source-bound selected-claim fields, never manufacture an author quote.
            _paper, original_claims = case()
            rows = []
            for i, text in enumerate(["The method reaches accuracy 90.", "The method uses batch size 16."]):
                row = extracted(original_claims[0].model_dump(mode="json"))
                row.update(text=text, conditions=[row["conditions"][i]])
                rows.append(row)
            assert original["conditions"] == [
                c.model_dump(mode="json") for c in original_claims[0].conditions
            ]
            return followup(p, [action(p, p["observations"], "split", rows)])
        if kw["module"] == "screening.claims.coverage_original_batch":
            raw = oracle(p)
            raw = {
                key: raw[key]
                for key in ("context_id", "window_id", "original_claim_reviews", "observation_links")
            }
            raw.update(
                schema_version="claim-original-recheck-batch-v1",
                parent_digest=p["parent_digest"],
                batch_id=p["batch_id"],
            )
            if corrupt:
                corrupt(raw, p, len([c for c in calls if c[0] == kw["module"]]))
            return raw
        assert kw["module"] == "screening.claims.coverage_batch_decisions"
        raw = oracle(p)
        raw = {
            key: raw[key] for key in ("context_id", "window_id", "change_decisions", "observation_decisions")
        }
        raw.update(
            schema_version="claim-batch-decisions-v1",
            parent_digest=p["parent_digest"],
            aggregate_digest=p["aggregate_digest"],
        )
        if corrupt:
            corrupt(raw, p, 0)
        return raw

    return call


def saved(path):
    return json.loads((path / "coverage.json").read_text(encoding="utf-8"))


def test_complete_23_obligations_exact_sources_conditions_and_default_compatibility(tmp_path):
    paper, claims = inputs(23)
    before = [c.model_dump(mode="json") for c in claims]
    calls = []
    result = m.review_claim_coverage(
        paper, claims, original_review_batch_size=3, call=callback(calls), output_dir=tmp_path / "batches"
    )
    batches = [p for module, p in calls if module.endswith("coverage_original_batch")]
    assert [len(p["required_original_claim_reviews"]) for p in batches] == [3] * 7 + [2]
    assert [r["claim_id"] for p in batches for r in p["required_original_claim_reviews"]] == [
        c.id for c in claims
    ]
    first = calls[0][1]
    expanded = m.unpack_scope_payload(first)
    for batch in batches:
        assert batch["review_only_block_ids"] == first["review_only_block_ids"]
        assert (
            batch["blocks"] == first["blocks"]
            and batch["visible_source_spans"] == first["visible_source_spans"]
        )
        assert "candidates" not in batch and "previous_unresolved_observations" not in batch
        b = m.unpack_scope_payload(batch)
        ids = [r["claim_id"] for r in b["required_original_claim_reviews"]]
        assert b["current_claims"] == [r for r in first["current_claims"] if r["claim_id"] in ids]
        assert b["scope_context"] == [r for r in expanded["scope_context"] if r["claim_id"] in ids]
        assert all(len(r["conditions"]) == 2 for r in b["current_claims"])
    assert len(calls) == 9 and result.coverage["budget"]["validation_calls"] == 8
    assert result.coverage["status"] == "complete" and not result.coverage["revisions"]
    window = result.coverage["windows"][0]
    assert window["original_recheck"]["completed"] == window["original_recheck"]["required"] == 23
    assert window["original_batches"]["planned_validation_calls"] == 8
    assert window["original_batches"]["remaining_required_claim_ids"] == []
    audit = saved(tmp_path / "batches")
    assert audit["coverage"]["windows"][0]["original_batches"]["parent_manifest"]["parent_provenance"]["required_original_digest"] == m._digest(
        [r for p in batches for r in p["required_original_claim_reviews"]]
    )
    assert audit["original_claims"] == audit["result_claims"] == before
    assert all("response_digest" in a for a in audit["attempts"][1:])
    defaults = []

    def old_call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        defaults.append(kw["module"])
        if kw["module"] == "screening.claims.coverage":
            return callback([])(**kw)
        return oracle(p)

    legacy_default = m.review_claim_coverage(paper, claims, call=old_call, output_dir=tmp_path / "default")
    assert defaults == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    assert legacy_default.coverage["status"] == "complete"


def test_failure_budget_and_invalid_batches_preserve_partial_and_zero_adoption(tmp_path):
    for mode in (
        "budget",
        "zero_budget",
        "invalid_link",
        "missing",
        "duplicate",
        "foreign",
        "digest",
        "context",
        "parent",
        "unresolved",
        "service",
    ):
        paper, claims = inputs(5)
        before = [c.model_dump(mode="json") for c in claims]
        calls = []

        def corrupt(raw, p, index):
            if index != 2:
                return
            if mode == "missing":
                raw["original_claim_reviews"].pop()
            elif mode == "duplicate":
                raw["original_claim_reviews"][1] = copy.deepcopy(raw["original_claim_reviews"][0])
            elif mode == "foreign":
                raw["original_claim_reviews"][0]["claim_id"] = "foreign"
            elif mode == "digest":
                raw["original_claim_reviews"][0]["claim_digest"] = "stale"
            elif mode == "context":
                raw["context_id"] = "stale"
            elif mode == "parent":
                raw["parent_digest"] = "stale"
            elif mode == "invalid_link":
                raw["observation_links"][0]["review_path"] = "/original_claim_reviews/99/assertion_groups"
            elif mode == "unresolved":
                row = raw["original_claim_reviews"][0]
                row["state"] = "unresolved"
                for group in row["scope_groups"]:
                    for dim in group["dimensions"].values():
                        dim.update(state="unresolved", atom_ids=[])
            elif mode == "service":
                raise RuntimeError("mock unavailable")

        result = m.review_claim_coverage(
            paper,
            claims,
            original_review_batch_size=2,
            max_validation_calls=0 if mode == "zero_budget" else 1 if mode == "budget" else 12,
            call=callback(calls, merge="claim_004", corrupt=corrupt),
            output_dir=tmp_path / mode,
        )
        assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
        assert [c.model_dump(mode="json") for c in result.claims] == before
        assert not any(module.endswith("coverage_batch_decisions") for module, _ in calls)
        assert result.coverage["budget"]["validation_calls"] <= (1 if mode == "budget" else 12)
        window = result.coverage["windows"][0]
        if mode == "invalid_link":
            assert window["original_recheck"]["completed"] == 5
            assert window["original_batches"]["aggregate"]["aggregation_errors"]
            assert window["original_recheck"]["findings"]
        else:
            assert window["original_recheck"]["completed"] < 5
            assert window["original_batches"]["remaining_required_claim_ids"]
        assert window["original_batches"]["planned_original_batches"] == 3
        assert window["original_batches"]["planned_validation_calls"] == 4
        if mode == "zero_budget":
            assert result.coverage["budget"]["validation_calls"] == 0
            assert len(window["original_batches"]["remaining_required_claim_ids"]) == 5
        assert saved(tmp_path / mode)["original_claims"] == before

    paper, claims = inputs(12)
    before = [c.model_dump(mode="json") for c in claims]
    calls = []
    invalid_pointer = "/original_claim_reviews/1０/assertion_groups"

    def unicode_index(raw, p, index):
        if index == 1:
            raw["observation_links"][0]["review_path"] = invalid_pointer

    result = m.review_claim_coverage(
        paper,
        claims,
        original_review_batch_size=12,
        call=callback(calls, merge="claim_011", corrupt=unicode_index),
        output_dir=tmp_path / "unicode_index",
    )
    assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
    assert [c.model_dump(mode="json") for c in result.claims] == before
    assert not any(module.endswith("coverage_batch_decisions") for module, _ in calls)
    window = result.coverage["windows"][0]
    assert window["original_recheck"]["completed"] == 12
    assert window["original_batches"]["aggregate"]["aggregation_errors"]
    assert window["original_recheck"]["findings"]
    assert "claim_011" in result.blocked_claim_ids
    audit = saved(tmp_path / "unicode_index")
    raw = next(a["response"] for a in audit["attempts"] if a.get("response", {}).get("schema_version") == "claim-original-recheck-batch-v1")
    assert raw["observation_links"][0]["review_path"] == invalid_pointer


def test_cross_batch_links_global_decisions_and_all_window_adoption_gate(tmp_path):
    for mode in ("ok", "missing_decision", "unresolved_decision", "stale_aggregate", "bad_link"):
        paper, claims = inputs(5)
        calls = []

        def corrupt(raw, p, index):
            if mode == "bad_link" and index == 2:
                raw["observation_links"][0]["review_path"] = "/original_claim_reviews/0/assertion_groups"
            if index != 0:
                return
            if mode == "missing_decision":
                raw["observation_decisions"] = []
            elif mode == "unresolved_decision":
                raw["observation_decisions"][0]["verdict"] = "unresolved"
            elif mode == "stale_aggregate":
                raw["aggregate_digest"] = "stale"

        result = m.review_claim_coverage(
            paper,
            claims,
            original_review_batch_size=2,
            call=callback(calls, merge="claim_004", corrupt=corrupt),
            output_dir=tmp_path / mode,
        )
        decision = [p for module, p in calls if module.endswith("coverage_batch_decisions")]
        assert len(decision) == 1 and len(decision[0]["current_claims"]) == 5
        assert len(decision[0]["candidates"]) == 1 and len(decision[0]["observations"]) == 1
        audit = saved(tmp_path / mode)
        aggregate = result.coverage["windows"][0]["original_batches"]["aggregate"]
        assert m._digest(aggregate) == decision[0]["aggregate_digest"]
        assert aggregate["observation_links"][0]["review_path"] == (
            "/original_claim_reviews/2/assertion_groups"
            if mode == "bad_link"
            else "/original_claim_reviews/3/assertion_groups"
        )
        assert aggregate["context_id"] == decision[0]["aggregate_context_id"]
        assert audit["attempts"][-1]["response"]["context_id"] == decision[0]["context_id"]
        assert aggregate["context_id"] != decision[0]["context_id"]
        if mode == "ok":
            assert len(result.claims) == 6 and len(result.coverage["revisions"]) == 1
            assert result.coverage["status"] == "complete" and not result.blocked_claim_ids
            recheck = result.coverage["windows"][0]["original_recheck"]
            assert recheck["findings"][0]["linked_observation"]
            assert recheck["completed"] == 5 and not recheck["errors"]
            assert set(recheck["lowering"]["observation_origins"]) == set(
                aggregate["generated_observation_ids"]
            )
        else:
            assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
            assert len(result.claims) == 5 and result.blocked_claim_ids == ["claim_004"]
