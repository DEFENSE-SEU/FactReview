"""Default adaptive output partitions; scientific reviews use explicit external mocks."""

import copy

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage import CFG
from tests.test_claim_first_batches import first_callback
from tests.test_claim_recheck_batches import inputs, oracle, saved


def test_default_auto_batches_large_windows_preserving_small_legacy_and_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External LLM forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: pytest.fail("HTTP forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))
    paper, claims = inputs(4)
    original_paper = paper.model_dump(mode="json")
    original_claims = [c.model_dump(mode="json") for c in claims]

    def run(name, selected_claims, **options):
        calls = []
        injected = first_callback(calls)

        def call(**kw):
            # The existing bulk validation still receives its original v5 envelope.
            if kw["module"].endswith("coverage_validation"):
                import json

                payload = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
                calls.append((kw["module"], copy.deepcopy(payload), {}))
                return oracle(payload)
            return injected(**kw)

        path = tmp_path / name
        result = m.review_claim_coverage(paper, selected_claims, call=call, output_dir=path, **options)
        audit = saved(path)
        assert paper.model_dump(mode="json") == original_paper
        assert [c.model_dump(mode="json") for c in claims] == original_claims
        assert audit["original_claims"] == audit["result_claims"]
        return result, calls, audit

    healthy, calls, audit = run("auto", claims)
    assert [module for module, _, _ in calls] == [
        "screening.claims.coverage_discovery", "screening.claims.coverage_first_batch",
        "screening.claims.coverage_first_batch", "screening.claims.coverage_original_batch",
        "screening.claims.coverage_original_batch",
    ]
    batches = [payload for module, payload, _ in calls if module.endswith("coverage_first_batch")]
    assert [len(p["required_claim_checks"]) for p in batches] == [3, 1]
    assert [r["claim_id"] for p in batches for r in p["required_claim_checks"]] == [c.id for c in claims]
    discovery = calls[0][1]
    assert all(p["current_claims"] == discovery["current_claims"] and p["blocks"] == discovery["blocks"]
               and p["scope_context"] == discovery["scope_context"] for p in batches)
    assert healthy.coverage["status"] == "complete"
    assert healthy.coverage["claim_checks_required"] == healthy.coverage["claim_checks_completed"] == 4
    assert healthy.coverage["original_claim_reviews_required"] == healthy.coverage["original_claim_reviews_completed"] == 4
    assert healthy.coverage["budget"]["review_calls"] == 3
    assert healthy.coverage["budget"]["validation_calls"] == 2
    assert len(audit["attempts"]) == len(calls)

    for name, selected, options in (
        ("small", claims[:1], {}),
        ("legacy", claims, {"first_review_batch_size": None, "original_review_batch_size": None}),
    ):
        result, requested, _ = run(name, selected, **options)
        assert [module for module, _, _ in requested] == [
            "screening.claims.coverage", "screening.claims.coverage_validation",
        ]
        assert result.coverage["status"] == "complete"

    capped, requested, capped_audit = run("capped", claims, max_review_calls=2)
    assert [module for module, _, _ in requested] == [
        "screening.claims.coverage_discovery", "screening.claims.coverage_first_batch",
    ]
    assert capped.coverage["status"] == "partial" and not capped.coverage["revisions"]
    assert capped.coverage["budget"]["review_calls"] == 2 and capped.coverage["budget"]["validation_calls"] == 0
    assert capped.coverage["windows"][0]["first_batches"]["remaining_required_claim_ids"]
    assert capped.coverage["windows"][0]["original_recheck"]["status"] == "not_run_incomplete_first"
    assert len(capped_audit["attempts"]) == len(requested)

    limited, requested, _ = run("validation-capped", claims, max_validation_calls=1)
    assert limited.coverage["status"] == "partial" and not limited.coverage["revisions"]
    assert limited.coverage["budget"]["validation_calls"] == 1
    assert len([mod for mod, _, _ in requested if mod.endswith("coverage_original_batch")]) == 1
    assert limited.coverage["original_claim_reviews_completed"] < limited.coverage["original_claim_reviews_required"]

    import sys

    from pipeline_full import parse_args

    for flags, expected in (([], "auto"), (["--claim-coverage-first-batch-size", "legacy",
                                           "--claim-coverage-original-batch-size", "legacy"], None),
                            (["--claim-coverage-first-batch-size", "2",
                              "--claim-coverage-original-batch-size", "2"], 2)):
        monkeypatch.setattr(sys, "argv", ["review", "paper.pdf", *flags])
        args = parse_args()
        assert args.claim_coverage_first_batch_size == args.claim_coverage_original_batch_size == expected
