"""The citation probe exercises production verification with offline boundaries."""

import asyncio
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials

spec = importlib.util.spec_from_file_location(
    "literature_probe", Path(__file__).resolve().parents[1] / "scripts/check_v2_literature.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

PAPER = {
    "arxiv_id": "1609.08144v2",
    "title": "Wordpiece vocabulary experiments",
    "published": "2016-09-26",
    "abstract": "An independent study of vocabulary choices for machine translation models across several language pairs.",
}
QUOTE = "We find vocabularies between 8k and 32k wordpieces useful for machine translation."
READ = {
    "success": True,
    "provider": "fixed",
    "items": [
        {
            "id": PAPER["arxiv_id"],
            "success": True,
            "paper": PAPER,
            "evidence": [{"page": 7, "text": QUOTE}],
            "reader_provider": "fixed-fulltext",
        }
    ],
}


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    probe._save(path, obj)


def _input(directory, name, *, unresolved=False, extra=False):
    text = (
        "This neural network experiment uses the cited method [AB20]."
        if unresolved
        else "This neural network experiment reports a wordpiece vocabulary range [1]."
    )
    bib = (
        "[AB20] Alpha Beta. A neural method. Proceedings, 2020."
        if unresolved
        else "[1] Wu et al. Vocabulary study. arXiv:1609.08144, 2016."
    )
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    markdown = text + "\n\n" + bib
    material = SharedMaterials(
        paper_key=name,
        title="Distinct submitted citation scope test",
        abstract="This document tests source attribution and bounds for an independently described machine translation system.",
        source_pdf="fixture:no-pdf",
        markdown=markdown,
        markdown_path=str(directory / "paper.md"),
        content_list_path="fixture:no-parser",
        provider="mock",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
        bibliography=[
            MaterialBlock(
                id="ref",
                text=bib,
                kind="ref_text",
                loc=ClaimLocation(page=2, char_start=len(text) + 2, char_end=len(markdown)),
            )
        ],
    )
    claim = Claim(
        id=name,
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[
            Condition(id="c1", description="Original wordpiece range", settings={"range": "8k to 32k"})
        ]
        + ([Condition(id="c2", description="An extra condition unrelated to the source")] if extra else []),
        needs=["Literature", "Code"],
    )
    return claim, material


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_default_adapter", lambda: pytest.fail("Unmocked retrieval"))
    monkeypatch.setattr(probe, "llm_json", lambda **kw: pytest.fail("Unmocked LLM"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Docker/process"))
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config",
        lambda: LLMConfig("mock", "model", "https://example.test", "fake-secret-value"),
    )
    cases, protected = [], {}
    for name in ("original-partial", "original-unresolved", "synthetic-positive", "synthetic-negative"):
        unresolved, synthetic = name.endswith("unresolved"), name.startswith("synthetic")
        source = tmp_path / name
        claim, material = _input(source, name, unresolved=unresolved, extra=name == "original-partial")
        positive = name == "synthetic-positive"
        expected = {
            "supported_conditions": ["c1"] if positive else [],
            "unsupported_conditions": [] if positive else [c.id for c in claim.conditions],
            "status_in": ["supported"] if positive else ["unverified", "questioned"],
        }
        expected.update(
            {"unresolved": [material.bibliography[0].text]}
            if unresolved
            else {"citations": [{"id": "1609.08144", "covered": ["c1"]}]}
        )
        case = {
            "name": name,
            "kind": "synthetic" if synthetic else "original",
            "modes": ["original-cache"] if unresolved else ["live"],
            "submission_deadline": "2021-01-01",
            "expectation": expected,
        }
        if synthetic:
            case.update(
                synthetic=True,
                claim=claim.model_dump(mode="json"),
                materials=material.model_dump(mode="json"),
            )
        else:
            materialpath, screeningpath = (
                source / "materials/materials.json",
                source / "screening/screening.json",
            )
            _json(materialpath, material.model_dump(mode="json"))
            _json(screeningpath, {"claims": [claim.model_dump(mode="json")]})
            case.update(source_run=str(source), raw_claim=claim.model_dump(mode="json"))
            protected.update({str(p): probe._hash(p) for p in [materialpath, screeningpath]})
        if unresolved:
            cache = {
                "submission_deadline": case["submission_deadline"],
                "claim_source_excerpt": claim.source_quote,
                "queries": [
                    {
                        "query": q,
                        "response": {"success": True, "provider": "fixed", "complete": False, "papers": []},
                    }
                    for q in probe.literature_queries(claim, material)
                ],
                "metadata_lookups": [],
                "reads": [],
                "comparisons": [],
            }
            cachepath = source / "verification/original-search-audit.json"
            _json(cachepath, cache)
            case["cache_path"] = str(cachepath)
            protected[str(cachepath)] = probe._hash(cachepath)
        cases.append(case)
    plan = {"version": 1, "cases": cases, "protected_files": protected}
    path = tmp_path / "plan.json"
    _json(path, plan)
    adapter = Mock()
    adapter.search = AsyncMock(
        return_value={"success": True, "provider": "fixed", "complete": False, "papers": []}
    )
    adapter.lookup_metadata = AsyncMock(return_value={"success": True, "paper": PAPER})
    adapter.read_papers = AsyncMock(return_value=READ)

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n", 1)[1])
        positive = payload["claim"]["id"] == "synthetic-positive"
        return {
            "status": "ok",
            "comparisons": [
                {
                    "paper_id": PAPER["arxiv_id"],
                    "purpose": "citation_support",
                    "relation": "supports" if positive else "partial",
                    "quote": QUOTE,
                    "covered": ["c1"],
                    "fully_supported_conditions": ["c1"] if positive else [],
                    "mechanism": "wordpiece model",
                    "setting": "machine translation",
                    "protocol": "reported vocabulary range",
                    "note": "Exact range only; other constraints are not established.",
                }
            ],
        }

    return path, plan, adapter, call, tmp_path / "output"


def run(setup, names, *, mode="live", adapter=None, call=None):
    path, _, default_adapter, default_call, out = setup
    kwargs = (
        {}
        if mode == "original-cache"
        else {"adapter": adapter or default_adapter, "call": call or default_call}
    )
    return asyncio.run(probe.run_probe(path, names, out, mode=mode, **kwargs))


def change_plan(setup, transform):
    transform(setup[1])
    _json(setup[0], setup[1])


def test_original_partial_and_separate_synthetic_positive_negative(setup):
    directory, result = run(setup, ["original-partial", "synthetic-positive", "synthetic-negative"])
    assert result["ok"] and result["source_unchanged"] and result["implementation_unchanged"]
    assert [c["assessed_status"] for c in result["cases"]] == ["unverified", "supported", "unverified"]
    assert [c["synthetic"] for c in result["cases"]] == [False, True, True]
    original = setup[1]["cases"][0]["raw_claim"]
    assert probe._load(directory / "case-001/claim.json") == original
    assert len(original["conditions"]) == 2 and original["needs"] == ["Literature", "Code"]
    assert result["training_runs"] == 0 and result["docker_calls"] == 0
    assert all(set(c["boundaries"].values()) == {"injected"} for c in result["cases"])
    assert len(list(directory.glob("case-*/request-*.json"))) == 18
    assert all(c["usage"]["token_usage"]["requests"] == 0 for c in result["cases"])


def test_configured_adapter_and_model_used_only_in_live(setup, monkeypatch):
    monkeypatch.setattr(probe, "_default_adapter", lambda: setup[2])
    monkeypatch.setattr(probe, "llm_json", setup[3])
    _, result = asyncio.run(probe.run_probe(setup[0], ["synthetic-positive"], setup[4]))
    assert result["ok"] and set(result["cases"][0]["boundaries"].values()) == {"live"}
    assert setup[2].search.call_count == 3 and setup[2].lookup_metadata.call_count == 1
    assert setup[2].read_papers.call_count == 1


def test_original_cache_is_exact_no_network_and_explicit_unresolved(setup):
    directory, result = run(setup, ["original-unresolved"], mode="original-cache")
    assert result["ok"] and set(result["cases"][0]["boundaries"].values()) == {"cached"}
    saved = probe._load(next((directory / "case-001/audit").glob("*.json")))
    assert saved["excluded"][0]["reason"] == "unresolved_identity_metadata"
    assert saved["queries"] == probe._load(Path(setup[1]["cases"][1]["cache_path"]))["queries"]
    assert saved["reads"] == [] and saved["comparisons"] == []
    assert setup[2].search.call_count == 0


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {"status": "error", "error": "offline"},
        {"status": "ok", "comparisons": []},
        {"status": "ok", "comparisons": [None]},
    ],
)
def test_empty_or_failed_model_cannot_pass_negative(setup, response):
    _, result = run(setup, ["synthetic-negative"], call=lambda **kw: response)
    assert not result["ok"] and not result["cases"][0]["expectations_passed"]


@pytest.mark.parametrize(
    "failure", ["search", "metadata", "read", "wrong_identity", "abstract_only", "empty_passages"]
)
def test_unhealthy_retrieval_or_reader_cannot_pass_negative(setup, failure):
    adapter = setup[2]
    if failure in {"search", "metadata", "read"}:
        getattr(
            adapter, {"search": "search", "metadata": "lookup_metadata", "read": "read_papers"}[failure]
        ).return_value = {"success": False, "error": "offline"}
    else:
        response = deepcopy(READ)
        if failure == "wrong_identity":
            response["items"][0]["paper"]["arxiv_id"] = "2001.00001"
        else:
            response["items"][0]["evidence"] = []
            response["items"][0]["answer"] = "" if failure == "empty_passages" else PAPER["abstract"]
        adapter.read_papers.return_value = response
    _, result = run(setup, ["synthetic-negative"])
    assert not result["ok"]


@pytest.mark.parametrize(
    "mutation",
    ["quote", "foreign", "other_paper", "wrong_purpose", "missing_dimensions", "invalid_full_flags"],
)
def test_nonempty_but_invalid_comparison_cannot_pass_negative(setup, mutation):
    def call(**kwargs):
        response = setup[3](**kwargs)
        row = response["comparisons"][0]
        if mutation == "quote":
            row["quote"] = "Not in the source"
        elif mutation == "foreign":
            row["covered"] = ["foreign"]
        elif mutation == "other_paper":
            row["paper_id"] = "2001.00001"
        elif mutation == "wrong_purpose":
            row["purpose"] = "related_work"
        elif mutation == "invalid_full_flags":
            row["fully_supported_conditions"] = "c1"
        else:
            row["setting"] = ""
        return response

    _, result = run(setup, ["synthetic-negative"], call=call)
    assert not result["ok"]


def test_cache_miss_is_failure_without_fallback(setup):
    case = setup[1]["cases"][1]
    cachepath = Path(case["cache_path"])
    cache = probe._load(cachepath)
    cache["queries"].pop()
    _json(cachepath, cache)
    change_plan(setup, lambda p: p["protected_files"].update({str(cachepath): probe._hash(cachepath)}))
    directory, result = run(setup, ["original-unresolved"], mode="original-cache")
    assert not result["ok"] and result["cases"][0]["failed_calls"] == 1
    assert any("Cache miss" in p.read_text(encoding="utf-8") for p in directory.rglob("request-*.json"))


@pytest.mark.parametrize(
    "mutation",
    ["source_hash", "raw_claim", "omit_condition", "overlap", "synthetic_label", "mode", "unknown_case"],
)
def test_bad_inputs_rejected_before_provider_calls_or_outputs(setup, mutation):
    names = ["original-partial"]
    if mutation == "source_hash":
        setup[1]["protected_files"][next(iter(setup[1]["protected_files"]))] = "0" * 64
    elif mutation == "raw_claim":
        setup[1]["cases"][0]["raw_claim"]["conditions"].pop()
    elif mutation == "omit_condition":
        setup[1]["cases"][0]["expectation"]["unsupported_conditions"].pop()
    elif mutation == "overlap":
        setup[1]["cases"][0]["expectation"]["supported_conditions"] = ["c1"]
    elif mutation == "synthetic_label":
        names = ["synthetic-positive"]
        setup[1]["cases"][2]["synthetic"] = False
    elif mutation == "mode":
        names = ["original-unresolved"]
    else:
        names = ["unknown"]
    _json(setup[0], setup[1])
    with pytest.raises(ValueError):
        run(setup, names)
    assert not setup[4].exists() and setup[2].search.call_count == 0


def test_outputs_cannot_be_added_inside_original_run(setup):
    with pytest.raises(ValueError):
        asyncio.run(
            probe.run_probe(
                setup[0],
                ["original-partial"],
                Path(setup[1]["cases"][0]["source_run"]) / "new",
                adapter=setup[2],
                call=setup[3],
            )
        )


def test_repeated_runs_preserve_originals_and_previous_records(setup):
    first, old = run(setup, ["synthetic-positive"])
    hashes = {p: probe._hash(p) for p in first.rglob("*") if p.is_file()}
    second, new = run(setup, ["synthetic-positive"])
    assert first != second and old["ok"] and new["ok"]
    assert all(probe._hash(p) == h for p, h in hashes.items())


def test_input_change_during_run_invalidates_observed_success(setup):
    source = Path(setup[1]["cases"][0]["source_run"]) / "screening/screening.json"

    def call(**kwargs):
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return setup[3](**kwargs)

    _, result = run(setup, ["synthetic-positive"], call=call)
    assert result["cases"][0]["expectations_passed"] and not result["source_unchanged"] and not result["ok"]


def test_credentials_removed_from_model_failure_and_audits(setup):
    def call(**kwargs):
        raise RuntimeError("server says fake-secret-value")

    directory, result = run(setup, ["synthetic-negative"], call=call)
    assert not result["ok"]
    assert all("fake-secret-value" not in p.read_text(encoding="utf-8") for p in directory.rglob("*.json"))
    assert "[redacted]" in next(
        p for p in directory.rglob("request-*.json") if '"kind": "model"' in p.read_text(encoding="utf-8")
    ).read_text(encoding="utf-8")


def test_interrupted_request_remains_running_and_never_claims_pass(setup):
    def call(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(setup, ["synthetic-positive"], call=call)
    saved = probe._load(next(setup[4].glob("*/summary.json")))
    assert saved["status"] == "running" and "ok" not in saved
    assert saved["cases"][0]["status"] == "running"


def test_cached_model_does_not_change_original_partial_or_full_flags():
    comparisons = [
        {
            "purpose": "citation_support",
            "relation": "partial",
            "covered": ["c1"],
            "fully_supported_conditions": [],
        }
    ]
    cache = probe.Cache({"comparisons": comparisons})
    response = cache.compare()
    assert response == {"status": "ok", "comparisons": comparisons}
    response["comparisons"][0]["fully_supported_conditions"].append("c1")
    assert comparisons[0]["fully_supported_conditions"] == []


def test_unresolved_without_the_exact_explanatory_issue_is_not_a_pass(setup, monkeypatch):
    original = probe.verify_literature

    async def missing_issue(*args, **kwargs):
        result = await original(*args, **kwargs)
        result.issues = []
        return result

    monkeypatch.setattr(probe, "verify_literature", missing_issue)
    _, result = run(setup, ["original-unresolved"], mode="original-cache")
    assert not result["ok"]


def test_implementation_change_invalidates_success_without_editing_source(setup, monkeypatch):
    original = probe._implementation
    calls = []

    def changing_snapshot():
        value = original()
        calls.append(True)
        if len(calls) > 1:
            value["changed-fixture.py"] = "0" * 64
        return value

    monkeypatch.setattr(probe, "_implementation", changing_snapshot)
    _, result = run(setup, ["synthetic-positive"])
    assert result["cases"][0]["expectations_passed"]
    assert not result["implementation_unchanged"] and not result["ok"]


def test_retrieval_diagnostics_redact_configured_endpoint_and_key(setup, monkeypatch):
    endpoint = "https://alice:transport-password@example.test/api?token=transport-token"
    monkeypatch.setenv("PAPER_READ_BASE_URL", endpoint)
    monkeypatch.setenv("PAPER_READ_API_KEY", "transport-key")
    setup[2].read_papers.side_effect = RuntimeError(endpoint + " transport-key")
    directory, result = run(setup, ["synthetic-negative"])
    assert not result["ok"]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in directory.rglob("*.json"))
    assert all(
        secret not in combined
        for secret in ("transport-password", "transport-token", "transport-key", "alice:")
    )


def test_cached_unrelated_reader_failure_is_visible_and_does_not_fake_unresolved(setup):
    case = setup[1]["cases"][1]
    cachepath = Path(case["cache_path"])
    cache = probe._load(cachepath)
    for query in cache["queries"]:
        query["response"]["papers"] = [PAPER]
    cache["reads"] = [
        {"id": PAPER["arxiv_id"], "response": {"success": False, "error": "Historical download failure"}}
    ]
    _json(cachepath, cache)
    change_plan(setup, lambda p: p["protected_files"].update({str(cachepath): probe._hash(cachepath)}))
    directory, result = run(setup, ["original-unresolved"], mode="original-cache")
    row = result["cases"][0]
    assert result["ok"] and row["failed_calls"] == 1 and row["failed_live_calls"] == 0
    assert any(r["status"] == "unsuccessful_response" and r["boundary"] == "cached" for r in row["calls"])
    assert "Historical download failure" in "\n".join(
        p.read_text(encoding="utf-8") for p in directory.rglob("request-*.json")
    )


@pytest.mark.parametrize("selected_contains_quote", [False, True])
def test_oracle_uses_only_the_reader_item_actually_sent_to_model(setup, selected_contains_quote):
    response = deepcopy(READ)
    unused = deepcopy(response["items"][0])
    if not selected_contains_quote:
        response["items"][0]["evidence"] = [{"page": 1, "text": "Only acknowledgments in the selected item."}]
    response["items"].append(unused)
    setup[2].read_papers.return_value = response
    directory, result = run(setup, ["synthetic-negative"])
    assert result["ok"] is selected_contains_quote
    saved = probe._load(directory / "case-001/result.json")
    if not selected_contains_quote:
        assert any("ungrounded quote" in issue for issue in saved["issues"])
        assert not saved["evidence"]
    requests = [probe._load(p) for p in (directory / "case-001").glob("request-*.json")]
    payload = json.loads(
        next(r for r in requests if r["kind"] == "model")["request"]["prompt"].split("\nDATA_JSON:\n", 1)[1]
    )
    actual = next(s for s in payload["sources"] if s["paper_id"] == PAPER["arxiv_id"])
    assert any(QUOTE in p["text"] for p in actual["passages"]) is selected_contains_quote


def test_cli_failure_returns_nonzero_without_automatic_live_fallback(setup, monkeypatch):
    monkeypatch.setattr(probe, "load_env_file", lambda *args: None)
    monkeypatch.setattr(probe, "_default_adapter", lambda: setup[2])
    monkeypatch.setattr(probe, "llm_json", lambda **kw: {"status": "ok", "comparisons": []})
    assert (
        probe.main(
            [
                "--plan",
                str(setup[0]),
                "--case",
                "synthetic-negative",
                "--mode",
                "live",
                "--output-root",
                str(setup[4]),
            ]
        )
        == 1
    )


def test_falsey_injected_boundaries_never_fall_back_to_live(setup):
    class FalseyAdapter:
        search = setup[2].search
        lookup_metadata = setup[2].lookup_metadata
        read_papers = setup[2].read_papers

        def __bool__(self):
            return False

    class FalseyModel:
        def __bool__(self):
            return False

        def __call__(self, **kwargs):
            return setup[3](**kwargs)

    _, result = asyncio.run(
        probe.run_probe(
            setup[0], ["synthetic-positive"], setup[4], adapter=FalseyAdapter(), call=FalseyModel()
        )
    )
    assert result["ok"]
    assert set(result["cases"][0]["boundaries"].values()) == {"injected"}
