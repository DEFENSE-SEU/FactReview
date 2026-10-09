"""Service failures retain their exact condition scope and operational responsibility."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from assessment import assess_claims
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition, EvidenceNeed
from schemas.materials import MaterialBlock, SharedMaterials
from schemas.review import FinalReview
from verification.dispatch import verify_claims
from verification.literature import verify_literature


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("All model/retrieval/execution boundaries must be mocked")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("verification.literature.llm_json", forbidden)
    monkeypatch.setattr("verification.literature._default_adapter", forbidden)
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr(
        "review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )


def inputs(tmp_path, *, shared=False, no_id=False):
    texts = ["The graph bound holds under assumption A [1].", "The graph bound holds under assumption B [2]."]
    md = "\n".join(texts)
    file = tmp_path / "paper.md"
    file.write_text(md, encoding="utf-8")
    blocks = []
    start = 0
    for i, text in enumerate(texts, 1):
        blocks.append(
            MaterialBlock(
                id=f"b{i}", text=text, loc=ClaimLocation(page=i, char_start=start, char_end=start + len(text))
            )
        )
        start += len(text) + 1
    material = SharedMaterials(
        paper_key="service",
        title="Our new graph bound study",
        abstract="Our submission studies finite graph bounds.",
        source_pdf=str(tmp_path / "unused.pdf"),
        markdown_path=str(file),
        markdown=md,
        content_list_path="",
        provider="fixture",
        blocks=blocks,
        bibliography=[
            MaterialBlock(
                id=f"r{i}",
                text=f"[{i}] Independent bound paper {i}. 2020." + ("" if no_id else f" arXiv:2001.0000{i}"),
                loc=ClaimLocation(page=3),
            )
            for i in (1, 2)
        ],
    )
    claim = Claim(
        id="claim",
        text="The graph bound holds under the stated assumptions.",
        loc=blocks[0].loc,
        source_block_id="b1",
        source_quote=texts[0],
        source_refs=[
            {
                "source_block_id": b.id,
                "source_quote": b.text,
                "loc": b.loc,
                "covered": ["c1" if shared else f"c{i}"],
            }
            for i, b in enumerate(blocks, 1)
        ],
        conditions=[
            Condition(id=f"c{i}", description=f"Bound under assumption {'A' if i == 1 else 'B'}")
            for i in ((1,) if shared else (1, 2))
        ],
        needs=["Literature"],
    )
    return claim, material


def paper(pid):
    return {
        "id": pid,
        "arxiv_id": pid,
        "title": "An independent bound " + pid,
        "published": "2020-01-01",
        "abstract": "A bounded result.",
    }


def read_response(pid):
    return {
        "success": True,
        "provider": "fixture-reader",
        "items": [
            {
                "id": pid,
                "success": True,
                "paper": paper(pid),
                "evidence": [{"text": "The bound holds under the stated assumptions.", "page": 2}],
            }
        ],
    }


def comparison(pid="2001.00002", cid="c2", relation="supports"):
    return {
        "paper_id": pid,
        "purpose": "citation_support",
        "relation": relation,
        "quote": "The bound holds under the stated assumptions.",
        "covered": [cid],
        "fully_supported_conditions": [cid] if relation == "supports" else [],
        "mechanism": "Bound",
        "setting": "The stated assumption",
        "protocol": "Derivation",
        "note": "Original passage comparison.",
    }


def boundaries(*, lookup_failure=None, read_failure=None, search_failure=None, complete=True, rows=None):
    async def lookup(identifier):
        if identifier == "2001.00001" and lookup_failure is not None:
            if isinstance(lookup_failure, Exception):
                raise lookup_failure
            return copy.deepcopy(lookup_failure)
        return {"success": True, "provider": "fixture-metadata", "paper": paper(identifier)}

    async def read(items):
        pid = items[0]["id"]
        if pid == "2001.00001" and read_failure is not None:
            if isinstance(read_failure, Exception):
                raise read_failure
            return copy.deepcopy(read_failure)
        return read_response(pid)

    searcher = SimpleNamespace(
        search_cfg=SimpleNamespace(provider="fixture-search"),
        search=AsyncMock(
            return_value={"success": True, "provider": "fixture-search", "complete": complete, "papers": []}
        ),
        lookup_metadata=AsyncMock(side_effect=lookup),
    )
    if search_failure is not None:
        searcher.search.side_effect = search_failure if isinstance(search_failure, Exception) else None
        searcher.search.return_value = copy.deepcopy(search_failure)
    reader = SimpleNamespace(
        read_cfg=SimpleNamespace(provider="fixture-reader"), read_papers=AsyncMock(side_effect=read)
    )
    call = Mock(return_value={"status": "ok", "comparisons": rows if rows is not None else [comparison()]})
    return searcher, reader, call


async def run(tmp_path, claim, material, boundaries):
    searcher, reader, call = boundaries
    frozen = claim.model_dump(mode="json"), material.model_dump(mode="json")
    result = await verify_literature(
        claim,
        material,
        submission_deadline="2021-01-01",
        searcher=searcher,
        reader=reader,
        call=call,
        output_dir=tmp_path / "audit",
    )
    assert frozen == (claim.model_dump(mode="json"), material.model_dump(mode="json"))
    audit = json.loads((tmp_path / "audit" / "claim-search-audit.json").read_text("utf-8"))
    return result, audit


def assert_system(result, cid, operation):
    limits = result.verification_limitations
    assert any(
        limitation.condition_ids == [cid]
        and limitation.stage == "Literature"
        and limitation.kind == "source_context_unavailable"
        and limitation.responsibility == "system"
        and operation in limitation.reason
        for limitation in limits
    )
    assert all(set(limitation.condition_ids) <= {cid} for limitation in limits)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["lookup_metadata", "read_papers"])
@pytest.mark.parametrize(
    "failure",
    [TimeoutError("read timed out"), {"success": False, "provider": "fixture", "error": "HTTP 429"}],
)
async def test_failure_scoped_to_original_citation_and_healthy_support_retained(tmp_path, operation, failure):
    claim, material = inputs(tmp_path)
    calls = boundaries(**{("lookup_failure" if operation == "lookup_metadata" else "read_failure"): failure})
    result, audit = await run(tmp_path, claim, material, calls)
    assert_system(result, "c1", operation)
    assert not result.questions
    assert any(e.sufficient and e.direction == "support" and e.covered == ["c2"] for e in result.evidence)
    event = next(
        e for e in audit["context_events"] if e["operation"] == operation and e["condition_ids"] == ["c1"]
    )
    assert (
        event["category"] == "service_failure"
        and event["identifier"] == "2001.00001"
        and event["provider"]
        and event["error"]
    )
    assert event["pointer"].startswith(str((tmp_path / "audit" / "claim-search-audit.json").resolve()) + "#/")
    if isinstance(failure, dict):
        assert (
            audit["metadata_lookups" if operation == "lookup_metadata" else "reads"][0]["response"] == failure
        )
    assert calls[0].search.await_count == 3 and calls[0].lookup_metadata.await_count == 2
    assert calls[1].read_papers.await_count == (1 if operation == "lookup_metadata" else 2)
    assert calls[2].call_count == 1


@pytest.mark.asyncio
async def test_c2_read_disagreement_keeps_scoped_author_question_despite_c1_service_failure(tmp_path):
    claim, material = inputs(tmp_path)
    result, _ = await run(
        tmp_path,
        claim,
        material,
        boundaries(lookup_failure=TimeoutError("timeout"), rows=[comparison(relation="unclear")]),
    )
    assert_system(result, "c1", "lookup_metadata")
    assert len(result.questions) == 1
    assert "c2" in result.questions[0].text + result.questions[0].reason
    assert "c1" not in result.questions[0].text + result.questions[0].reason
    assert result.evidence[0].concern


@pytest.mark.asyncio
async def test_same_condition_healthy_independent_support_survives_failed_reference(tmp_path):
    claim, material = inputs(tmp_path, shared=True)
    result, _ = await run(
        tmp_path,
        claim,
        material,
        boundaries(lookup_failure=TimeoutError("timeout"), rows=[comparison(cid="c1")]),
    )
    assert_system(result, "c1", "lookup_metadata")
    assert not result.questions and result.evidence[0].sufficient
    claim.evidence = result.evidence
    claim.verification_limitations = result.verification_limitations
    assert assess_claims([claim])[0].status == "supported"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing_item", "failed_item", "identity", "empty_passages"])
async def test_read_failures_and_content_unavailability_have_distinct_audit_reasons(tmp_path, mode):
    claim, material = inputs(tmp_path)
    response = read_response("2001.00001")
    category = {
        "missing_item": "reader_protocol_failure",
        "failed_item": "service_failure",
        "identity": "identity_conflict",
        "empty_passages": "no_passage",
    }[mode]
    if mode == "missing_item":
        response["items"] = []
    elif mode == "failed_item":
        response["items"][0].update(success=False, error="reader unavailable")
    elif mode == "identity":
        response["items"][0]["paper"]["arxiv_id"] = "2002.00002"
    else:
        response["items"][0]["evidence"] = []
        response["items"][0]["paper"]["abstract"] = ""
    result, audit = await run(tmp_path, claim, material, boundaries(read_failure=response))
    assert not result.questions
    assert audit["reads"][0]["response"] == response
    assert any(e["category"] == category and e["condition_ids"] == ["c1"] for e in audit["context_events"])
    if mode in {"missing_item", "failed_item"}:
        assert_system(result, "c1", "read_papers")


@pytest.mark.asyncio
@pytest.mark.parametrize("novelty", [False, True])
async def test_search_failure_scopes_only_explicit_novelty_conditions(tmp_path, novelty):
    claim, material = inputs(tmp_path)
    if novelty:
        claim.text += " The mechanism is novel."
        claim.conditions[0].description = "The mechanism is novel relative to prior work."
    result, audit = await run(
        tmp_path,
        claim,
        material,
        boundaries(
            search_failure=TimeoutError("search unavailable"),
            rows=[comparison("2001.00001", "c1"), comparison()],
        ),
    )
    if novelty:
        assert_system(result, "c1", "search")
    else:
        assert not result.verification_limitations
    assert not result.questions and len(result.evidence) == 2
    assert all(
        e["condition_ids"] == (["c1"] if novelty else [])
        for e in audit["context_events"]
        if e["operation"] == "search"
    )


@pytest.mark.asyncio
async def test_top_n_incomplete_search_is_not_service_failure(tmp_path):
    claim, material = inputs(tmp_path)
    result, audit = await run(
        tmp_path,
        claim,
        material,
        boundaries(complete=False, rows=[comparison("2001.00001", "c1"), comparison()]),
    )
    assert not result.verification_limitations
    assert not any(e["category"] == "service_failure" for e in audit["context_events"])
    assert any("complete=true" in i for i in result.issues)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,category",
    [
        (
            {
                "success": True,
                "provider": "fixture",
                "papers": [],
                "question_results": [{"success": False, "error": "HTTP 429"}],
            },
            "service_failure",
        ),
        ({"success": True, "provider": "fixture", "papers": "malformed"}, "search_protocol_failure"),
    ],
)
async def test_failed_nested_search_or_malformed_protocol_retains_novelty_scope(tmp_path, response, category):
    claim, material = inputs(tmp_path)
    claim.text += " The mechanism is novel."
    claim.conditions[0].description = "The mechanism is novel relative to prior work."
    calls = boundaries(search_failure=response, rows=[comparison("2001.00001", "c1"), comparison()])
    result, audit = await run(tmp_path, claim, material, calls)
    assert_system(result, "c1", "search")
    assert any(
        row["category"] == category and row["condition_ids"] == ["c1"] for row in audit["context_events"]
    )
    assert all(row["response"] == response for row in audit["queries"])
    assert len(result.evidence) == 2 and not result.questions


@pytest.mark.asyncio
async def test_no_id_is_unresolved_without_service_failure_or_hidden_lookup(tmp_path):
    claim, material = inputs(tmp_path, no_id=True)
    calls = boundaries()
    result, audit = await run(tmp_path, claim, material, calls)
    assert not result.verification_limitations and result.questions
    assert any(e["category"] == "unresolved_identifier" for e in audit["context_events"])
    calls[0].lookup_metadata.assert_not_awaited()
    calls[1].read_papers.assert_not_awaited()
    calls[2].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [TimeoutError("model timeout"), {"status": "error", "error": "provider unavailable"}]
)
async def test_comparison_failure_only_affects_sent_conditions_and_retains_raw(tmp_path, response):
    claim, material = inputs(tmp_path)
    claim.conditions.append(Condition(id="background", description="Unrelated background condition"))
    calls = boundaries()
    if isinstance(response, Exception):
        calls[2].side_effect = response
    else:
        calls[2].return_value = response
    result, audit = await run(tmp_path, claim, material, calls)
    assert not result.questions
    assert {cid for limitation in result.verification_limitations for cid in limitation.condition_ids} == {
        "c1",
        "c2",
    }
    assert any(
        e["operation"] == "comparison" and e["category"] == "service_failure" for e in audit["context_events"]
    )
    if isinstance(response, dict):
        assert audit["comparison_response"] == response


@pytest.mark.asyncio
async def test_valid_empty_comparison_does_not_become_author_content_question(tmp_path):
    claim, material = inputs(tmp_path)
    result, audit = await run(tmp_path, claim, material, boundaries(rows=[]))
    assert not result.questions
    assert any(e["category"] == "comparison_unresolved" for e in audit["context_events"])
    assert not any(e["category"] == "service_failure" for e in audit["context_events"])
    assert {cid for row in result.verification_limitations for cid in row.condition_ids} == {"c1", "c2"}
    assert all(row.kind == "evidence_validation_failed" for row in result.verification_limitations)


@pytest.mark.asyncio
async def test_valid_comparison_omission_is_scoped_and_cannot_cancel_healthy_support(tmp_path):
    claim, material = inputs(tmp_path)
    claim.conditions.append(Condition(id="background", description="Unrelated background condition"))
    result, _ = await run(tmp_path, claim, material, boundaries())
    assert result.evidence[0].sufficient and result.evidence[0].covered == ["c2"]
    assert not result.questions
    assert len(result.verification_limitations) == 1
    assert result.verification_limitations[0].condition_ids == ["c1"]
    assert result.verification_limitations[0].kind == "evidence_validation_failed"


@pytest.mark.asyncio
async def test_missing_comparison_for_one_reference_does_not_hide_same_condition_support(tmp_path):
    claim, material = inputs(tmp_path, shared=True)
    result, _ = await run(tmp_path, claim, material, boundaries(rows=[comparison(cid="c1")]))
    assert result.evidence[0].sufficient and result.evidence[0].covered == ["c1"]
    assert not result.questions and not result.verification_limitations


@pytest.mark.asyncio
async def test_read_content_question_is_not_removed_by_another_failed_reference_for_same_condition(tmp_path):
    claim, material = inputs(tmp_path, shared=True)
    result, _ = await run(
        tmp_path,
        claim,
        material,
        boundaries(lookup_failure=TimeoutError("timeout"), rows=[comparison(cid="c1", relation="unclear")]),
    )
    assert_system(result, "c1", "lookup_metadata")
    assert len(result.questions) == 1
    assert "2001.00002" in result.questions[0].text and "2001.00001" not in result.questions[0].text
    assert result.evidence[0].concern


@pytest.mark.asyncio
async def test_global_lookup_service_errors_cannot_create_claim_limitations(tmp_path):
    _, material = inputs(tmp_path)
    searcher, reader, call = boundaries(search_failure=TimeoutError("search unavailable"))
    result = await verify_literature(
        None,
        material,
        submission_deadline="2021-01-01",
        searcher=searcher,
        reader=reader,
        call=call,
        output_dir=tmp_path / "global",
    )
    assert not result.questions and not result.evidence and not result.verification_limitations
    assert result.issues


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["transport", "omitted_comparison"])
async def test_dispatcher_to_advice_preserves_system_responsibility(tmp_path, cause):
    from review.report.advice import generate_advice

    claim, material = inputs(tmp_path, shared=True)
    searcher, reader, call = boundaries(rows=[])
    if cause == "transport":
        searcher.lookup_metadata.side_effect = TimeoutError("timeout")
        reader.read_papers.side_effect = TimeoutError("timeout")

    async def branch(c, m):
        return await verify_literature(
            c,
            m,
            submission_deadline="2021-01-01",
            searcher=searcher,
            reader=reader,
            call=call,
            output_dir=tmp_path / "lit",
        )

    result = await verify_claims(
        [claim], material, tmp_path / "verification", branches={EvidenceNeed.LITERATURE: branch}
    )
    checked = assess_claims(result.claims)
    assert checked[0].status == "unverified" and not checked[0].questions
    assert checked[0].verification_limitations
    original = FinalReview(paper_key="fixture", run_id="offline", claims=checked)

    def answer(action):
        def model(**kwargs):
            data = json.loads(kwargs["prompt"].split("\nADVICE_DATA_JSON:\n", 1)[1])
            refs = [
                r for r in data["basis"] if r.startswith(("/verification_limitations/", "/coverage_gaps/"))
            ]
            return {
                "status": "ok",
                "claim_id": claim.id,
                "items": [
                    {
                        "text": "Retry the failed verification service.",
                        "action": action,
                        "condition_ids": ["c1"],
                        "basis_refs": refs,
                    }
                ],
            }

        return model

    assert generate_advice(original, tmp_path / "wrong", call=answer("author_question")).counts == {
        "generated": 0,
        "unavailable": 1,
    }
    good = generate_advice(original, tmp_path / "good", call=answer("verification_followup"))
    assert good.counts == {"generated": 1, "unavailable": 0}
    assert good.review.claims[0].advice.items[0].action == "verification_followup"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["lookup_exception", "reader_error", "comparison_error", "successful_comparison"]
)
async def test_diagnostic_copies_redact_all_provider_credentials_without_changing_raw(
    mode, tmp_path, monkeypatch
):
    claim, material = inputs(tmp_path)
    calls = boundaries(rows=[comparison("2001.00001", "c1"), comparison()])
    secret1, secret2 = "fixture-search-secret-long", "fixture-model-secret-long"
    url = "https://operator:fixture-password@provider.invalid/v1?api_key=fixture-query-secret"
    calls[0].search_cfg = SimpleNamespace(provider="fixture", base_url=url, api_key=secret1)
    calls[1].read_cfg = SimpleNamespace(base_url=url, api_key=secret1)
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", url, secret2)
    )
    message = f"{secret1} {secret2} {url}"
    if mode == "lookup_exception":
        calls[0].lookup_metadata.side_effect = TimeoutError(f"{secret1} {url}")
    elif mode == "reader_error":
        calls[1].read_papers.side_effect = None
        calls[1].read_papers.return_value = {
            "success": False,
            "error": f"{secret1} {url}",
            "nested": {secret1: "value"},
        }
    elif mode == "comparison_error":
        calls[2].return_value = {"status": "error", "error": message, "nested": {secret1: "a", secret2: "b"}}
    else:
        calls[2].return_value["comparisons"][0]["note"] = message
        calls[2].return_value["nested"] = {secret1: "a", secret2: "b"}
    raw = copy.deepcopy(calls[2].return_value)
    reader_raw = copy.deepcopy(calls[1].read_papers.return_value) if mode == "reader_error" else None
    result, audit = await run(tmp_path, claim, material, calls)
    saved = json.dumps(audit) + result.model_dump_json()
    for credential in (secret1, secret2, "fixture-password", "fixture-query-secret", "operator:"):
        assert credential not in saved
    assert calls[2].return_value == raw
    if reader_raw is not None:
        assert calls[1].read_papers.return_value == reader_raw
    if mode in {"comparison_error", "successful_comparison"}:
        nested = audit["comparison_response"]["nested"]
        assert nested["_audit_redaction"] and [row["value"] for row in nested["entries"]] == ["a", "b"]
    if mode == "successful_comparison":
        assert len(result.evidence) == 2 and all(e.sufficient for e in result.evidence)
        assert result.evidence[0].pointer.quote == comparison()["quote"]
        assert "[redacted]" in result.evidence[0].note


@pytest.mark.asyncio
async def test_redacted_quote_cannot_be_used_to_validate_original_passage(tmp_path, monkeypatch):
    claim, material = inputs(tmp_path)
    calls = boundaries(rows=[comparison("2001.00001", "c1"), comparison()])
    secret = "fixture-source-key-long"
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, secret)
    )
    raw_read = read_response("2001.00001")
    raw_read["items"][0]["evidence"][0]["text"] = f"A source mentioning {secret}."
    original_read = calls[1].read_papers.side_effect

    async def read(items):
        return copy.deepcopy(raw_read) if items[0]["id"] == "2001.00001" else await original_read(items)

    calls[1].read_papers.side_effect = read
    calls[2].return_value["comparisons"][0]["quote"] = "A source mentioning [redacted]."
    result, audit = await run(tmp_path, claim, material, calls)
    assert all("c1" not in e.covered for e in result.evidence)
    assert any("ungrounded quote" in issue for issue in result.issues)
    assert (
        audit["reads"][0]["response"]["items"][0]["evidence"][0]["text"] == "A source mentioning [redacted]."
    )
    assert secret in raw_read["items"][0]["evidence"][0]["text"]
