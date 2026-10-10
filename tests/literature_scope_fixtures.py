"""Explicit observed paging and scientific response fixtures, with no services."""

import copy
import hashlib
import json

from util.cutoff_date import filter_papers


def native_response(query, papers, cutoff, *, exhausted=True, provider="arxiv"):
    rows = copy.deepcopy(papers)
    kept, dropped = filter_papers(rows, cutoff)
    count = len(rows)
    total = count if exhausted else count + 1
    page = {
        "index": 0, "offset": 0, "limit": 8, "status": "ok", "raw_count": count,
        "normalized_count": count, "normalization_dropped_count": 0, "budget_dropped_count": 0,
        "provider_total": total, "exhausted": exhausted,
        "response_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "request_cursor_sha256": hashlib.sha256(b"*").hexdigest() if provider == "openalex" else None,
        "next_cursor_sha256": hashlib.sha256(b"next").hexdigest() if provider == "openalex" and not exhausted else None,
        "next_offset": count if provider != "openalex" and not exhausted and count else None,
    }
    coverage = {
        "version": "search-coverage-v1", "provider": provider, "query": query,
        "translated_query": query, "endpoint": "https://retrieval.fixture/works",
        "limits": {"page_size": 8, "max_pages": 1, "max_results": 8,
                   **({"provider_result_cap": 1_000} if provider == "semantic_scholar" else {})},
        "pages": [page], "raw_count": count, "normalized_count": count,
        "retained_count": len(kept), "filtered_out_count": len(dropped), "cutoff_date": cutoff.to_metadata(),
        "exhausted": exhausted,
        "stop_reason": "provider_exhausted" if exhausted else "request_budget" if count else "provider_unknown",
    }
    reported_provider = "arxiv_fallback" if provider == "arxiv" else provider
    return {
        "success": True, "partial": False, "provider": reported_provider, "papers": kept, "count": len(kept),
        "cutoff_date": cutoff.to_metadata(), "filtered_out_count": len(dropped),
        "question_results": [{"question": query, "success": True, "partial": False,
                              "provider": reported_provider, "papers": copy.deepcopy(kept), "count": len(kept),
                              "search_coverage": coverage}],
        "search_coverage": {"version": "search-coverage-v1", "queries": [coverage]},
    }


def scientific_response(payload, comparisons=(), *, states=None):
    response = {"status": "ok", "comparisons": copy.deepcopy(list(comparisons))}
    if payload.get("search_adequacy_requested"):
        scope = payload["search_scope"]
        response["search_adequacy"] = {
            "version": "literature-search-adequacy-v1", "claim_id": payload["claim"]["id"],
            "scope_digest": scope["digest"], "source_ids": [row["paper_id"] for row in scope["sources"]],
            "conditions": [
                {"condition_id": cid, "state": (states or {}).get(cid, "adequate"),
                 "queries": [{"query_id": row["query_id"], "intent": row["intent"], "coverage": "covered",
                              "reason": "This actual query addresses the stated condition under its specified dimension."}
                             for row in scope["queries"]],
                 "mechanism": "These queries cover the stated mechanism and its known alternative operators.",
                 "setting": "The target task and domain terms address the stated problem setting.",
                 "protocol": "The protocol and baseline query addresses comparable evaluation settings.",
                 "limitations": ["Work outside the recorded provider index and query vocabulary remains unsearched."],
                 "omission_risk": "Unindexed or differently named mechanisms may remain outside this scope."}
                for cid in scope["novelty_condition_ids"]
            ],
        }
    return response
