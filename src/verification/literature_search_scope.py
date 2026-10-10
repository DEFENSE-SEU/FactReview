"""Bind scientific search judgments to conservatively checked observed facts."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

_PROVIDERS = {"arxiv": "arxiv_fallback", "openalex": "openalex", "semantic_scholar": "semantic_scholar"}
_HASH = re.compile(r"[a-f0-9]{64}")


def _integer(value):
    return type(value) is int and value >= 0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _excerpt_snapshot(row):
    return {**{key: deepcopy(value) for key, value in row.items() if key != "source_quote"},
            "source_quote_sha256": hashlib.sha256(row["source_quote"].encode()).hexdigest(),
            "source_quote_characters": len(row["source_quote"])}


def _identity_snapshot(paper):
    return {key: deepcopy(paper[key]) for key in (
        "id", "arxiv_id", "doi", "title", "url", "abs_url", "pdf_url", "published", "updated", "year"
    ) if key in paper}


def _native_query(query, response, cutoff):
    """Check the current producer's complete page chain, never remote flags."""
    failures = []
    coverage = response.get("search_coverage")
    scopes = coverage.get("queries") if isinstance(coverage, dict) else None
    if not isinstance(scopes, list) or len(scopes) != 1 or coverage.get("version") != "search-coverage-v1":
        return False, ["unknown observed native scope"], None
    scope = scopes[0]
    if not isinstance(scope, dict):
        return False, ["malformed observed native scope"], scope
    provider = scope.get("provider")
    grouped = response.get("question_results")
    papers = response.get("papers")
    if (
        not isinstance(provider, str) or provider not in _PROVIDERS or response.get("provider") != _PROVIDERS.get(provider)
        or scope.get("version") != "search-coverage-v1" or scope.get("query") != query
        or not isinstance(scope.get("translated_query"), str) or not scope["translated_query"].strip()
        or not isinstance(scope.get("endpoint"), str) or not scope["endpoint"]
        or response.get("success") is not True or response.get("partial") is not False
        or any(response.get(key) for key in ("error", "truncated", "has_more"))
        or not isinstance(papers, list) or not all(isinstance(row, dict) for row in papers)
        or not _integer(response.get("count")) or response["count"] != len(papers)
        or not isinstance(grouped, list) or len(grouped) != 1
    ):
        failures.append("query/provider/result binding is incomplete")
    else:
        row = grouped[0]
        if (
            not isinstance(row, dict) or row.get("question") != query or row.get("success") is not True
            or row.get("partial") is not False or row.get("error")
            or row.get("provider") != response["provider"] or row.get("papers") != papers
            or not _integer(row.get("count")) or row["count"] != len(papers) or row.get("search_coverage") != scope
        ):
            failures.append("grouped query binding is incomplete")
    if not isinstance(provider, str) or provider not in _PROVIDERS:
        return False, failures, scope
    limits, pages = scope.get("limits"), scope.get("pages")
    if (
        not isinstance(limits, dict) or any(not _integer(limits.get(key)) or not limits[key]
                                          for key in ("page_size", "max_pages", "max_results"))
        or not isinstance(pages, list) or not pages or len(pages) > limits["max_pages"]
        or (provider == "semantic_scholar" and limits.get("provider_result_cap") != 1000)
    ):
        return False, [*failures, "page limits or observations are unavailable"], scope
    consumed, totals, cursor, seen_cursors = 0, set(), hashlib.sha256(b"*").hexdigest(), set()
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            failures.append("malformed page")
            break
        limit = min(limits["page_size"], limits["max_results"] - consumed,
                    16 if provider == "arxiv" else 100,
                    1000 - consumed if provider == "semantic_scholar" else limits["page_size"])
        count, total = page.get("raw_count"), page.get("provider_total")
        if (
            type(page.get("index")) is not int or page["index"] != index
            or type(page.get("offset")) is not int or page["offset"] != consumed
            or type(page.get("limit")) is not int or page["limit"] != limit or limit <= 0
            or page.get("status") != "ok" or page.get("error")
            or not _integer(count) or count > limit or not _integer(total)
            or page.get("normalized_count") != count
            or type(page.get("normalized_count")) is not int
            or type(page.get("normalization_dropped_count")) is not int or page["normalization_dropped_count"] != 0
            or type(page.get("budget_dropped_count")) is not int or page["budget_dropped_count"] != 0
            or not isinstance(page.get("response_sha256"), str) or not _HASH.fullmatch(page["response_sha256"])
        ):
            failures.append("page continuity/count/normalization facts are incomplete")
            break
        consumed += count
        totals.add(total)
        terminal = index == len(pages) - 1
        if total < consumed or len(totals) != 1 or page.get("exhausted") is not terminal:
            failures.append("page totals or exhaustion disagree")
        if terminal and total != consumed:
            failures.append("terminal page does not consume the observed total")
        if provider == "openalex":
            if page.get("request_cursor_sha256") != cursor or cursor in seen_cursors:
                failures.append("cursor chain disagrees")
            seen_cursors.add(cursor)
            next_cursor = page.get("next_cursor_sha256")
            if (terminal and next_cursor is not None) or (not terminal and (
                not isinstance(next_cursor, str) or not _HASH.fullmatch(next_cursor) or not count
            )) or page.get("next_offset") is not None:
                failures.append("cursor termination disagrees")
            cursor = next_cursor
        elif page.get("next_cursor_sha256") is not None or page.get("request_cursor_sha256") is not None or (
            page.get("next_offset") is not None if terminal
            else type(page.get("next_offset")) is not int or page["next_offset"] != consumed or not count
        ):
            failures.append("offset continuation disagrees")
    retained = len(papers) if isinstance(papers, list) else None
    filtered = scope.get("filtered_out_count")
    if (
        scope.get("exhausted") is not True or scope.get("stop_reason") != "provider_exhausted"
        or not _integer(scope.get("raw_count")) or scope["raw_count"] != consumed
        or not _integer(scope.get("normalized_count")) or scope["normalized_count"] != consumed
        or not _integer(scope.get("retained_count")) or scope["retained_count"] != retained
        or not _integer(filtered) or retained is None or filtered + retained != consumed
        or scope.get("cutoff_date") != cutoff or response.get("cutoff_date") != cutoff
        or not _integer(response.get("filtered_out_count")) or response["filtered_out_count"] != filtered
    ):
        failures.append("observed scope totals/cutoff/exhaustion are incomplete")
    return not failures, failures, scope


def build_search_scope(*, claim, queries, intents, query_records, cutoff, concurrent_start,
                       sources, source_excerpts, self_exclusion, source_guards, excluded, reads, novelty_ids,
                       public_copy=deepcopy):
    rows = []
    for index, query in enumerate(queries):
        response = query_records[index]["response"] if index < len(query_records) else {}
        valid, reasons, observed = _native_query(query, response, cutoff)
        if index >= len(query_records) or query_records[index].get("query") != query:
            valid = False
            reasons.append("actual query record disagrees with plan")
        rows.append({"query_id": f"q{index + 1}", "intent": intents[index], "query": query,
                     "native_exhausted": valid, "reasons": reasons, "observed": deepcopy(observed)})
    native = len(rows) == len(query_records) == len(intents) == 3 and len({row["query"] for row in rows}) == 3 and all(
        row["native_exhausted"] for row in rows
    )
    raw_zero = native and all(row["observed"]["raw_count"] == 0 for row in rows)
    # The comparison payload already supplies passage text. Bind that exact
    # text through hashes here rather than doubling full-text prompt cost.
    source_snapshot = [
        {**{key: deepcopy(row[key]) for key in ("paper_id", "period", "cited", "citation_condition_ids",
                                              "full_text", "reader_identity_verified", "in_bibliography")},
         "source_sha256": _digest(row), "paper": _identity_snapshot(row["paper"]),
         "citation_sources": [_excerpt_snapshot(item) for item in row["citation_sources"]],
         "passages": [{"page": passage.get("page"), "source": passage.get("source"),
                       "text_sha256": hashlib.sha256(passage["text"].encode()).hexdigest(),
                       "characters": len(passage["text"])} for passage in row["passages"]]}
        for row in sources
    ]
    claim_snapshot = claim.model_dump(mode="json") if claim else None
    if claim_snapshot:
        claim_snapshot = {key: claim_snapshot.get(key) for key in (
            "id", "text", "loc", "conditions", "source_block_id", "source_quote", "source_refs", "needs"
        )}
        claim_snapshot["source_quote_sha256"] = hashlib.sha256((claim_snapshot.pop("source_quote") or "").encode()).hexdigest()
        claim_snapshot["source_refs"] = [_excerpt_snapshot(row) for row in claim_snapshot["source_refs"]]
    scope = {"version": "literature-search-scope-v1", "policy": "strict-native-exhaustion-v1",
             "claim": claim_snapshot, "claim_sha256": _digest(claim.model_dump(mode="json")) if claim else None,
             "novelty_condition_ids": sorted(novelty_ids), "queries": rows,
             "cutoff": cutoff, "concurrent_start": concurrent_start,
             "sources": source_snapshot, "source_excerpts": [_excerpt_snapshot(row) for row in source_excerpts],
             "self_exclusion": deepcopy(self_exclusion), "source_guards": bool(source_guards),
             "excluded": [{"reason": row["reason"], "paper": _identity_snapshot(row["paper"]),
                           "paper_sha256": _digest(row["paper"])} for row in excluded], "reads": [
                 {"id": row.get("id"), "identity_verified": row.get("identity_verified"),
                  "identity_rejected": row.get("identity_rejected"),
                  "success": row["response"].get("success") if isinstance(row.get("response"), dict) else None}
                 for row in reads],
             "native_exhausted": native, "raw_zero_hits": raw_zero,
             "zero_scope_eligible": bool(raw_zero and source_guards and not excluded and not reads and not sources
                                         and novelty_ids and claim),
             "limitations": ["Only recorded provider indexes and the three recorded query intents are searched."]}
    # Content hashes bind the untouched scientific inputs. Public metadata is
    # redacted before computing the scope digest, so saved snapshots stay exact.
    scope = public_copy(scope)
    scope["digest"] = _digest(scope)
    return scope


def validate_search_adequacy(response, scope, claim_id):
    """Return validated scientific decisions; a decision cannot alter scope facts."""
    raw = response.get("search_adequacy") if isinstance(response, dict) else None
    errors = []
    if not isinstance(raw, dict) or set(raw) != {"version", "claim_id", "scope_digest", "source_ids", "conditions"}:
        return {"valid": False, "errors": ["missing or malformed search_adequacy"], "decisions": {}}
    expected_sources = [row["paper_id"] for row in scope["sources"]]
    ids = raw.get("source_ids")
    if (raw.get("version") != "literature-search-adequacy-v1" or raw.get("claim_id") != claim_id
        or raw.get("scope_digest") != scope["digest"] or not isinstance(ids, list)
        or not all(isinstance(value, str) for value in ids) or len(ids) != len(set(ids))
        or sorted(ids) != sorted(expected_sources)):
        errors.append("version/claim/digest/source binding disagrees")
    conditions = raw.get("conditions")
    decisions = {}
    if not isinstance(conditions, list):
        errors.append("condition table is missing")
        conditions = []
    for row in conditions:
        if not isinstance(row, dict) or set(row) != {"condition_id", "state", "queries", "mechanism", "setting", "protocol", "limitations", "omission_risk"}:
            errors.append("malformed condition judgment")
            continue
        cid = row.get("condition_id")
        if not isinstance(cid, str) or cid not in scope["novelty_condition_ids"] or cid in decisions:
            errors.append("unknown or duplicate condition")
            continue
        decisions[cid] = deepcopy(row)
        if not isinstance(row.get("state"), str) or row["state"] not in {"adequate", "inadequate", "unresolved"} or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("mechanism", "setting", "protocol", "omission_risk")
        ) or not isinstance(row.get("limitations"), list) or not row["limitations"] or not all(
            isinstance(value, str) and value.strip() for value in row["limitations"]
        ):
            errors.append("scientific dimensions/limitations are missing")
        query_rows = row.get("queries")
        seen = set()
        if not isinstance(query_rows, list):
            errors.append("query coverage table is missing")
            query_rows = []
        for query in query_rows:
            if not isinstance(query, dict) or set(query) != {"query_id", "intent", "coverage", "reason"}:
                errors.append("malformed query judgment")
                continue
            match = next((item for item in scope["queries"] if item["query_id"] == query.get("query_id")), None)
            qid = query.get("query_id")
            if (match is None or qid in seen or query.get("intent") != match["intent"]
                or not isinstance(query.get("coverage"), str) or query["coverage"] not in {"covered", "insufficient", "unresolved"}
                or not isinstance(query.get("reason"), str) or not query["reason"].strip()):
                errors.append("query identity/intent/semantic coverage disagrees")
            if isinstance(qid, str):
                seen.add(qid)
        if seen != {item["query_id"] for item in scope["queries"]}:
            errors.append("query table does not cover the current plan")
    if set(decisions) != set(scope["novelty_condition_ids"]) or not decisions:
        errors.append("condition table does not cover the novelty scope")
    return {"valid": not errors, "errors": errors, "decisions": decisions}
