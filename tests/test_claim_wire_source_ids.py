"""Legal multiple spans close through the real wire and scope consumers."""

import copy
from unittest.mock import Mock

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v3 import case
from tests.test_claim_coverage_v4 import missing_row, preserved_row
from tests.test_claim_coverage_wire_v5 import compact_row


@pytest.mark.parametrize("builder", [preserved_row, missing_row], ids=["preserved", "missing"])
def test_distinct_spans_keep_original_authority_and_close_block_views(builder, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("External model, retrieval or Docker forbidden"))
    monkeypatch.setattr(m, "llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    row, claim, context, blocks = builder()
    raw = compact_row(row)
    control, control_mapping = m.expand_scope_row(copy.deepcopy(raw))
    first = raw["scope_atoms"][0]["sources"][0]
    assert first["end"] > first["start"] + 1
    raw["scope_atoms"][0]["sources"].append({**first, "end": first["start"] + 1})
    original = copy.deepcopy(raw)
    state = copy.deepcopy((claim, context, {k: b.model_dump(mode="json") for k, b in blocks.items()}))
    generated, mapping = m.expand_scope_row(raw)
    m._check_scope(m.CurrentClaimReviewV4.model_validate(generated), claim, context, blocks)

    def lower(value):
        response = m.CoverageReviewV5.model_validate({
            "schema_version": "claim-coverage-v5", "context_id": "controlled", "window_id": "controlled",
            "reviewed_block_ids": list(blocks), "claim_reviews": [value], "new_findings": [],
            "explanation": "Explicit unchanged mock scientific judgments",
        })
        return m._lower_v3(response, [{"claim_id": claim["claim_id"], "digest": claim["digest"]}],
                           [claim], blocks, case()[0].markdown, set(), scope_context=[context])

    lowered, audit = lower(raw)
    assert audit["errors"] == [] and len(lowered.claim_checks) == 1
    expected = copy.deepcopy(control)
    expected["scope_atoms"][0]["sources"] = copy.deepcopy(raw["scope_atoms"][0]["sources"])
    assert generated == expected
    assert raw == original and len(generated["scope_atoms"][0]["sources"]) == 2
    assert mapping["raw_digest"] == m.wire_fingerprint(original)
    assert mapping["generated_digest"] == m.wire_fingerprint(generated)
    assert mapping["fields"] == control_mapping["fields"]
    assert audit["wire_mappings"] == [{"raw_path": "/claim_reviews/0", "claim_id": claim["claim_id"], **mapping}]
    assert audit["mappings"][0]["raw_digest"] == m._digest(generated)
    for corruption in ("duplicate_span", "duplicate_source_group", "duplicate_raw_source_ids"):
        broken = copy.deepcopy(raw)
        if corruption == "duplicate_span":
            broken["scope_atoms"][0]["sources"].append(copy.deepcopy(first))
        elif corruption == "duplicate_source_group":
            broken["source_review_groups"].append(copy.deepcopy(broken["source_review_groups"][0]))
        else:
            broken["source_block_ids"].append(broken["source_block_ids"][0])
        rejected, rejected_audit = lower(broken)
        assert rejected_audit["errors"] and not rejected.claim_checks and not rejected_audit["wire_mappings"]
    assert (claim, context, {k: b.model_dump(mode="json") for k, b in blocks.items()}) == state
    assert forbidden.call_count == 0
