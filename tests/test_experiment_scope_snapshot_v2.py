"""The sent request stays exact when decoding adds joint-source diagnostics."""

import copy
import json

import pytest

from common.run_stats import run_scope
from tests.test_experiment_joint_sources_v2 import offline as offline
from tests.test_experiment_joint_sources_v2 import paper, review_for, run


@pytest.mark.parametrize("invalid_role", [False, True])
def test_sent_scope_snapshot_is_separate_from_post_validation_trace(tmp_path, monkeypatch, invalid_role):
    monkeypatch.setattr("httpx.Client.send", lambda *a, **kw: pytest.fail("External network"))
    fixture = paper(tmp_path)
    response = review_for(fixture[4])
    if invalid_role:
        response["items"][0]["source_uses"][0]["roles"].append("metric_definition")
    original_response = copy.deepcopy(response)
    with run_scope(tmp_path / "stats.json"):
        result, calls = run(fixture, response)
    audit_files = list((tmp_path / "experiment_scope").glob("*.json"))
    assert len(audit_files) == 1 and len(calls) == 2
    audit = json.loads(audit_files[0].read_text("utf-8"))
    sent = json.loads(calls[1]["prompt"])
    assert audit["input"] == sent
    assert audit["response"] == original_response == response
    request_manifest = sent["joint_candidates"]["0"]["manifest"]
    trace = audit["joint_sources"]["0"]
    assert request_manifest["consumption"] == [] and request_manifest["semantic_only"] == []
    assert trace["consumption"]
    assert len(result.evidence) == 1
    if invalid_role:
        assert trace["errors"] and not result.evidence[0].sufficient
    else:
        assert not trace["errors"] and result.evidence[0].sufficient
