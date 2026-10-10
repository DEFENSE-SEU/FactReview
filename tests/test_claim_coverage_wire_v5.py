"""Five bounded wire control groups; model/retrieval/service calls are mocked."""

import copy
import json

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG, action, extracted, followup, validation
from tests.test_claim_coverage_v3 import case, review
from tests.test_claim_coverage_v4 import missing_row, preserved_row, scope_row


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External model forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))


def compact_row(row):
    """Fixture migration only: retains each explicit injected scientific choice."""
    r = copy.deepcopy(row)
    views, findings, sources = r.pop("preserved_qualifiers"), r.pop("findings"), r.pop("source_reviews")
    r["other_findings"] = [f for f in findings if f["kind"] != "missing_qualifier_or_condition"]
    r["source_review_groups"] = [
        {"source_ids": [s["block_id"]], **{k: v for k, v in s.items() if k != "block_id"}}
        for s in sources
    ]
    for atom in r["scope_atoms"]:
        atom["carriers"] = [
            {k: v for k, v in views[i].items() if k not in {"restriction", "source_block_ids"}}
            for i in atom.pop("preserved_indices")
        ]
        atom.pop("finding_index")
    return r


def compact_response(raw):
    r = copy.deepcopy(raw)
    key = "claim_reviews" if "claim_reviews" in r else "original_claim_reviews"
    r[key] = [compact_row(row) for row in r[key]]
    r["schema_version"] = "claim-coverage-v5" if key == "claim_reviews" else "claim-coverage-validation-v5"
    return r


def unpack(p):
    return m.unpack_scope_payload(p)


def test_catalog_exact_roundtrip_and_conflicting_or_foreign_metadata():
    assert hasattr(m, "pack_scope_payload")
    _row, claim, context, blocks = scope_row()
    payload = {"scope_context": [context, {**copy.deepcopy(context), "claim_id": "second"}],
               "current_claims": [claim], "blocks": [b.model_dump(mode="json") for b in blocks.values()]}
    packed = m.pack_scope_payload(payload)
    assert unpack(packed) == payload and packed["current_claims"] == payload["current_claims"]
    assert len(packed["source_catalog"]) < sum(len(c["sources"]) for c in payload["scope_context"])
    for kind in ["conflict", "foreign", "duplicate", "override"]:
        broken = copy.deepcopy(packed)
        source = broken["scope_context"][0]["sources"][0]
        if kind == "conflict":
            original = copy.deepcopy(payload)
            original["scope_context"][1]["sources"][0]["block_digest"] = "stale"
            with pytest.raises(ValueError):
                m.pack_scope_payload(original)
            continue
        if kind == "foreign":
            source["source_id"] = "unloaded"
        elif kind == "duplicate":
            broken["scope_context"][0]["sources"].append(copy.deepcopy(source))
        else:
            source["block_digest"] = "override"
        with pytest.raises(ValueError):
            unpack(broken)


def test_atom_schema_rejects_native_non_governing_placeholder():
    row, _claim, _context, _blocks = preserved_row()
    atom = compact_row(row)["scope_atoms"][0]
    atom.update(state="not_governing", carriers=[], effect=None)
    with pytest.raises(ValueError):
        m.ScopeAtomV5.model_validate(atom)
    assert "not_governing" not in m.ScopeAtomV5.model_json_schema()["properties"]["state"]["enum"]


def test_single_atom_authority_retains_carriers_effects_and_scope_guards():
    assert hasattr(m, "CurrentClaimReviewV5")
    for builder in [preserved_row, missing_row]:
        row, claim, context, blocks = builder()
        compact = compact_row(row)
        parsed = m.CurrentClaimReviewV5.model_validate(compact)
        generated, mapping = m.expand_scope_row(parsed.model_dump(mode="json"))
        checked = m.CurrentClaimReviewV4.model_validate(generated)
        m._check_scope(checked, claim, context, blocks)
        assert checked.scope_atoms[0].restriction == parsed.scope_atoms[0].restriction
        assert mapping["raw_digest"] == m.wire_fingerprint(compact)
        assert mapping["generated_digest"] == m.wire_fingerprint(generated)
        if builder is preserved_row:
            assert generated["preserved_qualifiers"][0]["claim_value"] == row["preserved_qualifiers"][0]["claim_value"]
        else:
            assert generated["findings"][0]["material_effect"] == row["findings"][0]["material_effect"]
            assert generated["scope_atoms"][0]["effect"] == row["scope_atoms"][0]["effect"]
    row, claim, context, blocks = preserved_row()
    for corruption in ["empty", "value", "metadata", "condition", "foreign_source", "omitted_source", "duplicate_source", "duplicate_carrier"]:
        r = compact_row(row)
        atom = r["scope_atoms"][0]
        if corruption == "empty":
            atom["carriers"] = []
        elif corruption == "value":
            atom["carriers"][0]["claim_value"] = "foreign scalar"
        elif corruption == "metadata":
            atom["carriers"][0].update(claim_path="/claim_id", claim_value=claim["claim_id"])
        elif corruption == "condition":
            atom["condition_ids"] = ["foreign"]
        elif corruption == "foreign_source":
            atom["sources"][0]["block_id"] = "unloaded"
        elif corruption == "omitted_source":
            r["source_review_groups"].pop()
        elif corruption == "duplicate_carrier":
            atom["carriers"].append(copy.deepcopy(atom["carriers"][0]))
        else:
            r["source_review_groups"].append(copy.deepcopy(r["source_review_groups"][0]))
        with pytest.raises(ValueError):
            generated, _ = m.expand_scope_row(m.CurrentClaimReviewV5.model_validate(r).model_dump(mode="json"))
            m._check_scope(m.CurrentClaimReviewV4.model_validate(generated), claim, context, blocks)


def test_native_three_stage_split_and_raw_generated_audit(tmp_path):
    assert hasattr(m, "CoverageReviewV5")
    paper, claims = case()
    original = [c.model_dump(mode="json") for c in claims]
    calls = []
    def call(**kw):
        wire = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        p = unpack(wire)
        calls.append((kw["module"], wire))
        if kw["module"] == "screening.claims.coverage":
            return compact_response(review(p, groups=2))
        if kw["module"] == "screening.claims.coverage_followup":
            props = []
            for i, text in enumerate(["The method reaches accuracy 90.", "The method uses batch size 16."]):
                raw = extracted(original[0])
                raw.update(text=text, conditions=[raw["conditions"][i]])
                props.append(raw)
            return followup(p, [action(p, p["observations"], "split", props)])
        return compact_response(validation(p))
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert len(calls) == 3 and len(result.claims) == 2 and not result.blocked_claim_ids
    assert result.coverage["status"] == "complete" and len(result.coverage["revisions"]) == 1
    assert result.coverage["windows"][0]["scope_protocol"] == "claim-coverage-v5"
    assert all("source_catalog" in p for _, p in calls)
    saved = json.loads((tmp_path / "coverage.json").read_text("utf-8"))
    assert saved["attempts"][0]["response"]["schema_version"] == "claim-coverage-v5"
    assert saved["attempts"][2]["response"]["schema_version"] == "claim-coverage-validation-v5"
    assert saved["attempts"][0]["v3_lowering"]["wire_mappings"]
    assert saved["attempts"][2]["original_recheck"]["lowering"]["wire_mappings"]
    assert [c.model_dump(mode="json") for c in claims] == original


def test_stale_context_claim_and_legacy_cannot_complete_or_adopt(tmp_path):
    assert hasattr(m, "CoverageValidationV5")
    for mode in ["context", "claim", "legacy"]:
        paper, claims = case()
        before = [c.model_dump(mode="json") for c in claims]
        def call(**kw):
            p = unpack(json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1]))
            if kw["module"] == "screening.claims.coverage":
                raw = review(p)
                if mode == "legacy":
                    return raw
                raw = compact_response(raw)
                if mode == "context":
                    raw["context_id"] = "stale"
                elif mode == "claim":
                    raw["claim_reviews"][0]["claim_digest"] = "stale"
                return raw
            raw = validation(p)
            return raw if mode == "legacy" else compact_response(raw)
        result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path / mode)
        assert result.coverage["status"] != "complete" and not result.coverage["revisions"]
        assert [c.model_dump(mode="json") for c in result.claims] == before
        if mode == "legacy":
            window = result.coverage["windows"][0]
            assert all(r["state"] == "unreviewed" for r in window["claim_checks"])
            assert window["original_recheck"]["completed"] == 0


def test_missing_effect_uncertainty_source_condition_and_dimensions_stay_closed():
    assert hasattr(m, "CurrentClaimReviewV5")
    row, claim, context, blocks = missing_row()
    for mode in ["excluded", "same", "background", "uncertain", "dimension", "source_condition"]:
        r = compact_row(row)
        if mode == "excluded":
            r["scope_atoms"][0]["effect"]["original_permits_alternative"] = False
        elif mode == "same":
            e = r["scope_atoms"][0]["effect"]
            e["alternative_setting"] = e["source_setting"]
        elif mode == "background":
            r["scope_atoms"][0]["effect"]["kind"] = "background_fact"
        elif mode == "uncertain":
            r["state"] = "unresolved"
        elif mode == "dimension":
            r["scope_groups"][0]["dimensions"].pop(m.SCOPE_DIMENSIONS[0])
        else:
            next(s for s in r["source_review_groups"] if s["source_ids"] == ["b2"])["condition_ids"] = ["c2"]
        raw_review = m.CoverageReviewV5.model_validate({
            "schema_version": "claim-coverage-v5", "context_id": "mock", "window_id": "mock",
            "reviewed_block_ids": list(blocks), "claim_reviews": [r], "new_findings": [],
            "explanation": "Explicit corrupted mock",
        })
        lowered, audit = m._lower_v3(raw_review, [{"claim_id": claim["claim_id"], "digest": claim["digest"]}],
            [claim], blocks, case()[0].markdown, set(), scope_context=[context])
        assert audit["errors"] and not lowered.claim_checks and not audit["wire_mappings"]
