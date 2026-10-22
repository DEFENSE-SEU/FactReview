"""Fresh image checks close every category while retaining grounded partial findings."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from review.delivery import checked_delivery
from schemas.review import FinalReview
from screening import checks
from screening.figures import FigureCheckRecord, check_figures
from screening.tables import TableCheckRecord, check_visual_tables
from tests.test_figure_context import context_response
from tests.test_figure_context import paper as figure_paper
from tests.test_table_context import decision
from tests.test_table_context import paper as table_paper


def test_fresh_visual_categories_close_without_losing_partial_or_source_guards(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External model/network/process forbidden by visual completion control")

    cfg = LLMConfig("mock", "visual-completion", None, None)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    monkeypatch.setattr(checks, "resolve_vlm_config", lambda **_: cfg)
    monkeypatch.setattr(checks, "llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)

    for kind, factory, review, record_type, consistency, context_reply in (
        ("figure", figure_paper, check_figures, FigureCheckRecord, "text_figure_consistency", context_response),
        ("table", table_paper, check_visual_tables, TableCheckRecord, "text_table_consistency", decision),
    ):
        for mode in ("missing", "clear", "multi", "empty", "mutated"):
            directory = tmp_path / f"{kind}-{mode}"
            directory.mkdir()
            materials = factory(directory, **({"caption_mode": "native"} if kind == "table" else {}))
            original = materials.model_dump(mode="json")
            material = getattr(materials, kind + "s")[0]
            image = Path(material.printed_crop_path)
            image_bytes = image.read_bytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            raw = {"findings": [
                {"category": "self_containedness", "disposition": "clear",
                 "text": "The visible labels and the supplied caption identify the plotted or tabulated quantities."},
                {"category": "legibility", "disposition": "clear",
                 "text": "The printed-size crop exposes identifiable numeric labels."},
                {"category": consistency, "disposition": "clear",
                 "text": "The crop and the supplied original-page caption describe the same recorded counts."},
            ]}
            if mode in {"missing", "multi", "mutated"}:
                raw["findings"][1:2] = [
                    {"category": "legibility", "disposition": "issue", "text": text}
                    for text in ("The mocked visual judgment identifies overlapping row or tick labels.",
                                 "The mocked visual judgment identifies an unreadable numeric label.")
                ]
            if mode == "missing":
                raw["findings"].pop()
            if mode == "empty":
                raw["findings"] = []
            saved_raw = copy.deepcopy(raw)
            calls, records = [], []

            def call(**request):
                payload = json.loads(request["prompt"])
                calls.append(request["module"])
                if request["module"].endswith(".context"):
                    return context_reply(payload)
                assert request["images"] == [str(image)]
                assert payload["printed_size_verified"] is True
                assert payload["references"] == [r.model_dump() for r in material.references]
                assert hashlib.sha256(image.read_bytes()).hexdigest() == image_hash
                if mode == "mutated":
                    image.write_bytes(b"corrupt source after its actual image request")
                return copy.deepcopy(raw)

            findings, issues = review(materials, call=call, recover_errors=True, records=records)
            record = records[0]
            assert record.crop_response == saved_raw and raw == saved_raw
            assert materials.model_dump(mode="json") == original
            assert len(records) == 1 and record.printed_size_verified
            assert len([m for m in calls if not m.endswith(".context")]) == 1
            assert len(calls) <= 2
            coverage = {"total": 1, **{
                status: int(record.status == status) for status in ("checked", "failed", "unavailable")}}
            delivery = checked_delivery(
                FinalReview(paper_key=materials.paper_key, run_id=mode, claims=[], findings=findings),
                **{kind + "_coverage": coverage},
            )
            if mode in {"missing", "empty"}:
                assert record.status == "unavailable" and coverage["checked"] == 0
                assert delivery.run_status == "partial" and "screening" in delivery.incomplete_stages
                assert any("unassessed categories" in issue for issue in issues)
                missing = {consistency} if mode == "missing" else {"self_containedness", "legibility", consistency}
                assert all(any(category in issue for issue in record.issues) for category in missing)
                assert len(findings) == (2 if mode == "missing" else 0)
            elif mode == "mutated":
                assert record.status == "failed" and not findings and record.finding_count == 0
                assert delivery.run_status == "partial"
                assert any("changed" in issue for issue in issues)
            else:
                assert record.status == "checked" and coverage["unavailable"] == 0
                assert delivery.run_status == "completed"
                assert not any("unassessed categories" in issue for issue in issues)
                assert len(findings) == (2 if mode == "multi" else 0)
            if findings:
                assert all(f.level == "legibility" and not f.evidence[0].affects_claim for f in findings)
                assert all(f.evidence[0].pointer.locator == str(image) for f in findings)
                assert all(not f.evidence[0].sufficient for f in findings)
                assert record.finding_count == len(findings)
            image.write_bytes(image_bytes)
            assert hashlib.sha256(image.read_bytes()).hexdigest() == image_hash
        historical = record_type.model_validate({
            kind + "_id": "historical", "status": "checked", "crop_response": {"findings": []},
        })
        assert historical.crop_response == {"findings": []} and historical.status == "checked"
