"""The opt-in live probe must not count unavailable visual checks as passing."""

import importlib.util
import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from llm.client import LLMConfig
from screening.figures import FigureCheckRecord

spec = importlib.util.spec_from_file_location(
    "visual_probe", Path(__file__).resolve().parents[1] / "scripts/check_v2_visual.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("status,verified", [("failed", False), ("unavailable", False), ("checked", False)])
def test_empty_findings_without_verified_pixels_cannot_pass_clear_case(
    tmp_path, monkeypatch, status, verified
):
    def check(materials, *, recover_errors, records):
        records.append(FigureCheckRecord(figure_id="figure_1", status=status, printed_size_verified=verified))
        return [], []

    monkeypatch.setattr(probe, "check_figures", check)
    result = probe.figure_probe(tmp_path / "case", *probe.FIGURE_CASES[0])
    assert result["passed"] is False


def test_pixel_transport_omits_ground_truth_from_text_input(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.secrets, "randbelow", lambda _: 12345)

    def ask(system, payload, *, module, images):
        assert "012345" not in system and payload == {}
        assert Path(images[0]).is_file()
        return {"code": "012345"}

    monkeypatch.setattr(probe, "ask", ask)
    result = probe.transport_probe(tmp_path / "transport")
    assert result["passed"] and result["sha256"]


@pytest.mark.parametrize(
    "category,disposition,passes",
    [
        ("legibility", "issue", True),
        ("legibility", "clear", False),
        ("legibility", "uncertain", False),
        ("self_containedness", "issue", False),
    ],
)
def test_tiny_printed_labels_use_real_pixels_and_require_legibility_issue(
    tmp_path, monkeypatch, category, disposition, passes
):
    calls = []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        assert "tiny_labels" not in kwargs["prompt"]
        assert "legibility" not in payload["caption"]
        assert payload["printed_size_verified"] is True
        calls.append(kwargs)
        return {
            "findings": [
                {
                    "category": category,
                    "disposition": disposition,
                    "text": "The axis and legend text is too small to read at this size.",
                }
            ]
        }

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda *, fallback: fallback)
    monkeypatch.setattr("screening.checks.llm_json", model)
    case = next(case for case in probe.FIGURE_CASES if case[0] == "tiny_labels")
    result = probe.figure_probe(tmp_path / "tiny", *case)
    assert result["passed"] is passes
    assert len(calls) == 1 and len(calls[0]["images"]) == 1
    info = result["inputs"][0]
    assert calls[0]["images"] == [info["image"]]
    with Image.open(info["image"]) as image:
        x1, y1, x2, y2 = info["bbox_points"]
        assert image.size == (round((x2 - x1) * 96 / 72), round((y2 - y1) * 96 / 72))
    with pymupdf.open(tmp_path / "tiny/source.pdf") as pdf:
        spans = [
            span
            for block in pdf[0].get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]
    assert all(
        span["size"] == 2 for span in spans if span["text"] in {"Time (s)", "Rate (items/s)", "Measured"}
    )
    assert {span["text"] for span in spans} >= {"Time (s)", "Rate (items/s)", "Measured"}
    assert info["sha256"] and info["printed_dpi"] == 96


def test_selected_figure_case_runs_without_transport_or_other_cases(tmp_path, monkeypatch):
    calls = []

    def figure(directory, name, *args):
        calls.append(name)
        return {"case": name, "passed": True}

    monkeypatch.setattr(probe, "load_env_file", lambda _: None)
    monkeypatch.setattr(probe, "figure_probe", figure)
    monkeypatch.setattr(probe, "transport_probe", lambda _: pytest.fail("Unselected probe must not run"))
    assert (
        probe.main(["--mode", "figures", "--figure-cases", "tiny_labels", "--run-root", str(tmp_path)]) == 0
    )
    assert calls == ["tiny_labels"]
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert summary["passed"] and len(summary["cases"]) == 1


def test_transport_mode_rejects_figure_case_selection(tmp_path):
    with pytest.raises(SystemExit) as exc:
        probe.main(["--mode", "transport", "--figure-cases", "tiny_labels", "--run-root", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
