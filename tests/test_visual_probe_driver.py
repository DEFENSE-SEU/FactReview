"""The opt-in live probe must not count unavailable visual checks as passing."""

import importlib.util
from pathlib import Path

import pytest

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
