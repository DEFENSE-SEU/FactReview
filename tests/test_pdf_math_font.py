"""Installed math fonts keep literal formula glyphs; missing fonts stay visible."""

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from review.report import pdf_renderer as renderer


@pytest.fixture(autouse=True)
def offline_and_uncached(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Font selection and PDF rendering must stay offline")

    monkeypatch.setattr(renderer, "_FONTS_CACHE", None)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("socket.socket.connect", blocked)
    monkeypatch.setattr("subprocess.Popen", blocked)
    monkeypatch.setattr("llm.client.llm_json", blocked)


def font_files(tmp_path, monkeypatch):
    directory = tmp_path / "Fonts"
    directory.mkdir()
    for filename in ("cambria.ttc", "seguisym.ttf"):
        (directory / filename).write_bytes(b"mock font bytes")
    monkeypatch.setenv("WINDIR", str(tmp_path))
    monkeypatch.setattr(renderer, "sys", SimpleNamespace(platform="win32"), raising=False)
    monkeypatch.setattr(renderer, "FONT_MONO_UNICODE_CANDIDATES", ())
    return directory


def test_windows_candidate_failure_uses_next_covered_font(tmp_path, monkeypatch):
    directory = font_files(tmp_path, monkeypatch)
    calls = []

    def register(name, path, *, quiet=False, required_chars=()):
        calls.append((name, path, set(required_chars)))
        return path.name == "seguisym.ttf"

    monkeypatch.setattr(renderer, "_register_ttf_font", register)
    fonts = renderer._resolve_report_fonts()
    assert [row[1] for row in calls] == [directory / "cambria.ttc", directory / "seguisym.ttf"]
    assert all(set("ℓλτ∑≥θℝ∀≈") <= row[2] for row in calls)
    assert fonts.mono == calls[1][0] and fonts.mono != "Courier"


def test_existing_dejavu_priority_has_no_new_math_gate(tmp_path, monkeypatch):
    font_files(tmp_path, monkeypatch)
    existing = tmp_path / "DejaVuSansMono.ttf"
    existing.write_bytes(b"mock Linux font")
    monkeypatch.setattr(renderer, "FONT_MONO_UNICODE_CANDIDATES", (existing,))
    calls = []

    def register(name, path, *, quiet=False, required_chars=()):
        calls.append((name, path, tuple(required_chars)))
        return True

    monkeypatch.setattr(renderer, "_register_ttf_font", register)
    assert renderer._resolve_report_fonts().mono == renderer.FONT_MONO_UNICODE_NAME
    assert calls == [(renderer.FONT_MONO_UNICODE_NAME, existing, ())]


def test_nonwindows_keeps_courier_fallback(tmp_path, monkeypatch, caplog):
    font_files(tmp_path, monkeypatch)
    monkeypatch.setattr(renderer, "sys", SimpleNamespace(platform="linux"), raising=False)
    monkeypatch.setattr(
        renderer, "_register_ttf_font", lambda *a, **k: pytest.fail("unexpected Windows font")
    )
    assert renderer._resolve_report_fonts().mono == "Courier"
    assert "mathematical glyphs may be unavailable" in caplog.text


def test_missing_windows_fonts_retains_fallback_and_explicit_limit(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WINDIR", str(tmp_path))
    monkeypatch.setattr(renderer, "sys", SimpleNamespace(platform="win32"), raising=False)
    monkeypatch.setattr(renderer, "FONT_MONO_UNICODE_CANDIDATES", ())
    assert renderer._resolve_report_fonts().mono == "Courier"
    assert "mathematical glyphs may be unavailable" in caplog.text


@pytest.mark.parametrize("mapping", [{}, {ord("ℓ"): 0}, {ord("ℓ"): 7}])
def test_glyph_registration_requires_nonzero_glyph_and_preserves_general_ttf_behavior(monkeypatch, mapping):
    font = SimpleNamespace(face=SimpleNamespace(charToGlyph=mapping))
    registered = []
    monkeypatch.setattr(renderer.pdfmetrics, "getRegisteredFontNames", lambda: [])
    monkeypatch.setattr(renderer, "TTFont", lambda *a: font)
    monkeypatch.setattr(renderer.pdfmetrics, "registerFont", registered.append)
    okay = renderer._register_ttf_font("test-required", Path("font.ttf"), required_chars="ℓ")
    assert okay is bool(mapping.get(ord("ℓ")))
    assert len(registered) == int(okay)
    assert renderer._register_ttf_font("test-general", Path("font.ttf"))
    assert registered[-1] is font


def test_cached_missing_glyph_does_not_bypass_coverage(monkeypatch):
    monkeypatch.setattr(renderer.pdfmetrics, "getRegisteredFontNames", lambda: ["cached-test"])
    monkeypatch.setattr(
        renderer.pdfmetrics,
        "getFont",
        lambda _: SimpleNamespace(face=SimpleNamespace(charToGlyph={ord("ℓ"): 0})),
    )
    assert not renderer._register_ttf_font("cached-test", Path("unused.ttf"), required_chars="ℓ")


@pytest.mark.skipif(os.name != "nt", reason="Windows installed-font PDF integration")
def test_installed_windows_font_renders_formula_glyphs_without_dingbat_substitution():
    directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if not any((directory / name).is_file() for name in ("cambria.ttc", "seguisym.ttf")):
        pytest.skip("No Windows math font installed; deterministic fallback covered separately")
    markdown = (
        r"Original objective: $\ell _ { s } + \lambda _ { u } \ell _ { u }$."
        + "\n\n"
        + r"Coverage: $\tau \sum \geq \theta \mathbb{R} \forall \approx$."
    )
    original = markdown
    content = renderer.build_review_report_pdf(
        workspace_title="Math glyph contract",
        source_pdf_name="source.pdf",
        run_id="offline",
        status="completed",
        decision=None,
        estimated_cost=0,
        actual_cost=None,
        exported_at=datetime(2026, 10, 9, tzinfo=UTC),
        meta_review={},
        reviewers=[],
        raw_output=None,
        final_report_markdown=markdown,
        implicit_math=False,
    )
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        fonts = [
            span["font"]
            for page in pdf
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]
    assert text.count("ℓ") == 2
    assert all(character in text for character in "λτ∑≥θℝ∀≈")
    assert not any("ZapfDingbats" in font for font in fonts)
    assert markdown == original
