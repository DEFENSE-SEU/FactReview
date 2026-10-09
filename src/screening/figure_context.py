"""Bound original-page confirmation of existing crop observations, excluding legibility."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from PIL import Image
from pydantic import Field

from schemas.claim import Contract, NonEmpty
from screening import checks
from screening.checks import ask

VERSION = "figure-context-v1"
Identifier = Annotated[str, Field(strict=True, min_length=1)]
_CAPTION = re.compile(r"^\s*fig(?:ure)?\.?\s+((?:[A-Za-z]\.?)?\d+(?:\.\d+)*)(?!\w)", re.I)


class PixelBindingError(ValueError):
    """An available original source disproves the submitted image's identity."""


def safe_response(value):
    """Retain model data with the same configured-credential redaction as its audit."""
    cfg = checks.resolve_vlm_config(fallback=checks.resolve_llm_config())
    return checks.redact_provider_details(value, cfg)


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    except OSError:
        return None


def _dump(value):
    return value.model_dump(mode="json")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class FigureSources:
    """Captured for every figure before any model callback, including later figures."""

    figure: dict
    pages: list[dict]
    neighbors: list[dict]
    source_pdf: str
    content_list_path: str
    markdown_path: str
    hashes: dict
    context_consumed: bool = False
    context: dict | None = None
    context_error: str = ""

    @classmethod
    def capture(cls, materials, figure, hash_cache):
        page = figure.loc.page if figure.loc else None
        pages = [_dump(p) for p in materials.pages if p.page == page]
        paths = [
            figure.printed_crop_path,
            materials.source_pdf,
            materials.content_list_path,
            materials.markdown_path,
        ]
        paths.extend(p["path"] for p in pages)
        for path in paths:
            if path not in hash_cache:
                hash_cache[path] = file_hash(path)
        return cls(
            _dump(figure),
            pages,
            [_dump(f) for f in materials.figures if f.loc and f.loc.page == page],
            materials.source_pdf,
            materials.content_list_path,
            materials.markdown_path,
            {path: hash_cache[path] for path in paths},
        )

    def check(self, materials, *, context=False):
        current = [_dump(f) for f in materials.figures if f.id == self.figure["id"]]
        if current != [self.figure]:
            raise ValueError("Figure metadata/reference identity changed or is duplicated")
        crop = self.figure["printed_crop_path"]
        if not self.hashes[crop] or file_hash(crop) != self.hashes[crop]:
            raise ValueError("Figure printed crop changed or was unavailable")
        if (
            materials.source_pdf != self.source_pdf
            or materials.content_list_path != self.content_list_path
            or materials.markdown_path != self.markdown_path
            or any(
                digest is not None and file_hash(path) != digest
                for path, digest in self.hashes.items()
                if path in {self.source_pdf, self.content_list_path, self.markdown_path}
            )
        ):
            raise ValueError("Figure source PDF/parser/reference artifact changed")
        if context or self.context_consumed:
            page = self.figure["loc"]["page"]
            if (
                materials.source_pdf != self.source_pdf
                or materials.content_list_path != self.content_list_path
                or [_dump(p) for p in materials.pages if p.page == page] != self.pages
                or [_dump(f) for f in materials.figures if f.loc and f.loc.page == page] != self.neighbors
                or any(file_hash(path) != digest for path, digest in self.hashes.items())
            ):
                raise ValueError("Figure original-page source changed during screening")

    def prepare(self, materials):
        if self.context is None and not self.context_error:
            try:
                self.check(materials, context=True)
                self.context = _build_context(self)
                self.check(materials, context=True)
            except PixelBindingError:
                raise
            except Exception as exc:
                self.context_error = f"{type(exc).__name__}: {exc}"
        return self.context


def _pixel_tuple(image):
    pixels = image.convert("RGB")
    return pixels.size, pixels.tobytes()


def _verify_pixels(actual_path, pixmap, size=None):
    expected = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    if size:
        expected = expected.resize(size, Image.Resampling.LANCZOS)
    with Image.open(actual_path) as actual:
        if _pixel_tuple(actual) != _pixel_tuple(expected):
            raise PixelBindingError(
                "Figure/page pixels do not match their original PDF page and bounding box"
            )


def _parts_from_original(sources, width, height):
    from preprocessing.materials import _bbox, _page, _row_text, figure_caption_parts

    figure = sources.figure
    if not sources.hashes.get(sources.content_list_path):
        raise ValueError("Figure original parser content list is unavailable")
    rows = json.loads(Path(sources.content_list_path).read_text(encoding="utf-8"))
    matched = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("type") not in {"image", "chart"}:
            continue
        if _page(row) != figure["loc"]["page"] or _row_text(row) != figure["caption"]:
            continue
        try:
            box = _bbox(row, width, height, figure["parser_bbox_space"] or "normalized_1000")
        except (ValueError, TypeError, OverflowError):
            continue
        if list(box) == figure["bbox_points"]:
            matched.append((index, row))
    if len(matched) != 1:
        raise ValueError("Figure parser row/page/box/caption association must be unique")
    index, row = matched[0]
    if figure["parser_row_index"] is not None and figure["parser_row_index"] != index:
        raise ValueError("Figure parser row index does not match its original row")
    if figure["block_id"] and figure["block_id"] != f"block_{index}":
        raise ValueError("Figure source block does not match its original parser row")
    parts = [_dump(part) for part in figure_caption_parts(row)]
    if figure["caption_parts"] and figure["caption_parts"] != parts:
        raise ValueError("Figure caption elements differ from original parser elements")
    if not parts:
        raise ValueError("Figure has no original caption elements")
    return parts


def _label(text):
    match = _CAPTION.match(text)
    return match.group(1).casefold() if match else ""


def _intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _distance(a, b):
    return math.hypot(max(b[0] - a[2], a[0] - b[2], 0), max(b[1] - a[3], a[1] - b[3], 0))


def _build_context(sources):
    import fitz

    f = sources.figure
    if len(sources.pages) != 1 or not sources.hashes.get(sources.source_pdf):
        raise ValueError("Figure requires a unique original PDF page image")
    p = sources.pages[0]
    if not sources.hashes.get(p["path"]) or f["bbox_points"] is None:
        raise ValueError("Figure page image or physical box is unavailable")
    with fitz.open(sources.source_pdf) as pdf:
        number = f["loc"]["page"]
        if number is None or not 1 <= number <= len(pdf):
            raise ValueError("Figure page is outside the original PDF")
        page = pdf[number - 1]
        box = f["bbox_points"]
        if (
            not all(math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= page.rect.width
            or not 0 <= box[1] < box[3] <= page.rect.height
            or p["width_points"] != page.rect.width
            or p["height_points"] != page.rect.height
            or p["dpi"] != 200
            or f["printed_dpi"] != 96
        ):
            raise ValueError("Figure physical page/box/DPI metadata does not match the PDF")
        _verify_pixels(p["path"], page.get_pixmap(dpi=200))
        size = (max(1, round((box[2] - box[0]) * 96 / 72)), max(1, round((box[3] - box[1]) * 96 / 72)))
        _verify_pixels(f["printed_crop_path"], page.get_pixmap(dpi=200, clip=fitz.Rect(box)), size)
        parts = _parts_from_original(sources, page.rect.width, page.rect.height)
        spans = []
        for index, block in enumerate(page.get_text("dict")["blocks"]):
            if "lines" not in block:
                continue
            text = "\n".join("".join(s["text"] for s in line["spans"]) for line in block["lines"])
            if text.strip():
                spans.append(
                    {"id": f"page:{number}:block:{index}", "text": text, "bbox_points": list(block["bbox"])}
                )
    labels = [_label(part["text"]) for part in parts if _label(part["text"])]
    if len(labels) != 1:
        raise ValueError("Figure requires one uniquely numbered caption element")
    label = labels[0]
    if f["anchor"] and f["anchor"].casefold() != label:
        raise ValueError("Figure numbered caption and parser anchor disagree")
    # A repeated numbered caption anywhere on the page has unresolved assignment.
    if len([s for s in spans if _label(s["text"]) == label]) != 1:
        raise ValueError("Figure numbered caption is absent or duplicated on the original page")
    known_labels = {
        _label(line)
        for neighbor in sources.neighbors
        for line in neighbor["caption"].splitlines()
        if _label(line)
    }
    if any(_label(s["text"]) and _label(s["text"]) not in known_labels for s in spans):
        raise ValueError("Original page has a numbered figure outside the supplied figure assignments")
    others = []
    for neighbor in sources.neighbors:
        if neighbor["id"] == f["id"]:
            continue
        other_box = neighbor["bbox_points"]
        if other_box is None or _intersects(box, other_box):
            raise ValueError("Same-page figure boxes are missing or overlap; target identity is unresolved")
        others.append({"figure_id": neighbor["id"], "bbox_points": other_box})
    for part in parts:
        matches = [s for s in spans if s["text"].split() == part["text"].split()]
        part["matching_span_ids"] = [s["id"] for s in matches] if len(matches) == 1 else []
        part["caption_label"] = _label(part["text"])
        part["alignment"] = (
            "exact"
            if len(matches) == 1 and matches[0]["text"] == part["text"]
            else "whitespace_layout"
            if len(matches) == 1
            else "unavailable"
        )
    caption = next(part for part in parts if part["caption_label"])
    if not caption["matching_span_ids"]:
        raise ValueError(
            "Numbered caption cannot be uniquely located with unchanged tokens on the original page"
        )
    allowed = []
    part_spans = {identifier for part in parts for identifier in part["matching_span_ids"]}
    for span in spans:
        rect = span["bbox_points"]
        if _label(span["text"]) and _label(span["text"]) != label:
            continue
        if any(
            _intersects(rect, n["bbox_points"]) or _distance(rect, n["bbox_points"]) <= _distance(rect, box)
            for n in others
        ):
            continue
        if _intersects(rect, box) or span["id"] in part_spans:
            allowed.append(span["id"])
    if not set(caption["matching_span_ids"]).issubset(allowed):
        raise ValueError("Caption location cannot be separated from another figure")
    context = {
        "figure_id": f["id"],
        "page": number,
        "bbox_points": box,
        "source_pdf": sources.source_pdf,
        "page_path": p["path"],
        "printed_crop_path": f["printed_crop_path"],
        "source_hashes": sources.hashes,
        "caption_ambiguous": f["caption_ambiguous"],
        "caption_parts": parts,
        "page_spans": spans,
        "allowed_witness_span_ids": allowed,
        "other_figures": others,
        "references": f["references"],
    }
    context["context_id"] = _digest(context)
    return context


class PartRole(Contract):
    part_id: Identifier
    role: Literal["caption", "axis_label", "legend", "panel_label", "unrelated", "uncertain"]
    page_span_ids: list[Identifier]
    reason: NonEmpty


class Decision(Contract):
    candidate_id: Identifier
    classification: Literal["manuscript_issue", "crop_artifact", "no_manuscript_issue", "uncertain"]
    witness_span_ids: list[Identifier]
    reason: NonEmpty


class ContextResponse(Contract):
    schema_version: Literal["figure-context-v1"]
    context_id: Identifier
    figure_id: Identifier
    target: Literal["matched", "uncertain", "mismatch"]
    part_roles: list[PartRole]
    decisions: list[Decision]


def _exact_set(values, expected, name):
    if len(values) != len(set(values)) or set(values) != set(expected):
        raise ValueError(f"Figure context {name} must identify each supplied item exactly once")


def confirm(context, candidates, *, call=None):
    payload = {**context, "candidates": candidates, "output_schema": ContextResponse.model_json_schema()}
    raw = ask(
        "Confirm only the supplied crop candidates against the bound original PDF page. "
        "The first image is the original printed crop; the second is high-resolution PAGE CONTEXT. "
        "This request does not assess legibility, font size, clarity at printed size, or new defects. "
        "Keep the target figure's box and numbered caption distinct from every other figure on the page. "
        "Choose only supplied exact IDs. Return every original caption part's visual role and every candidate's decision. "
        "A caption part can be an axis label rather than a caption; preserve the parser's ambiguous flag. "
        "Use target=uncertain if the correspondence of crop, plotted area and numbered caption cannot be established. "
        "For matched, identify exactly the numbered caption and its unchanged original-page span. "
        "Use crop_artifact only with visible same-figure evidence that refutes the candidate, e.g. labels outside the crop. "
        "First determine the figure's display purpose from its caption and EVERY body reference. "
        "A training-example gallery may have a caption reporting an experiment performed using those examples; "
        "the gallery need not plot that accuracy value or a performance curve. Caption statements can supply "
        "additional context. A text_figure_consistency defect requires conflicting propositions about what the "
        "figure actually depicts or explicitly claims to depict, such as an incompatible plotted value. "
        "Use no_manuscript_issue to reject a candidate caused by misreading that display purpose or demanding "
        "that every caption assertion be drawn inside the image. Preserve the original candidate and explain "
        "the exact same-figure sources. This classification does not verify the underlying scientific result. "
        "Use manuscript_issue only for an original-page defect belonging to this figure, with visible witnesses; "
        "uncertain is required when assignment or evidence is incomplete. Neighbor proximity alone cannot prove ownership. "
        "Never borrow another figure's labels, caption or legend. Whitespace-layout alignment retains both exact source strings. "
        "Do not change candidate category or claim scientific validity. Record uncertainty in visual interpretation.",
        payload,
        module="screening_figures.context",
        call=call,
        images=[context["printed_crop_path"], context["page_path"]],
    )
    raw = safe_response(raw)
    response = ContextResponse.model_validate(raw)
    if response.context_id != context["context_id"] or response.figure_id != context["figure_id"]:
        raise ValueError("Figure context response changed target identity")
    parts = {p["id"]: p for p in context["caption_parts"]}
    _exact_set([p.part_id for p in response.part_roles], parts, "part roles")
    _exact_set(
        [d.candidate_id for d in response.decisions], [c["candidate_id"] for c in candidates], "decisions"
    )
    allowed = set(context["allowed_witness_span_ids"])
    for part in response.part_roles:
        if len(part.page_span_ids) != len(set(part.page_span_ids)):
            raise ValueError("Figure context repeats a part span")
        expected = parts[part.part_id]["matching_span_ids"]
        if part.page_span_ids and (part.page_span_ids != expected or not set(expected).issubset(allowed)):
            raise ValueError("Figure context part points outside its uniquely bound original text")
        if part.role not in {"uncertain", "unrelated"} and not part.page_span_ids:
            raise ValueError("Figure context assigned role without original-page text")
    for decision in response.decisions:
        if len(decision.witness_span_ids) != len(set(decision.witness_span_ids)) or not set(
            decision.witness_span_ids
        ).issubset(allowed):
            raise ValueError("Figure context witnesses are duplicated, unknown or belong to another figure")
        if decision.classification != "uncertain" and not decision.witness_span_ids:
            raise ValueError("Figure context decision lacks original-page witnesses")
    assigned = [p for p in response.part_roles if p.role == "caption"]
    assignment = (
        response.target == "matched"
        and len(assigned) == 1
        and bool(parts[assigned[0].part_id]["caption_label"])
        and all(p.role != "uncertain" for p in response.part_roles)
    )
    if not assignment and any(d.classification != "uncertain" for d in response.decisions):
        raise ValueError("Figure context decisive observation lacks confirmed target/caption assignment")
    return response, {
        "schema_version": VERSION,
        "context_id": context["context_id"],
        "source_hashes": context["source_hashes"],
        "page": context["page"],
        "bbox_points": context["bbox_points"],
        "caption_assignment": "confirmed" if assignment else "uncertain",
        "caption_parts": context["caption_parts"],
        "page_spans": context["page_spans"],
        "response": raw,
        "interpretation_scope": "model_visual_association",
    }
