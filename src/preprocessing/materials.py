"""Prepare v2 inputs once, without starting the legacy review agent.

MinerU content-list boxes use a 0..1000 coordinate system by default. Callers
with PDF-point boxes must explicitly select ``bbox_space="pdf_points"`` (or
set ``bbox_space`` on a row); coordinate magnitudes are never used to guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

from preprocessing.parse.mineru_adapter import MineruAdapter, MineruConfig, MineruParseResult
from schemas.claim import ClaimLocation
from schemas.materials import (
    FigureMaterial,
    MaterialBlock,
    PageImage,
    RepositoryFile,
    RepositoryIndex,
    SharedMaterials,
)

BBoxSpace = Literal["normalized_1000", "pdf_points"]
_HEADINGS = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$")
_ANCHOR_NUMBER = r"(?:[A-Za-z]\.?)?\d+(?:\.\d+)*"
_PANEL_SUFFIX = r"(?:[a-z]|\s*\(\s*[a-z]\s*\))?"
_FIGURE_ANCHOR = rf"{_ANCHOR_NUMBER}{_PANEL_SUFFIX}(?![\w]|\.\d)"
_FIGURE_REF = re.compile(
    rf"\bfig(?:ure)?s?\.?\s*({_FIGURE_ANCHOR}"
    rf"(?:\s*(?:,\s*(?:(?:and|&)\s*)?|(?:and|&|[-–])\s*){_FIGURE_ANCHOR})*)",
    re.IGNORECASE,
)
_REFERENCE_HEADING = re.compile(r"^(?:\d+[.\s]+)?(?:references|bibliography)$", re.IGNORECASE)
_METADATA_ROW_TYPES = {"header", "footer", "page_number", "page_num", "page_footnote", "aside_text"}
_IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
_TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".ps1",
    ".bat",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".c",
    ".cpp",
    ".h",
    ".js",
    ".ts",
    ".ipynb",
    ".r",
    ".jl",
    ".java",
    ".go",
    ".rs",
    ".cu",
    ".cuh",
    ".cc",
    ".cxx",
    ".hpp",
    ".m",
    ".f",
    ".f90",
    ".f95",
    ".scala",
    ".sql",
}


def index_repository(root: Path) -> RepositoryIndex:
    """Read a released-repository snapshot; never execute or modify its files."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    index = RepositoryIndex(root=str(root))
    paths = []
    for directory, directories, names in os.walk(root, followlinks=False):
        directories[:] = [
            name
            for name in directories
            if name not in _IGNORED_DIRS
            and not (Path(directory) / name).is_symlink()
            and not getattr(Path(directory) / name, "is_junction", lambda: False)()
        ]
        paths.extend(Path(directory) / name for name in names)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRS for part in relative.parts) or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        suffix, name = path.suffix.lower(), path.name.lower()
        kind = "asset"
        if suffix in {".md", ".rst"} or name.startswith(("readme", "license")):
            kind = "documentation"
        elif suffix in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"} or name in {
            "requirements.txt",
            "dockerfile",
            "makefile",
            "environment.yml",
        }:
            kind = "config"
        elif suffix in {".sh", ".ps1", ".bat"} or (
            suffix == ".py"
            and re.search(r"(?:^|_)(?:main|run|train|eval|evaluate|infer|predict)(?:_|$)", path.stem)
        ):
            kind = "entry"
        elif suffix in _TEXT_SUFFIXES:
            kind = "source"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        line_count = None
        if suffix in _TEXT_SUFFIXES or kind in {"config", "documentation"}:
            try:
                line_count = len(path.read_text(encoding="utf-8").splitlines())
            except (UnicodeError, OSError):
                pass
        row = RepositoryFile(
            path=relative.as_posix(), kind=kind, line_count=line_count, sha256=digest.hexdigest()
        )
        index.files.append(row)
        if kind == "entry":
            index.entry_scripts.append(row.path)
        if kind == "config":
            index.configs.append(row.path)
    return index


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _content_text(item)))
    if isinstance(value, dict):
        return _content_text(value.get("text") or value.get("content"))
    return ""


def _row_text(row: dict[str, Any]) -> str:
    kind = str(row.get("type") or "text")
    keys = {
        "image": ("image_caption", "caption"),
        "chart": ("chart_caption", "caption", "content"),
        "table": ("table_caption", "table_body", "text"),
        "equation": ("text", "equation"),
        "list": ("list_items", "text"),
    }.get(kind, ("text",))
    parts = []
    for key in keys:
        value = _content_text(row.get(key))
        if value:
            parts.append(value)
    return "\n".join(parts)


def _page(row: dict[str, Any]) -> int | None:
    idx = row.get("page_idx")
    if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
        return idx + 1
    number = row.get("page") or row.get("page_number")
    if isinstance(number, int) and not isinstance(number, bool) and number > 0:
        return number
    return None


def _span(markdown: str, text: str, cursor: int) -> tuple[int, int] | None:
    start = markdown.find(text, cursor)
    if start >= 0:
        return start, start + len(text)
    # MinerU can normalize whitespace between markdown and content-list text.
    pattern = r"\s+".join(re.escape(word) for word in text.split())
    if pattern:
        match = re.search(pattern, markdown[cursor:])
        if match:
            return cursor + match.start(), cursor + match.end()
    return None


def _anchors(text: str) -> set[str]:
    anchors = set()
    for match in _FIGURE_REF.finditer(text):
        group = match.group(1)
        # Panel labels identify part of the same figure and are kept in the
        # original reference text, while linkage uses its parent figure number.
        anchors.update(
            value.lower()
            for value in re.findall(rf"({_ANCHOR_NUMBER}){_PANEL_SUFFIX}(?![\w]|\.\d)", group, re.IGNORECASE)
        )
        for start, end in re.findall(r"\b(\d+)\s*[-–]\s*(\d+)\b", group):
            if 0 < int(end) - int(start) <= 30:
                anchors.update(str(value) for value in range(int(start), int(end) + 1))
    return anchors


def _sentences(block: MaterialBlock) -> list[MaterialBlock]:
    results = []
    start = 0
    spans = []
    for boundary in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", block.text):
        prefix = block.text[: boundary.start()]
        if re.search(r"\b(?:figs?|figures?|eqs?|secs?|e\.g|i\.e|et al)\.$|\b[A-Z]\.$", prefix, re.IGNORECASE):
            continue
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(block.text)))
    for start, end in spans:
        raw = block.text[start:end]
        text = raw.strip()
        if not text:
            continue
        loc = block.loc.model_copy() if block.loc else None
        if loc and loc.char_start is not None:
            leading = len(raw) - len(raw.lstrip())
            loc.char_start += start + leading
            loc.char_end = loc.char_start + len(text)
        results.append(
            MaterialBlock(
                id=f"{block.id}.s{len(results) + 1}",
                text=text,
                kind="text",
                loc=loc,
                citations=block.citations,
            )
        )
    return results


def _blocks(markdown: str, rows: list[dict[str, Any]], issues: list[str]) -> list[MaterialBlock]:
    headings = list(_HEADINGS.finditer(markdown))
    blocks = []
    cursor = 0
    parser_section = None
    if not rows:
        rows = [
            {"type": "text", "text": match.group(), "_span": (match.start(), match.end())}
            for match in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]+)*", markdown)
        ]
        issues.append("Parser returned no content list; text blocks have no parser page locations.")
    material_rows = []
    for idx, row in enumerate(rows, 1):
        items = row.get("list_items")
        if row.get("type") == "list" and isinstance(items, list) and items:
            # Keep item locations independent: Markdown may insert bullet
            # markers between items that are absent from parser list_items.
            for item_idx, item in enumerate(items, 1):
                item_text = _content_text(item)
                if item and not item_text:
                    issues.append(
                        f"block_{idx}.item_{item_idx}: unsupported list item content; inspect content_list.json."
                    )
                material_rows.append(
                    (
                        f"block_{idx}.item_{item_idx}",
                        {
                            **row,
                            "list_items": [],
                            "text": item_text,
                        },
                    )
                )
        else:
            material_rows.append((f"block_{idx}", row))
    for block_id, row in material_rows:
        kind = str(row.get("type") or "text")
        metadata = kind in _METADATA_ROW_TYPES
        text = _row_text(row)
        if not metadata and kind not in {
            "text",
            "list",
            "table",
            "equation",
            "image",
            "chart",
            "ref_text",
            "title",
            "heading",
        }:
            issues.append(f"{block_id}: unsupported parser row type {kind!r}; inspect content_list.json.")
        if not text:
            content_keys = set(row) - {
                "type",
                "page_idx",
                "page",
                "page_number",
                "bbox",
                "bbox_space",
                "text_level",
                "img_path",
                "image_path",
                "id",
                "list_items",
            }
            if any(row[key] for key in content_keys):
                issues.append(
                    f"{block_id}: parser content could not be converted to text; inspect content_list.json."
                )
            continue
        # Headers, page numbers and marginal notes may be absent from Markdown.
        # Their repeated text must not consume a later body passage or image hash.
        span = None if metadata else row.get("_span") or _span(markdown, text, cursor)
        if span:
            cursor = span[1]
            text = markdown[span[0] : span[1]]
        elif not metadata:
            issues.append(f"{block_id}: parser text could not be aligned to markdown; char span unavailable.")
        if row.get("text_level") and not metadata:
            parser_section = f"{block_id}: {text.lstrip('# ').strip()}"
        section = parser_section
        for number, heading in enumerate(headings, 1):
            if span and heading.start() <= span[0]:
                section = f"section_{number}: {heading.group(1).strip()}"
        page = _page(row)
        if page is None and section:
            page_match = re.search(r": Page (\d+)$", section, re.IGNORECASE)
            page = int(page_match.group(1)) if page_match else None
        loc = (
            ClaimLocation(
                page=page,
                section=section,
                char_start=span[0] if span else None,
                char_end=span[1] if span else None,
            )
            if (page or section or span)
            else None
        )
        if loc is None:
            issues.append(f"{block_id}: parser supplied no verifiable location.")
        if not metadata and (
            row.get("text_level") or any(span and h.start() <= span[0] < h.end() for h in headings)
        ):
            kind = "heading"
        blocks.append(
            MaterialBlock(
                id=block_id, text=text, kind=kind, loc=loc, citations=re.findall(r"\[[^\]\n]+\]", text)
            )
        )
    return blocks


def _bbox(row: dict[str, Any], width: float, height: float, default: BBoxSpace) -> tuple[float, ...]:
    value = row.get("bbox")
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("missing or invalid parser bbox")
    coords = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coords):
        raise ValueError("non-finite parser bbox")
    space = row.get("bbox_space", default)
    if space == "normalized_1000":
        if not all(0 <= item <= 1000 for item in coords):
            raise ValueError("normalized bbox outside 0..1000")
        coords = tuple(item * (width if i % 2 == 0 else height) / 1000 for i, item in enumerate(coords))
    elif space != "pdf_points":
        raise ValueError(f"unsupported bbox space: {space}")
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox is outside the page or has non-positive area")
    return coords


def build_materials(
    parsed: MineruParseResult,
    *,
    paper_pdf: Path,
    output_dir: Path,
    paper_key: str,
    repo_root: Path | None = None,
    bbox_space: BBoxSpace = "normalized_1000",
) -> SharedMaterials:
    """Materialize parser output and real PDF crops; retain unavailable inputs as issues."""
    import fitz
    from PIL import Image

    if bbox_space not in {"normalized_1000", "pdf_points"}:
        raise ValueError(f"unsupported bbox space: {bbox_space}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_pdf = paper_pdf.resolve(strict=True)
    markdown_path = output_dir / "paper.md"
    content_path = output_dir / "content_list.json"
    markdown_path.write_text(parsed.markdown, encoding="utf-8", newline="")
    rows = [row for row in (parsed.content_list or []) if isinstance(row, dict)]
    content_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    issues = [parsed.warning] if parsed.warning else []
    blocks = _blocks(parsed.markdown, rows, issues)
    heading_matches = list(_HEADINGS.finditer(parsed.markdown))
    title = heading_matches[0].group(1).strip() if heading_matches else ""
    if re.fullmatch(r"(?:abstract|page \d+|introduction)", title, re.IGNORECASE):
        title = ""
    abstract = ""
    for idx, heading in enumerate(heading_matches):
        if heading.group(1).strip().lower() == "abstract":
            end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(parsed.markdown)
            abstract = parsed.markdown[heading.end() : end].strip()
            break
    material = SharedMaterials(
        paper_key=paper_key,
        title=title,
        abstract=abstract,
        source_pdf=str(paper_pdf),
        markdown=parsed.markdown,
        markdown_path=str(markdown_path),
        content_list_path=str(content_path),
        provider=parsed.provider,
        blocks=blocks,
        issues=issues,
        repository=index_repository(repo_root) if repo_root else None,
    )
    in_bibliography = False
    for block in blocks:
        if block.kind == "heading":
            in_bibliography = bool(_REFERENCE_HEADING.fullmatch(block.text.lstrip("# ").strip()))
        elif block.kind == "ref_text" or (in_bibliography and block.kind in {"text", "list"}):
            material.bibliography.append(block)

    assets = output_dir / "assets"
    linked_images = []
    for name, data in (parsed.image_files or {}).items():
        destination = (assets / name.replace("\\", "/")).resolve()
        if not destination.is_relative_to(assets.resolve()):
            material.issues.append(f"Skipped unsafe parser image path: {name}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        # Keep the parser assets and the original Markdown image URLs. Editing
        # Markdown here would invalidate every previously derived character span.
        linked = (output_dir / name.replace("\\", "/")).resolve()
        if linked.is_relative_to((output_dir / "images").resolve()):
            linked_images.append((linked, data))

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    image_rows = [(i, row) for i, row in enumerate(rows, 1) if row.get("type") in {"image", "chart"}]
    if not image_rows:
        image_rows = [
            (0, {"type": "image", "caption": match.group(1), "img_path": match.group(2)})
            for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", parsed.markdown)
        ]
    with fitz.open(paper_pdf) as pdf:
        for idx, page in enumerate(pdf, 1):
            page_path = images_dir / f"page_{idx}.png"
            page.get_pixmap(dpi=200).save(page_path)
            material.pages.append(
                PageImage(
                    page=idx,
                    path=str(page_path),
                    width_points=page.rect.width,
                    height_points=page.rect.height,
                )
            )
        for idx, (row_number, row) in enumerate(image_rows, 1):
            caption = _row_text(row)
            first_reference = _FIGURE_REF.search(caption)
            caption_anchors = set()
            for line in caption.splitlines():
                label = _FIGURE_REF.match(line.lstrip())
                if label:
                    caption_anchors.update(_anchors(label.group()))
            if not caption_anchors and first_reference:
                caption_anchors.add(re.match(_ANCHOR_NUMBER, first_reference.group(1)).group().lower())
            raw_captions = row.get("chart_caption") or row.get("image_caption") or row.get("caption")
            caption_ambiguous = len(caption_anchors) > 1 or (
                isinstance(raw_captions, list)
                and sum(bool(_content_text(value)) for value in raw_captions) > 1
            )
            anchor = (
                next(iter(caption_anchors)) if len(caption_anchors) == 1 and not caption_ambiguous else ""
            )
            block = next((b for b in blocks if b.id == f"block_{row_number}"), None)
            figure = FigureMaterial(
                id=f"figure_{idx}",
                anchor=anchor,
                caption=caption,
                caption_ambiguous=caption_ambiguous,
                loc=block.loc if block else None,
            )
            if not caption:
                material.issues.append(f"{figure.id}: parser supplied no caption.")
            if len(caption_anchors) > 1:
                material.issues.append(
                    f"{figure.id}: parser combined captions for figure anchors {', '.join(sorted(caption_anchors))}; "
                    "crop-to-caption assignment is ambiguous."
                )
            elif caption_ambiguous:
                material.issues.append(
                    f"{figure.id}: parser supplied multiple captions; crop-to-caption assignment is ambiguous."
                )
            elif not anchor:
                material.issues.append(f"{figure.id}: caption has no figure citation anchor.")
            for body in blocks:
                if body.kind not in {"text", "list"} or body in material.bibliography:
                    continue
                figure.references.extend(
                    sentence for sentence in _sentences(body) if caption_anchors & _anchors(sentence.text)
                )
            try:
                page_number = _page(row)
                if page_number is None or page_number > len(pdf):
                    raise ValueError("missing or out-of-range parser page")
                page = pdf[page_number - 1]
                box = _bbox(row, page.rect.width, page.rect.height, bbox_space)
                figure.bbox_points = box
                crop = images_dir / f"{figure.id}.png"
                printed = images_dir / f"{figure.id}_printed.png"
                page.get_pixmap(dpi=200, clip=fitz.Rect(box)).save(crop)
                # Resize to the physical box dimensions: renderer clip rounding
                # can otherwise add two extra pixels at non-integer origins.
                size = (
                    max(1, round((box[2] - box[0]) * 96 / 72)),
                    max(1, round((box[3] - box[1]) * 96 / 72)),
                )
                with Image.open(crop) as image:
                    image.resize(size, Image.Resampling.LANCZOS).save(printed, dpi=(96, 96))
                figure.crop_path = str(crop)
                figure.printed_crop_path = str(printed)
                if figure.loc is None:
                    figure.loc = ClaimLocation(page=page_number)
            except (ValueError, TypeError, OverflowError) as exc:
                material.issues.append(f"{figure.id}: crop/printed-size input unavailable: {exc}")
            material.figures.append(figure)
    for linked, data in linked_images:
        if linked.exists() and linked.read_bytes() != data:
            material.issues.append(
                f"Skipped parser image link colliding with a generated artifact: {linked.name}"
            )
            continue
        linked.parent.mkdir(parents=True, exist_ok=True)
        linked.write_bytes(data)
    (output_dir / "materials.json").write_text(material.model_dump_json(indent=2), encoding="utf-8")
    return material


async def parse_materials(
    *,
    paper_pdf: Path,
    output_dir: Path,
    paper_key: str,
    repo_root: Path | None = None,
    parser: MineruAdapter | None = None,
    bbox_space: BBoxSpace = "normalized_1000",
) -> SharedMaterials:
    """Call MinerU directly; no claim extraction, retrieval, or report work runs here."""
    if parser is None:
        from common.config import get_settings

        settings = get_settings()
        parser = MineruAdapter(
            MineruConfig(
                base_url=settings.mineru_base_url,
                api_token=settings.mineru_api_token,
                model_version=settings.mineru_model_version,
                upload_endpoint=settings.mineru_upload_endpoint,
                poll_endpoint_templates=settings.mineru_poll_templates(),
                poll_interval_seconds=settings.mineru_poll_interval_seconds,
                poll_timeout_seconds=settings.mineru_poll_timeout_seconds,
                allow_local_fallback=settings.mineru_allow_local_fallback,
            )
        )
    parsed = await parser.parse_pdf(pdf_path=paper_pdf, data_id=paper_key)
    return build_materials(
        parsed,
        paper_pdf=paper_pdf,
        output_dir=output_dir,
        paper_key=paper_key,
        repo_root=repo_root,
        bbox_space=bbox_space,
    )
