from __future__ import annotations

import glob
import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from util.fs import ensure_dir, write_text
from util.subprocess_runner import CommandResult, persist_command_result, run_command


@dataclass(frozen=True)
class MinerUResult:
    success: bool
    command: list[str]
    output_md: str
    stdout_log: str
    stderr_log: str
    command_log: str
    note: str = ""


def mineru_available() -> bool:
    """
    MinerU's PDF extractor is distributed as the `mineru` CLI.
    The dependency is optional to install, but the evaluation workflow may require it
    (see prepare node behavior; can be bypassed with --no-pdf-extract).
    """
    return shutil.which("mineru") is not None


def extract_with_mineru(
    *,
    pdf_path: str | Path,
    out_dir: str | Path,
    logs_dir: str | Path,
    timeout_sec: int = 1800,
) -> MinerUResult:
    """
    Run MinerU to convert a paper PDF into markdown.

    We use the official `mineru` CLI:
    - mineru -p <input_pdf> -o <output_dir>

    Note: on Windows, user-site packages (AppData\\Roaming\\Python\\...) may shadow conda env deps
    and cause import errors. We set PYTHONNOUSERSITE=1 for this subprocess by default.
    """
    pdfp = Path(pdf_path)
    outd = ensure_dir(out_dir)
    logsd = ensure_dir(logs_dir)

    # Preferred output path (our convention); some CLI versions write to a directory instead.
    preferred_out_md = outd / "paper.mineru.md"

    # Prefer pipeline backend for broad compatibility (CPU-friendly).
    backend = str(os.getenv("MINERU_LOCAL_BACKEND") or "pipeline").strip()
    device = str(os.getenv("MINERU_LOCAL_DEVICE") or "cpu").strip()
    source = str(os.getenv("MINERU_LOCAL_SOURCE") or "").strip()  # huggingface/modelscope/local

    base_cmd: list[str] = ["mineru", "-p", str(pdfp), "-o", str(outd)]
    if backend:
        base_cmd += ["-b", backend]
    if device and backend == "pipeline":
        base_cmd += ["-d", device]
    if source:
        base_cmd += ["--source", source]

    # Retry a couple variants (some environments dislike table/formula on CPU).
    attempts: list[list[str]] = [
        base_cmd,
        [*base_cmd, "-t", "false"],
        [*base_cmd, "-f", "false"],
        [*base_cmd, "-t", "false", "-f", "false"],
    ]
    res: CommandResult | None = None
    cmd: list[str] = []
    for c in attempts:
        cmd = c
        env = os.environ.copy()
        env.setdefault("PYTHONNOUSERSITE", "1")
        res = run_command(cmd=cmd, cwd=str(pdfp.parent), timeout_sec=timeout_sec, env=env)
        # Always persist the last attempt (single prefix for stable filenames)
        persist_command_result(res, logsd, prefix="pdf_mineru")
        if res.returncode == 0:
            break

    # Some versions may choose to write to a directory or ignore -o; keep a hint file for debugging.
    hint = (
        f"preferred_out_md: {preferred_out_md}\n"
        f"preferred_exists: {preferred_out_md.exists()}\n"
        f"cwd: {pdfp.parent}\n"
        f"cmd: {' '.join(cmd)}\n"
        f"rc: {res.returncode if res else 'N/A'}\n"
    )
    write_text(outd / "mineru_hint.txt", hint)

    cmd_log = str(Path(logsd) / "pdf_mineru_command.txt")
    stdout_log = str(Path(logsd) / "pdf_mineru_stdout.log")
    stderr_log = str(Path(logsd) / "pdf_mineru_stderr.log")
    # Determine actual output markdown and flatten MinerU's default output structure.
    #
    # MinerU typically writes: <out_dir>/<pdf_stem>/auto/<pdf_stem>.md (+ json/pdf/images).
    # User preference: keep a single folder (out_dir) without nested long-name subfolders.
    #
    # Strategy:
    # - locate the largest markdown under out_dir
    # - treat its parent as the "auto" directory
    # - copy key artifacts into out_dir with stable names
    # - copy images into out_dir/images
    # - keep the preferred_out_md as the main entry point
    out_md_path = preferred_out_md
    cands: list[Path] = []
    for pat in ["*.md", "**/*.md", "*.markdown", "**/*.markdown"]:
        cands.extend([Path(p) for p in glob.glob(str(outd / pat), recursive=True)])
    cands = [p for p in cands if p.exists() and p.is_file()]
    # Prefer MinerU's raw output markdown (typically under */auto/*) over our stable copy
    # to ensure we can locate the full artifact set to flatten.
    cands = [p for p in cands if p.resolve() != preferred_out_md.resolve()]
    auto_dir: Path | None = None
    if cands:

        def _score(p: Path) -> tuple[int, int, int]:
            # is_auto: prioritize files under an "auto" directory
            parts = [x.lower() for x in p.parts]
            is_auto = 1 if "auto" in parts else 0
            try:
                size = int(p.stat().st_size)
            except Exception:
                size = 0
            depth = len(parts)
            return (is_auto, size, depth)

        try:
            cands.sort(key=_score, reverse=True)
        except Exception:
            pass
        out_md_path = cands[0]
        auto_dir = out_md_path.parent
        # Always copy the main md to our stable path.
        try:
            preferred_out_md.write_text(
                out_md_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8", errors="ignore"
            )
            out_md_path = preferred_out_md
        except Exception:
            pass

    # Copy selected artifacts next to paper.mineru.md (best-effort).
    if auto_dir and auto_dir.exists():
        # Copy json/pdf siblings with stable filenames.
        for src in auto_dir.glob("*.json"):
            name = src.name.lower()
            if name.endswith("_content_list.json"):
                dst = outd / "paper.mineru.content_list.json"
            elif name.endswith("_middle.json"):
                dst = outd / "paper.mineru.middle.json"
            elif name.endswith("_model.json"):
                dst = outd / "paper.mineru.model.json"
            else:
                continue
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        for src in auto_dir.glob("*.pdf"):
            name = src.name.lower()
            if name.endswith("_origin.pdf"):
                dst = outd / "paper.mineru.origin.pdf"
            elif name.endswith("_layout.pdf"):
                dst = outd / "paper.mineru.layout.pdf"
            elif name.endswith("_span.pdf"):
                dst = outd / "paper.mineru.span.pdf"
            else:
                continue
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        # Copy images folder.
        img_src = auto_dir / "images"
        if img_src.exists() and img_src.is_dir():
            img_dst = outd / "images"
            try:
                if img_dst.exists():
                    shutil.rmtree(img_dst, ignore_errors=True)
                shutil.copytree(img_src, img_dst)
            except Exception:
                pass

        # Remove the nested MinerU output folder (<out_dir>/<pdf_stem>/...) to keep only one folder.
        # We only remove it if it is inside outd and not the outd itself.
        try:
            # Typically: <out_dir>/<pdf_stem>/auto. Remove <out_dir>/<pdf_stem>.
            root_candidate = auto_dir.parent if auto_dir.name.lower() == "auto" else auto_dir
            if root_candidate.exists() and root_candidate.is_dir():
                if (
                    str(root_candidate.resolve()).lower().startswith(str(outd.resolve()).lower())
                    and root_candidate.resolve() != outd.resolve()
                ):
                    shutil.rmtree(root_candidate, ignore_errors=True)
        except Exception:
            pass

    def _postprocess_assets_and_tables(root: Path) -> None:
        """
        After flattening, further organize:
        - keep only referenced images under images/
        - move other images to assets_misc/images/ for traceability
        - extract HTML tables into tables/ as md (HTML + Markdown)
        """
        md_path = root / "paper.mineru.md"
        if not md_path.exists():
            return
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        # 1) Image routing: keep only referenced images in images/
        img_dir = root / "images"
        if img_dir.exists() and img_dir.is_dir():
            # Collect referenced filenames from markdown
            refs = set()
            for m in re.finditer(r"!\[[^\]]*\]\(images/([^)]+)\)", md_text):
                fn = (m.group(1) or "").strip()
                if fn:
                    refs.add(fn)
            misc_dir = root / "assets_misc" / "images"
            try:
                misc_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                misc_dir = None  # type: ignore
            for p in img_dir.glob("*"):
                if not p.is_file():
                    continue
                if p.name in refs:
                    continue
                if misc_dir is None:
                    continue
                try:
                    shutil.move(str(p), str(misc_dir / p.name))
                except Exception:
                    pass

        # 2) Table extraction
        tables_root = root / "tables"
        tables_html_dir = tables_root / "html"
        tables_md_dir = tables_root / "md"
        try:
            tables_html_dir.mkdir(parents=True, exist_ok=True)
            tables_md_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        # Remove legacy combined files (older format: tables/table_###.md)
        try:
            for old in tables_root.glob("table_*.md"):
                if old.is_file():
                    old.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

        # Find HTML <table> blocks (MinerU emits raw HTML in markdown).
        blocks = list(re.finditer(r"<table>[\s\S]*?</table>", md_text))
        if not blocks:
            return

        def _strip_tags(s: str) -> str:
            # Remove nested tags in cell values; keep text.
            s2 = re.sub(r"<[^>]+>", "", s or "")
            return html.unescape(s2).strip()

        def _html_table_to_rows(table_html: str) -> list[list[str]]:
            rows: list[list[str]] = []
            for tr in re.findall(r"<tr[^>]*>[\s\S]*?</tr>", table_html):
                cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr)
                if cells:
                    rows.append([_strip_tags(c) for c in cells])
            return rows

        def _rows_to_markdown(rows: list[list[str]]) -> str:
            if not rows:
                return ""
            # Normalize width
            w = max((len(r) for r in rows), default=0)
            norm = [r + [""] * (w - len(r)) for r in rows]
            header = norm[0]
            body = norm[1:] if len(norm) > 1 else []

            def esc(x: str) -> str:
                return (x or "").replace("\n", " ").replace("|", "\\|").strip()

            lines = []
            lines.append("| " + " | ".join(esc(x) for x in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for r in body:
                lines.append("| " + " | ".join(esc(x) for x in r) + " |")
            return "\n".join(lines) + "\n"

        index: list[dict] = []
        for i, b in enumerate(blocks, 1):
            table_html = b.group(0)
            # Caption: look for "Table N:" in a small window after the table, else before.
            caption = ""
            after = md_text[b.end() : b.end() + 400]
            mcap = re.search(r"(Table\s+\d+\s*:\s*[^\n]+)", after)
            if not mcap:
                before = md_text[max(0, b.start() - 400) : b.start()]
                mcap = re.search(r"(Table\s+\d+\s*:\s*[^\n]+)", before)
            if mcap:
                caption = (mcap.group(1) or "").strip()

            rows = _html_table_to_rows(table_html)
            md_table = _rows_to_markdown(rows)

            table_id = f"table_{i:03d}"
            out_html_path = tables_html_dir / f"{table_id}.html"
            out_md_path = tables_md_dir / f"{table_id}.md"
            try:
                # HTML: store raw table (with a minimal wrapper for readability).
                out_html_path.write_text(
                    "<!-- generated by execution-stage MinerU postprocess -->\n"
                    + (f"<!-- {caption} -->\n" if caption else "")
                    + table_html
                    + "\n",
                    encoding="utf-8",
                    errors="ignore",
                )
                # Markdown: store best-effort markdown plus caption.
                out_md_path.write_text(
                    f"# {table_id}\n\n"
                    + (f"**Caption**: {caption}\n\n" if caption else "")
                    + (
                        "(Note: rowspan/colspan may not be represented perfectly in this Markdown view.)\n\n"
                        if ("rowspan" in table_html or "colspan" in table_html)
                        else ""
                    )
                    + (md_table if md_table.strip() else "(empty table)\n"),
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            index.append(
                {
                    "id": table_id,
                    "caption": caption,
                    "path_html": str(out_html_path),
                    "path_md": str(out_md_path),
                    "has_rowspan_colspan": bool(("rowspan" in table_html) or ("colspan" in table_html)),
                }
            )

        try:
            (tables_root / "index.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", errors="ignore"
            )
        except Exception:
            pass

    # Always postprocess after successful run attempt (best-effort).
    try:
        _postprocess_assets_and_tables(outd)
    except Exception:
        pass

    # Some versions may print a fatal exception but still exit 0. Treat that as failure.
    stderr_txt = (res.stderr or "") if res else ""
    stdout_txt = (res.stdout or "") if res else ""
    has_traceback = ("Traceback (most recent call last)" in stderr_txt) or (
        "Traceback (most recent call last)" in stdout_txt
    )
    has_fatal_error = ("FileNotFoundError" in stderr_txt) or ("FileNotFoundError" in stdout_txt)
    ok = (
        (res is not None)
        and (res.returncode == 0)
        and (not has_traceback)
        and (not has_fatal_error)
        and out_md_path.exists()
    )
    note = ""
    if res is not None and res.returncode == 0 and (has_traceback or has_fatal_error):
        note = "mineru_printed_error_but_exit_0"
    elif res is not None and res.returncode == 0 and (not out_md_path.exists()):
        note = "mineru_returned_success_but_output_missing"
    if res is not None and res.returncode != 0:
        note = "mineru_failed"

    return MinerUResult(
        success=ok,
        command=cmd,
        output_md=str(out_md_path),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        command_log=cmd_log,
        note=note,
    )
