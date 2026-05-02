from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse
from urllib.request import Request, urlopen

from util.run_layout import slugify_run_key

PDF_MAGIC = b"%PDF-"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PaperInput:
    source: str
    source_type: str
    path: Path
    downloaded: bool = False


def is_url(value: str | None) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def looks_like_pdf(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.read_bytes()[: len(PDF_MAGIC)] == PDF_MAGIC
    except Exception:
        return False


def infer_paper_key(source: str | None, *, fallback: str = "paper") -> str:
    token = str(source or "").strip()
    if not token:
        return fallback
    if is_url(token):
        parsed = urlparse(token)
        arxiv_id = _arxiv_id_from_path(parsed.path)
        if arxiv_id:
            return arxiv_id.replace("/", "_")
        name = Path(unquote(parsed.path)).name
        if name:
            stem = name[:-4] if name.lower().endswith(".pdf") else Path(name).stem
            if stem:
                return stem
        return parsed.netloc.split(":")[0] or fallback
    return Path(token).expanduser().stem or fallback


def materialize_paper_pdf(
    source: str | Path, destination_dir: str | Path, *, paper_key: str = ""
) -> PaperInput:
    raw_source = str(source).strip()
    if not raw_source:
        raise ValueError("paper PDF input is required")

    destination = Path(destination_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if is_url(raw_source):
        pdf_url = _normalize_pdf_url(raw_source)
        filename = _filename_for_url(pdf_url, paper_key=paper_key)
        target = _dedupe_path(destination / filename)
        _download_pdf(pdf_url, target)
        return PaperInput(source=raw_source, source_type="url", path=target.resolve(), downloaded=True)

    source_path = Path(raw_source).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"paper pdf not found: {source_path}")
    if not looks_like_pdf(source_path):
        raise ValueError(f"paper input is not a valid PDF: {source_path}")

    filename = _safe_pdf_filename(source_path.name, paper_key=paper_key)
    target = destination / filename
    if source_path != target.resolve():
        target = _dedupe_path(target)
        shutil.copy2(source_path, target)
    else:
        target = source_path
    return PaperInput(source=raw_source, source_type="path", path=target.resolve(), downloaded=False)


def _normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("arxiv.org"):
        arxiv_id = _arxiv_id_from_path(parsed.path)
        if arxiv_id:
            pdf_path = f"/pdf/{arxiv_id}"
            if not pdf_path.lower().endswith(".pdf"):
                pdf_path = f"{pdf_path}.pdf"
            return urlunparse((parsed.scheme, parsed.netloc, pdf_path, "", "", ""))
    return url


def _arxiv_id_from_path(path: str) -> str:
    clean = unquote(path or "").strip("/")
    for prefix in ("abs/", "pdf/"):
        if clean.startswith(prefix):
            token = clean[len(prefix) :].strip("/")
            if token.lower().endswith(".pdf"):
                token = token[:-4]
            return token
    return ""


def _filename_for_url(url: str, *, paper_key: str = "") -> str:
    parsed = urlparse(url)
    arxiv_id = _arxiv_id_from_path(parsed.path)
    if arxiv_id:
        return _safe_pdf_filename(arxiv_id.replace("/", "_"), paper_key=paper_key)
    name = _safe_pdf_filename(Path(unquote(parsed.path)).name, paper_key=paper_key)
    if name != "paper.pdf":
        return name
    return name


def _safe_pdf_filename(name: str, *, paper_key: str = "") -> str:
    token = unquote(str(name or "")).replace("\\", "/").split("/")[-1].strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("._-")
    if not token:
        token = slugify_run_key(paper_key, fallback="paper")
    if not token.lower().endswith(".pdf"):
        token = f"{token}.pdf"
    return token


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".pdf"
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not allocate destination path under {path.parent}")


def _download_pdf(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "FactReview/0.1"})
    temp = target.with_suffix(f"{target.suffix}.part")
    try:
        with urlopen(request, timeout=60) as response, open(temp, "wb") as fh:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
        if not looks_like_pdf(temp):
            raise ValueError(f"downloaded content is not a valid PDF: {url}")
        temp.replace(target)
    except Exception:
        try:
            temp.unlink()
        except Exception:
            pass
        raise
