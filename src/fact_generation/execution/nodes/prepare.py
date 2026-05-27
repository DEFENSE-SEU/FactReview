from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from preprocessing.parse.mineru import extract_with_mineru, mineru_available
from util.fs import copy_file_if_exists, ensure_dir, write_text
from util.paper_input import infer_paper_key, is_url, materialize_paper_pdf
from util.recorder import append_event
from util.run_layout import build_run_dir, ensure_run_subdirs, make_run_id, slugify_run_key
from util.subprocess_runner import persist_command_result, run_command

from ..tools.docker import _collect_repo_requirements_text, docker_ensure_paper_image, docker_strategy


class DownloadLimitError(RuntimeError):
    pass


class DownloadIncompleteError(OSError):
    pass


DEFAULT_GIT_CLONE_TIMEOUT_SEC = 900


def _repo_root() -> Path:
    """Return the FactReview repository root (where ``demos/`` and ``runs/`` live)."""
    return Path(__file__).resolve().parents[4]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_yaml_or_json(path: Path, data: Any) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        write_text(path, text)
        return
    except Exception:
        pass
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _task_risk_level(task: dict[str, Any]) -> str:
    """
    Heuristic task risk classification for auditability.
    - smoke: fast, no real training/data downloads
    - heavy: likely training / large downloads / long runtimes
    - unknown: can't tell
    """
    cmd = task.get("cmd")
    if not isinstance(cmd, list):
        return "unknown"
    s = " ".join([str(x) for x in cmd]).lower()
    timeout = int(task.get("timeout_sec") or 0)
    if "--help" in s or " -h" in s or "print('ok')" in s or 'print("ok")' in s:
        return "smoke"
    heavy_tokens = [
        "train",
        "finetune",
        "fine-tune",
        "download",
        "wget",
        "curl",
        "pip install",
        "conda install",
        "make",
    ]
    if any(t in s for t in heavy_tokens):
        return "heavy"
    if timeout >= 3600:
        return "heavy"
    return "unknown"


def _write_tasks_risk_report(tasks_path: Path, logs_dir: Path) -> None:
    try:
        import yaml  # type: ignore

        raw = tasks_path.read_text(encoding="utf-8", errors="ignore")
        tasks = yaml.safe_load(raw)
        if not isinstance(tasks, list):
            return
        report = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            report.append(
                {
                    "id": str(t.get("id") or ""),
                    "enabled": bool(t.get("enabled", True)),
                    "timeout_sec": int(t.get("timeout_sec") or 0),
                    "risk": _task_risk_level(t),
                    "cmd": t.get("cmd"),
                }
            )
        write_text(
            logs_dir / "tasks_risk_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        return


def _parse_requirements_pins(req_text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (req_text or "").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if "==" not in s:
            continue
        name, ver = s.split("==", 1)
        name = name.strip()
        ver = ver.strip()
        if name and ver:
            pins[name] = ver
    return pins


def _infer_python_spec_from_requirements_text(txt: str) -> str:
    pins = _parse_requirements_pins(txt)
    torch_ver = pins.get("torch") or pins.get("pytorch") or ""
    if torch_ver.startswith("1.4.") or torch_ver == "1.4.0":
        return "3.7"
    if torch_ver.startswith(
        (
            "1.8.",
            "1.9.",
            "1.10.",
            "1.11.",
            "1.12.",
            "1.13.",
        )
    ):
        return "3.10"
    # Conservative: old numpy pins often imply Python <= 3.7 for many research repos.
    numpy_ver = pins.get("numpy") or ""
    if numpy_ver.startswith("1.16.") or numpy_ver.startswith("1.17."):
        return "3.7"
    return "3.11"


def _infer_python_spec_from_requirements(req_path: Path) -> str:
    txt = _read_text(req_path) if req_path.exists() else ""
    return _infer_python_spec_from_requirements_text(txt)


def _python_spec_from_requires_python(specifier: str) -> str:
    s = str(specifier or "").replace(" ", "")
    if not s:
        return ""
    if re.search(r"(>=|==|~=)3\.12", s) or re.search(r">3\.11", s):
        return "3.12"
    if re.search(r"(>=|==|~=)3\.11", s) or re.search(r">3\.10", s):
        return "3.11"
    if re.search(r"(==|~=)3\.10", s):
        return "3.10"
    if re.search(r"(==|~=)3\.7", s):
        return "3.7"
    if re.search(r"<3\.8", s):
        return "3.7"
    if re.search(r"<3\.11", s):
        return "3.10"
    return ""


def _infer_python_spec_from_pyproject(repo_root: Path) -> str:
    pyproject = Path(repo_root) / "pyproject.toml"
    if not pyproject.exists():
        return ""
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    project = data.get("project") if isinstance(data, dict) else {}
    if not isinstance(project, dict):
        return ""
    spec = _python_spec_from_requires_python(str(project.get("requires-python") or ""))
    if spec:
        return spec
    classifiers = project.get("classifiers")
    if isinstance(classifiers, list):
        versions = {str(item).rsplit("::", 1)[-1].strip() for item in classifiers}
        if "3.12" in versions and not any(v in versions for v in {"3.7", "3.8", "3.9", "3.10", "3.11"}):
            return "3.12"
    return ""


def _environment_file_candidates(repo_root: Path, *, max_files: int = 20) -> list[Path]:
    root = Path(repo_root)
    patterns = ["environment.yml", "environment.yaml", "conda.yml", "conda.yaml", "env.yml", "env.yaml"]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(p for p in root.glob(pattern) if p.is_file())
    if len(out) < max_files:
        for p in root.rglob("*"):
            if len(out) >= max_files:
                break
            if p.is_file() and p.name.lower() in patterns and p not in out:
                out.append(p)
    return out[:max_files]


def _infer_python_spec_from_environment_files(repo_root: Path) -> str:
    for path in _environment_file_candidates(repo_root):
        text = _read_text(path)
        for line in text.splitlines():
            m = re.match(
                r"^\s*-\s*python\s*(?:=|==|>=|<=|~=)?\s*([0-9]+\.[0-9]+)",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                return _normalize_python_version(m.group(1))
        m = re.search(r"\bpython\s*=\s*['\"]?([0-9]+\.[0-9]+)", text, flags=re.IGNORECASE)
        if m:
            return _normalize_python_version(m.group(1))
    return ""


def _normalize_python_version(version: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)", str(version or "").strip())
    return f"{m.group(1)}.{m.group(2)}" if m else ""


def _infer_python_spec_from_repo(repo_root: Path) -> str:
    pyproject_spec = _infer_python_spec_from_pyproject(repo_root)
    if pyproject_spec:
        return pyproject_spec
    environment_spec = _infer_python_spec_from_environment_files(repo_root)
    if environment_spec:
        return environment_spec
    try:
        txt = _collect_repo_requirements_text(repo_root)
    except Exception:
        txt = ""
    return _infer_python_spec_from_requirements_text(txt)


_GITHUB_URL_PATTERN = re.compile(
    # Tolerate whitespace introduced by PDF -> markdown line wrapping inside URLs,
    # e.g. ``github.com/ org/repo`` produced by MinerU when the URL straddled a line break.
    r"(https?://)?github\.com/\s*[A-Za-z0-9_.-]+\s*/\s*[A-Za-z0-9_.-]+",
    flags=re.IGNORECASE,
)


def _harvest_github_urls(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _GITHUB_URL_PATTERN.finditer(text):
        raw = (m.group(0) or "").strip()
        # Collapse intra-URL whitespace from PDF wrapping artifacts.
        raw = re.sub(r"\s+", "", raw)
        raw = raw.rstrip(").,;:]}'\"")
        if not raw:
            continue
        if not raw.lower().startswith("http"):
            raw = "https://" + raw
        # Drop anchors / query / fragments and normalise trailing slash.
        raw = raw.split("#", 1)[0].split("?", 1)[0].rstrip("/")
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _extract_repo_urls_from_pdf(
    pdf_path: Path, max_pages: int = 0, extracted_md: Path | None = None
) -> list[str]:
    """
    Extract GitHub repository URLs from a paper.

    Looks at the parsed markdown first (when ``extracted_md`` is provided) since
    it preserves links from across the entire paper, and falls back to scanning
    the PDF directly. ``max_pages`` of ``0`` (default) scans all pages.
    """
    candidates: list[str] = []

    if extracted_md is not None:
        try:
            md_path = Path(extracted_md)
            if md_path.exists():
                md_text = md_path.read_text(encoding="utf-8", errors="ignore")
                candidates.extend(_harvest_github_urls(md_text))
        except Exception:
            pass

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        pages = reader.pages if not max_pages else reader.pages[:max_pages]
        texts: list[str] = []
        for page in pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        candidates.extend(_harvest_github_urls("\n".join(texts)))
    except Exception:
        pass

    # Deduplicate while preserving order (markdown-derived URLs first).
    seen: set[str] = set()
    deduped: list[str] = []
    for url in candidates:
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(url)
    return deduped


_RECURSIVE_COPY_IGNORED = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "wandb",
    # MinerU image dumps are not consumed downstream and can create very deep
    # paths inside execution run directories on Windows.
    "mineru_assets",
}

_ROOT_GENERATED_COPY_IGNORED = {
    "runs",
    "outputs",
    "output",
    "checkpoints",
    "checkpoint",
    "logs",
    "log",
}

_ROOT_GENERATED_COPY_IGNORED_PREFIXES = (
    "runs-",
    "runs_",
    "outputs-",
    "outputs_",
    "output-",
    "output_",
    "checkpoints-",
    "checkpoints_",
    "checkpoint-",
    "checkpoint_",
    "logs-",
    "logs_",
    "log-",
    "log_",
)


def _generated_archive_dir_ignored(part: str) -> bool:
    name = str(part or "").strip().lower()
    return name in _ROOT_GENERATED_COPY_IGNORED or name.startswith(_ROOT_GENERATED_COPY_IGNORED_PREFIXES)


def _source_dir_has_payload(path: Path) -> bool:
    """Return true when a cloned/downloaded source dir has files outside VCS metadata."""
    if not path.exists() or not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.name == ".git":
                continue
            return True
    except Exception:
        return False
    return False


def _source_dir_looks_partial_clone(path: Path) -> bool:
    return (path / ".git").exists() and not _source_dir_has_payload(path)


def _remove_tree_best_effort(path: Path) -> bool:
    if not path.exists():
        return True

    def onerror(func: Any, raw_path: str, exc_info: Any) -> None:
        try:
            os.chmod(raw_path, stat.S_IWRITE)
            func(raw_path)
        except Exception:
            pass

    for target in [path, _extended_windows_path(path) if os.name == "nt" else path]:
        try:
            shutil.rmtree(target, ignore_errors=False, onerror=onerror)
        except Exception:
            pass
        if not path.exists():
            return True
    return not path.exists()


def _robocopy_tree(src: Path, dst: Path) -> bool:
    """Copy a source tree with robocopy on Windows."""
    if os.name != "nt":
        return False
    robocopy = shutil.which("robocopy")
    if not robocopy:
        return False

    ensure_dir(dst)
    cmd = [
        robocopy,
        str(src),
        str(dst),
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
    ]
    ignored_dirs = sorted(_RECURSIVE_COPY_IGNORED | _ROOT_GENERATED_COPY_IGNORED)
    if ignored_dirs:
        cmd.extend(["/XD", *ignored_dirs])
    cmd.extend(["/XF", "*.pyc", ".DS_Store"])
    result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    if result.returncode <= 7:
        return True
    msg = "\n".join(x for x in [result.stdout.strip(), result.stderr.strip()] if x)
    raise RuntimeError(f"robocopy_failed exit_code={result.returncode}: {msg[:2000]}")


def _copy_tree(src: Path, dst: Path) -> None:
    """
    Copy src -> dst in a Windows-friendly way.

    On Windows, `shutil.rmtree(..., ignore_errors=True)` can silently fail (file locks),
    leaving the destination directory behind and causing `copytree` to raise FileExistsError.
    Prefer an explicit delete; if it fails, fall back to merge-copy when possible.
    """
    if dst.exists():
        try:
            shutil.rmtree(dst, ignore_errors=False)
        except Exception:
            # Best-effort fallback: merge into existing dir (Python 3.8+).
            try:
                if _robocopy_tree(src, dst):
                    return
                shutil.copytree(src, dst, ignore=_copy_ignore_patterns(src), dirs_exist_ok=True)
                return
            except Exception:
                # Re-raise the original intent: caller will record copy_source_failed.
                raise
    if _robocopy_tree(src, dst):
        return
    shutil.copytree(src, dst, ignore=_copy_ignore_patterns(src), dirs_exist_ok=True)


def _copy_ignore_patterns(src_root: Path):
    src_root = src_root.resolve()
    recursive_ignored = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "wandb",
        # MinerU dumps every figure as a hex-named JPG under
        # ``paper_extracted/mineru_assets/images/`` — those nested paths blow past
        # Windows' 260-char MAX_PATH when re-copied into deep run subtrees, and
        # nothing downstream consumes them. Skip the whole tree.
        "mineru_assets",
    }
    root_generated_ignored = {
        "runs",
        "outputs",
        "output",
        "checkpoints",
        "checkpoint",
        "logs",
        "log",
    }

    def ignore(current: str, names: list[str]) -> set[str]:
        ignored = set(names).intersection(recursive_ignored)
        try:
            if Path(current).resolve() == src_root:
                ignored.update(set(names).intersection(root_generated_ignored))
        except Exception:
            pass
        return ignored

    return ignore


def _configured_demo_dir(paper_key: str) -> Path | None:
    """Locate a bundled demo fixture for a paper key.

    Supports two layouts under ``demos/``:
    - flat:        ``demos/<key>/`` (legacy)
    - categorized: ``demos/<category>/<key>/`` (e.g. ``demos/Text/bert``)

    For categorized demos, ``paper_key`` may arrive as either ``<category>_<name>``
    or just ``<name>``; both are resolved by scanning one level deep.
    """
    raw_key = str(paper_key or "paper").strip()
    keys: list[str] = []
    for key in (raw_key, slugify_run_key(raw_key)):
        if key and key not in keys:
            keys.append(key)

    # Also accept the trailing component of "<category>_<name>"-style keys so
    # that e.g. ``Text_bert`` matches ``demos/Text/bert``.
    suffix_keys: list[str] = []
    for key in list(keys):
        if "_" in key:
            tail = key.split("_", 1)[1]
            for variant in (tail, tail.replace("_", "-"), slugify_run_key(tail)):
                if variant and variant not in keys and variant not in suffix_keys:
                    suffix_keys.append(variant)
    keys.extend(suffix_keys)

    demos_root = _repo_root() / "demos"

    # 1) Direct flat lookup.
    for key in keys:
        candidate = demos_root / key
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    # 2) Categorized lookup: demos/<*>/<key>.
    if demos_root.exists():
        for category in demos_root.iterdir():
            if not category.is_dir():
                continue
            for key in keys:
                candidate = category / key
                if candidate.exists() and candidate.is_dir():
                    return candidate.resolve()

    return None


def _select_demo_source(demo_dir: Path | None) -> Path | None:
    if demo_dir is None:
        return None
    for candidate in (demo_dir / "execution" / "repo",):
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate.resolve()
    return None


def _load_demo_execution_config(execution_dir: Path, demo_dir: Path) -> dict[str, Any]:
    for path in (
        execution_dir / "config.json",
        execution_dir / "execution_config.json",
        demo_dir / "execution_config.json",
    ):
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def _materialize_demo_fixture(demo_dir: Path | None, baseline_dir: Path) -> dict[str, Any]:
    if demo_dir is None:
        return {}
    execution_dir = demo_dir / "execution"

    def _copy_first(dst_name: str, candidates: tuple[Path, ...]) -> None:
        dst = baseline_dir / dst_name
        if dst.exists():
            return
        for src in candidates:
            if copy_file_if_exists(src, dst):
                return

    _copy_first(
        "tasks.yaml",
        (
            execution_dir / "tasks.yaml",
            execution_dir / "tasks.yml",
            demo_dir / "tasks.yaml",
            demo_dir / "tasks.yml",
        ),
    )
    _copy_first(
        "baseline.json",
        (
            execution_dir / "checks.json",
            execution_dir / "baseline.json",
            demo_dir / "baseline.json",
        ),
    )
    copy_file_if_exists(demo_dir / "paper.pdf", baseline_dir / "paper.pdf")
    wheelhouse = execution_dir / "wheelhouse_linux"
    if wheelhouse.exists() and wheelhouse.is_dir():
        try:
            _copy_tree(wheelhouse, baseline_dir / "wheelhouse_linux")
        except Exception:
            pass
    return _load_demo_execution_config(execution_dir, demo_dir)


def _copy_prepared_extract(prepared_extract_dir: str, baseline_dir: Path) -> str:
    token = str(prepared_extract_dir or "").strip()
    if not token:
        return ""
    src = Path(token).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        return ""
    dst = baseline_dir / "paper_extracted"
    if src.resolve() != dst.resolve():
        _copy_tree(src, dst)
    md = dst / "paper.mineru.md"
    return str(md.resolve()) if md.exists() else ""


def _anonymous_4open_repo_id(raw_url: str) -> str:
    try:
        parsed = urlparse(str(raw_url or "").strip())
    except Exception:
        return ""
    if parsed.netloc.lower() != "anonymous.4open.science":
        return ""
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] not in {"r", "repository"}:
        return ""
    return parts[1].strip()


def _anonymous_4open_get_json(path: str, timeout_sec: int = 60) -> Any:
    url = "https://anonymous.4open.science" + path
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "FactReview execution"})
            with urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            last_exc = exc
            if exc.code not in {404, 429, 500, 502, 503, 504} or attempt >= 2:
                raise
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _anonymous_4open_get_bytes(path: str, timeout_sec: int = 120) -> bytes:
    url = "https://anonymous.4open.science" + path
    return _download_url_bytes(url, timeout_sec=timeout_sec)


def _download_anonymous_4open_repo(raw_url: str, dest: Path, logs_dir: Path) -> dict[str, Any]:
    repo_id = _anonymous_4open_repo_id(raw_url)
    if not repo_id:
        raise ValueError("not_anonymous_4open_url")

    repo_q = quote(repo_id, safe="")
    errors: list[str] = []

    try:
        blob = _anonymous_4open_get_bytes(f"/api/repo/{repo_q}/zip", timeout_sec=300)
        archive_manifest = _extract_archive_bytes(blob, dest)
        manifest = {
            "source": raw_url,
            "repo_id": repo_id,
            "files": archive_manifest.get("files", 0),
            "sample_files": archive_manifest.get("sample_files", []),
            "method": "zip",
        }
        write_text(
            logs_dir / "anonymous_4open_download.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        return manifest
    except HTTPError as exc:
        errors.append(f"zip_http_{exc.code}")
    except Exception as exc:
        errors.append(f"zip_failed:{type(exc).__name__}:{exc}")

    options: dict[str, Any] = {}
    try:
        raw_options = _anonymous_4open_get_json(f"/api/repo/{repo_q}/options")
    except HTTPError as exc:
        errors.append(f"options_http_{exc.code}")
    except (URLError, TimeoutError, OSError) as exc:
        errors.append(f"options_unavailable:{exc}")
    else:
        if isinstance(raw_options, dict):
            options = raw_options
        else:
            errors.append("options_invalid")

    last_update = str(options.get("lastUpdateDate") or "")
    downloaded: list[str] = []
    visited: set[str] = set()

    def list_dir(rel_dir: str) -> list[dict[str, Any]]:
        if rel_dir in visited:
            return []
        visited.add(rel_dir)
        path_q = quote(rel_dir, safe="")
        version_q = quote(last_update, safe="")
        url = f"/api/repo/{repo_q}/files/?path={path_q}&v={version_q}"
        try:
            data = _anonymous_4open_get_json(url)
        except HTTPError as exc:
            errors.append(f"files_http_{exc.code}:{rel_dir or '/'}")
            return []
        except (URLError, TimeoutError, OSError) as exc:
            errors.append(f"files_unavailable:{rel_dir or '/'}:{exc}")
            return []
        return data if isinstance(data, list) else []

    def walk(rel_dir: str) -> None:
        for item in list_dir(rel_dir):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parent = str(item.get("path") or rel_dir or "").replace("\\", "/").strip("/")
            rel = f"{parent}/{name}".strip("/")
            if not rel or rel.startswith("../") or "/../" in rel or Path(rel).is_absolute():
                continue
            safe_rel = _safe_archive_member_path(rel)
            if safe_rel is None:
                continue
            rel = str(safe_rel).replace("\\", "/")
            size = item.get("size")
            sha = str(item.get("sha") or "")
            if sha and size is not None:
                rel_q = quote(rel, safe="/")
                version_q = quote(sha, safe="")
                try:
                    blob = _anonymous_4open_get_bytes(f"/api/repo/{repo_q}/file/{rel_q}?v={version_q}")
                except HTTPError as exc:
                    errors.append(f"file_http_{exc.code}:{rel}")
                    continue
                except (URLError, TimeoutError, OSError) as exc:
                    errors.append(f"file_unavailable:{rel}:{exc}")
                    continue
                out_path = dest / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(blob)
                downloaded.append(rel)
            else:
                walk(rel)

    if dest.exists():
        _remove_tree_best_effort(dest)
    dest.mkdir(parents=True, exist_ok=True)
    walk("")
    if not downloaded:
        detail = "; ".join(errors[:8]) if errors else "no_files_listed"
        raise RuntimeError(f"anonymous_4open_no_files_downloaded: {detail}")
    manifest = {
        "source": raw_url,
        "repo_id": repo_id,
        "files": len(downloaded),
        "last_update": last_update,
        "method": "files_api",
        "errors": errors[:20],
    }
    write_text(logs_dir / "anonymous_4open_download.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _openreview_forum_id(raw_url: str) -> str:
    try:
        parsed = urlparse(str(raw_url or "").strip())
    except Exception:
        return ""
    if parsed.netloc.lower() != "openreview.net":
        return ""
    query = parse_qs(parsed.query or "")
    forum_id = str((query.get("id") or [""])[0]).strip()
    if forum_id:
        return forum_id
    if parsed.path.rstrip("/").endswith("/pdf"):
        return str((query.get("id") or [""])[0]).strip()
    return ""


def _is_openreview_attachment_url(raw_url: str) -> bool:
    try:
        parsed = urlparse(str(raw_url or "").strip())
    except Exception:
        return False
    return parsed.netloc.lower() == "openreview.net" and (
        parsed.path.rstrip("/") == "/attachment" or parsed.path.startswith("/attachment/")
    )


def _openreview_content_value(item: Any) -> Any:
    if isinstance(item, dict) and "value" in item:
        return item.get("value")
    return item


def _openreview_note_metadata(forum_id: str, *, timeout_sec: int = 30) -> dict[str, Any]:
    fid = str(forum_id or "").strip()
    if not fid:
        return {}
    url = f"https://api2.openreview.net/notes?id={quote(fid, safe='')}"
    try:
        payload = json.loads(_download_url_bytes(url, timeout_sec=timeout_sec).decode("utf-8", errors="ignore"))
    except Exception:
        return {}
    notes = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(notes, list) or not notes or not isinstance(notes[0], dict):
        return {}
    content = notes[0].get("content")
    if not isinstance(content, dict):
        return {}
    return content


def _openreview_candidate_source_urls(content: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    source_key_re = re.compile(
        r"(?:code|software|source|repo|repository|artifact|supplement|supplementary|data|dataset)",
        flags=re.IGNORECASE,
    )
    url_re = re.compile(r"https?://[^\s<>'\")]+|/attachment(?:\?[^\s<>'\")]+|/[^\s<>'\")]+)")

    def url_looks_source(url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host in {
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "huggingface.co",
            "anonymous.4open.science",
            "zenodo.org",
            "figshare.com",
            "osf.io",
            "www.kaggle.com",
            "kaggle.com",
        }:
            return True
        if host == "openreview.net" and (
            parsed.path.rstrip("/") == "/attachment" or parsed.path.startswith("/attachment/")
        ):
            query = parse_qs(parsed.query or "")
            name = str((query.get("name") or [""])[0])
            return parsed.path.startswith("/attachment/") or bool(source_key_re.search(name))
        return False

    for key, raw_value in (content or {}).items():
        value = _openreview_content_value(raw_value)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item or "")
            if not text:
                continue
            for match in url_re.finditer(text):
                url = match.group(0).rstrip(".,;")
                if url.startswith("/attachment"):
                    url = "https://openreview.net" + url
                if not (source_key_re.search(str(key)) or url_looks_source(url)):
                    continue
                if url not in seen:
                    candidates.append(url)
                    seen.add(url)
    return candidates


def _normalize_cloneable_repo_url(raw_url: str) -> str:
    try:
        parsed = urlparse(str(raw_url or "").strip())
    except Exception:
        return ""
    host = parsed.netloc.lower()
    if host not in {"github.com", "gitlab.com", "bitbucket.org"}:
        return ""
    parts = [unquote(p).strip() for p in parsed.path.split("/") if p.strip()]
    if len(parts) < 2:
        return ""
    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return ""
    return f"https://{host}/{owner}/{repo}"


def _count_files_for_manifest(root: Path, *, limit: int = 20) -> tuple[int, list[str]]:
    count = 0
    sample: list[str] = []
    try:
        for path in root.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue
            count += 1
            if len(sample) < limit:
                sample.append(str(path.relative_to(root)).replace("\\", "/"))
    except Exception:
        return count, sample
    return count, sample


def _download_openreview_candidate_source(candidate_url: str, dest: Path, logs_dir: Path, index: int) -> dict[str, Any]:
    url = str(candidate_url or "").strip()
    if not url:
        raise RuntimeError("candidate_url_empty")
    if _is_openreview_attachment_url(url) or urlparse(url).path.lower().endswith(
        (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
    ):
        blob = _download_url_bytes(url, timeout_sec=_openreview_attachment_timeout_sec())
        archive_manifest = _extract_archive_bytes(blob, dest)
        return {
            "method": "candidate_archive",
            "candidate_url": url,
            "files": archive_manifest.get("files", 0),
            "sample_files": archive_manifest.get("sample_files", []),
        }

    anonymous_4open_id = _anonymous_4open_repo_id(url)
    if anonymous_4open_id:
        manifest = _download_anonymous_4open_repo(url, dest, logs_dir)
        manifest["method"] = f"candidate_anonymous_4open:{manifest.get('method') or 'unknown'}"
        manifest["candidate_url"] = url
        return manifest

    clone_url = _normalize_cloneable_repo_url(url)
    if not clone_url:
        raise RuntimeError("candidate_url_unsupported")
    if dest.exists():
        _remove_tree_best_effort(dest)
    ensure_dir(dest.parent)
    use_blob_filter = (
        str(os.getenv("EXECUTION_GIT_CLONE_FILTER_BLOB_NONE") or "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    clone_cmd = ["git", "clone", "--depth", "1"]
    if use_blob_filter:
        clone_cmd.extend(["--filter", "blob:none"])
    clone_cmd.extend([clone_url, str(dest)])
    clone_timeout = _git_clone_timeout_sec({})
    res = run_command(cmd=clone_cmd, cwd=str(dest.parent), timeout_sec=clone_timeout)
    persist_command_result(res, logs_dir, prefix=f"openreview_candidate_{index}_clone")
    if res.returncode != 0 and res.returncode != 124 and use_blob_filter:
        _remove_tree_best_effort(dest)
        fallback_cmd = ["git", "clone", "--depth", "1", clone_url, str(dest)]
        res = run_command(cmd=fallback_cmd, cwd=str(dest.parent), timeout_sec=clone_timeout)
        persist_command_result(res, logs_dir, prefix=f"openreview_candidate_{index}_clone_fallback")
    if res.returncode != 0:
        _remove_tree_best_effort(dest)
        if res.returncode == 124:
            raise RuntimeError(f"candidate_git_clone_timeout:{clone_timeout}")
        raise RuntimeError(f"candidate_git_clone_failed:{res.returncode}")
    files, sample = _count_files_for_manifest(dest)
    return {
        "method": "candidate_git_clone",
        "candidate_url": url,
        "clone_url": clone_url,
        "files": files,
        "sample_files": sample,
    }


def _download_openreview_candidate_sources(
    *,
    raw_url: str,
    forum_id: str,
    attachment_url: str,
    attachment_status: int,
    content: dict[str, Any],
    candidate_urls: list[str],
    dest: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, str]] = []
    for index, candidate_url in enumerate(candidate_urls, start=1):
        try:
            manifest = _download_openreview_candidate_source(candidate_url, dest, logs_dir, index)
        except Exception as exc:
            attempts.append(
                {
                    "url": str(candidate_url),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        manifest.update(
            {
                "source": raw_url,
                "attachment_url": attachment_url,
                "attachment_http_status": attachment_status,
                "forum_id": forum_id,
                "content_keys": sorted(str(k) for k in content),
                "candidate_source_urls": candidate_urls,
                "candidate_attempts": attempts,
            }
        )
        write_text(
            logs_dir / "openreview_supplementary_download.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        return manifest
    manifest = {
        "source": raw_url,
        "forum_id": forum_id,
        "content_keys": sorted(str(k) for k in content),
        "candidate_source_urls": candidate_urls,
        "attachment_url": attachment_url,
        "http_status": attachment_status,
        "candidate_attempts": attempts,
    }
    write_text(
        logs_dir / "openreview_supplementary_metadata.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    detail = "; ".join(f"{a['url']} -> {a['error']}" for a in attempts[:5]) or "no_supported_candidate"
    raise RuntimeError(f"openreview_candidate_source_download_failed: {detail}")


def _openreview_no_source_error(content_keys: list[str]) -> str:
    if not content_keys:
        return "openreview_metadata_empty_or_unavailable"
    return f"openreview_no_supplementary_or_code_url: content_keys={','.join(content_keys)}"


def _openreview_candidate_urls_for_retry(primary_url: str, candidate_urls: list[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for url in [primary_url, *list(candidate_urls or [])]:
        token = str(url or "").strip()
        if not token or token in seen:
            continue
        urls.append(token)
        seen.add(token)
    return urls


def _download_max_bytes() -> int:
    raw = os.getenv("EXECUTION_DOWNLOAD_MAX_BYTES") or os.getenv("FACTREVIEW_DOWNLOAD_MAX_BYTES") or ""
    if raw:
        try:
            return max(1_000_000, int(raw))
        except Exception:
            pass
    return 300 * 1024 * 1024


def _git_clone_timeout_sec(cfg: dict[str, Any]) -> int:
    raw = cfg.get("git_clone_timeout_sec") or os.getenv("EXECUTION_GIT_CLONE_TIMEOUT_SEC") or ""
    if str(raw).strip():
        try:
            return max(int(raw), 1)
        except Exception:
            pass
    return DEFAULT_GIT_CLONE_TIMEOUT_SEC


def _download_backend() -> str:
    return str(
        os.getenv("EXECUTION_DOWNLOAD_BACKEND")
        or os.getenv("FACTREVIEW_DOWNLOAD_BACKEND")
        or "python"
    ).strip().lower()


def _openreview_attachment_timeout_sec() -> int:
    raw = (
        os.getenv("EXECUTION_OPENREVIEW_ATTACHMENT_TIMEOUT_SEC")
        or os.getenv("FACTREVIEW_OPENREVIEW_ATTACHMENT_TIMEOUT_SEC")
        or os.getenv("EXECUTION_DOWNLOAD_TIMEOUT_SEC")
        or "600"
    )
    try:
        return max(30, int(raw))
    except Exception:
        return 600


def _download_url_bytes_curl(url: str, timeout_sec: int = 180) -> bytes:
    exe = shutil.which("curl.exe") or shutil.which("curl")
    if not exe:
        raise OSError("curl_unavailable")
    timeout = max(1, int(timeout_sec or 180))
    max_bytes = _download_max_bytes()
    fd, tmp_name = tempfile.mkstemp(prefix="factreview-download-", suffix=".bin")
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        exe,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(min(30, timeout)),
        "--max-time",
        str(timeout),
        "--max-filesize",
        str(max_bytes),
        "--output",
        str(tmp_path),
        "--user-agent",
        "FactReview execution",
        url,
    ]
    try:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"download_curl_timeout: {url}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace")
            if result.returncode == 63 or "maximum file size exceeded" in stderr.lower():
                raise DownloadLimitError(f"download_too_large: {url} max_bytes={max_bytes}")
            http_match = re.search(r"returned error:\s*(\d{3})", stderr, flags=re.IGNORECASE)
            if http_match:
                raise HTTPError(url, int(http_match.group(1)), stderr.strip() or "curl_http_error", hdrs=None, fp=None)
            raise OSError(f"download_curl_failed: rc={result.returncode} {stderr.strip()}")
        size = tmp_path.stat().st_size if tmp_path.exists() else 0
        if size > max_bytes:
            raise DownloadLimitError(f"download_too_large: {url} bytes_read={size} max_bytes={max_bytes}")
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _download_url_bytes(url: str, timeout_sec: int = 180) -> bytes:
    if _download_backend() in {"curl", "curl.exe"}:
        return _download_url_bytes_curl(url, timeout_sec=timeout_sec)
    last_exc: Exception | None = None
    max_bytes = _download_max_bytes()
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "FactReview execution"})
            deadline = time.monotonic() + max(30, int(timeout_sec or 180))
            chunks: list[bytes] = []
            with urlopen(req, timeout=min(10, max(1, int(timeout_sec or 180)))) as resp:
                content_length = str(resp.headers.get("Content-Length") or "").strip()
                expected_length: int | None = None
                if content_length:
                    try:
                        expected_length = int(content_length)
                        if expected_length > max_bytes:
                            raise DownloadLimitError(
                                f"download_too_large: {url} content_length={content_length} max_bytes={max_bytes}"
                            )
                    except ValueError:
                        pass
                total = 0
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"download_total_timeout: {url}")
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        if expected_length is not None and total < expected_length:
                            raise DownloadIncompleteError(
                                f"download_incomplete: {url} bytes_read={total} content_length={expected_length}"
                            )
                        return b"".join(chunks)
                    total += len(chunk)
                    if total > max_bytes:
                        raise DownloadLimitError(
                            f"download_too_large: {url} bytes_read={total} max_bytes={max_bytes}"
                        )
                    chunks.append(chunk)
        except HTTPError:
            raise
        except DownloadLimitError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _safe_archive_member_path(name: str) -> Path | None:
    clean = unquote(str(name or "")).replace("\\", "/").strip("/")
    if not clean or clean.startswith("__MACOSX/"):
        return None
    parts = [p for p in clean.split("/") if p and p not in {".", ".."}]
    if not parts or len(parts) != len([p for p in clean.split("/") if p]):
        return None
    lowered = [part.lower() for part in parts]
    if any(part == ".DS_Store" or part.startswith("._") for part in parts):
        return None
    if any(part in _RECURSIVE_COPY_IGNORED or _generated_archive_dir_ignored(part) for part in lowered):
        return None
    rel = Path(*parts)
    return None if rel.is_absolute() else rel


def _extended_windows_path(path: Path) -> str:
    raw = str(path.resolve())
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw.lstrip("\\")
    return "\\\\?\\" + raw


def _mkdir_for_archive_member(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        os.makedirs(_extended_windows_path(path), exist_ok=True)


def _write_archive_member_bytes(path: Path, data: bytes) -> None:
    _mkdir_for_archive_member(path.parent)
    try:
        path.write_bytes(data)
    except OSError:
        with open(_extended_windows_path(path), "wb") as fh:
            fh.write(data)


def _remove_archive_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except Exception:
        if os.name == "nt":
            shutil.rmtree(_extended_windows_path(path), ignore_errors=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def _shorten_archive_component(part: str, max_len: int = 96) -> str:
    value = str(part or "")
    if len(value) <= max_len:
        return value
    suffix = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:10]
    path = Path(value)
    ext = path.suffix if 0 < len(path.suffix) <= 16 else ""
    budget = max(12, max_len - len(suffix) - len(ext) - 1)
    return f"{value[:budget]}~{suffix}{ext}"


def _archive_short_path_limit() -> int:
    raw = os.getenv("EXECUTION_ARCHIVE_MAX_PATH") or os.getenv("FACTREVIEW_ARCHIVE_MAX_PATH") or "240"
    try:
        return max(120, int(raw))
    except Exception:
        return 240


def _shorten_archive_member_path(dest: Path, rel: Path) -> tuple[Path, str]:
    """
    Keep archive extraction usable on Windows run directories with deep names.

    Returns ``(relative_path_to_write, original_relative_path_if_rewritten)``.
    """

    original = str(rel).replace("\\", "/")
    force = str(os.getenv("EXECUTION_ARCHIVE_FORCE_SHORT_PATHS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    limit = _archive_short_path_limit()
    parts = [_shorten_archive_component(part) for part in rel.parts]
    shortened = Path(*parts)
    shortened_rel = str(shortened).replace("\\", "/")

    if not force and (os.name != "nt" or len(str(dest / rel)) < limit):
        return rel, ""

    if shortened != rel and len(shortened_rel) < limit:
        return shortened, original

    if shortened == rel and len(original) < max(96, limit // 2):
        return rel, ""

    digest = hashlib.sha1(original.encode("utf-8", errors="ignore")).hexdigest()[:16]
    leaf = _shorten_archive_component(rel.name or "member", max_len=80)
    prefix = Path(rel.parts[0]) if len(rel.parts) > 1 else Path()
    shortened = prefix / "__longpaths__" / digest / leaf
    return shortened, original if shortened != rel else ""


def _flatten_single_extracted_root(dest: Path) -> str:
    children = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(children) != 1 or not children[0].is_dir():
        return ""
    nested = children[0]
    nested_name = nested.name
    tmp = dest.with_name(dest.name + "_flat_tmp")
    if tmp.exists():
        _remove_archive_tree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for child in nested.iterdir():
        target = tmp / child.name
        try:
            shutil.move(str(child), str(target))
        except OSError:
            if child.is_dir():
                _copy_tree(child, target)
                _remove_archive_tree(child)
            else:
                _write_archive_member_bytes(target, child.read_bytes())
                child.unlink(missing_ok=True)
    _remove_archive_tree(dest)
    tmp.rename(dest)
    return nested_name


def _strip_flattened_root(path: str, root_name: str) -> str:
    value = str(path or "").replace("\\", "/")
    root = str(root_name or "").strip().strip("/")
    if root and value.startswith(root + "/"):
        return value[len(root) + 1 :]
    return value


def _extract_archive_bytes(blob: bytes, dest: Path) -> dict[str, Any]:
    if dest.exists():
        _remove_archive_tree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    path_rewrites: list[dict[str, str]] = []
    stream = BytesIO(blob)
    if zipfile.is_zipfile(stream):
        stream.seek(0)
        with zipfile.ZipFile(stream) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = _safe_archive_member_path(info.filename)
                if rel is None:
                    continue
                rel, rewritten_from = _shorten_archive_member_path(dest, rel)
                out_path = dest / rel
                _write_archive_member_bytes(out_path, zf.read(info))
                files.append(str(rel).replace("\\", "/"))
                if rewritten_from:
                    path_rewrites.append({"from": rewritten_from, "to": str(rel).replace("\\", "/")})
    else:
        stream.seek(0)
        try:
            with tarfile.open(fileobj=stream, mode="r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    rel = _safe_archive_member_path(member.name)
                    if rel is None:
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    rel, rewritten_from = _shorten_archive_member_path(dest, rel)
                    out_path = dest / rel
                    _write_archive_member_bytes(out_path, fh.read())
                    files.append(str(rel).replace("\\", "/"))
                    if rewritten_from:
                        path_rewrites.append({"from": rewritten_from, "to": str(rel).replace("\\", "/")})
        except tarfile.TarError as exc:
            raise RuntimeError("archive_format_unsupported") from exc
    if not files:
        raise RuntimeError("archive_no_files_extracted")
    flattened_root = _flatten_single_extracted_root(dest)
    if flattened_root:
        files = [_strip_flattened_root(path, flattened_root) for path in files]
        for row in path_rewrites:
            row["to"] = _strip_flattened_root(row.get("to", ""), flattened_root)
    manifest = {"files": len(files), "sample_files": files[:20]}
    if path_rewrites:
        manifest["path_rewrite_count"] = len(path_rewrites)
        manifest["path_rewrites"] = path_rewrites[:50]
    return manifest


def _download_openreview_supplementary(raw_url: str, dest: Path, logs_dir: Path) -> dict[str, Any]:
    forum_id = _openreview_forum_id(raw_url)
    write_text(
        logs_dir / "openreview_supplementary_status.json",
        json.dumps(
            {
                "status": "metadata_start",
                "source": raw_url,
                "forum_id": forum_id,
                "ts": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    content = _openreview_note_metadata(forum_id) if forum_id else {}
    candidate_urls = _openreview_candidate_source_urls(content)
    if _is_openreview_attachment_url(raw_url):
        url = raw_url
    elif forum_id:
        url = f"https://openreview.net/attachment?id={quote(forum_id, safe='')}&name=supplementary_material"
    else:
        raise ValueError("not_openreview_url")
    write_text(
        logs_dir / "openreview_supplementary_status.json",
        json.dumps(
            {
                "status": "attachment_download_start",
                "source": raw_url,
                "attachment_url": url,
                "forum_id": forum_id,
                "candidate_source_urls": candidate_urls,
                "content_keys": sorted(str(k) for k in content) if content else [],
                "ts": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    try:
        blob = _download_url_bytes(url, timeout_sec=_openreview_attachment_timeout_sec())
    except HTTPError as exc:
        if forum_id and candidate_urls:
            try:
                return _download_openreview_candidate_sources(
                    raw_url=raw_url,
                    forum_id=forum_id,
                    attachment_url=url,
                    attachment_status=int(exc.code),
                    content=content,
                    candidate_urls=candidate_urls,
                    dest=dest,
                    logs_dir=logs_dir,
                )
            except RuntimeError as candidate_exc:
                raise candidate_exc from exc
        if forum_id:
            manifest = {
                "source": raw_url,
                "forum_id": forum_id,
                "content_keys": sorted(str(k) for k in content),
                "candidate_source_urls": candidate_urls,
                "attachment_url": url,
                "http_status": exc.code,
            }
            write_text(
                logs_dir / "openreview_supplementary_metadata.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            if not candidate_urls:
                raise RuntimeError(_openreview_no_source_error(manifest["content_keys"])) from exc
        raise RuntimeError(f"openreview_supplementary_http_{exc.code}") from exc
    except (URLError, TimeoutError, OSError, DownloadLimitError) as exc:
        retry_urls = _openreview_candidate_urls_for_retry(url, candidate_urls)
        if forum_id and retry_urls:
            try:
                return _download_openreview_candidate_sources(
                    raw_url=raw_url,
                    forum_id=forum_id,
                    attachment_url=url,
                    attachment_status=0,
                    content=content,
                    candidate_urls=retry_urls,
                    dest=dest,
                    logs_dir=logs_dir,
                )
            except RuntimeError as candidate_exc:
                raise candidate_exc from exc
        raise RuntimeError(f"openreview_supplementary_unavailable: {exc}") from exc
    archive_path = logs_dir / "openreview_supplementary.archive"
    archive_path.write_bytes(blob)
    write_text(
        logs_dir / "openreview_supplementary_status.json",
        json.dumps(
            {
                "status": "archive_downloaded",
                "source": raw_url,
                "attachment_url": url,
                "forum_id": forum_id,
                "bytes": len(blob),
                "ts": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    try:
        manifest = _extract_archive_bytes(blob, dest)
    except RuntimeError as exc:
        raise RuntimeError(f"openreview_supplementary_extract_failed: {exc}") from exc
    manifest.update({"source": raw_url, "attachment_url": url, "forum_id": forum_id})
    if content:
        manifest["content_keys"] = sorted(str(k) for k in content)
    if candidate_urls:
        manifest["candidate_source_urls"] = candidate_urls
    write_text(
        logs_dir / "openreview_supplementary_download.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _normalize_shell_script_line_endings(
    root: Path, *, max_files: int = 200, max_bytes: int = 5_000_000
) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    changed = 0
    for path in root.rglob("*.sh"):
        if changed >= max_files:
            break
        if ".git" in path.parts:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            data = path.read_bytes()
        except Exception:
            continue
        if b"\r" not in data:
            continue
        normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized == data:
            continue
        try:
            path.write_bytes(normalized)
            changed += 1
        except Exception:
            continue
    return changed


_PY_LITERAL_ASSIGN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>{name})\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)(?P<tail>\s*(?:#.*)?)$",
    re.MULTILINE,
)
_SH_LITERAL_ASSIGN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<export>export\s+)?(?P<name>{name})\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)(?P<tail>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def _literal_assign_re(*names: str) -> re.Pattern[str]:
    escaped = "|".join(re.escape(name) for name in names)
    return re.compile(_PY_LITERAL_ASSIGN_RE.pattern.format(name=escaped), re.MULTILINE)


def _shell_literal_assign_re(*names: str) -> re.Pattern[str]:
    escaped = "|".join(re.escape(name) for name in names)
    return re.compile(_SH_LITERAL_ASSIGN_RE.pattern.format(name=escaped), re.MULTILINE)


def _env_fallback_expr(env_names: list[str], fallback: str) -> str:
    parts = [f"__import__('os').environ.get({name!r})" for name in env_names]
    parts.append(repr(fallback))
    return " or ".join(parts)


def _shell_env_fallback_expr(env_names: list[str], fallback: str) -> str:
    expr = fallback
    for name in reversed(env_names):
        expr = f"${{{name}:-{expr}}}"
    return f'"{expr}"'


def _looks_like_api_key_placeholder(value: str) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if not token:
        return True
    if token in {"yourapikey", "yourkey", "apikey", "api", "key", "xxx", "empty", "changeme", "none", "null"}:
        return True
    return token.startswith("your") and ("api" in token or "key" in token)


def _patch_api_placeholders_for_env(root: Path, *, max_files: int = 300, max_bytes: int = 300_000) -> list[str]:
    """
    Patch copied paper-code snapshots that contain obvious API placeholders.

    The patch never writes secrets. It only changes placeholder constants such
    as ``API_KEY = 'YOUR_API_KEY'`` to read from runtime environment variables
    first, preserving the original literal as a fallback for auditability.
    """
    if not root.exists() or not root.is_dir():
        return []

    key_re = _literal_assign_re("API_KEY", "OPENAI_API_KEY", "api_key", "openai_api_key")
    base_re = _literal_assign_re("BASE_URL", "OPENAI_BASE_URL", "base_url", "api_base", "openai_api_base")
    model_re = _literal_assign_re("MODEL_NAME", "OPENAI_MODEL", "MODEL", "model_name")
    sh_key_re = _shell_literal_assign_re("API_KEY", "OPENAI_API_KEY")
    sh_base_re = _shell_literal_assign_re("BASE_URL", "OPENAI_BASE_URL")
    sh_model_re = _shell_literal_assign_re("MODEL_NAME", "OPENAI_MODEL", "MODEL")
    changed: list[str] = []
    scanned = 0

    candidates = list(root.rglob("*.py")) + list(root.rglob("*.sh"))
    for path in candidates:
        if scanned >= max_files:
            break
        scanned += 1
        if ".git" in path.parts:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        patched_key = False

        if path.suffix.lower() == ".sh":

            def repl_sh_key(match: re.Match[str]) -> str:
                nonlocal patched_key
                value = match.group("value")
                if not _looks_like_api_key_placeholder(value):
                    return match.group(0)
                patched_key = True
                expr = _shell_env_fallback_expr(
                    ["EXECUTION_OPENAI_API_KEY", "OPENAI_API_KEY", "API_KEY", "LLM_API_KEY"],
                    value,
                )
                export = match.group("export") or ""
                return f"{match.group('indent')}{export}{match.group('name')}={expr}{match.group('tail')}"

            new_text = sh_key_re.sub(repl_sh_key, text)
            if not patched_key:
                continue

            def repl_sh_base(match: re.Match[str]) -> str:
                expr = _shell_env_fallback_expr(
                    ["EXECUTION_OPENAI_BASE_URL", "OPENAI_BASE_URL", "BASE_URL", "LLM_BASE_URL"],
                    match.group("value"),
                )
                export = match.group("export") or ""
                return f"{match.group('indent')}{export}{match.group('name')}={expr}{match.group('tail')}"

            def repl_sh_model(match: re.Match[str]) -> str:
                expr = _shell_env_fallback_expr(
                    ["EXECUTION_OPENAI_MODEL", "OPENAI_MODEL", "MODEL", "LLM_MODEL"],
                    match.group("value"),
                )
                export = match.group("export") or ""
                return f"{match.group('indent')}{export}{match.group('name')}={expr}{match.group('tail')}"

            new_text = sh_base_re.sub(repl_sh_base, new_text)
            new_text = sh_model_re.sub(repl_sh_model, new_text)
            if new_text == text:
                continue
            try:
                path.write_text(new_text, encoding="utf-8", errors="ignore")
                changed.append(str(path.relative_to(root)).replace("\\", "/"))
            except Exception:
                continue
            continue

        def repl_key(match: re.Match[str]) -> str:
            nonlocal patched_key
            value = match.group("value")
            if not _looks_like_api_key_placeholder(value):
                return match.group(0)
            patched_key = True
            expr = _env_fallback_expr(
                ["EXECUTION_OPENAI_API_KEY", "OPENAI_API_KEY", "API_KEY", "LLM_API_KEY"],
                value,
            )
            return f"{match.group('indent')}{match.group('name')} = {expr}{match.group('tail')}"

        new_text = key_re.sub(repl_key, text)
        if not patched_key:
            continue

        def repl_base(match: re.Match[str]) -> str:
            expr = _env_fallback_expr(
                ["EXECUTION_OPENAI_BASE_URL", "OPENAI_BASE_URL", "BASE_URL", "LLM_BASE_URL"],
                match.group("value"),
            )
            return f"{match.group('indent')}{match.group('name')} = {expr}{match.group('tail')}"

        def repl_model(match: re.Match[str]) -> str:
            expr = _env_fallback_expr(
                ["EXECUTION_OPENAI_MODEL", "OPENAI_MODEL", "MODEL", "LLM_MODEL"],
                match.group("value"),
            )
            return f"{match.group('indent')}{match.group('name')} = {expr}{match.group('tail')}"

        new_text = base_re.sub(repl_base, new_text)
        new_text = model_re.sub(repl_model, new_text)
        if new_text == text:
            continue
        try:
            path.write_text(new_text, encoding="utf-8", errors="ignore")
            changed.append(str(path.relative_to(root)).replace("\\", "/"))
        except Exception:
            continue

    return changed


def _ensure_default_baseline(baseline_path: Path) -> None:
    if baseline_path.exists():
        return
    write_text(baseline_path, json.dumps({"checks": []}, ensure_ascii=False, indent=2) + "\n")


def _git_reset_if_possible(repo_root: Path, logs_dir: Path) -> None:
    """
    Keep the repo folder reusable without carrying local patches across runs.
    If it is a git repo, reset to HEAD and clean untracked files.
    """
    if not (repo_root / ".git").exists():
        return
    try:
        r1 = run_command(["git", "reset", "--hard"], cwd=str(repo_root), timeout_sec=120)
        persist_command_result(r1, logs_dir, prefix="git_reset")
        r2 = run_command(["git", "clean", "-fd"], cwd=str(repo_root), timeout_sec=120)
        persist_command_result(r2, logs_dir, prefix="git_clean")
    except Exception:
        pass


def _git_head_sha(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return ""
    try:
        r = run_command(["git", "rev-parse", "HEAD"], cwd=str(repo_root), timeout_sec=30)
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip().splitlines()[0].strip()
    except Exception:
        return ""


def _write_run_manifest(*, run_dir: Path, cfg: dict[str, Any], baseline_dir: Path) -> None:
    """
    Write a compact, deterministic manifest for auditability and cross-run comparison.
    This intentionally duplicates some fields from meta.json, but adds paper/baseline pointers.
    """
    try:
        paper_key = str(cfg.get("paper_key") or "paper")
        paper_root = str(cfg.get("paper_root") or "")
        manifest = {
            "paper_key": paper_key,
            "paper_pdf": str(cfg.get("paper_pdf") or ""),
            "paper_repo_url": str(cfg.get("paper_repo_url") or ""),
            "paper_root": paper_root,
            "paper_git_head": _git_head_sha(Path(paper_root)) if paper_root else "",
            "paper_extracted": {
                "md_path": str(cfg.get("paper_pdf_extracted_md") or ""),
                "tables_dir": str((baseline_dir / "paper_extracted" / "tables").resolve()),
            },
            "wrapper_config": {
                "tasks_path": str(cfg.get("tasks_path") or ""),
                "baseline_path": str(cfg.get("baseline_path") or ""),
            },
            "docker": {
                "enabled": bool(cfg.get("docker_enabled", True)),
                "strategy": str(cfg.get("docker_strategy") or ""),
                "python_spec": str(cfg.get("python_spec") or ""),
                "paper_image": str(cfg.get("docker_paper_image") or ""),
                "gpus": str(cfg.get("docker_gpus") or os.environ.get("EXECUTION_DOCKER_GPUS") or ""),
                "shm_size": str(
                    cfg.get("docker_shm_size") or os.environ.get("EXECUTION_DOCKER_SHM_SIZE") or ""
                ),
                "ipc": str(cfg.get("docker_ipc") or os.environ.get("EXECUTION_DOCKER_IPC") or ""),
            },
            "llm": {
                "no_llm": bool(cfg.get("no_llm")),
                "provider": str(cfg.get("llm_provider") or ""),
                "model": str(cfg.get("llm_model") or ""),
                "base_url": str(cfg.get("llm_base_url") or ""),
                "judge_mode": str(cfg.get("llm_judge_mode") or ""),
            },
        }
        write_text(run_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        return


def prepare_node(state: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = state.get("config", {}) or {}
    run_root = str(cfg.get("run_root") or (_repo_root() / "runs" / "execution"))

    paper_pdf = str(cfg.get("paper_pdf") or "").strip()
    paper_root_in = str(cfg.get("paper_root") or "").strip()
    paper_key = str(cfg.get("paper_key") or "").strip()
    local_source_path = str(cfg.get("local_source_path") or "").strip()
    no_pdf_extract = bool(cfg.get("no_pdf_extract"))
    dry_run = bool(cfg.get("dry_run"))
    strategy = docker_strategy(cfg)

    pdf_path = Path(paper_pdf).resolve() if (paper_pdf and not is_url(paper_pdf)) else None

    if not paper_key:
        if paper_pdf:
            paper_key = infer_paper_key(paper_pdf)
        elif paper_root_in:
            paper_key = Path(paper_root_in).resolve().name
        else:
            paper_key = "paper"

    run_id = str(cfg.get("run_id") or "").strip() or make_run_id()
    run_dir = (
        Path(str(cfg.get("run_dir") or "")).resolve()
        if str(cfg.get("run_dir") or "").strip()
        else build_run_dir(run_root, paper_key, run_id)
    )
    layout = ensure_run_subdirs(run_dir)
    logs_dir = ensure_dir(layout["logs"])
    artifacts_dir = ensure_dir(layout["artifacts"])
    fixes_dir = ensure_dir(run_dir / "fixes")
    inputs_dir = ensure_dir(layout["inputs"])
    workspace_dir = ensure_dir(layout["workspace"])

    state["run"] = {
        "id": run_id,
        "dir": str(run_dir),
        "logs_dir": str(logs_dir),
        "artifacts_dir": str(artifacts_dir),
        "fixes_dir": str(fixes_dir),
    }

    paper_pdf_source = paper_pdf
    if paper_pdf:
        try:
            materialized = materialize_paper_pdf(
                paper_pdf,
                inputs_dir / "source_pdf",
                paper_key=paper_key,
            )
            paper_pdf = str(materialized.path)
            pdf_path = materialized.path
        except Exception as exc:
            msg = f"paper_pdf_unavailable: {type(exc).__name__}: {exc}"
            append_event(
                run_dir,
                "prepare_error",
                {"error": msg, "paper_pdf": paper_pdf_source},
            )
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state

    append_event(
        run_dir,
        "prepare_start",
        {
            "paper_key": paper_key,
            "paper_pdf": paper_pdf,
            "paper_pdf_source": paper_pdf_source,
            "paper_root": paper_root_in,
        },
    )
    state.setdefault("history", []).append(
        {"kind": "prepare_start", "data": {"paper_key": paper_key, "paper_pdf": paper_pdf}}
    )

    explicit_source_requested = bool(
        paper_root_in or local_source_path or str(cfg.get("paper_repo_url") or "").strip()
    )
    use_demo_fixture = (
        str(cfg.get("use_demo_fixture") or os.getenv("EXECUTION_USE_DEMO_FIXTURE") or "")
        .strip()
        .lower()
        in {"1", "true", "yes", "y", "on"}
    )
    demo_dir = _configured_demo_dir(paper_key) if ((not explicit_source_requested) or use_demo_fixture) else None
    baseline_dir = (
        Path(str(cfg.get("baseline_dir") or "")).resolve()
        if str(cfg.get("baseline_dir") or "").strip()
        else (inputs_dir / "baseline" / slugify_run_key(paper_key)).resolve()
    )
    ensure_dir(baseline_dir)
    demo_cfg = _materialize_demo_fixture(demo_dir, baseline_dir)
    for key, value in demo_cfg.items():
        if key not in cfg or cfg.get(key) in ("", None):
            cfg[key] = value

    source_dir = (workspace_dir / "source").resolve()
    demo_source_dir = _select_demo_source(demo_dir)

    if paper_root_in:
        source_origin = Path(paper_root_in).resolve()
    elif local_source_path:
        source_origin = Path(local_source_path).resolve()
    elif demo_source_dir is not None:
        source_origin = demo_source_dir
    else:
        source_origin = source_dir

    if pdf_path and pdf_path.exists():
        try:
            dst_pdf = baseline_dir / "paper.pdf"
            if not dst_pdf.exists():
                shutil.copy2(pdf_path, dst_pdf)
        except Exception:
            pass

    if paper_root_in or local_source_path or demo_source_dir is not None:
        if not source_origin.exists():
            msg = f"source_not_found: {source_origin}"
            append_event(run_dir, "prepare_error", {"error": msg})
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state
        try:
            _copy_tree(source_origin, source_dir)
        except Exception as exc:
            msg = f"copy_source_failed: {type(exc).__name__}: {exc}"
            append_event(
                run_dir,
                "prepare_error",
                {"error": msg, "source": str(source_origin), "dest": str(source_dir)},
            )
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state
        paper_root = source_dir.resolve()
        append_event(
            run_dir,
            "prepare_source_snapshot",
            {
                "source": str(source_origin),
                "dest": str(paper_root),
                "demo_fixture": str(demo_dir or ""),
            },
        )
        state.setdefault("history", []).append(
            {
                "kind": "prepare_source_snapshot",
                "data": {"source": str(source_origin), "dest": str(paper_root)},
            }
        )
    else:
        paper_root = source_dir.resolve()
        if _source_dir_looks_partial_clone(source_dir):
            append_event(
                run_dir,
                "prepare_partial_clone_removed",
                {"source_dir": str(source_dir), "reason": "git_metadata_without_payload"},
            )
            _remove_tree_best_effort(source_dir)
        need_clone = not _source_dir_has_payload(source_dir)
        if need_clone:
            repo_url = str(cfg.get("paper_repo_url") or "").strip()
            candidates: list[str] = []
            if not repo_url and pdf_path and pdf_path.exists():
                # Prefer the parsed markdown (set by an earlier parse stage or
                # via ``paper_extracted_dir``) since PDFs often hide links in
                # cross-page footnotes that pypdf splits awkwardly.
                md_hint = str(cfg.get("paper_pdf_extracted_md") or "").strip()
                md_path: Path | None = Path(md_hint) if md_hint else None
                if md_path is None:
                    extracted_dir = str(cfg.get("paper_extracted_dir") or "").strip()
                    if extracted_dir:
                        guess = Path(extracted_dir) / "paper.mineru.md"
                        if guess.exists():
                            md_path = guess
                candidates = _extract_repo_urls_from_pdf(pdf_path, extracted_md=md_path)
                write_text(
                    logs_dir / "repo_url_candidates.txt", "\n".join(candidates) + ("\n" if candidates else "")
                )
                repo_url = candidates[0] if candidates else ""

            if not repo_url:
                msg = "repo_url_not_found"
                append_event(run_dir, "prepare_skipped", {"reason": msg, "candidates": candidates})
                state.setdefault("history", []).append(
                    {
                        "kind": "prepare_skipped",
                        "data": {
                            "reason": msg,
                            "candidates": candidates,
                            "message": "No cloneable GitHub repository URL was found in the paper text.",
                        },
                    }
                )
                state["run_result"] = {
                    "success": False,
                    "skipped": True,
                    "reason": msg,
                    "message": "No cloneable repository URL was found, so execution could not run code.",
                    "tasks": [],
                }
                state["status"] = "skipped"
                return state

            ensure_dir(source_dir.parent)
            if source_dir.exists():
                _remove_tree_best_effort(source_dir)
            anonymous_4open_id = _anonymous_4open_repo_id(repo_url)
            if anonymous_4open_id:
                try:
                    manifest = _download_anonymous_4open_repo(repo_url, source_dir, logs_dir)
                except Exception as exc:
                    msg = f"anonymous_4open_download_failed: {type(exc).__name__}: {exc}"
                    append_event(
                        run_dir,
                        "prepare_error",
                        {"error": msg, "repo_url": repo_url, "repo_id": anonymous_4open_id},
                    )
                    state.setdefault("history", []).append(
                        {"kind": "prepare_error", "data": {"error": msg, "repo_url": repo_url}}
                    )
                    state["status"] = "failed"
                    return state
                cfg["paper_repo_url"] = repo_url
                append_event(
                    run_dir,
                    "prepare_anonymous_4open_download_ok",
                    {"repo_url": repo_url, "dest": str(source_dir), **manifest},
                )
                need_clone = False
            openreview_id = _openreview_forum_id(repo_url)
            if need_clone and (openreview_id or _is_openreview_attachment_url(repo_url)):
                try:
                    append_event(
                        run_dir,
                        "prepare_openreview_supplementary_download_start",
                        {"repo_url": repo_url, "dest": str(source_dir), "forum_id": openreview_id},
                    )
                    manifest = _download_openreview_supplementary(repo_url, source_dir, logs_dir)
                except Exception as exc:
                    msg = f"openreview_supplementary_download_failed: {type(exc).__name__}: {exc}"
                    append_event(
                        run_dir,
                        "prepare_error",
                        {"error": msg, "repo_url": repo_url, "forum_id": openreview_id},
                    )
                    state.setdefault("history", []).append(
                        {"kind": "prepare_error", "data": {"error": msg, "repo_url": repo_url}}
                    )
                    state["status"] = "failed"
                    return state
                cfg["paper_repo_url"] = repo_url
                append_event(
                    run_dir,
                    "prepare_openreview_supplementary_download_ok",
                    {"repo_url": repo_url, "dest": str(source_dir), **manifest},
                )
                need_clone = False
            if not need_clone:
                pass
            else:
                use_blob_filter = str(
                    cfg.get("git_clone_filter_blob_none")
                    or os.getenv("EXECUTION_GIT_CLONE_FILTER_BLOB_NONE")
                    or "1"
                ).strip().lower() not in {"0", "false", "no", "off"}
                clone_cmd = ["git", "clone", "--depth", "1"]
                if use_blob_filter:
                    clone_cmd.extend(["--filter", "blob:none"])
                clone_cmd.extend([repo_url, str(source_dir)])
                clone_timeout = _git_clone_timeout_sec(cfg)
                res = run_command(cmd=clone_cmd, cwd=str(baseline_dir), timeout_sec=clone_timeout)
                persist_command_result(res, logs_dir, prefix="clone")
                if res.returncode != 0 and res.returncode != 124 and use_blob_filter:
                    _remove_tree_best_effort(source_dir)
                    fallback_cmd = ["git", "clone", "--depth", "1", repo_url, str(source_dir)]
                    res = run_command(cmd=fallback_cmd, cwd=str(baseline_dir), timeout_sec=clone_timeout)
                    persist_command_result(res, logs_dir, prefix="clone_fallback_depth_only")
                if res.returncode != 0:
                    msg = "git_clone_timeout" if res.returncode == 124 else "git_clone_failed"
                    _remove_tree_best_effort(source_dir)
                    append_event(
                        run_dir,
                        "prepare_error",
                        {
                            "error": msg,
                            "repo_url": repo_url,
                            "rc": res.returncode,
                            "timeout_sec": clone_timeout,
                        },
                    )
                    state.setdefault("history", []).append(
                        {
                            "kind": "prepare_error",
                            "data": {"error": msg, "repo_url": repo_url, "timeout_sec": clone_timeout},
                        }
                    )
                    state["status"] = "failed"
                    return state
                cfg["paper_repo_url"] = repo_url
                append_event(run_dir, "prepare_clone_ok", {"repo_url": repo_url, "dest": str(source_dir)})

    _git_reset_if_possible(paper_root, logs_dir)
    normalized_scripts = _normalize_shell_script_line_endings(paper_root)
    if normalized_scripts:
        append_event(
            run_dir,
            "prepare_normalized_shell_scripts",
            {"count": normalized_scripts, "root": str(paper_root)},
        )
    api_placeholder_files = _patch_api_placeholders_for_env(paper_root)
    if api_placeholder_files:
        append_event(
            run_dir,
            "prepare_api_placeholders_env_patched",
            {"count": len(api_placeholder_files), "files": api_placeholder_files[:50], "root": str(paper_root)},
        )

    prepared_md = _copy_prepared_extract(str(cfg.get("paper_extracted_dir") or ""), baseline_dir)
    if prepared_md:
        cfg["paper_pdf_extracted_md"] = prepared_md
        append_event(run_dir, "pdf_extract_reuse_pipeline_snapshot", {"output_md": prepared_md})

    if (not no_pdf_extract) and pdf_path and pdf_path.exists():
        out_dir = baseline_dir / "paper_extracted"
        existing_md = out_dir / "paper.mineru.md"
        if str(cfg.get("paper_pdf_extracted_md") or "").strip():
            append_event(
                run_dir,
                "pdf_extract_reuse_configured",
                {"output_md": str(cfg.get("paper_pdf_extracted_md"))},
            )
        elif existing_md.exists():
            cfg["paper_pdf_extracted_md"] = str(existing_md)
            append_event(run_dir, "pdf_extract_reuse_existing", {"output_md": str(existing_md)})
        else:
            if not mineru_available():
                msg = "pdf_extract_required_but_mineru_unavailable"
                append_event(
                    run_dir,
                    "prepare_error",
                    {
                        "error": msg,
                        "hint": (
                            "Install MinerU and ensure `mineru` is on PATH. "
                            "Or rerun with --no-pdf-extract to bypass."
                        ),
                    },
                )
                state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
                state["status"] = "failed"
                return state

        if "paper_pdf_extracted_md" not in cfg:
            r = extract_with_mineru(
                pdf_path=str(pdf_path), out_dir=out_dir, logs_dir=logs_dir, timeout_sec=1800
            )
            append_event(
                run_dir,
                "pdf_extract_mineru",
                {"success": r.success, "output_md": r.output_md, "note": r.note},
            )
            if not r.success:
                msg = "pdf_extract_failed"
                append_event(
                    run_dir,
                    "prepare_error",
                    {
                        "error": msg,
                        "note": r.note,
                        "stdout_log": r.stdout_log,
                        "stderr_log": r.stderr_log,
                        "command_log": r.command_log,
                    },
                )
                state.setdefault("history", []).append(
                    {"kind": "prepare_error", "data": {"error": msg, "note": r.note}}
                )
                state["status"] = "failed"
                return state
            cfg["paper_pdf_extracted_md"] = r.output_md
    else:
        if pdf_path and pdf_path.exists():
            append_event(run_dir, "pdf_extract_skipped", {"reason": "disabled"})

    python_spec = str(cfg.get("python_spec") or os.getenv("EXECUTION_PYTHON_SPEC") or "").strip()
    if not python_spec:
        python_spec = _infer_python_spec_from_repo(paper_root)
    cfg["python_spec"] = python_spec
    docker_enabled_raw = cfg.get("docker_enabled", os.getenv("EXECUTION_DOCKER_ENABLED", "true"))
    cfg["docker_enabled"] = str(docker_enabled_raw).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    cfg["docker_strategy"] = strategy

    tasks_path = str(cfg.get("tasks_path") or "").strip()
    baseline_path = str(cfg.get("baseline_path") or "").strip()
    if not tasks_path:
        tasks_path = str((baseline_dir / "tasks.yaml").resolve())
    if not baseline_path:
        baseline_path = str((baseline_dir / "baseline.json").resolve())
    cfg["tasks_path"] = tasks_path
    cfg["baseline_path"] = baseline_path

    cfg["paper_key"] = paper_key
    cfg["paper_pdf"] = paper_pdf
    cfg["paper_root"] = str(paper_root)
    cfg["baseline_dir"] = str(baseline_dir)
    cfg["paper_extracted_dir"] = str((baseline_dir / "paper_extracted").resolve())
    cfg["paper_extracted_tables_dir"] = str((baseline_dir / "paper_extracted" / "tables").resolve())
    state["config"] = cfg

    if not dry_run and bool(cfg.get("docker_enabled", True)):
        docker_build_timeout_raw = cfg.get("docker_build_timeout_sec")
        if docker_build_timeout_raw in (None, ""):
            docker_build_timeout_raw = os.getenv("EXECUTION_DOCKER_BUILD_TIMEOUT_SEC", "3600")
        docker_build_timeout = int(
            docker_build_timeout_raw
        )
        append_event(
            run_dir,
            "prepare_docker_image_build_start",
            {"paper_key": paper_key, "python_spec": python_spec, "timeout_sec": docker_build_timeout},
        )
        ok_img, img_or_msg = docker_ensure_paper_image(
            cfg,
            paper_key=paper_key,
            paper_root_host=str(paper_root),
            python_spec=python_spec,
            timeout_sec=docker_build_timeout,
        )
        append_event(
            run_dir,
            "prepare_docker_image_build_done",
            {"ok": ok_img, "detail": img_or_msg if ok_img else str(img_or_msg)[-1200:]},
        )
        if not ok_img:
            err = "docker_paper_image_build_failed"
            append_event(run_dir, "prepare_error", {"error": err, "detail": img_or_msg})
            state.setdefault("history", []).append(
                {"kind": "prepare_error", "data": {"error": err, "detail": img_or_msg}}
            )
            state["status"] = "failed"
            return state
        cfg["docker_paper_image"] = img_or_msg

    append_event(run_dir, "prepare_ok", {"paper_root": str(paper_root), "python_spec": python_spec})
    state.setdefault("history", []).append(
        {"kind": "prepare_ok", "data": {"paper_root": str(paper_root), "python_spec": python_spec}}
    )
    state["status"] = "running"
    return state
