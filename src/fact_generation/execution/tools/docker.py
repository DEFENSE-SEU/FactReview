from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import socket
import warnings
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from util.run_layout import slugify_run_key
from util.subprocess_runner import run_command


def docker_cmd(args: list[str]) -> list[str]:
    # `docker` works with shell=False on Windows and non-Windows.
    return ["docker", *args]


def _repo_root() -> Path:
    """Return the FactReview repository root used as the docker command cwd."""
    return Path(__file__).resolve().parents[4]


def docker_strategy(cfg: dict) -> str:
    """
    docker_strategy:
    - paper_image: build one image per paper repo (like mcp-repo-output) and run tasks inside it
    """
    # Always force per-paper image mode.
    # If an old env var/config sets something else, silently ignore and use paper_image.
    v = str(
        cfg.get("docker_strategy") or os.environ.get("EXECUTION_DOCKER_STRATEGY") or "paper_image"
    ).strip()
    return "paper_image" if v != "paper_image" else v


def _paper_image_prefix(cfg: dict) -> str:
    return str(
        cfg.get("docker_paper_image_prefix")
        or os.environ.get("EXECUTION_DOCKER_PAPER_IMAGE_PREFIX")
        or "factreview-paper"
    ).strip()


def _normalize_python_spec_for_image(python_spec: str) -> str:
    """
    Convert python spec into a docker image tag suffix.
    We keep it simple: '3.7.12' -> '3.7', '3.11' -> '3.11'.
    """
    s = str(python_spec or "").strip()
    m = re.match(r"^(\d+)\.(\d+)", s)
    if not m:
        return "3.10"
    return f"{m.group(1)}.{m.group(2)}"


def _image_exists(image: str) -> bool:
    if not image:
        return False
    try:
        r = run_command(docker_cmd(["image", "inspect", image]), cwd=str(_repo_root()), timeout_sec=30)
        return r.returncode == 0
    except Exception:
        return False


def _select_python_image(cfg: dict, py_tag: str) -> str:
    explicit = str(
        cfg.get("docker_paper_python_image") or os.environ.get("EXECUTION_DOCKER_PAPER_PYTHON_IMAGE") or ""
    ).strip()
    if explicit:
        return explicit

    preferred = f"python:{py_tag}"
    if _image_exists(preferred):
        return preferred

    # Prefer locally available images before asking Docker to pull from the
    # network. This keeps execution usable when Docker Desktop has a stale proxy
    # or the host is offline.
    local_fallbacks = []
    if py_tag.startswith("3.7"):
        local_fallbacks.append("code-eval-paper-verify:3.7")
    for candidate in local_fallbacks:
        if _image_exists(candidate):
            return candidate

    return preferred


_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "datasets": "datasets",
    "dgl": "dgl",
    "faiss": "faiss-cpu",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "openai": "openai",
    "pandas": "pandas",
    "PIL": "pillow",
    "scipy": "scipy",
    "schemdraw": "schemdraw",
    "seaborn": "seaborn",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
    "spacy": "spacy",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "yaml": "pyyaml",
}

_README_DEP_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<name>pytorch|torch|dgl|numpy|scipy|sklearn|scikit-learn|pandas|"
    r"torch[-_]?geometric|torch[-_]?sparse|torch[-_]?scatter|opencv-python|cv2|"
    r"tqdm|networkx|transformers|datasets|openai|matplotlib|seaborn)"
    r"\s*(?P<op>==|>=|<=|~=|>|<)?\s*(?P<version>[A-Za-z0-9_.+*-]+)?",
    flags=re.IGNORECASE,
)


def _read_text_limited(path: Path, max_bytes: int = 500_000) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def _canonical_package_name(raw: str) -> str:
    name = str(raw or "").strip().lower().replace("_", "-")
    aliases = {
        "pytorch": "torch",
        "cv2": "opencv-python",
        "pil": "pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
    }
    return aliases.get(name, name)


def _requirement_key(line: str) -> str:
    token = str(line or "").strip()
    if not token or token.startswith(("-", "#")):
        return ""
    name = re.split(r"\s*(?:==|>=|<=|~=|>|<|;|\[)", token, maxsplit=1)[0].strip()
    return _canonical_package_name(name)


def _normalise_dependency_line(name: str, op: str = "", version: str = "") -> str:
    pkg = _canonical_package_name(name)
    op = str(op or "").strip()
    if op == "=":
        op = "=="
    version = str(version or "").strip().rstrip(".,;)")
    if pkg == "dgl" and "+cu" in version:
        # DGL CUDA local-version wheels need a custom find-links URL. Use the
        # base version as a portable first pass; runtime can still request GPU.
        version = version.split("+", 1)[0]
    if op and version and "*" not in version:
        return f"{pkg}{op}{version}"
    return pkg


_CONDA_SKIP_NAMES = {
    "python",
    "pip",
    "setuptools",
    "wheel",
    "ca-certificates",
    "certifi",
    "openssl",
    "readline",
    "sqlite",
    "tk",
    "xz",
    "zlib",
    "ld_impl_linux-64",
    "libgcc-ng",
    "libstdcxx-ng",
    "pytorch-cuda",
    "pytorch-mutex",
}

_CONDA_SKIP_PREFIXES = (
    "_",
    "cuda-",
    "libc",
    "libd",
    "libf",
    "libg",
    "libi",
    "libj",
    "libn",
    "libp",
    "libstd",
    "libt",
    "libu",
    "libw",
    "mkl",
    "intel-",
    "llvm-",
    "ncurses",
)

_CONDA_TO_PIP = {
    "pytorch": "torch",
    "opencv": "opencv-python",
    "opencv-python-headless": "opencv-python-headless",
    "pillow": "pillow",
    "pyyaml": "pyyaml",
    "sklearn": "scikit-learn",
}


def _normalise_conda_dependency_name(name: str) -> str:
    raw = str(name or "").strip().lower().replace("_", "-")
    if not raw or raw in _CONDA_SKIP_NAMES or any(raw.startswith(prefix) for prefix in _CONDA_SKIP_PREFIXES):
        return ""
    mapped = _CONDA_TO_PIP.get(raw, raw)
    # Conda build strings and channels do not map cleanly to PyPI. Keep these
    # as broad package requirements; explicit requirements.txt/pyproject pins
    # still win during de-duplication.
    return _canonical_package_name(mapped)


def _dedupe_requirement_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for raw in lines:
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        key = _requirement_key(line)
        if not key:
            if line not in out:
                out.append(line)
            continue
        if key in seen:
            # Prefer pinned/specified requirements over bare import-derived names.
            old = out[seen[key]]
            old_specific = any(op in old for op in ("==", ">=", "<=", "~=", ">", "<"))
            new_specific = any(op in line for op in ("==", ">=", "<=", "~=", ">", "<"))
            if new_specific and not old_specific:
                out[seen[key]] = line
            continue
        seen[key] = len(out)
        out.append(line)
    return out


def _read_requirement_file_lines(repo_root: Path) -> list[str]:
    lines: list[str] = []
    candidates = []
    skip_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "deployment",
        "outputs",
        "logs",
        "results",
        "checkpoints",
        "simpletransformers",
        "transformers",
        "venv",
    }
    for pattern in ("requirements*.txt", "environment*.yml", "environment*.yaml"):
        candidates.extend(repo_root.rglob(pattern))
    unique = {p.resolve(): p for p in candidates if p.exists()}
    for path in sorted(unique.values(), key=lambda p: (len(p.relative_to(repo_root).parts), p.as_posix()))[:40]:
        if not path.is_file():
            continue
        try:
            rel_parts = [part.lower() for part in path.relative_to(repo_root).parts]
        except Exception:
            rel_parts = [part.lower() for part in path.parts]
        if any(part in skip_parts for part in rel_parts):
            continue
        text = _read_text_limited(path)
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if path.suffix.lower() in {".yml", ".yaml"}:
                m = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*([<>=~!]{1,2})?\s*([A-Za-z0-9_.+*-]+)?", raw)
                if not m:
                    continue
                line = _normalise_conda_dependency_name(m.group(1))
                if not line:
                    continue
            lines.append(line)
    return lines


def _pyproject_dependency_lines(repo_root: Path) -> list[str]:
    pyproject = Path(repo_root) / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    project = data.get("project") if isinstance(data, dict) else {}
    if not isinstance(project, dict):
        return []
    lines: list[str] = []
    deps = project.get("dependencies")
    if isinstance(deps, list):
        lines.extend(str(dep).strip() for dep in deps if str(dep).strip())
    return lines


def _readme_requirement_lines(repo_root: Path) -> list[str]:
    lines: list[str] = []
    candidates = list(repo_root.glob("README*")) + list(repo_root.glob("INSTALL*"))
    for path in sorted(p for p in candidates if p.is_file()):
        text = _read_text_limited(path)
        if not text:
            continue
        dependency_context_budget = 0
        for raw in text.splitlines():
            line = raw.strip()
            lower = line.lower()
            if not line:
                dependency_context_budget = 0
                continue
            if re.match(r"^#{1,6}\s+", line) and any(
                key in lower for key in ["install", "setup", "requirement", "dependenc", "package"]
            ):
                dependency_context_budget = 40
                continue
            in_dependency_context = dependency_context_budget > 0 or any(
                key in lower
                for key in [
                    "pip install",
                    "conda install",
                    "mamba install",
                    "requirements.txt",
                    "environment.yml",
                    "environment.yaml",
                ]
            )
            if not in_dependency_context:
                continue
            for m in _README_DEP_RE.finditer(line):
                name = m.group("name") or ""
                op = m.group("op") or ""
                version = m.group("version") or ""
                lines.append(_normalise_dependency_line(name, op, version))
            if dependency_context_budget > 0 and line:
                dependency_context_budget -= 1
    return lines


def _local_python_modules(repo_root: Path) -> set[str]:
    local = {p.stem for p in repo_root.glob("*.py") if p.is_file()}
    for p in repo_root.iterdir() if repo_root.exists() else []:
        if p.is_dir() and ((p / "__init__.py").exists() or any(p.glob("*.py"))):
            local.add(p.name)
    return local


def _import_requirement_lines(repo_root: Path, *, max_files: int = 500, max_bytes: int = 300_000) -> list[str]:
    if not repo_root.exists():
        return []
    local = _local_python_modules(repo_root)
    found: set[str] = set()
    scanned = 0
    skip_parts = {
        ".git",
        "__pycache__",
        "deployment",
        "outputs",
        "logs",
        "results",
        "checkpoints",
        "simpletransformers",
        "transformers",
    }
    for path in repo_root.rglob("*.py"):
        if scanned >= max_files:
            break
        if any(part in skip_parts for part in path.parts):
            continue
        scanned += 1
        try:
            if path.stat().st_size > max_bytes:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        found.update(_requirements_from_python_source(source, local))
    return sorted(found)


def _requirements_from_python_source(source: str, local: set[str]) -> set[str]:
    found: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source or "")
    except Exception:
        return found
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module.split(".", 1)[0])
        for name in names:
            if name in local:
                continue
            pkg = _IMPORT_TO_PIP.get(name)
            if pkg:
                found.add(pkg)
    return found


_NOTEBOOK_SKIP_PARTS = {
        ".git",
        ".ipynb_checkpoints",
        "__MACOSX",
        "__pycache__",
        "deployment",
        "outputs",
        "logs",
        "results",
        "checkpoints",
        "simpletransformers",
        "transformers",
}


def _notebook_presence_requirement_lines(repo_root: Path, *, max_files: int = 200) -> list[str]:
    if not repo_root.exists():
        return []
    scanned = 0
    for path in repo_root.rglob("*.ipynb"):
        if scanned >= max_files:
            break
        if any(part in _NOTEBOOK_SKIP_PARTS for part in path.parts):
            continue
        scanned += 1
        return ["nbformat"]
    return []


def _notebook_requirement_lines(repo_root: Path, *, max_files: int = 80, max_bytes: int = 2_000_000) -> list[str]:
    if not repo_root.exists():
        return []
    local = _local_python_modules(repo_root)
    found: set[str] = set()
    scanned = 0
    has_notebook = False
    for path in repo_root.rglob("*.ipynb"):
        if scanned >= max_files:
            break
        if any(part in _NOTEBOOK_SKIP_PARTS for part in path.parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        scanned += 1
        has_notebook = True
        cells = data.get("cells") if isinstance(data, dict) else None
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            source = cell.get("source")
            if isinstance(source, list):
                text = "".join(str(x) for x in source)
            else:
                text = str(source or "")
            found.update(_requirements_from_python_source(text, local))
    if has_notebook:
        found.update({"ipykernel", "nbconvert", "nbformat"})
    return sorted(found)


def _collect_repo_requirements_text(repo_root: Path, *, include_notebook_runtime: bool = True) -> str:
    lines = []
    lines.extend(_read_requirement_file_lines(repo_root))
    lines.extend(_pyproject_dependency_lines(repo_root))
    lines.extend(_readme_requirement_lines(repo_root))
    lines.extend(_import_requirement_lines(repo_root))
    lines.extend(_notebook_presence_requirement_lines(repo_root))
    if include_notebook_runtime:
        lines.extend(_notebook_requirement_lines(repo_root))
    deduped = _dedupe_requirement_lines(lines)
    return "\n".join(deduped) + ("\n" if deduped else "")


def _paper_dockerfile_text(*, python_image: str) -> str:
    """
    Paper image Dockerfile (same style as mcp-repo-output):
    - base image is python:<version>
    - install requirements at build time, but handle torch/torch-scatter ordering generically
    - copy repo into /app
    """
    return (
        f"FROM {python_image}\n"
        "\n"
        "ARG PIP_INDEX_URL=\n"
        "ARG PIP_EXTRA_INDEX_URL=\n"
        "ARG PIP_TRUSTED_HOST=\n"
        "ARG EXECUTION_DOCKER_EXTRA_PIP_PACKAGES=\n"
        "ARG HTTP_PROXY=\n"
        "ARG HTTPS_PROXY=\n"
        "ARG NO_PROXY=\n"
        "\n"
        "USER root\n"
        "RUN { echo '[global]'; \\\n"
        "      if [ -n \"$PIP_INDEX_URL\" ]; then echo \"index-url = $PIP_INDEX_URL\"; fi; \\\n"
        "      if [ -n \"$PIP_EXTRA_INDEX_URL\" ]; then echo \"extra-index-url = $PIP_EXTRA_INDEX_URL\"; fi; \\\n"
        "      if [ -n \"$PIP_TRUSTED_HOST\" ]; then echo \"trusted-host = $PIP_TRUSTED_HOST\"; fi; \\\n"
        "    } > /etc/pip.conf \\\n"
        " && (id -u user >/dev/null 2>&1 || useradd -m -u 1000 user) \\\n"
        " && python -m pip install --upgrade pip\n"
        'ENV PATH="/home/user/.local/bin:$PATH"\n'
        "ENV EXECUTION_DOCKER_EXTRA_PIP_PACKAGES=${EXECUTION_DOCKER_EXTRA_PIP_PACKAGES}\n"
        "ENTRYPOINT []\n"
        'CMD ["python"]\n'
        "\n"
        "WORKDIR /app\n"
        "\n"
        "COPY --chown=user ./deployment/requirements.txt requirements.txt\n"
        "COPY --chown=user ./deployment/install_deps.py deployment/install_deps.py\n"
        "RUN export HTTP_PROXY=\"$HTTP_PROXY\" HTTPS_PROXY=\"$HTTPS_PROXY\" NO_PROXY=\"$NO_PROXY\" \\\n"
        " && export http_proxy=\"$HTTP_PROXY\" https_proxy=\"$HTTPS_PROXY\" no_proxy=\"$NO_PROXY\" \\\n"
        " && python deployment/install_deps.py || echo install_deps_failed_continuing\n"
        "\n"
        "COPY --chown=user . /app\n"
    )


def _paper_install_deps_py_text() -> str:
    return (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import re\n"
        "import shlex\n"
        "import struct\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def _run(cmd: list[str]) -> int:\n"
        "    p = subprocess.run(cmd, check=False)\n"
        "    return int(p.returncode)\n"
        "\n"
        "\n"
        "def _pip_check() -> tuple[int, str]:\n"
        "    p = subprocess.run(\n"
        "        [sys.executable, '-m', 'pip', 'check'],\n"
        "        check=False,\n"
        "        text=True,\n"
        "        stdout=subprocess.PIPE,\n"
        "        stderr=subprocess.STDOUT,\n"
        "    )\n"
        "    return int(p.returncode), (p.stdout or '')\n"
        "\n"
        "\n"
        "def _repair_python_package_consistency() -> None:\n"
        "    rc, out = _pip_check()\n"
        "    if rc == 0:\n"
        "        return\n"
        "    low = out.lower()\n"
        "    if 'pydantic' in low and ('pydantic-core' in low or 'typing-extensions' in low):\n"
        "        fix_rc = _run([\n"
        "            sys.executable,\n"
        "            '-m',\n"
        "            'pip',\n"
        "            'install',\n"
        "            '--no-cache-dir',\n"
        "            '--upgrade',\n"
        "            '--force-reinstall',\n"
        "            'pydantic',\n"
        "            'typing-extensions',\n"
        "        ])\n"
        "        if fix_rc == 0:\n"
        "            rc2, out2 = _pip_check()\n"
        "            if rc2 != 0:\n"
        "                print('pip_check_after_pydantic_repair_failed_continuing', out2[-2000:])\n"
        "            else:\n"
        "                print('pip_check_pydantic_repaired')\n"
        "            return\n"
        "    print('pip_check_failed_continuing', out[-2000:])\n"
        "\n"
        "\n"
        "def _base_name(raw: str) -> str:\n"
        "    s = (raw or '').split('#', 1)[0].strip()\n"
        "    for sep in ['==', '>=', '<=', '~=', '>', '<']:\n"
        "        if sep in s:\n"
        "            s = s.split(sep, 1)[0].strip()\n"
        "            break\n"
        "    return s.strip().lower().replace('-', '_')\n"
        "\n"
        "\n"
        "def _clear_torch_executable_stack() -> bool:\n"
        "    # Some old PyTorch wheels mark libtorch_cpu.so as requiring an\n"
        "    # executable stack, which fails under Docker Desktop/WSL kernels.\n"
        "    # Clear PF_X on PT_GNU_STACK in-place, like `execstack -c`, without\n"
        "    # requiring distro packages inside the paper image.\n"
        "    try:\n"
        "        import site\n"
        "        roots = []\n"
        "        try:\n"
        "            roots.extend(site.getsitepackages())\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            roots.append(site.getusersitepackages())\n"
        "        except Exception:\n"
        "            pass\n"
        "        changed = False\n"
        "        for root in roots:\n"
        "            p = Path(root) / 'torch' / 'lib' / 'libtorch_cpu.so'\n"
        "            if not p.exists():\n"
        "                continue\n"
        "            data = bytearray(p.read_bytes())\n"
        "            if data[:4] != b'\\x7fELF':\n"
        "                continue\n"
        "            is64 = data[4] == 2\n"
        "            endian = '<' if data[5] == 1 else '>'\n"
        "            if is64:\n"
        "                e_phoff = struct.unpack_from(endian + 'Q', data, 32)[0]\n"
        "                e_phentsize = struct.unpack_from(endian + 'H', data, 54)[0]\n"
        "                e_phnum = struct.unpack_from(endian + 'H', data, 56)[0]\n"
        "                p_flags_off = 4\n"
        "            else:\n"
        "                e_phoff = struct.unpack_from(endian + 'I', data, 28)[0]\n"
        "                e_phentsize = struct.unpack_from(endian + 'H', data, 42)[0]\n"
        "                e_phnum = struct.unpack_from(endian + 'H', data, 44)[0]\n"
        "                p_flags_off = 24\n"
        "            for i in range(e_phnum):\n"
        "                off = e_phoff + i * e_phentsize\n"
        "                p_type = struct.unpack_from(endian + 'I', data, off)[0]\n"
        "                if p_type != 0x6474E551:  # PT_GNU_STACK\n"
        "                    continue\n"
        "                flags_off = off + p_flags_off\n"
        "                flags = struct.unpack_from(endian + 'I', data, flags_off)[0]\n"
        "                if flags & 0x1:  # PF_X\n"
        "                    struct.pack_into(endian + 'I', data, flags_off, flags & ~0x1)\n"
        "                    p.write_bytes(data)\n"
        "                    print('torch_execstack_cleared', p)\n"
        "                    changed = True\n"
        "                break\n"
        "        return changed\n"
        "    except Exception as exc:\n"
        "        print('torch_execstack_clear_failed', type(exc).__name__, exc)\n"
        "        return False\n"
        "\n"
        "\n"
        "def _repo_uses_torch_scatter(repo_root: Path, max_files: int = 400, max_bytes: int = 200_000) -> bool:\n"
        "    n = 0\n"
        "    for p in repo_root.rglob('*.py'):\n"
        "        n += 1\n"
        "        if n > max_files:\n"
        "            break\n"
        "        try:\n"
        "            b = p.read_bytes()\n"
        "        except Exception:\n"
        "            continue\n"
        "        if not b:\n"
        "            continue\n"
        "        if len(b) > max_bytes:\n"
        "            b = b[:max_bytes]\n"
        "        s = b.decode('utf-8', errors='ignore')\n"
        "        if 'torch_scatter' in s:\n"
        "            return True\n"
        "    return False\n"
        "\n"
        "\n"
        "def _install_torch_scatter_fallback() -> bool:\n"
        "    try:\n"
        "        import site\n"
        "        from pathlib import Path\n"
        "\n"
        "        roots = []\n"
        "        try:\n"
        "            roots.extend(site.getsitepackages())\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            roots.append(site.getusersitepackages())\n"
        "        except Exception:\n"
        "            pass\n"
        "        sp = next((x for x in roots if x), '')\n"
        "        if not sp:\n"
        "            return False\n"
        "        pkg = Path(sp) / 'torch_scatter'\n"
        "        pkg.mkdir(parents=True, exist_ok=True)\n"
        "        (pkg / '__init__.py').write_text(\n"
        '            "import torch\\n\\n"\n'
        '            "def _expand_index(index, src, dim):\\n"\n'
        '            "    if index.dtype != torch.long: index = index.long()\\n"\n'
        '            "    if dim < 0: dim = src.dim() + dim\\n"\n'
        '            "    if index.dim() == 1 and src.dim() > 1:\\n"\n'
        '            "        shape = [1] * src.dim()\\n"\n'
        '            "        shape[dim] = index.numel()\\n"\n'
        '            "        index = index.view(*shape)\\n"\n'
        '            "    return index.expand_as(src)\\n\\n"\n'
        '            "def scatter_add(src, index, dim=0, out=None, dim_size=None):\\n"\n'
        '            "    if out is None:\\n"\n'
        '            "        if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\\n"\n'
        '            "        out_shape = list(src.shape); out_shape[dim] = dim_size\\n"\n'
        '            "        out = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)\\n"\n'
        '            "    idx = _expand_index(index, src, dim)\\n"\n'
        '            "    return out.scatter_add(dim, idx, src)\\n\\n"\n'
        '            "def scatter_max(src, index, dim=0, out=None, dim_size=None):\\n"\n'
        '            "    if index.dtype != torch.long: index = index.long()\\n"\n'
        '            "    if dim < 0: dim = src.dim() + dim\\n"\n'
        '            "    if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\\n"\n'
        '            "    if src.dim() == 1:\\n"\n'
        "            \"        outv = torch.full((dim_size,), -float('inf'), dtype=src.dtype, device=src.device)\\n\"\n"
        '            "        arg = torch.full((dim_size,), -1, dtype=torch.long, device=src.device)\\n"\n'
        '            "        for i in range(src.numel()):\\n"\n'
        '            "            j = int(index[i].item()); v = src[i]\\n"\n'
        '            "            if v > outv[j]: outv[j] = v; arg[j] = i\\n"\n'
        '            "        return outv, arg\\n"\n'
        '            "    dims = list(range(src.dim())); dims[0], dims[dim] = dims[dim], dims[0]\\n"\n'
        '            "    inv = [0] * len(dims)\\n"\n'
        '            "    for i, d in enumerate(dims): inv[d] = i\\n"\n'
        '            "    srcp = src.permute(dims)\\n"\n'
        "            \"    outp = torch.full((dim_size, *srcp.shape[1:]), -float('inf'), dtype=src.dtype, device=src.device)\\n\"\n"
        '            "    argp = torch.full((dim_size, *srcp.shape[1:]), -1, dtype=torch.long, device=src.device)\\n"\n'
        '            "    for i in range(srcp.shape[0]):\\n"\n'
        '            "        j = int(index[i].item()); v = srcp[i]\\n"\n'
        '            "        better = v > outp[j]\\n"\n'
        '            "        outp[j] = torch.where(better, v, outp[j])\\n"\n'
        '            "        argp[j] = torch.where(better, torch.full_like(argp[j], i), argp[j])\\n"\n'
        '            "    return outp.permute(inv), argp.permute(inv)\\n\\n"\n'
        "            \"def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):\\n\"\n"
        "            \"    if reduce in {'sum', 'add'}: return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)\\n\"\n"
        "            \"    if reduce == 'mean':\\n\"\n"
        '            "        outv = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)\\n"\n'
        '            "        cnt = scatter_add(torch.ones_like(src), index, dim=dim, out=None, dim_size=dim_size).clamp(min=1)\\n"\n'
        '            "        return outv / cnt\\n"\n'
        "            \"    if reduce == 'max': return scatter_max(src, index, dim=dim, out=out, dim_size=dim_size)\\n\"\n"
        "            \"    raise ValueError('unsupported reduce')\\n\"\n"
        "        )\n"
        "        print('torch_scatter_fallback_installed', pkg)\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    req = Path('requirements.txt')\n"
        "    txt = req.read_text(encoding='utf-8', errors='ignore') if req.exists() else ''\n"
        "    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith('#')]\n"
        "\n"
        "    torch_lines = []\n"
        "    rest_lines = []\n"
        "    scatter_requested = False\n"
        "    scatter_pin = ''\n"
        "    for ln in lines:\n"
        "        n = _base_name(ln)\n"
        "        if n in {'torch', 'pytorch', 'numpy'}:\n"
        "            torch_lines.append(ln)\n"
        "            continue\n"
        "        if n == 'torch_scatter':\n"
        "            scatter_requested = True\n"
        "            m = re.search(r'==\\s*([^\\s]+)\\s*$', ln)\n"
        "            scatter_pin = (m.group(1).strip() if m else '')\n"
        "            continue\n"
        "        rest_lines.append(ln)\n"
        "\n"
        "    # Some repos import torch_scatter without listing it.\n"
        "    if (not scatter_requested) and _repo_uses_torch_scatter(Path('.')):\n"
        "        scatter_requested = True\n"
        "\n"
        "    Path('requirements.codegen.torch.txt').write_text('\\n'.join(torch_lines) + ('\\n' if torch_lines else ''), encoding='utf-8', errors='ignore')\n"
        "    Path('requirements.codegen.rest.txt').write_text('\\n'.join(rest_lines) + ('\\n' if rest_lines else ''), encoding='utf-8', errors='ignore')\n"
        "\n"
        "    # Install torch/numpy first to satisfy build-time imports for extension packages.\n"
        "    torch_pin = ''\n"
        "    numpy_line = ''\n"
        "    other_first = []\n"
        "    for ln in torch_lines:\n"
        "        n = _base_name(ln)\n"
        "        if n == 'torch':\n"
        "            m = re.search(r'==\\s*([^\\s]+)\\s*$', ln)\n"
        "            torch_pin = (m.group(1).strip() if m else '')\n"
        "            continue\n"
        "        if n == 'numpy':\n"
        "            numpy_line = ln\n"
        "            continue\n"
        "        other_first.append(ln)\n"
        "\n"
        "    extra = shlex.split(os.getenv('EXECUTION_DOCKER_EXTRA_PIP_PACKAGES', '').strip())\n"
        "    if extra:\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', *extra])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "\n"
        "    if numpy_line:\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', numpy_line])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "\n"
        "    if torch_pin:\n"
        "        # Prefer CPU wheels for broad compatibility.\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])\n"
        "        if rc != 0:\n"
        "            rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}'])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "        _clear_torch_executable_stack()\n"
        "\n"
        "    if other_first:\n"
        "        tmp = Path('requirements.codegen.first_rest.txt')\n"
        "        tmp.write_text('\\n'.join(other_first) + '\\n', encoding='utf-8', errors='ignore')\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', str(tmp)])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "    rest_failed = []\n"
        "    if rest_lines:\n"
        "        for dep in rest_lines:\n"
        "            dep = dep.strip()\n"
        "            if not dep or dep.startswith(('-', '--')):\n"
        "                continue\n"
        "            tmp = Path('requirements.codegen.single_rest.txt')\n"
        "            tmp.write_text(dep + '\\n', encoding='utf-8', errors='ignore')\n"
        "            rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', str(tmp)])\n"
        "            if rc != 0:\n"
        "                rest_failed.append(dep)\n"
        "        if rest_failed:\n"
        "            print('install_deps_rest_failed_continuing', rest_failed)\n"
        "\n"
        "    if scatter_requested:\n"
        "        # Try a wheel index matched to torch version and (cpu/cu) when available.\n"
        "        tv = ''\n"
        "        cuda = ''\n"
        "        try:\n"
        "            import torch\n"
        "            tv = (torch.__version__ or '').split('+', 1)[0].strip()\n"
        "            cuda = str(getattr(torch.version, 'cuda', '') or '').strip()\n"
        "        except Exception:\n"
        "            tv = ''\n"
        "            cuda = ''\n"
        "        cu_tag = ''\n"
        "        if cuda and cuda != 'None':\n"
        "            cu_tag = 'cu' + cuda.replace('.', '')\n"
        "\n"
        "        pkgs = []\n"
        "        if scatter_pin:\n"
        "            pkgs.append(f'torch-scatter=={scatter_pin}')\n"
        "        pkgs.append('torch-scatter')\n"
        "\n"
        "        urls = []\n"
        "        if tv and cu_tag:\n"
        "            urls.append(f'https://data.pyg.org/whl/torch-{tv}+{cu_tag}.html')\n"
        "        if tv:\n"
        "            urls.append(f'https://data.pyg.org/whl/torch-{tv}+cpu.html')\n"
        "            urls.append(f'https://data.pyg.org/whl/torch-{tv}.html')\n"
        "\n"
        "        ok = False\n"
        "        for pkg in pkgs:\n"
        "            if ok:\n"
        "                break\n"
        "            for url in urls:\n"
        "                rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', pkg, '-f', url])\n"
        "                if rc == 0:\n"
        "                    ok = True\n"
        "                    break\n"
        "\n"
        "        if not ok:\n"
        "            for pkg in pkgs:\n"
        "                rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', pkg])\n"
        "                if rc == 0:\n"
        "                    ok = True\n"
        "                    break\n"
        "        allow = str(os.getenv('EXECUTION_ALLOW_TORCH_SCATTER_FALLBACK', '1')).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}\n"
        "        if (not ok) and allow:\n"
        "            ok = _install_torch_scatter_fallback()\n"
        "        if not ok:\n"
        "            return 1\n"
        "\n"
        "    _clear_torch_executable_stack()\n"
        "    _repair_python_package_consistency()\n"
        "    print('install_deps_ok')\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def _cfg_or_env(cfg: dict, cfg_key: str, *env_names: str) -> str:
    value = str(cfg.get(cfg_key) or "").strip()
    if value:
        return value
    for name in env_names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _inherited_proxy_usable(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    try:
        parts = urlsplit(raw)
    except Exception:
        return True
    host = (parts.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return True
    if not parts.port:
        return True
    try:
        with socket.create_connection((host, int(parts.port)), timeout=0.25):
            return True
    except OSError:
        return False


def _cfg_or_proxy_env(cfg: dict, cfg_key: str, explicit_env: str, *inherited_env_names: str) -> str:
    value = str(cfg.get(cfg_key) or "").strip()
    if value:
        return value
    value = str(os.environ.get(explicit_env) or "").strip()
    if value:
        return value
    for name in inherited_env_names:
        value = str(os.environ.get(name) or "").strip()
        if value and _inherited_proxy_usable(value):
            return value
    return ""


def _docker_info_field(field: str) -> str:
    try:
        r = run_command(docker_cmd(["info", "--format", f"{{{{.{field}}}}}"]), cwd=str(_repo_root()), timeout_sec=15)
        if r.returncode != 0:
            return ""
        value = (r.stdout or "").strip()
        return "" if value in {"<no value>", "null", "None"} else value
    except Exception:
        return ""


def _normalize_container_proxy(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return raw
    host = (parts.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return raw
    netloc = "host.docker.internal"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment))


def _docker_proxy_env(cfg: dict | None = None) -> dict[str, str]:
    cfg = cfg or {}
    http_proxy = _cfg_or_proxy_env(
        cfg,
        "docker_http_proxy",
        "EXECUTION_DOCKER_HTTP_PROXY",
        "HTTP_PROXY",
        "http_proxy",
    )
    https_proxy = _cfg_or_proxy_env(
        cfg,
        "docker_https_proxy",
        "EXECUTION_DOCKER_HTTPS_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
    )
    no_proxy = _cfg_or_proxy_env(
        cfg,
        "docker_no_proxy",
        "EXECUTION_DOCKER_NO_PROXY",
        "NO_PROXY",
        "no_proxy",
    )

    if not http_proxy:
        http_proxy = _docker_info_field("HTTPProxy")
    if not https_proxy:
        https_proxy = _docker_info_field("HTTPSProxy")
    if not no_proxy:
        no_proxy = _docker_info_field("NoProxy")

    out: dict[str, str] = {}
    http_proxy = _normalize_container_proxy(http_proxy)
    https_proxy = _normalize_container_proxy(https_proxy)
    if http_proxy:
        out["HTTP_PROXY"] = http_proxy
        out["http_proxy"] = http_proxy
    if https_proxy:
        out["HTTPS_PROXY"] = https_proxy
        out["https_proxy"] = https_proxy
    if no_proxy:
        out["NO_PROXY"] = no_proxy
        out["no_proxy"] = no_proxy
    return out


def _docker_cli_env(cfg: dict) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        env.pop(key, None)
    env.update(_docker_proxy_env(cfg))
    return env


def _docker_probe_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        env.pop(key, None)
    return env


def _docker_build_args(cfg: dict) -> list[str]:
    proxy_values = _docker_proxy_env(cfg)
    values = {
        "PIP_INDEX_URL": _cfg_or_env(cfg, "docker_pip_index_url", "EXECUTION_DOCKER_PIP_INDEX_URL", "PIP_INDEX_URL"),
        "PIP_EXTRA_INDEX_URL": _cfg_or_env(
            cfg, "docker_pip_extra_index_url", "EXECUTION_DOCKER_PIP_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL"
        ),
        "PIP_TRUSTED_HOST": _cfg_or_env(
            cfg, "docker_pip_trusted_host", "EXECUTION_DOCKER_PIP_TRUSTED_HOST", "PIP_TRUSTED_HOST"
        ),
        "EXECUTION_DOCKER_EXTRA_PIP_PACKAGES": _cfg_or_env(
            cfg, "docker_extra_pip_packages", "EXECUTION_DOCKER_EXTRA_PIP_PACKAGES"
        ),
        "HTTP_PROXY": proxy_values.get("HTTP_PROXY", ""),
        "HTTPS_PROXY": proxy_values.get("HTTPS_PROXY", ""),
        "NO_PROXY": proxy_values.get("NO_PROXY", ""),
    }
    args: list[str] = []
    for key, value in values.items():
        if value:
            args.extend(["--build-arg", f"{key}={value}"])
    return args


def _paper_image_tag(*, cfg: dict, paper_key: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{_paper_image_prefix(cfg)}:{slugify_run_key(paper_key)}-{h}"


def _docker_include_notebook_requirements(cfg: dict) -> bool:
    raw = str(
        cfg.get("docker_include_notebook_requirements")
        or os.environ.get("EXECUTION_DOCKER_INCLUDE_NOTEBOOK_REQUIREMENTS")
        or ""
    ).strip()
    if raw:
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    mode = str(cfg.get("auto_tasks_mode") or os.environ.get("EXECUTION_AUTO_TASKS_MODE") or "").strip().lower()
    return mode != "smoke"


def _docker_daemon_unavailable(result: object) -> bool:
    text = " ".join(
        [
            str(getattr(result, "stdout", "") or ""),
            str(getattr(result, "stderr", "") or ""),
        ]
    ).lower()
    return any(
        token in text
        for token in [
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "dockerdesktoplinuxengine",
            "docker desktop linux engine",
            "open //./pipe/docker",
            "open \\\\.\\pipe\\docker",
        ]
    )


def _docker_preflight_unavailable() -> str:
    result = run_command(
        docker_cmd(["version", "--format", "{{.Server.Version}}"]),
        cwd=str(_repo_root()),
        timeout_sec=15,
        env=_docker_probe_env(),
    )
    if result.returncode == 0:
        return ""
    tail = (result.stderr or result.stdout or "")[-1200:].replace("\r", "")
    if _docker_daemon_unavailable(result):
        return f"docker_daemon_unavailable: {tail}"
    if result.returncode == 127:
        return f"docker_cli_unavailable: {tail}"
    return ""


def docker_ensure_paper_image(
    cfg: dict, *, paper_key: str, paper_root_host: str, python_spec: str, timeout_sec: int = 3600
) -> tuple[bool, str]:
    """
    Build a per-paper image using the paper repo as build context.
    The generated Dockerfile is stored under <paper_root>/deployment/Dockerfile (inside build context).
    """
    pr = Path(paper_root_host).resolve()
    if not pr.exists():
        return False, f"paper_root_not_found: {pr}"
    unavailable = _docker_preflight_unavailable()
    if unavailable:
        return False, unavailable
    # Build tag is derived from dockerfile template + synthesized requirements
    # hash + python_spec. Many research repos document deps only in README or
    # rely on implicit lab environments, so root requirements.txt alone is too
    # weak for reproducibility.
    try:
        include_notebook_runtime = _docker_include_notebook_requirements(cfg)
        req_text = _collect_repo_requirements_text(
            pr, include_notebook_runtime=include_notebook_runtime
        )
        req_bytes = req_text.encode("utf-8", errors="ignore")
    except Exception:
        include_notebook_runtime = True
        req_bytes = b""
    py_tag = _normalize_python_spec_for_image(python_spec)
    python_image = _select_python_image(cfg, py_tag)
    dockerfile_text = _paper_dockerfile_text(python_image=python_image)
    install_deps_text = _paper_install_deps_py_text()
    proxy_env = _docker_proxy_env(cfg)
    proxy_env_hash = hashlib.sha256(repr(sorted(proxy_env.items())).encode("utf-8")).hexdigest()
    payload = (
        f"paper_key={paper_key}\npython_image={python_image}\npython_spec={python_spec}\n"
        f"req_sha256={hashlib.sha256(req_bytes).hexdigest()}\n"
        f"install_deps_sha256={hashlib.sha256(install_deps_text.encode('utf-8', errors='ignore')).hexdigest()}\n"
        f"pip_index_url={_cfg_or_env(cfg, 'docker_pip_index_url', 'EXECUTION_DOCKER_PIP_INDEX_URL', 'PIP_INDEX_URL')}\n"
        f"pip_extra_index_url={_cfg_or_env(cfg, 'docker_pip_extra_index_url', 'EXECUTION_DOCKER_PIP_EXTRA_INDEX_URL', 'PIP_EXTRA_INDEX_URL')}\n"
        f"pip_trusted_host={_cfg_or_env(cfg, 'docker_pip_trusted_host', 'EXECUTION_DOCKER_PIP_TRUSTED_HOST', 'PIP_TRUSTED_HOST')}\n"
        f"extra_pip_packages={_cfg_or_env(cfg, 'docker_extra_pip_packages', 'EXECUTION_DOCKER_EXTRA_PIP_PACKAGES')}\n"
        f"include_notebook_runtime={include_notebook_runtime}\n"
        f"proxy_env_sha256={proxy_env_hash}\n"
        f"Dockerfile={dockerfile_text}\n"
    )
    image = _paper_image_tag(cfg=cfg, paper_key=paper_key, payload=payload)

    # Fast path: if image exists, skip build.
    docker_env = _docker_cli_env(cfg)
    r = run_command(docker_cmd(["image", "inspect", image]), cwd=str(_repo_root()), timeout_sec=60, env=docker_env)
    if r.returncode == 0:
        return True, image
    if _docker_daemon_unavailable(r):
        tail = (r.stderr or r.stdout or "")[-1200:].replace("\r", "")
        return False, f"docker_daemon_unavailable: {tail}"

    deployment_dir = pr / "deployment"
    legacy_deployment_dir = pr.parent / "deployment"
    dockerfile_path = deployment_dir / "Dockerfile"
    try:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(dockerfile_text, encoding="utf-8", errors="ignore")
        (deployment_dir / "requirements.txt").write_bytes(req_bytes)
        (deployment_dir / "install_deps.py").write_text(
            _paper_install_deps_py_text(), encoding="utf-8", errors="ignore"
        )
        # Best-effort: keep legacy location in sync for old runs/logs.
        try:
            legacy_deployment_dir.mkdir(parents=True, exist_ok=True)
            (legacy_deployment_dir / "Dockerfile").write_text(
                dockerfile_text, encoding="utf-8", errors="ignore"
            )
            (legacy_deployment_dir / "requirements.txt").write_bytes(req_bytes)
        except Exception:
            pass
    except Exception:
        return False, f"write_dockerfile_failed: {dockerfile_path}"

    build = run_command(
        docker_cmd(["build", *_docker_build_args(cfg), "-t", image, "-f", str(dockerfile_path), "."]),
        cwd=str(pr),
        timeout_sec=timeout_sec,
        env=docker_env,
    )
    if build.returncode != 0:
        tail = (build.stderr or "")[-1200:].replace("\r", "")
        return False, f"paper_docker_build_failed: rc={build.returncode}\n{tail}"
    return True, image


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _docker_run_user_args() -> list[str]:
    raw = str(os.environ.get("EXECUTION_DOCKER_USER") or "").strip()
    if raw:
        if raw.lower() in {"auto"}:
            pass
        elif raw.lower() in {"image", "default", "none", "off", "false", "0"}:
            return []
        else:
            return ["--user", raw]
    if not _truthy(os.environ.get("EXECUTION_DOCKER_USER_AUTO", "1")):
        return []
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if not callable(getuid) or not callable(getgid):
        return []
    uid = int(getuid())
    gid = int(getgid())
    if uid <= 0:
        return []
    return ["--user", f"{uid}:{gid}"]


def _docker_env_passthrough(cfg: dict | None = None) -> list[str]:
    cfg = cfg or {}
    raw = str(
        cfg.get("docker_env_passthrough") or os.environ.get("EXECUTION_DOCKER_ENV_PASSTHROUGH") or ""
    ).strip()
    default_names = [
        "EXECUTION_MODEL_PROVIDER",
        "MODEL_PROVIDER",
        "EXECUTION_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_API_KEY",
        "EXECUTION_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "BASE_URL",
        "LLM_BASE_URL",
        "EXECUTION_OPENAI_MODEL",
        "OPENAI_MODEL",
        "MODEL",
        "LLM_MODEL",
        "HF_ENDPOINT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ]
    items = re.split(r"[\s,;]+", raw)
    out: list[str] = []
    seen: set[str] = set()
    for item in [*default_names, *items]:
        name = item.strip()
        if not name or name in seen or not _ENV_NAME_RE.match(name):
            continue
        if name not in os.environ:
            continue
        out.append(name)
        seen.add(name)
    return out


def docker_run_paper_image(
    *,
    image: str,
    paper_root_host: str,
    run_dir_host: str,
    cwd_container: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
    env_passthrough: list[str] | None = None,
    gpus: str | None = None,
    shm_size: str | None = None,
    ipc: str | None = None,
) -> list[str]:
    """
    Run a command inside a per-paper image.
    Commands execute using the image's default python environment.
    """
    env = dict(env or {})
    proxy_env = _docker_proxy_env({})
    run_dir_host = str(Path(run_dir_host).resolve())
    paper_root_host = str(Path(paper_root_host).resolve())
    run_dir_container = "/workspace/run_dir"
    paper_root_container = "/app"
    mount_source = str(os.getenv("EXECUTION_DOCKER_MOUNT_SOURCE", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    args: list[str] = [
        "run",
        "--rm",
    ]
    args.extend(_docker_run_user_args())
    if gpus:
        # e.g. "all" or "device=0"
        args.extend(["--gpus", str(gpus)])
    if shm_size:
        # Avoid DataLoader shared-memory crashes (common in ML workloads).
        args.extend(["--shm-size", str(shm_size)])
    if ipc:
        # e.g. "host" (Linux only)
        args.extend(["--ipc", str(ipc)])
    args.extend(
        [
            "-v",
            f"{run_dir_host}:{run_dir_container}",
            "-w",
            cwd_container,
            "-e",
            f"EXECUTION_RUN_DIR={run_dir_container}",
            "-e",
            f"EXECUTION_ARTIFACT_DIR={run_dir_container}/artifacts",
            "-e",
            f"EXECUTION_PAPER_DIR={paper_root_container}",
            "-e",
            f"EXECUTION_PAPER_ROOT={paper_root_container}",
        ]
    )
    if mount_source:
        args[2:2] = ["-v", f"{paper_root_host}:{paper_root_container}"]
    default_env = {
        "PYTHONPATH": paper_root_container,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONPYCACHEPREFIX": f"{run_dir_container}/.pycache",
        "XDG_CACHE_HOME": f"{run_dir_container}/.cache",
        "HF_HOME": f"{run_dir_container}/.cache/huggingface",
        "MPLCONFIGDIR": f"{run_dir_container}/.cache/matplotlib",
    }
    if env.get("PYTHONPATH"):
        parts = [part for part in str(env["PYTHONPATH"]).split(":") if part]
        if paper_root_container not in parts:
            env["PYTHONPATH"] = f"{paper_root_container}:{env['PYTHONPATH']}"
    merged_env = {**default_env, **proxy_env, **env}
    for k, v in merged_env.items():
        if not k:
            continue
        args.extend(["-e", f"{k}={v}"])
    passthrough = env_passthrough if env_passthrough is not None else _docker_env_passthrough({})
    explicit_env_names = set(merged_env)
    for k in passthrough:
        name = str(k or "").strip()
        if not name or name in explicit_env_names or not _ENV_NAME_RE.match(name):
            continue
        args.extend(["-e", name])
        explicit_env_names.add(name)
    args.extend([image, *cmd])
    return docker_cmd(args)
