from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

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
        "RUN useradd -m -u 1000 user && python -m pip install --upgrade pip\n"
        "USER user\n"
        'ENV PATH="/home/user/.local/bin:$PATH"\n'
        "\n"
        "WORKDIR /app\n"
        "\n"
        "COPY --chown=user ./requirements.txt requirements.txt\n"
        "COPY --chown=user ./deployment/install_deps.py deployment/install_deps.py\n"
        "RUN python deployment/install_deps.py\n"
        "\n"
        "COPY --chown=user . /app\n"
    )


def _paper_install_deps_py_text() -> str:
    return (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import re\n"
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
        "def _base_name(raw: str) -> str:\n"
        "    s = (raw or '').split('#', 1)[0].strip()\n"
        "    for sep in ['==', '>=', '<=', '~=', '>', '<']:\n"
        "        if sep in s:\n"
        "            s = s.split(sep, 1)[0].strip()\n"
        "            break\n"
        "    return s.strip().lower().replace('-', '_')\n"
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
        "        sp = site.getusersitepackages() or site.getsitepackages()[0]\n"
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
        "    if torch_pin:\n"
        "        # Prefer CPU wheels for broad compatibility.\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])\n"
        "        if rc != 0:\n"
        "            rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}'])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "\n"
        "    if numpy_line:\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', numpy_line])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "\n"
        "    if other_first:\n"
        "        tmp = Path('requirements.codegen.first_rest.txt')\n"
        "        tmp.write_text('\\n'.join(other_first) + '\\n', encoding='utf-8', errors='ignore')\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', str(tmp)])\n"
        "        if rc != 0:\n"
        "            return rc\n"
        "    if rest_lines:\n"
        "        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', 'requirements.codegen.rest.txt'])\n"
        "        if rc != 0:\n"
        "            return rc\n"
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
        "    print('install_deps_ok')\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def _paper_image_tag(*, cfg: dict, paper_key: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{_paper_image_prefix(cfg)}:{slugify_run_key(paper_key)}-{h}"


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
    req = pr / "requirements.txt"
    if not req.exists():
        return False, f"requirements_not_found: {req}"

    # Build tag is derived from dockerfile template + requirements hash + python_spec.
    try:
        req_bytes = req.read_bytes()
    except Exception:
        req_bytes = b""
    py_tag = _normalize_python_spec_for_image(python_spec)
    python_image = str(
        cfg.get("docker_paper_python_image")
        or os.environ.get("EXECUTION_DOCKER_PAPER_PYTHON_IMAGE")
        or f"python:{py_tag}"
    ).strip()
    dockerfile_text = _paper_dockerfile_text(python_image=python_image)
    install_deps_text = _paper_install_deps_py_text()
    payload = (
        f"paper_key={paper_key}\npython_image={python_image}\npython_spec={python_spec}\n"
        f"req_sha256={hashlib.sha256(req_bytes).hexdigest()}\n"
        f"install_deps_sha256={hashlib.sha256(install_deps_text.encode('utf-8', errors='ignore')).hexdigest()}\n"
        f"Dockerfile={dockerfile_text}\n"
    )
    image = _paper_image_tag(cfg=cfg, paper_key=paper_key, payload=payload)

    # Fast path: if image exists, skip build.
    r = run_command(docker_cmd(["image", "inspect", image]), cwd=str(_repo_root()), timeout_sec=60)
    if r.returncode == 0:
        return True, image

    deployment_dir = pr / "deployment"
    legacy_deployment_dir = pr.parent / "deployment"
    dockerfile_path = deployment_dir / "Dockerfile"
    try:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(dockerfile_text, encoding="utf-8", errors="ignore")
        (deployment_dir / "install_deps.py").write_text(
            _paper_install_deps_py_text(), encoding="utf-8", errors="ignore"
        )
        # Best-effort: keep legacy location in sync for old runs/logs.
        try:
            legacy_deployment_dir.mkdir(parents=True, exist_ok=True)
            (legacy_deployment_dir / "Dockerfile").write_text(
                dockerfile_text, encoding="utf-8", errors="ignore"
            )
        except Exception:
            pass
    except Exception:
        return False, f"write_dockerfile_failed: {dockerfile_path}"

    build = run_command(
        docker_cmd(["build", "-t", image, "-f", str(dockerfile_path), "."]),
        cwd=str(pr),
        timeout_sec=timeout_sec,
    )
    if build.returncode != 0:
        tail = (build.stderr or "")[-1200:].replace("\r", "")
        return False, f"paper_docker_build_failed: rc={build.returncode}\n{tail}"
    return True, image


def docker_run_paper_image(
    *,
    image: str,
    paper_root_host: str,
    run_dir_host: str,
    cwd_container: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
    gpus: str | None = None,
    shm_size: str | None = None,
    ipc: str | None = None,
) -> list[str]:
    """
    Run a command inside a per-paper image.
    Commands execute using the image's default python environment.
    """
    env = env or {}
    run_dir_host = str(Path(run_dir_host).resolve())
    paper_root_host = str(Path(paper_root_host).resolve())
    run_dir_container = "/workspace/run_dir"
    paper_root_container = "/app"
    args: list[str] = [
        "run",
        "--rm",
    ]
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
            f"{paper_root_host}:{paper_root_container}",
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
    for k, v in env.items():
        if not k:
            continue
        args.extend(["-e", f"{k}={v}"])
    args.extend([image, *cmd])
    return docker_cmd(args)
