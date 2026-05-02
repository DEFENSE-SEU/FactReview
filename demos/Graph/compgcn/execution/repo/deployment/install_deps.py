from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def _base_name(raw: str) -> str:
    s = (raw or '').split('#', 1)[0].strip()
    for sep in ['==', '>=', '<=', '~=', '>', '<']:
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    return s.strip().lower().replace('-', '_')


def _repo_uses_torch_scatter(repo_root: Path, max_files: int = 400, max_bytes: int = 200_000) -> bool:
    n = 0
    for p in repo_root.rglob('*.py'):
        n += 1
        if n > max_files:
            break
        try:
            b = p.read_bytes()
        except Exception:
            continue
        if not b:
            continue
        if len(b) > max_bytes:
            b = b[:max_bytes]
        s = b.decode('utf-8', errors='ignore')
        if 'torch_scatter' in s:
            return True
    return False


def _install_torch_scatter_fallback() -> bool:
    try:
        import site
        from pathlib import Path

        sp = site.getusersitepackages() or site.getsitepackages()[0]
        pkg = Path(sp) / 'torch_scatter'
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / '__init__.py').write_text(
            "import torch\n\n"
            "def _expand_index(index, src, dim):\n"
            "    if index.dtype != torch.long: index = index.long()\n"
            "    if dim < 0: dim = src.dim() + dim\n"
            "    if index.dim() == 1 and src.dim() > 1:\n"
            "        shape = [1] * src.dim()\n"
            "        shape[dim] = index.numel()\n"
            "        index = index.view(*shape)\n"
            "    return index.expand_as(src)\n\n"
            "def scatter_add(src, index, dim=0, out=None, dim_size=None):\n"
            "    if out is None:\n"
            "        if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\n"
            "        out_shape = list(src.shape); out_shape[dim] = dim_size\n"
            "        out = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)\n"
            "    idx = _expand_index(index, src, dim)\n"
            "    return out.scatter_add(dim, idx, src)\n\n"
            "def scatter_max(src, index, dim=0, out=None, dim_size=None):\n"
            "    if index.dtype != torch.long: index = index.long()\n"
            "    if dim < 0: dim = src.dim() + dim\n"
            "    if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\n"
            "    if src.dim() == 1:\n"
            "        outv = torch.full((dim_size,), -float('inf'), dtype=src.dtype, device=src.device)\n"
            "        arg = torch.full((dim_size,), -1, dtype=torch.long, device=src.device)\n"
            "        for i in range(src.numel()):\n"
            "            j = int(index[i].item()); v = src[i]\n"
            "            if v > outv[j]: outv[j] = v; arg[j] = i\n"
            "        return outv, arg\n"
            "    dims = list(range(src.dim())); dims[0], dims[dim] = dims[dim], dims[0]\n"
            "    inv = [0] * len(dims)\n"
            "    for i, d in enumerate(dims): inv[d] = i\n"
            "    srcp = src.permute(dims)\n"
            "    outp = torch.full((dim_size, *srcp.shape[1:]), -float('inf'), dtype=src.dtype, device=src.device)\n"
            "    argp = torch.full((dim_size, *srcp.shape[1:]), -1, dtype=torch.long, device=src.device)\n"
            "    for i in range(srcp.shape[0]):\n"
            "        j = int(index[i].item()); v = srcp[i]\n"
            "        better = v > outp[j]\n"
            "        outp[j] = torch.where(better, v, outp[j])\n"
            "        argp[j] = torch.where(better, torch.full_like(argp[j], i), argp[j])\n"
            "    return outp.permute(inv), argp.permute(inv)\n\n"
            "def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):\n"
            "    if reduce in {'sum', 'add'}: return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)\n"
            "    if reduce == 'mean':\n"
            "        outv = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)\n"
            "        cnt = scatter_add(torch.ones_like(src), index, dim=dim, out=None, dim_size=dim_size).clamp(min=1)\n"
            "        return outv / cnt\n"
            "    if reduce == 'max': return scatter_max(src, index, dim=dim, out=out, dim_size=dim_size)\n"
            "    raise ValueError('unsupported reduce')\n"
        )
        print('torch_scatter_fallback_installed', pkg)
        return True
    except Exception:
        return False


def main() -> int:
    req = Path('requirements.txt')
    txt = req.read_text(encoding='utf-8', errors='ignore') if req.exists() else ''
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith('#')]

    torch_lines = []
    rest_lines = []
    scatter_requested = False
    scatter_pin = ''
    for ln in lines:
        n = _base_name(ln)
        if n in {'torch', 'pytorch', 'numpy'}:
            torch_lines.append(ln)
            continue
        if n == 'torch_scatter':
            scatter_requested = True
            m = re.search(r'==\s*([^\s]+)\s*$', ln)
            scatter_pin = (m.group(1).strip() if m else '')
            continue
        rest_lines.append(ln)

    # Some repos import torch_scatter without listing it.
    if (not scatter_requested) and _repo_uses_torch_scatter(Path('.')):
        scatter_requested = True

    Path('requirements.codegen.torch.txt').write_text('\n'.join(torch_lines) + ('\n' if torch_lines else ''), encoding='utf-8', errors='ignore')
    Path('requirements.codegen.rest.txt').write_text('\n'.join(rest_lines) + ('\n' if rest_lines else ''), encoding='utf-8', errors='ignore')

    # Install torch/numpy first to satisfy build-time imports for extension packages.
    torch_pin = ''
    numpy_line = ''
    other_first = []
    for ln in torch_lines:
        n = _base_name(ln)
        if n == 'torch':
            m = re.search(r'==\s*([^\s]+)\s*$', ln)
            torch_pin = (m.group(1).strip() if m else '')
            continue
        if n == 'numpy':
            numpy_line = ln
            continue
        other_first.append(ln)

    if torch_pin:
        # Prefer CPU wheels for broad compatibility.
        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if rc != 0:
            rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', f'torch=={torch_pin}'])
        if rc != 0:
            return rc

    if numpy_line:
        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', numpy_line])
        if rc != 0:
            return rc

    if other_first:
        tmp = Path('requirements.codegen.first_rest.txt')
        tmp.write_text('\n'.join(other_first) + '\n', encoding='utf-8', errors='ignore')
        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', str(tmp)])
        if rc != 0:
            return rc
    if rest_lines:
        rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', '-r', 'requirements.codegen.rest.txt'])
        if rc != 0:
            return rc

    if scatter_requested:
        # Try a wheel index matched to torch version and (cpu/cu) when available.
        tv = ''
        cuda = ''
        try:
            import torch
            tv = (torch.__version__ or '').split('+', 1)[0].strip()
            cuda = str(getattr(torch.version, 'cuda', '') or '').strip()
        except Exception:
            tv = ''
            cuda = ''
        cu_tag = ''
        if cuda and cuda != 'None':
            cu_tag = 'cu' + cuda.replace('.', '')

        pkgs = []
        if scatter_pin:
            pkgs.append(f'torch-scatter=={scatter_pin}')
        pkgs.append('torch-scatter')

        urls = []
        if tv and cu_tag:
            urls.append(f'https://data.pyg.org/whl/torch-{tv}+{cu_tag}.html')
        if tv:
            urls.append(f'https://data.pyg.org/whl/torch-{tv}+cpu.html')
            urls.append(f'https://data.pyg.org/whl/torch-{tv}.html')

        ok = False
        for pkg in pkgs:
            if ok:
                break
            for url in urls:
                rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', pkg, '-f', url])
                if rc == 0:
                    ok = True
                    break

        if not ok:
            for pkg in pkgs:
                rc = _run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', pkg])
                if rc == 0:
                    ok = True
                    break
        allow = str(os.getenv('EXECUTION_ALLOW_TORCH_SCATTER_FALLBACK', '1')).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        if (not ok) and allow:
            ok = _install_torch_scatter_fallback()
        if not ok:
            return 1

    print('install_deps_ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
