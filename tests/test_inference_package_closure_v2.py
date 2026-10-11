"""One public control for static package closure; all external work is forbidden."""

import hashlib
import os
import subprocess
from types import SimpleNamespace

import httpx
import requests

import llm.client
from fact_generation.execution import v2
from fact_generation.execution.runtime_science import read_inference_sources


def test_indexed_static_package_import_closure(tmp_path, monkeypatch):
    external_calls = []

    def forbidden(*args, **kwargs):
        external_calls.append((args, kwargs))
        raise AssertionError("external work is forbidden by this public control")

    for owner, name in ((llm.client, "llm_json"), (requests.sessions.Session, "request"),
                        (httpx.Client, "send"), (httpx.AsyncClient, "send"),
                        (subprocess, "run"), (subprocess, "Popen"),
                        (v2, "run_command"), (v2, "docker_runner"), (v2, "docker_ensure_paper_image")):
        monkeypatch.setattr(owner, name, forbidden)

    cases = []

    def add(label, sources, expected=None, **options):
        cases.append((label, {"README.md": "", **sources}, expected, options))

    add("absolute-parent-transitive", {
        "eval.py": "import pkg.deep.model as model\nimport pkg.deep.model\n",
        "pkg/__init__.py": "import helper\n",
        "pkg/deep/__init__.py": "from .model import Model\n",
        "pkg/deep/model.py": "class Model: pass\n",
        "helper.py": "VALUE = None\n",
    }, {"eval.py", "README.md", "pkg/__init__.py", "pkg/deep/__init__.py",
        "pkg/deep/model.py", "helper.py"}, resources=True)
    add("relative-parent-and-local-torch", {
        "eval.py": "from pkg.inner.worker import Worker\n",
        "pkg/__init__.py": "from . import sibling\n",
        "pkg/sibling.py": "VALUE = 1\n",
        "pkg/torch.py": "class Thing: pass\n",
        "pkg/inner/__init__.py": "",
        "pkg/inner/worker.py": "from .. import sibling\nfrom ..torch import Thing\nclass Worker: pass\n",
    }, {"eval.py", "README.md", "pkg/__init__.py", "pkg/sibling.py", "pkg/torch.py",
        "pkg/inner/__init__.py", "pkg/inner/worker.py"})
    add("symbol-reexport-and-child-fallback", {
        "eval.py": "from pkg import Model, VALUE, sub, token\n",
        "pkg/__init__.py": "from .core import Model\nVALUE = None\ntoken = 1\n",
        "pkg/core.py": "class Model: pass\n",
        "pkg/sub.py": "import json\n",
        "pkg/token.py": "raise AssertionError('a bound attribute takes precedence')\n",
    }, {"eval.py", "README.md", "pkg/__init__.py", "pkg/core.py", "pkg/sub.py"})
    add("entry-parent-root", {
        "scripts/eval.py": "from pkg.mod import VALUE\n",
        "scripts/pkg/__init__.py": "",
        "scripts/pkg/mod.py": "VALUE = 1\n",
    }, {"scripts/eval.py", "README.md", "scripts/pkg/__init__.py", "scripts/pkg/mod.py"},
        entry="scripts/eval.py")
    add("script-relative", {"pkg/eval.py": "from . import sibling\n",
                             "pkg/__init__.py": "", "pkg/sibling.py": ""}, entry="pkg/eval.py")
    add("relative-overflow", {"eval.py": "import pkg.mod\n", "pkg/__init__.py": "",
                              "pkg/mod.py": "from .. import sibling\n", "sibling.py": ""})
    add("module-package-ambiguity", {"eval.py": "import pkg\n", "pkg.py": "", "pkg/__init__.py": ""})
    add("search-root-ambiguity", {"scripts/eval.py": "import pkg\n", "pkg.py": "",
                                 "scripts/pkg.py": ""}, entry="scripts/eval.py")
    for shadow, source in (("scripts/pkg/__init__.py", ""), ("scripts/pkg.py", "")):
        add("parent-root-shadow-" + str(len(cases)), {"scripts/eval.py": "import pkg.mod\n",
            "pkg/__init__.py": "", "pkg/mod.py": "VALUE = 1\n", shadow: source}, entry="scripts/eval.py")
    add("no-importer-directory-search", {"eval.py": "import pkg.worker\n", "pkg/__init__.py": "",
                                          "pkg/worker.py": "import sibling\n", "pkg/sibling.py": ""})
    for label, initializer in (
        ("missing-symbol", ""), ("annotation-only", "VALUE: int\n"),
        ("conditional-symbol", "if True:\n    VALUE = 1\n"),
        ("rebound-symbol", "VALUE = 1\nVALUE = 1\n"),
        ("deleted-symbol", "VALUE = 1\ndel VALUE\n"),
        ("unknown-value", "VALUE = factory()\n"),
    ):
        add(label, {"eval.py": "from pkg import VALUE\n", "pkg/__init__.py": initializer})
    add("cyclic-unestablished-reexport", {"eval.py": "from pkg import Model\n",
                                          "pkg/__init__.py": "from .core import Model\n",
                                          "pkg/core.py": "from pkg import Model\n"})
    add("parent-attribute-deletion", {"eval.py": "from pkg import VALUE\n",
        "pkg/__init__.py": "VALUE = 1\nfrom . import mutator\n",
        "pkg/mutator.py": "import pkg\ndel pkg.VALUE\n", "pkg/VALUE.py": "import foreign_dependency\n"})
    add("parent-attribute-alias-rebind", {"eval.py": "from pkg import VALUE\n",
        "pkg/__init__.py": "VALUE = 1\nfrom . import mutator\n",
        "pkg/mutator.py": "import pkg as original\nalias = original\nalias.VALUE = factory()\n"})
    add("selected-context-conflict", {"eval.py": "import config\n", "config.py": "VALUE = 1\n"},
        config="config.py")
    add("star-reexport", {"eval.py": "from pkg import Model\n", "pkg/__init__.py": "from .core import *\n",
                           "pkg/core.py": "class Model: pass\n"})
    add("namespace-package", {"eval.py": "import pkg.mod\n", "pkg/mod.py": "VALUE = 1\n"})
    add("unindexed-parent", {"eval.py": "import pkg.mod\n", "pkg/__init__.py": "",
                              "pkg/mod.py": "VALUE = 1\n"}, unindexed={"pkg/__init__.py"})
    add("unindexed-stdlib-shadow", {"eval.py": "import json\n", "json.py": ""}, unindexed={"json.py"})
    add("unindexed-native-torch-shadow", {"eval.py": "import torch\n", "torch.pyd": b"\x00"},
        unindexed={"torch.pyd"})
    add("unindexed-native-stdlib-shadow", {"eval.py": "import json\n", "json.cpython-311.so": b"\x00"},
        unindexed={"json.cpython-311.so"})
    add("changed-parent", {"eval.py": "import pkg.mod\n", "pkg/__init__.py": "",
                            "pkg/mod.py": "VALUE = 1\n"}, drift="pkg/__init__.py")
    add("linked-source", {"eval.py": "import linked\n", "linked.py": "VALUE = 1\n"}, hardlink="linked.py")
    for label, text in (
        ("dynamic-getattr", "def __getattr__(name): return 1\n"),
        ("dynamic-package-path", "__path__ = ['other']\n"),
        ("dynamic-search-path", "import sys\nsys.path.append('other')\n"),
        ("dynamic-import", "__import__('other')\n"),
        ("aliased-dynamic-import", "from builtins import __import__ as load\nload('other')\n"),
        ("assigned-dynamic-import", "load = __import__\nload('other')\n"),
        ("getter-alias-search-path", "import sys as s\nget = getattr\nget(s, 'path').append('other')\n"),
        ("getter-reexport-alias", "import sys\nfrom builtins import getattr as read\nalias = read\nalias(sys, 'modules')\n"),
        ("getter-dynamic-import", "import builtins\nload = getattr(builtins, '__import__')\nload('other')\n"),
        ("attribute-getter-search-path", "import builtins\nimport sys\nbuiltins.getattr(sys, 'path').append('other')\n"),
        ("foreign-import", "import foreign_dependency\n"),
        ("compiled-torch-guard", "import torch\nmodel = torch.jit.load('weights.pt')\n"),
        ("dynamic-code-guard", "eval('1')\n"),
    ):
        add(label, {"eval.py": "import pkg\n", "pkg/__init__.py": text})
    add("reexported-sys-path-mutation", {"eval.py": "from pkg import system\np = system.path\np.append('other')\n",
        "pkg/__init__.py": "import sys as system\n"})
    add("ordinary-os-path", {"eval.py": "import os\nVALUE = os.path.join('a', 'b')\n"}, {"eval.py", "README.md"})

    for location, native in (("bare", "mod.pyd"), ("leaf", "pkg/mod.cpython-311.so"),
                             ("package-native", "pkg/__init__.cpython-311.so"),
                             ("package-bytecode", "pkg/__init__.pyc")):
        for indexed in (True, False):
            sources = {"eval.py": "import mod\n", "mod.py": "VALUE = 1\n"} if location == "bare" else {
                "eval.py": "import pkg.mod\n", "pkg/__init__.py": "", "pkg/mod.py": "VALUE = 1\n"}
            sources[native] = b"\x00"
            add(f"local-shadow-{location}-{indexed}", sources, unindexed=set() if indexed else {native})
    for label, alias in (("tuple-module-alias", "alias, = (pkg,)\n"),
                         ("list-module-alias", "[alias] = [pkg]\n"),
                         ("marked-container-alias", "bundle = (pkg,)\nalias, = bundle\n")):
        add(label, {"eval.py": "from pkg import VALUE\n",
            "pkg/__init__.py": "VALUE = 1\nfrom . import mutator\n",
            "pkg/mutator.py": "import pkg\n" + alias + "del alias.VALUE\n",
            "pkg/VALUE.py": "import foreign_dependency\n"})
    add("nested-list-sys-alias", {"eval.py": "import sys\nignored, [s] = (0, [sys])\ns.path.append('other')\n"})
    add("tuple-getter-alias", {"eval.py": "import sys\nget, = (getattr,)\nget(sys, 'path').append('other')\n"})
    add("call-carried-module-alias", {"eval.py": "from pkg import VALUE\n",
        "pkg/__init__.py": "VALUE = 1\nfrom . import mutator\n",
        "pkg/mutator.py": "import pkg\nbundle = tuple((pkg,))\nalias, = bundle\ndel alias.VALUE\n",
        "pkg/VALUE.py": "import foreign_dependency\n"})
    add("ordinary-batch-destructuring", {
        "eval.py": "batch = (0, 1, 2)\nids, features, labels = batch\n"
                   "def evaluate(batch):\n    features, labels = batch\n    return features\n",
    }, {"eval.py", "README.md"})

    for count in (14, 15):
        sources = {"eval.py": "".join(f"import m{i}\n" for i in range(count)),
                   **{f"m{i}.py": "" for i in range(count)}}
        add(f"count-{count + 2}", sources,
            set(sources) | {"README.md"} if count == 14 else None)

    def padded(text, size):
        return text + "#" + "x" * (size - len(text) - 1)

    for size in (65536, 65537):
        add(f"single-bytes-{size}", {"eval.py": padded("", size)},
            {"eval.py", "README.md"} if size == 65536 else None)
    for extra in (0, 1):
        text = "import a\nimport b\nimport c\n" + ("import tail\n" if extra else "")
        sources = {"eval.py": padded(text, 65536),
                   **{f"{name}.py": padded("", 65536) for name in "abc"}}
        if extra:
            sources["tail.py"] = "#"
        add(f"total-bytes-{262144 + extra}", sources,
            set(sources) | {"README.md"} if not extra else None)

    for label, sources, expected, options in cases:
        workspace = tmp_path / label
        workspace.mkdir()
        if options.get("resources"):
            sources.update({"data.csv": "0,1\n", "weights.bin": b"\x00\xff"})
        for name, text in sources.items():
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode() if isinstance(text, str) else text)
        index = [SimpleNamespace(path=name, sha256=hashlib.sha256((workspace / name).read_bytes()).hexdigest())
                 for name in sources if name not in options.get("unindexed", set())]
        if options.get("drift"):
            (workspace / options["drift"]).write_text("# drift\n", encoding="utf-8")
        if options.get("hardlink"):
            os.link(workspace / options["hardlink"], workspace / "unindexed-hardlink-copy.py")
        before = {name: (workspace / name).read_bytes() for name in sources}
        task = SimpleNamespace(entry_script=options.get("entry", "eval.py"), config=options.get("config"),
                               data_paths=["data.csv"] if options.get("resources") else [],
                               weight_paths=["weights.bin"] if options.get("resources") else [])
        plan = SimpleNamespace(task=task)
        materials = SimpleNamespace(repository=SimpleNamespace(files=index))
        files, scope = read_inference_sources(plan, materials, workspace)
        if expected is None:
            assert scope["unavailable"], label
        else:
            assert not scope["unavailable"], (label, scope)
            assert set(files) == expected, (label, set(files), expected)
            assert all(files[name] == before[name].decode().replace("\r\n", "\n").replace("\r", "\n")
                       for name in files), label
        assert scope["scope"] == "indexed_local_python_import_closure", label
        assert scope["byte_limit_per_source"] == 65536 and scope["total_byte_limit"] == 262144, label
        assert scope["source_count_limit"] == 16 and scope["dynamic_imports"] == "unresolved", label
        assert scope["read"] == list(files), label
        expected_resources = {name: {"sha256": hashlib.sha256(before[name]).hexdigest(), "bytes": len(before[name])}
                              for name in [*task.data_paths, *task.weight_paths]}
        assert scope["selected_resource_metadata"] == expected_resources, label
        assert not {"scientific_qualification", "alignment", "support"}.intersection(scope), label
        assert before == {name: (workspace / name).read_bytes() for name in sources}, label
        assert not external_calls, label
