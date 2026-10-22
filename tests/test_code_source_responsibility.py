"""An inspection limit must not become an author-artifact request."""

import hashlib
import json

import pytest

from llm.client import LLMConfig
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, RepositoryIndex, SharedMaterials
from verification.code import verify_code


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Code source responsibility tests must not contact services or execute code")

    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", "200000")
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("common.run_stats.stats_path", lambda: tmp_path / "run_stats.json")


def inputs(tmp_path, *, content=b"optimizer = 'Adam'\n", repository="present"):
    text = "Our method uses Adam with the stated settings."
    paper = tmp_path / "paper.md"
    paper.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, section="Methods", char_start=0, char_end=len(text))
    repo = tmp_path / "repo"
    index = None
    if repository in {"present", "empty"}:
        repo.mkdir()
        if repository == "present":
            (repo / "config.py").write_bytes(content)
        index = index_repository(repo)
    elif repository == "missing_root":
        index = RepositoryIndex(root=str(repo), files=[])
    materials = SharedMaterials(
        paper_key="code-responsibility",
        source_pdf=str(tmp_path / "unused.pdf"),
        markdown=text,
        markdown_path=str(paper),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="method", text=text, loc=loc)],
        repository=index,
    )
    claim = Claim(
        id="claim-code",
        text=text,
        loc=loc,
        source_block_id="method",
        source_quote=text,
        conditions=[
            Condition(id="c-optimizer", description="The optimizer is Adam"),
            Condition(id="c-settings", description="The implementation uses the stated settings"),
        ],
        needs=["Code"],
    )
    return claim, materials


@pytest.mark.parametrize(
    "content,budget,reason",
    [
        (b"optimizer = 'Adam'\n", 4, "source_budget"),
        (b"optimizer = 'Adam'\n", 40, "source_budget"),
        (b"# -*- coding: latin-1 -*-\n# caf\xe9\noptimizer = 'Adam'\n", 200000, "not_utf8"),
    ],
)
def test_unselected_existing_sources_are_system_limitations(tmp_path, monkeypatch, content, budget, reason):
    claim, materials = inputs(tmp_path, content=content)
    original = (claim.model_dump(mode="json"), materials.model_dump(mode="json"))
    monkeypatch.setenv("CODE_SOURCE_MAX_BYTES", str(budget))
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        pytest.fail("No source was selected, so no model call is appropriate")

    result = verify_code(claim, materials, call=model)

    assert not calls and result.evidence == [] and result.questions == []
    assert len(result.verification_limitations) == 1
    limitation = result.verification_limitations[0]
    assert limitation.claim_id == claim.id
    assert limitation.condition_ids == [condition.id for condition in claim.conditions]
    assert (limitation.stage, limitation.kind, limitation.responsibility) == (
        "Code",
        "source_context_unavailable",
        "system",
    )
    assert limitation.action == "repair_or_retry_verification"
    assert reason in limitation.reason
    assert result.issues[0] == "No readable released-repository source fits the configured inspection scope."
    summary = json.loads(result.issues[1].split(": ", 1)[1])
    manifest_path = next((tmp_path / "code_scopes").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["manifest"] == str(manifest_path)
    assert summary["budget_bytes"] == budget and summary["selected_count"] == 0
    assert summary["omission_reasons"] == {reason: 1}
    assert manifest["omitted_files"] == [{"path": "config.py", "reason": reason}]
    assert (tmp_path / "repo/config.py").read_bytes() == content
    assert materials.repository.files[0].sha256 == hashlib.sha256(content).hexdigest()
    assert original == (claim.model_dump(mode="json"), materials.model_dump(mode="json"))


@pytest.mark.parametrize("repository", ["none", "empty"])
def test_missing_author_artifacts_preserve_the_original_request(tmp_path, repository):
    claim, materials = inputs(tmp_path, repository=repository)
    result = verify_code(claim, materials)
    assert result.evidence == [] and result.verification_limitations == []
    assert len(result.questions) == 1 and result.questions[0].claim_id == claim.id
    assert result.questions[0].text == (
        "Can the released source/configuration needed to check this claim be provided?"
    )
    assert result.questions[0].reason == result.issues[0]


def test_sufficient_budget_keeps_the_existing_readable_source_path(tmp_path):
    claim, materials = inputs(tmp_path)
    calls = []

    def model(**kwargs):
        calls.append(json.loads(kwargs["prompt"]))
        return {"items": [], "issues": []}

    result = verify_code(claim, materials, call=model)
    assert len(calls) == 1
    assert calls[0]["files"] == {"config.py": [{"line": 1, "text": "optimizer = 'Adam'"}]}
    assert calls[0]["source_scope"]["selected_count"] == 1
    assert result.questions == [] and result.verification_limitations == []


def test_missing_indexed_root_remains_an_exception(tmp_path):
    claim, materials = inputs(tmp_path, repository="missing_root")
    with pytest.raises(FileNotFoundError):
        verify_code(claim, materials)


def test_partial_selection_keeps_existing_scope_behavior(tmp_path):
    claim, materials = inputs(tmp_path)
    repo = tmp_path / "repo"
    (repo / "legacy.py").write_bytes(b"# caf\xe9\n")
    materials.repository = index_repository(repo)
    calls = []

    def model(**kwargs):
        calls.append(json.loads(kwargs["prompt"]))
        return {"items": [], "issues": []}

    result = verify_code(claim, materials, call=model)
    assert len(calls) == 1 and list(calls[0]["files"]) == ["config.py"]
    assert calls[0]["source_scope"]["omission_reasons"] == {"not_utf8": 1}
    assert result.issues and "not_utf8" in result.issues[0]
    assert result.questions == [] and result.verification_limitations == []
