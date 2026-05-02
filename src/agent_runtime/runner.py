from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fitz
from agents import Agent, ModelSettings, OpenAIProvider, RunConfig, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from agent_runtime.agent_prompt import build_review_agent_system_prompt
from agent_runtime.agent_tools import ReviewRuntimeContext, build_review_tools
from common.config import get_settings
from common.state import (
    ensure_artifact_paths,
    fail_job,
    load_job_state,
    mutate_job_state,
    set_status,
)
from common.storage import append_event, read_json, write_json_atomic, write_text_atomic
from common.types import AnnotationItem, JobStatus
from fact_generation.positioning.paper_search import (
    PaperReadConfig,
    PaperSearchAdapter,
    PaperSearchConfig,
)
from fact_generation.positioning.semantic_scholar import SemanticScholarAdapter, SemanticScholarConfig
from llm.codex_auth import get_codex_auth
from llm.codex_client import resolve_codex_base_url, resolve_codex_model
from llm.provider_capabilities import provider_capabilities
from preprocessing.parse.markdown_parser import build_page_index
from preprocessing.parse.mineru_adapter import MineruAdapter, MineruConfig
from review.report.final_report_audit import audit_and_refine_final_report
from review.report.pdf_renderer import build_review_report_pdf
from review.report.source_annotations import build_source_annotations_for_export


def _resolved_api_key() -> str:
    settings = get_settings()
    return str(settings.openai_api_key or "EMPTY")


def _uses_codex_subscription_backend() -> bool:
    return provider_capabilities(get_settings().model_provider).uses_codex_subscription


def _resolved_agent_model() -> str:
    settings = get_settings()
    if not _uses_codex_subscription_backend():
        return str(settings.agent_model or "").strip()

    explicit_agent_model = str(os.getenv("AGENT_MODEL") or "").strip()
    explicit_codex_model = str(settings.openai_codex_model or "").strip()
    return resolve_codex_model(explicit_agent_model or explicit_codex_model)


def _build_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()
    if not _uses_codex_subscription_backend():
        return AsyncOpenAI(
            api_key=_resolved_api_key(),
            base_url=settings.openai_base_url,
        )

    auth = get_codex_auth(allow_browser_login=sys.stdin.isatty())
    headers = {"User-Agent": "FactReview"}
    if auth.account_id:
        headers["ChatGPT-Account-Id"] = auth.account_id
    return AsyncOpenAI(
        api_key=auth.access_token,
        base_url=resolve_codex_base_url(settings.openai_codex_base_url),
        default_headers=headers,
    )


def _build_mineru_adapter() -> MineruAdapter:
    settings = get_settings()
    return MineruAdapter(
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


def _build_paper_adapter() -> PaperSearchAdapter:
    settings = get_settings()
    return PaperSearchAdapter(
        search_cfg=PaperSearchConfig(
            enabled=settings.paper_search_enabled,
            base_url=settings.paper_search_base_url,
            api_key=settings.paper_search_api_key,
            endpoint=settings.paper_search_endpoint,
            timeout_seconds=settings.paper_search_timeout_seconds,
            health_endpoint=settings.paper_search_health_endpoint,
            health_timeout_seconds=settings.paper_search_health_timeout_seconds,
        ),
        read_cfg=PaperReadConfig(
            base_url=settings.paper_read_base_url,
            api_key=settings.paper_read_api_key,
            endpoint=settings.paper_read_endpoint,
            timeout_seconds=settings.paper_read_timeout_seconds,
        ),
    )


def _build_semantic_scholar_adapter() -> SemanticScholarAdapter:
    settings = get_settings()
    return SemanticScholarAdapter(
        SemanticScholarConfig(
            enabled=settings.semantic_scholar_enabled,
            base_url=settings.semantic_scholar_base_url,
            api_key=settings.semantic_scholar_api_key,
            timeout_seconds=settings.semantic_scholar_timeout_seconds,
            top_k=settings.semantic_scholar_top_k,
        )
    )


def _extract_title_hint(markdown_text: str, fallback_name: str) -> str:
    lines = [line.strip() for line in str(markdown_text or "").splitlines() if line.strip()]
    for line in lines[:30]:
        cleaned = re.sub(r"^[#>\-\*\d\.\s]+", "", line).strip()
        if not cleaned:
            continue
        if len(cleaned) < 6:
            continue
        if "abstract" in cleaned.lower():
            continue
        return cleaned
    stem = Path(str(fallback_name or "paper")).stem.replace("_", " ").strip()
    return stem or "paper"


def _format_semantic_scholar_context(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "Not available."
    success = bool(payload.get("success"))
    query = str(payload.get("query") or "").strip()
    papers = payload.get("papers") if isinstance(payload.get("papers"), list) else []
    if not success or not papers:
        msg = str(payload.get("message") or "No results").strip()
        return (
            f"success: false\n"
            f"query: {query or '(empty)'}\n"
            f"message: {msg or 'No results'}\n"
            "papers: []\n"
            "strict_rule: objective_retrieval_unavailable_do_not_invent_papers"
        )

    lines = ["success: true", f"query: {query or '(empty)'}", "papers:"]
    for row in papers:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "").strip() or "R?"
        title = str(row.get("title") or "").strip() or "Unknown title"
        year = row.get("year")
        c = int(row.get("citationCount") or 0)
        venue = str(row.get("venue") or "").strip()
        url = str(row.get("url") or "").strip()
        parts = [f"{pid}", title]
        if year:
            parts.append(str(year))
        parts.append(f"citations={c}")
        if venue:
            parts.append(f"venue={venue}")
        if url:
            parts.append(f"url={url}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def _build_run_config() -> RunConfig:
    settings = get_settings()
    if _uses_codex_subscription_backend():
        provider = OpenAIProvider(
            openai_client=_build_async_openai_client(),
            use_responses=True,
        )
        return RunConfig(model_provider=provider)

    provider = OpenAIProvider(
        api_key=_resolved_api_key(),
        base_url=settings.openai_base_url,
        use_responses=settings.openai_use_responses_api,
    )
    return RunConfig(model_provider=provider)


def _build_agent_model() -> OpenAIChatCompletionsModel | OpenAIResponsesModel:
    settings = get_settings()
    client = _build_async_openai_client()
    if _uses_codex_subscription_backend() or settings.openai_use_responses_api:
        return OpenAIResponsesModel(
            model=_resolved_agent_model(),
            openai_client=client,
        )
    return OpenAIChatCompletionsModel(
        model=_resolved_agent_model(),
        openai_client=client,
    )


def _build_agent_model_settings(*, tool_choice: str | None = None) -> ModelSettings:
    settings = get_settings()
    model_name = _resolved_agent_model().strip().lower()
    capabilities = provider_capabilities(settings.model_provider)
    uses_codex = capabilities.uses_codex_subscription
    use_xhigh_reasoning = model_name in {"gpt-5.4", "gpt-5.3", "gpt-5.2"}

    return ModelSettings(
        temperature=None if uses_codex else settings.agent_temperature,
        max_tokens=settings.agent_max_tokens if capabilities.supports_max_output_tokens else None,
        tool_choice=tool_choice,
        parallel_tool_calls=False if not capabilities.supports_parallel_tool_calls else None,
        response_include=["reasoning.encrypted_content"]
        if uses_codex and capabilities.supports_response_include
        else None,
        store=False if not capabilities.supports_store else None,
        reasoning=Reasoning(summary="auto")
        if uses_codex
        else (Reasoning(effort="xhigh") if use_xhigh_reasoning else None),
    )


async def _run_agent_once(
    agent: Agent[Any],
    *,
    input_payload: str | list[Any],
    context: ReviewRuntimeContext,
    max_turns: int,
    run_config: RunConfig,
) -> Any:
    if not _uses_codex_subscription_backend():
        return await Runner.run(
            agent,
            input=input_payload,
            context=context,
            max_turns=max_turns,
            run_config=run_config,
        )

    result = Runner.run_streamed(
        agent,
        input=input_payload,
        context=context,
        max_turns=max_turns,
        run_config=run_config,
    )
    async for _event in result.stream_events():
        pass
    return result


def _sync_token_usage(job_id: str, usage: Any) -> None:
    requests = int(getattr(usage, "requests", 0) or 0)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    def apply(job):
        job.usage.token.requests = requests
        job.usage.token.input_tokens = input_tokens
        job.usage.token.output_tokens = output_tokens
        job.usage.token.total_tokens = total_tokens

    mutate_job_state(job_id, apply)


def _coerce_dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _load_content_list(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None

    if isinstance(payload, dict):
        rows = payload.get("content_list")
        extracted = _coerce_dict_rows(rows)
        return extracted or None
    extracted = _coerce_dict_rows(payload)
    return extracted or None


def _load_annotations_payload(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []

    if isinstance(payload, dict):
        return _coerce_dict_rows(payload.get("annotations"))
    return _coerce_dict_rows(payload)


def _token_usage_payload_from_state(state: Any) -> dict[str, int]:
    usage = getattr(state, "usage", None)
    token = getattr(usage, "token", None)
    return {
        "requests": int(getattr(token, "requests", 0) or 0),
        "input_tokens": int(getattr(token, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(token, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(token, "total_tokens", 0) or 0),
    }


_OVERVIEW_FIGURE_SECTION_PATTERN = re.compile(r"(?ims)^##\s+Overview Figure\s*$\n(?P<body>.*?)(?=^##\s+|\Z)")
_OVERVIEW_FIGURE_PAGE_PATTERN = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:overview figure\s+)?page\s*:\s*(\d+)\s*$"
)
_TECHNICAL_POSITIONING_SECTION_PATTERN = re.compile(
    r"(?ims)^##\s+2\.\s+Technical Positioning\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)
_TECHNICAL_POSITIONING_PAGE_PATTERN = re.compile(r"(?im)^\s*(?:[-*]\s*)?Overview Figure Page\s*:\s*(\d+)\s*$")
_TECHNICAL_POSITIONING_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(?:\[(?:Figure Placeholder|Overview Figure)\]|Overview Figure Page\s*:\s*.*)\s*$"
)
_SECTION_BLOCK_PATTERN = re.compile(r"(?ims)^##\s+(?P<title>.+?)\s*$\n(?P<body>.*?)(?=^##\s+|\Z)")


def _extract_section(markdown_text: str, aliases: tuple[str, ...]) -> str:
    text = str(markdown_text or "")
    for match in _SECTION_BLOCK_PATTERN.finditer(text):
        title = str(match.group("title") or "").strip().lower()
        if any(alias in title for alias in aliases):
            return str(match.group("body") or "").strip()
    return ""


def _parse_key_value_line(text: str, key: str) -> str:
    pattern = re.compile(rf"(?im)^\s*(?:[-•*]\s*)?(?:\*\*)?{re.escape(key)}(?:\*\*)?\s*:\s*(.+?)\s*$")
    match = pattern.search(str(text or ""))
    if not match:
        return "Not found in manuscript"
    value = str(match.group(1) or "").strip()
    return value or "Not found in manuscript"


def _parse_caption_line(text: str) -> str:
    pattern = re.compile(r"(?im)^\s*Figure\s*1\s*:\s*(.+?)\s*$")
    match = pattern.search(str(text or ""))
    if match:
        value = str(match.group(1) or "").strip()
        return value or "Not found in manuscript"
    return "Not found in manuscript"


def _extract_scope_line(text: str, prefix: str) -> str:
    pattern = re.compile(rf"(?im)^\s*{re.escape(prefix)}\s*(.+?)\s*$")
    match = pattern.search(str(text or ""))
    return str(match.group(1) or "").strip() if match else "Not found in manuscript"


def _collect_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            # skip markdown separator line
            if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                continue
            rows.append(cells)
            in_table = True
        else:
            if in_table:
                in_table = False
    return rows


def _rows_for_header(rows: list[list[str]], expected: tuple[str, ...]) -> list[list[str]]:
    lowered_expected = [x.strip().lower() for x in expected]
    for idx, row in enumerate(rows):
        lowered_row = [c.strip().lower() for c in row]
        if (
            len(lowered_row) >= len(lowered_expected)
            and lowered_row[: len(lowered_expected)] == lowered_expected
        ):
            return rows[idx + 1 :]
    return []


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    def pad(row: list[str], n: int) -> list[str]:
        if len(row) >= n:
            return row[:n]
        return row + (["Not found in manuscript"] * (n - len(row)))

    width = len(headers)
    normalized_rows = [pad([str(c or "").strip() for c in row], width) for row in rows]
    if not normalized_rows:
        normalized_rows = [pad([], width)]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in normalized_rows)
    return "\n".join([head, sep, body])


def _inject_overview_figure_image(*, markdown_text: str, source_pdf_path: Path, job_dir: Path) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text

    match = _OVERVIEW_FIGURE_SECTION_PATTERN.search(text)
    if not match:
        return text

    body = match.group("body")
    if "![" in body:
        return text

    page_match = _OVERVIEW_FIGURE_PAGE_PATTERN.search(body)
    if not page_match:
        return text

    try:
        page_no = int(page_match.group(1))
    except Exception:
        return text

    if page_no <= 0 or not source_pdf_path.exists():
        return text

    image_path = job_dir / f"overview_figure_page_{page_no}.png"
    try:
        doc = fitz.open(str(source_pdf_path))
        if page_no > doc.page_count:
            return text
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        pix.save(str(image_path))
    except Exception:
        return text

    figure_markdown = f"\n\n![Overview Figure]({image_path})\n"
    return text[: match.end("body")] + figure_markdown + text[match.end("body") :]


def _resolve_mineru_image_path(*, image_ref: str, job_dir: Path) -> Path | None:
    token = str(image_ref or "").strip()
    if not token:
        return None
    candidate = Path(token)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidate_rel = (job_dir / token).resolve()
    if candidate_rel.exists():
        return candidate_rel
    candidate_assets = (job_dir / "mineru_assets" / token).resolve()
    if candidate_assets.exists():
        return candidate_assets
    return None


def _pick_overview_mineru_image(
    *,
    content_list: list[dict[str, Any]] | None,
    job_dir: Path,
) -> Path | None:
    rows = [row for row in (content_list or []) if isinstance(row, dict)]
    if not rows:
        return None
    best_path: Path | None = None
    best_score = -1
    first_path: Path | None = None
    overview_tokens = ("overview", "framework", "architecture", "pipeline", "model", "method", "network")
    non_overview_tokens = (
        "ablation",
        "result",
        "comparison",
        "attention map",
        "training curve",
        "loss curve",
    )
    for row in rows:
        if str(row.get("type") or "").strip().lower() != "image":
            continue
        image_ref = str(row.get("img_path") or "").strip()
        if not image_ref:
            continue
        resolved = _resolve_mineru_image_path(image_ref=image_ref, job_dir=job_dir)
        if resolved is None:
            continue
        if first_path is None:
            first_path = resolved
        caption = " ".join(str(x) for x in (row.get("image_caption") or []) if str(x).strip()).lower()
        score = 0
        if "figure 1" in caption or "fig. 1" in caption:
            score += 5
        if any(tok in caption for tok in overview_tokens):
            score += 4
        if any(tok in caption for tok in non_overview_tokens):
            score -= 5
        if score > best_score:
            best_score = score
            best_path = resolved
    return best_path or first_path


def _row_image_caption_text(row: dict[str, Any]) -> str:
    captions = row.get("image_caption")
    if isinstance(captions, list):
        merged = " ".join(str(cap or "").strip() for cap in captions if str(cap or "").strip())
        if merged.strip():
            return merged.strip()
    text = str(row.get("caption") or "").strip()
    return text


def _pick_overview_mineru_figure_bundle(
    *,
    content_list: list[dict[str, Any]] | None,
    job_dir: Path,
) -> tuple[list[Path], str | None]:
    rows = [row for row in (content_list or []) if isinstance(row, dict)]
    if not rows:
        return ([], None)

    entries: list[tuple[int, Path, str, int]] = []
    overview_tokens = ("overview", "framework", "architecture", "pipeline", "model", "method", "network")
    non_overview_tokens = (
        "ablation",
        "result",
        "comparison",
        "attention map",
        "training curve",
        "loss curve",
    )
    for idx, row in enumerate(rows):
        if str(row.get("type") or "").strip().lower() != "image":
            continue
        image_ref = str(row.get("img_path") or "").strip()
        if not image_ref:
            continue
        resolved = _resolve_mineru_image_path(image_ref=image_ref, job_dir=job_dir)
        if resolved is None:
            continue
        caption = _row_image_caption_text(row)
        low = caption.lower()
        score = 0
        if "figure 1" in low or "fig. 1" in low:
            score += 6
        if any(tok in low for tok in overview_tokens):
            score += 4
        if any(tok in low for tok in non_overview_tokens):
            score -= 5
        if caption:
            score += 1
        entries.append((idx, resolved, caption, score))

    if not entries:
        return ([], None)

    best = max(entries, key=lambda x: x[3])
    best_idx = best[0]
    by_idx = {idx: (path, caption) for idx, path, caption, _ in entries}

    selected_indices = {best_idx}

    # Include contiguous previous image segments without independent caption (common for split panels).
    j = best_idx - 1
    while j in by_idx:
        _, cap = by_idx[j]
        if cap.strip():
            break
        selected_indices.add(j)
        j -= 1

    # Include contiguous next image segments without independent caption.
    j = best_idx + 1
    while j in by_idx:
        _, cap = by_idx[j]
        if cap.strip():
            break
        selected_indices.add(j)
        j += 1

    ordered = sorted(selected_indices)
    paths = [by_idx[i][0] for i in ordered]
    caption = best[2].strip() or None
    if not caption:
        for i in ordered:
            c = by_idx[i][1].strip()
            if c:
                caption = c
                break
    return (paths, caption)


def _pick_mineru_image_caption(
    *,
    picked_path: Path,
    content_list: list[dict[str, Any]] | None,
    job_dir: Path,
) -> str | None:
    target = picked_path.resolve()
    for row in content_list or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "").strip().lower() != "image":
            continue
        image_ref = str(row.get("img_path") or "").strip()
        if not image_ref:
            continue
        resolved = _resolve_mineru_image_path(image_ref=image_ref, job_dir=job_dir)
        if resolved is None or resolved.resolve() != target:
            continue
        captions = row.get("image_caption")
        if isinstance(captions, list):
            for cap in captions:
                text = str(cap or "").strip()
                if text:
                    return text
        text = str(row.get("caption") or "").strip()
        if text:
            return text
    return None


def _fallback_overview_images_from_assets(*, job_dir: Path, max_images: int = 2) -> list[Path]:
    assets_root = (job_dir / "mineru_assets").resolve()
    if not assets_root.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    candidates: list[Path] = []
    for p in assets_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        candidates.append(p)
    if not candidates:
        return []

    scored: list[tuple[int, Path]] = []
    for p in sorted(candidates):
        name = p.name.lower()
        score = 0
        if "fig1" in name or "figure1" in name:
            score += 8
        if "figure" in name or "fig" in name:
            score += 4
        if "overview" in name or "framework" in name or "architecture" in name:
            score += 4
        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    picked = [p for _, p in scored[: max(1, max_images)]]
    return picked


def _abbreviate_figure_caption(raw_caption: str, *, max_words: int = 32) -> str:
    text = re.sub(r"\s+", " ", str(raw_caption or "").strip())
    if not text:
        return "Not found in manuscript"
    text = re.sub(r"(?im)^\s*figure\s*1\s*:\s*", "", text).strip()
    if not text:
        return "Not found in manuscript"

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    candidate = first_sentence or text

    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words]).rstrip(".,;:") + "..."

    return candidate.strip() or "Not found in manuscript"


def _compose_side_by_side_image(
    *,
    image_paths: list[Path],
    job_dir: Path,
) -> Path | None:
    if len(image_paths) < 2:
        return image_paths[0] if image_paths else None

    try:
        from PIL import Image
    except Exception:
        return None

    images: list[Any] = []
    try:
        for p in image_paths:
            if not p.exists():
                continue
            images.append(Image.open(p).convert("RGB"))
        if len(images) < 2:
            return None

        target_h = max(img.height for img in images)
        resized: list[Any] = []
        for img in images:
            if img.height != target_h:
                w = max(1, int(img.width * (target_h / img.height)))
                resized.append(img.resize((w, target_h)))
            else:
                resized.append(img)

        total_w = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += img.width

        out = job_dir / "overview_figure_combined.jpg"
        canvas.save(out, quality=95)
        return out
    except Exception:
        return None


def _materialize_canonical_overview_image(
    *,
    image_paths: list[Path],
    job_dir: Path,
) -> Path | None:
    if not image_paths:
        return None
    canonical = (job_dir / "overview_figure.jpg").resolve()
    try:
        if len(image_paths) == 1:
            src = image_paths[0]
            if not src.exists() or not src.is_file():
                return None
            shutil.copy2(src, canonical)
            return canonical
        combined = _compose_side_by_side_image(image_paths=image_paths, job_dir=job_dir)
        if combined is None or not combined.exists():
            return None
        shutil.copy2(combined, canonical)
        return canonical
    except Exception:
        return None


def _stabilize_experiment_section(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = re.search(r"(?ims)^##\s+5\.\s+Experiment\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")
    body = re.sub(r"(?im)^\s*Main Result\s*$", "### Main Result", body)
    body = re.sub(r"(?im)^\s*Ablation Result\s*$", "### Ablation Result", body)
    body = _dedupe_experiment_subsections(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = body.strip("\n") + "\n\n"
    return text[: sec.start("body")] + body + text[sec.end("body") :]


def _dedupe_experiment_subsections(section_body: str) -> str:
    body = str(section_body or "")
    for label in ("Main Result", "Ablation Result"):
        pattern = re.compile(rf"(?im)^###\s+{re.escape(label)}\s*$")
        matches = list(pattern.finditer(body))
        if len(matches) <= 1:
            continue
        for match in reversed(matches[1:]):
            next_heading = re.search(r"(?im)^###\s+", body[match.end() :])
            end = (match.end() + next_heading.start()) if next_heading else len(body)
            body = body[: match.start()] + body[end:]
    return body


def _ensure_experiment_contract(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = re.search(r"(?ims)^##\s+5\.\s+Experiment\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text

    body = sec.group("body").strip("\n")
    body = re.sub(r"(?im)^\s*Main Result\s*$", "### Main Result", body)
    body = re.sub(r"(?im)^\s*Ablation Result\s*$", "### Ablation Result", body)

    has_main = bool(re.search(r"(?i)\bMain Result\b", body))
    has_ablation = bool(re.search(r"(?i)\bAblation Result\b", body))

    if not has_main:
        body += (
            "\n\n### Main Result\n"
            "Location: Not found in manuscript\n\n"
            "| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |\n"
            "|---|---|---|---|---|---|\n"
            "| Not found in manuscript | Not found in manuscript | Not found in manuscript | "
            "Not found in manuscript | Not found in manuscript | Not found in manuscript |\n"
        )

    _ABLATION_PLACEHOLDER = (
        "### Ablation Result\n"
        "Location: Not found in manuscript\n\n"
        "| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |\n"
        "|---|---|---|---|---|\n"
        "| Optimal setup | Not found in manuscript | Not found in manuscript | Not found in manuscript | 0 |\n"
    )

    if not has_ablation:
        body += "\n\n" + _ABLATION_PLACEHOLDER

    # Prevent heading sticking to previous table row.
    body = re.sub(r"(?m)(\|[^\n]*\|)\s*(###\s*Ablation Result)", r"\1\n\n\2", body)

    # If both a real and placeholder ablation section exist, keep only the real one.
    if body.count("### Ablation Result") > 1:
        body = body.replace(_ABLATION_PLACEHOLDER, "")

    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + body + text[sec.end("body") :]


def _first_float(text: str) -> float | None:
    s = str(text or "").replace(",", "")
    m = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _last_float(text: str) -> float | None:
    s = str(text or "").replace(",", "")
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", s)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _float_candidates(text: str) -> list[float]:
    s = str(text or "").replace(",", "")
    tokens = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", s)
    out: list[float] = []
    for token in tokens:
        try:
            out.append(float(token))
        except Exception:
            continue
    return out


def _metric_aware_value(text: str, *, metric_hint: str = "") -> float | None:
    candidates = _float_candidates(text)
    if not candidates:
        return None

    metric_raw = str(metric_hint or "").strip().lower()
    metric_key = _norm_metric_key(metric_hint)
    non_year = [v for v in candidates if not (float(v).is_integer() and 1900 <= abs(v) <= 2100)]
    pool = non_year or candidates

    if metric_key == "mr":
        bounded = [v for v in pool if abs(v) < 1_000_000]
        return bounded[-1] if bounded else pool[-1]

    is_perf_metric = metric_key in {"mrr", "hits@10", "hits@3", "hits@1", "accuracy"} or any(
        token in metric_raw
        for token in ("bleu", "f1", "rouge", "map", "auc", "wer", "cer", "precision", "recall")
    )
    if is_perf_metric:
        bounded_100 = [v for v in pool if -100.0 <= v <= 100.0]
        if bounded_100:
            return bounded_100[-1]
        bounded_unit = [v for v in pool if -1.5 <= v <= 1.5]
        if bounded_unit:
            return bounded_unit[-1]
        return pool[-1]

    bounded_default = [v for v in pool if abs(v) <= 1000.0]
    if bounded_default:
        return bounded_default[-1]
    return pool[-1]


def _fmt_value(v: float | None, *, metric_key: str = "") -> str:
    if v is None:
        return "Not found in manuscript"
    if metric_key == "mr":
        return str(round(v))
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


def _hard_validate_experiment_tables(
    markdown_text: str,
    *,
    content_list: list[dict[str, Any]] | None,
) -> str:
    _ = content_list  # reserved for future use; current logic normalizes model-extracted tables only.
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = re.search(r"(?ims)^##\s+5\.\s+Experiment\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")

    def _rewrite_subsection_table(*, section_body: str, label: str, headers: list[str]) -> str:
        heading_re = re.compile(
            rf"(?ims)^###\s+(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*$\n(?P<chunk>.*?)(?=^###\s+|\Z)"
        )
        m = heading_re.search(section_body)
        if not m:
            return section_body
        chunk = str(m.group("chunk") or "")

        table_re = re.compile(r"(?ims)^\|[^\n]*\|\n\|[-:| ]+\|\n(?:\|[^\n]*\|\n?)+")
        t = table_re.search(chunk)
        if t:
            table_block = str(t.group(0) or "")
            rows = _collect_table_rows(table_block)
            data_rows = rows[1:] if len(rows) > 1 else []
            normalized = _format_table(headers, data_rows)
            chunk = chunk[: t.start()] + normalized + chunk[t.end() :]
        else:
            normalized = _format_table(headers, [])
            chunk = chunk.rstrip() + "\n\n" + normalized + "\n"

        return section_body[: m.start("chunk")] + chunk + section_body[m.end("chunk") :]

    body = _rewrite_subsection_table(
        section_body=body,
        label="Main Result",
        headers=["Task", "Dataset", "Metric", "Best Baseline", "Paper Result", "Difference (Δ)"],
    )
    body = _rewrite_subsection_table(
        section_body=body,
        label="Ablation Result",
        headers=["Ablation Dimension", "Configuration", "Full Model", "Paper Result", "Difference (Δ)"],
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + body + text[sec.end("body") :]


def _demote_experiment_child_headings(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = re.search(
        r"(?ims)^##\s+(?:\*\*)?5\.\s+Experiment(?:\*\*)?\s*$\n"
        r"(?P<body>.*?)(?=^##\s+(?:\*\*)?(?:1\.|2\.|3\.|4\.|5\.)|\Z)",
        text,
    )
    if not sec:
        return text
    body = sec.group("body")
    body = re.sub(
        r"(?im)^\s{0,3}#{1,6}\s+(?:\*\*)?(Main Result|Ablation Result)(?:\*\*)?\s*$",
        r"### \1",
        body,
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + body + text[sec.end("body") :]


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _global_eval_status(summary: dict[str, Any], alignment: dict[str, Any]) -> tuple[str, str]:
    status = str(summary.get("status") or "").strip().lower()
    run_ok = bool((summary.get("run_result") or {}).get("success")) if isinstance(summary, dict) else False
    matched = int(alignment.get("matched") or 0)
    failed = int(alignment.get("failed") or 0)
    if status in {"failed", "error"} or not run_ok:
        return ("In conflict", "Execution failed or strongly conflicts with reported experiment behavior.")
    if matched > 0 and failed == 0:
        return ("Supported", "Execution-alignment supports reported experimental trends within tolerance.")
    if matched > 0 and failed > 0:
        return ("Inconclusive", "Execution evidence is mixed: some aligned and some mismatched metrics.")
    return ("Inconclusive", "Execution finished but deterministic alignment evidence is insufficient.")


def _metric_higher_is_better(metric: str) -> bool:
    key = _norm_metric_key(metric)
    if key in {"mr", "error", "loss", "wer", "cer", "perplexity"}:
        return False
    return True


def _norm_metric_key(metric: str) -> str:
    s = str(metric or "").strip().lower()
    if "mrr" in s:
        return "mrr"
    if s in {"mr", "mean rank"} or "mean rank" in s:
        return "mr"
    if "hits@10" in s or "h@10" in s:
        return "hits@10"
    if "hits@3" in s or "h@3" in s:
        return "hits@3"
    if "hits@1" in s or "h@1" in s:
        return "hits@1"
    if "acc" in s:
        return "accuracy"
    if "wer" in s:
        return "wer"
    if "cer" in s:
        return "cer"
    if "perplex" in s or "ppl" in s:
        return "perplexity"
    if "loss" in s:
        return "loss"
    if "error" in s or "err" in s:
        return "error"
    return ""


def _lookup_observed_metric(*, dataset: str, metric: str, alignment: dict[str, Any]) -> float | None:
    matches = alignment.get("matches") if isinstance(alignment.get("matches"), list) else []
    ds = str(dataset or "").strip().lower().replace(" ", "")
    k = _norm_metric_key(metric)
    if not k:
        return None
    for m in matches:
        if not isinstance(m, dict):
            continue
        mds = str(m.get("dataset") or "").strip().lower().replace(" ", "")
        if ds and mds and ds not in mds and mds not in ds:
            continue
        observed = m.get("observed") if isinstance(m.get("observed"), dict) else {}
        if k in observed:
            obs = observed.get(k)
            if isinstance(obs, (int, float)):
                return float(obs)
            try:
                return float(str(obs).strip())
            except Exception:
                return None
    return None


def _row_eval_cell(
    *,
    dataset: str,
    metric: str,
    paper_result: str,
    alignment: dict[str, Any],
    observed_override: float | None = None,
) -> str:
    observed = observed_override
    if observed is None:
        observed = _lookup_observed_metric(dataset=dataset, metric=metric, alignment=alignment)
    if observed is None:
        return "Inconclusive"
    paper_val = _metric_aware_value(paper_result, metric_hint=metric)
    if paper_val is None:
        return "Inconclusive"

    mk = _norm_metric_key(metric)
    # Normalize scale for percentage-style paper values.
    if mk != "mr":
        if paper_val > 1.0 and observed <= 1.0:
            paper_val = paper_val / 100.0
        elif observed > 1.0 and paper_val <= 1.0:
            observed = observed / 100.0

    # Directional rule:
    # supported: performance drop within 1 * threshold
    # inconclusive: drop within (1, 2] * threshold
    # in conflict: drop > 2 * threshold
    threshold = max(float(get_settings().eval_status_threshold), 0.0)
    if _metric_higher_is_better(metric):
        performance_drop = paper_val - observed
    else:
        performance_drop = observed - paper_val

    if performance_drop <= threshold:
        return f"Supported ({_fmt_value(observed, metric_key=mk)})"
    if performance_drop <= (2.0 * threshold):
        return f"Inconclusive ({_fmt_value(observed, metric_key=mk)})"
    return f"In conflict ({_fmt_value(observed, metric_key=mk)})"


def _status_with_symbol(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    lower = raw.lower()
    if lower.startswith("supported"):
        return f"✓ {raw}"
    if lower.startswith("partially supported"):
        return f"⚠ {raw}"
    if lower.startswith("inconclusive"):
        return f"⚠ {raw}"
    if lower.startswith("in conflict"):
        return f"✗ {raw}"
    return raw


def _looks_like_data_evidence(text: str) -> bool:
    raw = str(text or "")
    if re.search(
        r"(?i)\b(mrr|mean rank|hits?@\d+|h@\d+|accuracy|acc|f1|bleu|rouge|map|auc|error rate|top-1|top-5)\b",
        raw,
    ):
        return True
    return bool(
        re.search(r"(?i)\b(table|figure|experiment|result|dataset|benchmark|baseline|ablation)\b", raw)
        and _first_float(raw) is not None
    )


def _normalize_claim_type_label(value: str) -> str:
    raw = _strip_inline_formatting(value).strip().lower()
    if not raw:
        return ""
    if "experiment" in raw or "empirical" in raw:
        return "Experimental"
    if "theor" in raw or "proof" in raw or "formal" in raw:
        return "Theoretical"
    if "method" in raw or "model" in raw or "architect" in raw or "algorithm" in raw:
        return "Methodological"
    return ""


def _infer_claim_type_label(*, claim: str, evidence: str, location: str) -> str:
    joined = f"{claim}\n{evidence}\n{location}"
    if _looks_like_data_evidence(joined):
        return "Experimental"
    lowered = joined.lower()
    if re.search(r"(?i)\b(theorem|proposition|lemma|corollary|proof|derivation|bound)\b", joined):
        return "Theoretical"
    if re.search(
        r"(?i)\b(module|architecture|algorithm|design|mechanism|encoder|decoder|attention)\b", joined
    ):
        return "Methodological"
    if "equation" in lowered or "objective" in lowered or "loss" in lowered:
        return "Theoretical"
    return "Methodological"


def _resolve_claim_type_label(*, model_claim_type: str, claim: str, evidence: str, location: str) -> str:
    normalized = _normalize_claim_type_label(model_claim_type)
    if normalized:
        return normalized
    return _infer_claim_type_label(claim=claim, evidence=evidence, location=location)


def _claims_status_legend_colored() -> str:
    return (
        "(Status legend: "
        '<span style="color: green;">✓ Supported</span>, '
        '<span style="color: #E6B800;">⚠ Partially supported</span>, '
        '<span style="color: red;">✗ In conflict</span>.)'
    )


def _experiment_status_legend_colored() -> str:
    return (
        "(Status legend: "
        '<span style="color: green;">✓ Supported</span>, '
        '<span style="color: #E6B800;">⚠ Inconclusive</span>, '
        '<span style="color: red;">✗ In conflict</span>.)'
    )


def _claim_dataset_metric_hint(
    text: str,
    *,
    alignment: dict[str, Any] | None = None,
) -> tuple[str, str]:
    s = str(text or "").lower()
    dataset = ""
    metric = ""
    if isinstance(alignment, dict):
        matches = alignment.get("matches") if isinstance(alignment.get("matches"), list) else []
        text_norm = re.sub(r"[^a-z0-9]+", "", s)
        candidates: list[str] = []
        for item in matches:
            if not isinstance(item, dict):
                continue
            ds = str(item.get("dataset") or "").strip()
            if ds and ds not in candidates:
                candidates.append(ds)
        candidates.sort(key=len, reverse=True)
        for ds in candidates:
            ds_norm = re.sub(r"[^a-z0-9]+", "", ds.lower())
            if ds_norm and ds_norm in text_norm:
                dataset = ds
                break

    if "mrr" in s:
        metric = "MRR"
    elif re.search(r"\bmr\b|mean rank", s):
        metric = "MR"
    elif "h@10" in s or "hits@10" in s:
        metric = "H@10"
    elif "h@3" in s or "hits@3" in s:
        metric = "H@3"
    elif "h@1" in s or "hits@1" in s:
        metric = "H@1"
    elif "acc" in s or "accuracy" in s:
        metric = "Accuracy"
    return dataset, metric


def _status_from_paper_observed(
    *, paper_val: float | None, observed: float | None, metric: str
) -> tuple[str, str]:
    if paper_val is None or observed is None:
        return ("Inconclusive", "Insufficient numeric evidence for deterministic comparison.")
    mk = _norm_metric_key(metric)
    p = float(paper_val)
    o = float(observed)
    if mk != "mr":
        if p > 1.0 and o <= 1.0:
            p = p / 100.0
        elif o > 1.0 and p <= 1.0:
            o = o / 100.0
    threshold = max(float(get_settings().eval_status_threshold), 0.0)
    if _metric_higher_is_better(metric):
        performance_drop = p - o
    else:
        performance_drop = o - p

    note = f"Delta={abs(o - p):.4f}, performance_drop={performance_drop:.4f}, threshold={threshold:.4f}"
    if performance_drop <= threshold:
        return ("Supported", note)
    if performance_drop <= (2.0 * threshold):
        return ("Partially supported", note)
    return ("In conflict", note)


def _score_paper_evidence(*, claim: str, evidence: str, location: str, claim_type: str) -> tuple[int, list[str]]:
    """Score how strongly the paper's own text supports a claim.

    Returns (score, explanation_parts). Score >= 3 → strong, -1 < score < 3 → partial, <= -1 → conflict.
    claim_type should be 'theoretical', 'methodological', or 'experimental'.
    """
    text = f"{claim}\n{evidence}\n{location}"
    lower = text.lower()

    has_location = bool(str(location or "").strip() and "not found in manuscript" not in lower)
    has_missing_marker = bool(
        re.search(r"(?i)\b(not found in manuscript|not provided|missing|unclear|unspecified)\b", text)
    )
    has_contradiction_marker = bool(
        re.search(r"(?i)\b(contradict|inconsistent|does not hold|invalid|fails?)\b", text)
    )

    score = 0
    parts: list[str] = []

    if claim_type == "theoretical":
        has_anchor = bool(
            re.search(r"(?i)\b(theorem|proposition|lemma|corollary|proof|equation|derivation)\b", text)
        )
        has_signal = bool(
            re.search(r"(?i)(\\[a-z]+|\bO\(|=|<=|>=|->|⇒|∇|argmin|argmax|loss|objective)", text)
        )
        if has_anchor:
            score += 2
            parts.append("The paper gives a formal anchor for the reasoning.")
        else:
            parts.append("The paper gives no clear theorem/proposition/proof anchor.")
        if has_signal:
            score += 1
            parts.append("The evidence contains explicit mathematical derivation signals.")
        else:
            parts.append("The evidence lacks explicit derivation details.")
    elif claim_type == "methodological":
        has_design_anchor = bool(
            re.search(
                r"(?i)\b(module|architecture|algorithm|design|pipeline|encoder|decoder|attention|component)\b",
                text,
            )
        )
        has_impl_anchor = bool(
            re.search(r"(?i)\b(section|table|figure|equation|appendix|implementation)\b", text)
        )
        if has_design_anchor:
            score += 2
            parts.append("The claim has concrete method-design anchors in the manuscript.")
        else:
            parts.append("The claim lacks explicit method-design anchors.")
        if has_impl_anchor:
            score += 1
            parts.append("Supporting evidence references implementation-level manuscript artifacts.")
        else:
            parts.append("Supporting evidence lacks direct implementation-level anchors.")
    else:  # experimental / no execution
        has_data_anchor = bool(
            re.search(
                r"(?i)\b(table|figure|experiment|result|dataset|benchmark|baseline|ablation)\b", text
            )
        )
        has_numeric = _first_float(text) is not None
        if has_data_anchor:
            score += 2
            parts.append("The paper cites experimental tables or figures as evidence.")
        else:
            parts.append("The paper provides no direct experimental table or figure reference.")
        if has_numeric:
            score += 1
            parts.append("Concrete numeric values are present in the evidence.")
        else:
            parts.append("No concrete numeric values found in the evidence.")

    if has_location:
        score += 1
        parts.append(f"The cited location is {location.strip()}.")
    else:
        parts.append("No reliable manuscript location is provided.")
    if has_missing_marker:
        score -= 2
        parts.append("The evidence indicates missing or incomplete support.")
    if has_contradiction_marker:
        score -= 2
        parts.append("The evidence text suggests an unresolved inconsistency risk.")

    return score, parts


def _paper_status_from_score(score: int) -> str:
    if score >= 3:
        return "Supported"
    if score <= -1:
        return "In conflict"
    return "Partially supported"


def _build_experimental_claim_assessment(
    *,
    claim: str,
    evidence: str,
    location: str,
    alignment: dict[str, Any],
) -> tuple[str, str]:
    joined = f"{claim} {evidence} {location}"
    dataset, metric = _claim_dataset_metric_hint(joined, alignment=alignment)
    observed = (
        _lookup_observed_metric(dataset=dataset, metric=metric, alignment=alignment)
        if (dataset and metric)
        else None
    )
    paper_val = _metric_aware_value(evidence, metric_hint=metric) or _metric_aware_value(
        claim, metric_hint=metric
    )

    has_execution = bool(alignment)

    if has_execution and paper_val is not None and observed is not None:
        # Both paper value and reproduced value available — apply the 2×2 matrix.
        exec_status, delta_note = _status_from_paper_observed(
            paper_val=paper_val,
            observed=observed,
            metric=metric or "metric",
        )
        exec_passes = exec_status == "Supported"
        paper_score, _ = _score_paper_evidence(
            claim=claim, evidence=evidence, location=location, claim_type="experimental"
        )
        paper_strong = paper_score >= 3
        paper_conflict = paper_score <= -1
        norm_metric = _norm_metric_key(metric)
        base = (
            f"Claim mapped to {dataset} / {metric}. "
            f"Paper={_fmt_value(paper_val, metric_key=norm_metric)}, "
            f"Reproduced={_fmt_value(observed, metric_key=norm_metric)}. "
            f"{delta_note}."
        )
        if paper_strong and exec_passes:
            return base, "Supported"
        if paper_strong and not exec_passes:
            return base + " Paper evidence is strong but reproduction did not align.", "Partially supported"
        if paper_conflict:
            return base + " Paper evidence is internally inconsistent.", "In conflict"
        # paper partial
        if exec_passes:
            return base + " Results reproduced but paper evidence is incomplete.", "Partially supported"
        return base + " Incomplete paper evidence and reproduction did not align.", "In conflict"

    # No execution data or values could not be extracted — fall back to paper-only scoring.
    paper_score, parts = _score_paper_evidence(
        claim=claim, evidence=evidence, location=location, claim_type="experimental"
    )
    return " ".join(parts), _paper_status_from_score(paper_score)


def _build_theoretical_claim_assessment(*, claim: str, evidence: str, location: str) -> tuple[str, str]:
    score, parts = _score_paper_evidence(
        claim=claim, evidence=evidence, location=location, claim_type="theoretical"
    )
    return " ".join(parts), _paper_status_from_score(score)


def _build_methodological_claim_assessment(*, claim: str, evidence: str, location: str) -> tuple[str, str]:
    score, parts = _score_paper_evidence(
        claim=claim, evidence=evidence, location=location, claim_type="methodological"
    )
    return " ".join(parts), _paper_status_from_score(score)


def _augment_claims_with_assessment_status(
    markdown_text: str,
    *,
    summary: dict[str, Any],
    alignment: dict[str, Any],
) -> str:
    text = str(markdown_text or "")
    sec = re.search(r"(?ims)^##\s+3\.\s+Claims\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")
    lines = body.splitlines()
    header_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("|") and "Claim" in s and "Evidence" in s and "Location" in s:
            header_idx = i
            break
    if header_idx < 0:
        return text

    def _cell_safe(v: str) -> str:
        # Prevent markdown table column break due to literal pipe in generated text.
        return str(v or "").replace("|", "/").strip()

    legend = _claims_status_legend_colored()
    if legend not in body:
        lines.insert(header_idx, legend)
        lines.insert(header_idx + 1, "")
        header_idx += 2

    header_cells = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    normalized_headers = [_strip_inline_formatting(c).strip().lower() for c in header_cells]

    def _header_index(*candidates: str, exact: str = "") -> int:
        if exact:
            exact_norm = str(exact or "").strip().lower()
            for idx, name in enumerate(normalized_headers):
                if name == exact_norm:
                    return idx
        for idx, name in enumerate(normalized_headers):
            if any(candidate in name for candidate in candidates):
                return idx
        return -1

    claim_type_idx = _header_index("claim type", exact="claim type")
    claim_idx = _header_index("claim", exact="claim")
    evidence_idx = _header_index("evidence")
    assessment_idx = _header_index("assessment")
    location_idx = _header_index("location")

    def _cell_value(cells: list[str], idx: int, default: str = "") -> str:
        if idx < 0 or idx >= len(cells):
            return default
        return str(cells[idx] or "").strip()

    def _is_meaningful_assessment(value: str) -> bool:
        plain = _strip_inline_formatting(value).strip().lower()
        if not plain:
            return False
        return plain not in {
            "not found in manuscript",
            "not found",
            "n/a",
            "na",
            "none",
            "unknown",
            "not provided",
            "unspecified",
        }

    # separator line expected right after header
    sep_idx = header_idx + 1
    row_start = header_idx + 2
    row_end = row_start
    while row_end < len(lines):
        s = lines[row_end].strip()
        if s.startswith("|") and s.endswith("|"):
            row_end += 1
            continue
        break

    new_header = "| Claim | Evidence | Assessment | Status | Location |"
    new_sep = "|---|---|---|---|---|"
    new_rows: list[str] = []
    for ln in lines[row_start:row_end]:
        s = ln.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        claim = _cell_value(cells, claim_idx, _cell_value(cells, 0, "Not found in manuscript"))
        evidence = _cell_value(cells, evidence_idx, _cell_value(cells, 1, "Not found in manuscript"))
        default_location = _cell_value(cells, len(cells) - 1, "Not found in manuscript")
        location = _cell_value(cells, location_idx, default_location)
        authored_assessment = _cell_value(cells, assessment_idx, "")
        model_claim_type = _cell_value(cells, claim_type_idx, "")
        resolved_claim_type = _resolve_claim_type_label(
            model_claim_type=model_claim_type,
            claim=claim,
            evidence=evidence,
            location=location,
        )
        if resolved_claim_type == "Experimental":
            _generated_assess, stat = _build_experimental_claim_assessment(
                claim=claim,
                evidence=evidence,
                location=location,
                alignment=alignment,
            )
        elif resolved_claim_type == "Theoretical":
            _generated_assess, stat = _build_theoretical_claim_assessment(
                claim=claim,
                evidence=evidence,
                location=location,
            )
        else:
            _generated_assess, stat = _build_methodological_claim_assessment(
                claim=claim,
                evidence=evidence,
                location=location,
            )
        # The final assessment is system-derived and type-aware:
        # experimental claims compare with aligned execution evidence,
        # while theoretical/methodological claims are judged from manuscript anchors.
        assess = _generated_assess or (
            authored_assessment
            if _is_meaningful_assessment(authored_assessment)
            else "Not found in manuscript"
        )
        claim = _cell_safe(claim)
        evidence = _cell_safe(evidence)
        assess = _cell_safe(assess)
        stat = _cell_safe(_status_with_symbol(stat))
        location = _cell_safe(location)
        new_rows.append(f"| {claim} | {evidence} | {assess} | {stat} | {location} |")

    lines[header_idx] = new_header
    if sep_idx < len(lines):
        lines[sep_idx] = new_sep
    lines[row_start:row_end] = new_rows

    new_body = "\n".join(lines).strip("\n") + "\n\n"
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


def _augment_experiment_with_eval_status(
    markdown_text: str,
    *,
    summary: dict[str, Any],
    alignment: dict[str, Any],
) -> str:
    text = str(markdown_text or "")
    sec = re.search(r"(?ims)^##\s+5\.\s+Experiment\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")
    # Main result legend
    main_legend = _experiment_status_legend_colored()
    if re.search(r"(?im)^###\s+(?:\*\*)?Main Result(?:\*\*)?\s*$", body) and main_legend not in body:
        body = re.sub(
            r"(?im)^###\s+(?:\*\*)?Main Result(?:\*\*)?\s*$",
            f"### Main Result\n\n{main_legend}",
            body,
            count=1,
        )

    # Main table
    main_match = re.search(
        r"(?im)^(\|\s*Task\s*\|[^\n]*Difference\s*\(Δ\)\s*\|)\n(\|[-:\| ]+\|)\n(?P<rows>(?:\|[^\n]*\|\n?)*)",
        body,
    )
    if main_match:
        header = "| **Task** | **Dataset** | **Metric** | **Best Baseline** | **Paper Result** | **Difference (Δ)** | **Evaluation Status** |"
        sep = "|---|---|---|---|---|---|---|"
        rows_raw = main_match.group("rows")
        new_rows: list[str] = []
        for ln in rows_raw.splitlines():
            s = ln.strip()
            if not (s.startswith("|") and s.endswith("|")):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) < 6:
                continue
            dataset = cells[1]
            metric = cells[2]
            paper_result = cells[4]
            cell = _row_eval_cell(
                dataset=dataset,
                metric=metric,
                paper_result=paper_result,
                alignment=alignment,
            )
            new_rows.append("| " + " | ".join([*cells[:6], _status_with_symbol(cell)]) + " |")
        rebuilt = "\n".join([header, sep, *new_rows]) + "\n"
        body = body[: main_match.start()] + rebuilt + body[main_match.end() :]

    # Ablation legend and table
    if re.search(r"(?im)^###\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$", body):
        tail_match = re.search(r"(?ims)^###\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$\n(?P<tail>.*)$", body)
        tail = str(tail_match.group("tail") or "") if tail_match else ""
        if main_legend not in tail[:300]:
            body = re.sub(
                r"(?im)^###\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$",
                f"### Ablation Result\n\n{main_legend}",
                body,
                count=1,
            )

    abl_match = re.search(
        r"(?im)^(\|\s*Ablation Dimension\s*\|[^\n]*Difference\s*\(Δ\)\s*\|)\n(\|[-:\| ]+\|)\n(?P<rows>(?:\|[^\n]*\|\n?)*)",
        body,
    )
    if abl_match:
        header = "| **Ablation Dimension** | **Configuration** | **Full Model** | **Paper Result** | **Difference (Δ)** | **Evaluation Status** |"
        sep = "|---|---|---|---|---|---|"
        rows_raw = abl_match.group("rows")
        parsed_rows: list[list[str]] = []
        for ln in rows_raw.splitlines():
            s = ln.strip()
            if not (s.startswith("|") and s.endswith("|")):
                continue
            parsed_rows.append([c.strip() for c in s.strip("|").split("|")])
        table_metric_hint = _infer_metric_hint_from_table(
            headers=["Ablation Dimension", "Configuration", "Full Model", "Paper Result", "Difference (Δ)"],
            rows=parsed_rows,
            block_text=body[abl_match.start() : abl_match.end()],
        )
        new_rows: list[str] = []
        for cells in parsed_rows:
            if len(cells) < 5:
                continue
            paper_result = cells[3]
            dataset_hint, metric_hint = _claim_dataset_metric_hint(" ".join(cells), alignment=alignment)
            if not metric_hint:
                metric_hint = table_metric_hint or "metric"
            if not dataset_hint:
                dataset_hint = ""
            cell = _row_eval_cell(
                dataset=dataset_hint,
                metric=metric_hint,
                paper_result=paper_result,
                alignment=alignment,
            )
            new_rows.append("| " + " | ".join([*cells[:5], _status_with_symbol(cell)]) + " |")
        rebuilt = "\n".join([header, sep, *new_rows]) + "\n"
        body = body[: abl_match.start()] + rebuilt + body[abl_match.end() :]

    new_body = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


def _compress_experiment_note(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = re.search(r"(?ims)^##\s+5\.\s+Experiment\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")
    # Remove Note lines; ablation context should be carried by the Location line.
    body = re.sub(r"(?im)^\s*Note\s*:\s*.*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return text[: sec.start("body")] + body + text[sec.end("body") :]


def _strip_inline_formatting(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = s.replace("✓", "").replace("⚠", "").replace("✗", "")
    return " ".join(s.split()).strip()


def _bold_label_line(text: str, label: str) -> str:
    pattern = re.compile(rf"(?im)^(\s*)(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*(.*?)\s*$")

    def _repl(match: re.Match[str]) -> str:
        indent = str(match.group(1) or "")
        value = str(match.group(2) or "").strip()
        # Avoid duplicated emphasis markers when line already contains bolded values.
        if value.startswith("**"):
            value = value[2:].lstrip()
        if value.endswith("**"):
            value = value[:-2].rstrip()
        return f"{indent}**{label}:** {value}"

    return pattern.sub(_repl, str(text or ""))


def _as_status_label(value: str) -> str:
    raw = _strip_inline_formatting(value).lower()
    if "paper-supported" in raw or "paper supported" in raw or "supported by the paper" in raw:
        return "paper-supported"
    if "partial" in raw:
        return "Partially supported"
    if "support" in raw:
        return "Supported"
    if "conflict" in raw or "fail" in raw:
        return "In conflict"
    return "Inconclusive"


def _parse_numeric_delta(text: str) -> float | None:
    raw = _strip_inline_formatting(text)
    if not raw or "not found" in raw.lower():
        return None
    return _first_float(raw)


def _colorize_difference_cell(*, diff_text: str, metric_text: str = "") -> tuple[str, str]:
    delta = _parse_numeric_delta(diff_text)
    if delta is None:
        return (diff_text, "Inconclusive")
    higher_better = _metric_higher_is_better(metric_text) if metric_text else True
    if abs(delta) <= 1e-12:
        clean = _strip_inline_formatting(diff_text) or "0"
        return (f"**{clean}**", "Inconclusive")
    improved = delta > 0 if higher_better else delta < 0
    clean = _strip_inline_formatting(diff_text) or str(delta)
    if improved:
        return (f'<span style="color: green;">{clean}</span>', "Supported")
    return (f'<span style="color: red;">{clean}</span>', "In conflict")


def _colorize_ablation_difference_cell(*, diff_text: str) -> tuple[str, str]:
    # Product rule requested by user:
    # in Ablation table, negative delta means an effective change -> green.
    delta = _parse_numeric_delta(diff_text)
    if delta is None:
        return (diff_text, "Inconclusive")
    clean = _strip_inline_formatting(diff_text) or str(delta)
    if abs(delta) <= 1e-12:
        return (f"**{clean}**", "Inconclusive")
    if delta < 0:
        return (f'<span style="color: green;">{clean}</span>', "Supported")
    return (f'<span style="color: red;">{clean}</span>', "In conflict")


def _fmt_float_trim(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _extract_baseline_model_name(raw_cell: str) -> str:
    raw = _strip_inline_formatting(raw_cell)
    if not raw:
        return "Baseline"
    m_name_before_num = re.match(r"^\s*(?P<name>.+?)\s*\(\s*[-+]?(?:\d+\.\d+|\d+|\.\d+)\s*\)\s*$", raw)
    if m_name_before_num:
        candidate = str(m_name_before_num.group("name") or "").strip()
        if candidate:
            return candidate
    m_num_before_name = re.match(r"^\s*[-+]?(?:\d+\.\d+|\d+|\.\d+)\s*\(\s*(?P<name>.+?)\s*\)\s*$", raw)
    if m_num_before_name:
        candidate = str(m_num_before_name.group("name") or "").strip()
        if candidate:
            return candidate
    # Remove standalone numeric tokens while preserving alphanumeric model names like ConvS2S.
    no_num = re.sub(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?![A-Za-z])", " ", raw)
    no_num = re.sub(r"[()\[\]{}]", " ", no_num)
    no_num = re.sub(r"[,;:]", " ", no_num)
    no_num = " ".join(no_num.split()).strip()
    if not no_num or no_num.lower() in {"not found in manuscript", "n/a", "na", "not found"}:
        return "Baseline"
    return no_num


def _best_baseline_numeric_model(
    *,
    baseline_cell: str,
    paper_result_cell: str,
    diff_cell: str,
    metric_hint: str = "",
) -> str:
    base_raw = _strip_inline_formatting(baseline_cell)
    model = _extract_baseline_model_name(base_raw)
    baseline_val = _metric_aware_value(base_raw, metric_hint=metric_hint)
    paper_val = _metric_aware_value(_strip_inline_formatting(paper_result_cell), metric_hint=metric_hint)
    diff_val = _parse_numeric_delta(diff_cell)
    reconstructed = None
    if paper_val is not None and diff_val is not None:
        reconstructed = paper_val - diff_val

    # If baseline text is missing or inconsistent with paper-diff arithmetic,
    # prefer deterministic reconstruction from `paper_result - difference`.
    if reconstructed is not None:
        if baseline_val is None:
            baseline_val = reconstructed
        else:
            mismatch = abs((paper_val - baseline_val) - diff_val) if paper_val is not None else 0.0
            tolerance = max(0.02, abs(diff_val) * 0.02)
            if mismatch > tolerance:
                baseline_val = reconstructed
    if baseline_val is None:
        return f"Not found({model})"
    return f"{_fmt_float_trim(baseline_val)}({model})"


def _style_status_value(value: str) -> str:
    normalized = _as_status_label(value)
    if normalized == "Supported":
        return '<span style="color: green;">✓ Supported</span>'
    if normalized == "paper-supported":
        return '<span style="color: #1E5EFF;">☑ Paper-supported</span>'
    if normalized == "Partially supported":
        return '<span style="color: #E6B800;">⚠ Partially supported</span>'
    if normalized == "In conflict":
        return '<span style="color: red;">✗ In conflict</span>'
    return '<span style="color: #E6B800;">⚠ Inconclusive</span>'


def _style_experiment_eval_status(*, status_label: str, raw_value: str) -> str:
    value = _strip_inline_formatting(raw_value)
    metric_value = _first_float(value)
    normalized = _as_status_label(status_label)
    if normalized in {"Supported", "In conflict"} and metric_value is None:
        # Without numeric evidence, avoid presenting deterministic pass/fail.
        normalized = "Inconclusive"
    if normalized == "Supported":
        icon = "✓"
        color = "green"
    elif normalized == "In conflict":
        icon = "✗"
        color = "red"
    else:
        icon = "⚠"
        color = "#E6B800"
    if metric_value is None:
        return f'<span style="color: {color};">{icon}()</span>'
    return f'<span style="color: {color};">{icon}({_fmt_float_trim(metric_value)})</span>'


def _infer_metric_hint_from_table(*, headers: list[str], rows: list[list[str]], block_text: str) -> str:
    corpus_parts: list[str] = []
    corpus_parts.extend(headers)
    corpus_parts.append(block_text)
    for row in rows[:8]:
        corpus_parts.extend(row)
    text = " ".join(str(x or "") for x in corpus_parts).lower()
    if "mean rank" in text or re.search(r"(?<![a-z])mr(?!r)(?![a-z])", text):
        return "MR"
    if "mrr" in text:
        return "MRR"
    if "hits@10" in text or "h@10" in text:
        return "H@10"
    if "hits@3" in text or "h@3" in text:
        return "H@3"
    if "hits@1" in text or "h@1" in text:
        return "H@1"
    if "bleu" in text:
        return "BLEU"
    if "f1" in text:
        return "F1"
    if "acc" in text or "accuracy" in text:
        return "Accuracy"
    return ""


def _format_metric_value_for_cell(value: float, *, metric_hint: str) -> str:
    mk = _norm_metric_key(metric_hint)
    if mk == "mr":
        return str(round(value))
    return _fmt_float_trim(value)


def _normalize_experiment_tables_in_block(block: str) -> tuple[str, list[str]]:
    lines = str(block or "").splitlines()
    row_statuses: list[str] = []
    i = 0
    while i + 1 < len(lines):
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and header.endswith("|") and re.fullmatch(r"\|[ :\-|]+\|", sep)):
            i += 1
            continue
        j = i + 2
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("|") and s.endswith("|"):
                j += 1
                continue
            break

        headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows: list[list[str]] = []
        for k in range(i + 2, j):
            s = lines[k].strip()
            if not (s.startswith("|") and s.endswith("|")):
                continue
            rows.append([c.strip() for c in s.strip("|").split("|")])

        lower_headers = [_strip_inline_formatting(h).lower() for h in headers]
        diff_idx = next((idx for idx, h in enumerate(lower_headers) if "difference" in h), -1)
        metric_idx = next((idx for idx, h in enumerate(lower_headers) if h == "metric"), -1)
        best_baseline_idx = next((idx for idx, h in enumerate(lower_headers) if "best baseline" in h), -1)
        full_model_idx = next((idx for idx, h in enumerate(lower_headers) if h == "full model"), -1)
        paper_result_idx = next((idx for idx, h in enumerate(lower_headers) if h == "paper result"), -1)
        status_idx = next((idx for idx, h in enumerate(lower_headers) if "status" in h), -1)
        table_metric_hint = _infer_metric_hint_from_table(headers=headers, rows=rows, block_text=block)
        is_ablation_table = any("ablation dimension" in h for h in lower_headers)
        if status_idx < 0:
            headers.append("Evaluation Status")
            status_idx = len(headers) - 1
            for row in rows:
                row.append("Inconclusive")

        for ridx, row in enumerate(rows):
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            metric_text = (
                row[metric_idx]
                if metric_idx >= 0 and metric_idx < len(row) and str(row[metric_idx]).strip()
                else table_metric_hint
            )
            raw_status_cell = row[status_idx] if status_idx < len(row) else ""
            if (
                best_baseline_idx >= 0
                and paper_result_idx >= 0
                and diff_idx >= 0
                and best_baseline_idx < len(row)
                and paper_result_idx < len(row)
                and diff_idx < len(row)
            ):
                row[best_baseline_idx] = _best_baseline_numeric_model(
                    baseline_cell=row[best_baseline_idx],
                    paper_result_cell=row[paper_result_idx],
                    diff_cell=row[diff_idx],
                    metric_hint=metric_text,
                )
            if diff_idx >= 0 and diff_idx < len(row):
                if is_ablation_table:
                    colored_diff, _ = _colorize_ablation_difference_cell(diff_text=row[diff_idx])
                else:
                    colored_diff, _ = _colorize_difference_cell(
                        diff_text=row[diff_idx],
                        metric_text=metric_text,
                    )
                row[diff_idx] = colored_diff
            row[status_idx] = _as_status_label(raw_status_cell)
            # Experiment section is restricted to 3 statuses only.
            if row[status_idx] == "paper-supported":
                row[status_idx] = "Inconclusive"
            row_statuses.append(_as_status_label(row[status_idx]))
            row[status_idx] = _style_experiment_eval_status(
                status_label=row[status_idx],
                raw_value=raw_status_cell,
            )
            rows[ridx] = row

        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = ["| " + " | ".join(r[: len(headers)]) + " |" for r in rows]
        replacement = [header_line, sep_line, *data_lines]
        lines[i:j] = replacement
        i += len(replacement)
    return "\n".join(lines), row_statuses


def _apply_experiment_hard_requirements(markdown_text: str) -> str:
    text = str(markdown_text or "")
    sec = re.search(r"(?ims)^##\s+(?:\*\*)?5\.\s+Experiment(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    if not sec:
        return text
    body = sec.group("body")
    subsections = list(re.finditer(r"(?im)^###\s+(?:\*\*)?(Main Result|Ablation Result)(?:\*\*)?\s*$", body))
    if not subsections:
        body = body.strip("\n")
        if body:
            body += "\n\n**Location:** Not found in manuscript\n"
        else:
            body = "**Location:** Not found in manuscript\n"
        return text[: sec.start("body")] + body + text[sec.end("body") :]

    rebuilt: list[str] = []
    cursor = 0
    for idx, subsection in enumerate(subsections):
        start = subsection.start()
        end = subsections[idx + 1].start() if idx + 1 < len(subsections) else len(body)
        rebuilt.append(body[cursor:start])
        block = body[start:end]
        block = _bold_label_line(block, "Location")
        # Experiment keeps Location line; subsection Status line is intentionally removed.
        block = re.sub(r"(?im)^\s*(?:\*\*)?Status(?:\*\*)?\s*:\s*.*$", "", block)
        has_location = bool(re.search(r"(?im)^\s*\*\*Location:\*\*\s*.+$", block))
        if not has_location:
            block = re.sub(
                r"(?im)^(###\s+(?:\*\*)?(?:Main Result|Ablation Result)(?:\*\*)?\s*)$",
                r"\1\n\n**Location:** Not found in manuscript",
                block,
                count=1,
            )

        block = re.sub(
            r"(?im)^###\s+(?:\*\*)?(Main Result|Ablation Result)(?:\*\*)?\s*$",
            r"### **\1**",
            block,
        )

        block, _row_statuses = _normalize_experiment_tables_in_block(block)

        # Ensure no subsection-level Status line remains.
        block = re.sub(r"(?im)^\s*\*\*Status:\*\*\s*.*$", "", block)
        rebuilt.append(block)
        cursor = end
    rebuilt.append(body[cursor:])
    new_body = "".join(rebuilt)
    new_body = re.sub(r"\n{3,}", "\n\n", new_body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


def _bold_markdown_table_headers(markdown_text: str) -> str:
    lines = str(markdown_text or "").splitlines()
    i = 0
    while i + 1 < len(lines):
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if header.startswith("|") and header.endswith("|") and re.fullmatch(r"\|[ :\-|]+\|", sep):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            bolded: list[str] = []
            for c in cells:
                plain = _strip_inline_formatting(c)
                if not plain:
                    bolded.append(c)
                    continue
                if plain.lower().startswith("difference"):
                    plain = "Difference (Δ)"
                bolded.append(f"**{plain}**")
            lines[i] = "| " + " | ".join(bolded) + " |"
            i += 2
            continue
        i += 1
    return "\n".join(lines)


def _colorize_status_fields(markdown_text: str) -> str:
    text = str(markdown_text or "")
    # Status label lines.
    text = re.sub(
        r"(?im)^(\s*\*\*Status:\*\*\s*)(.+?)\s*$",
        lambda m: m.group(1) + _style_status_value(m.group(2)),
        text,
    )
    # Status columns in markdown tables.
    lines = text.splitlines()
    i = 0
    while i + 1 < len(lines):
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and header.endswith("|") and re.fullmatch(r"\|[ :\-|]+\|", sep)):
            i += 1
            continue
        headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        status_indices: list[int] = []
        for idx, h in enumerate(headers):
            plain = _strip_inline_formatting(h).lower()
            if "status" not in plain:
                continue
            # Experiment evaluation status is already normalized in dedicated pass.
            if "evaluation status" in plain:
                continue
            status_indices.append(idx)
        j = i + 2
        while j < len(lines):
            s = lines[j].strip()
            if not (s.startswith("|") and s.endswith("|")):
                break
            if status_indices:
                cells = [c.strip() for c in s.strip("|").split("|")]
                for idx in status_indices:
                    if idx < len(cells):
                        cells[idx] = _style_status_value(cells[idx])
                lines[j] = "| " + " | ".join(cells) + " |"
            j += 1
        i = j
    return "\n".join(lines)


def _normalize_status_legends(markdown_text: str) -> str:
    text = str(markdown_text or "")
    claims_sec = re.search(
        r"(?ims)^##\s+(?:\*\*)?3\.\s+Claims(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text
    )
    if claims_sec:
        body = claims_sec.group("body")
        if re.search(r"(?im)^\s*\(Status legend:.*\)\s*$", body):
            body = re.sub(r"(?im)^\s*\(Status legend:.*\)\s*$", _claims_status_legend_colored(), body)
        else:
            body = _claims_status_legend_colored() + "\n\n" + body.lstrip("\n")
        text = text[: claims_sec.start("body")] + body + text[claims_sec.end("body") :]

    exp_sec = re.search(
        r"(?ims)^##\s+(?:\*\*)?5\.\s+Experiment(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", text
    )
    if exp_sec:
        body = exp_sec.group("body")
        body = re.sub(r"(?im)^\s*\(Status legend:.*\)\s*$", _experiment_status_legend_colored(), body)
        text = text[: exp_sec.start("body")] + body + text[exp_sec.end("body") :]
    return text


def _apply_hard_formatting_requirements(markdown_text: str) -> str:
    text = str(markdown_text or "")
    # Required section titles in bold while preserving heading structure.
    text = re.sub(
        r"(?im)^##\s+(?:\*\*)?(1\.\s*Metadata|2\.\s*Technical Positioning|3\.\s*Claims|4\.\s*Summary|5\.\s*Experiment)(?:\*\*)?\s*$",
        r"## **\1**",
        text,
    )
    text = _normalize_status_legends(text)
    for label in ("Paper scope", "Evaluation scope", "Strengths", "Weaknesses", "Location", "Status"):
        text = _bold_label_line(text, label)
    text = _apply_experiment_hard_requirements(text)
    text = _bold_markdown_table_headers(text)
    text = _colorize_status_fields(text)
    return text


def _compact_ref_label_from_title(*, title: str, year: str | None, rid: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(title or ""))
    stop = {
        "the",
        "with",
        "for",
        "and",
        "from",
        "based",
        "using",
        "on",
        "of",
        "to",
        "in",
        "a",
        "an",
        "multi",
        "relational",
        "graph",
        "convolutional",
        "networks",
        "network",
        "towards",
        "via",
        "learning",
        "representation",
        "representations",
        "knowledge",
        "graphs",
    }
    picked = [t for t in tokens if t.lower() not in stop]
    if not picked:
        picked = [t for t in tokens if t]
    compact_words = picked[:2]
    if not compact_words:
        return rid
    label = " ".join((w[:8] if len(w) > 8 else w) for w in compact_words)
    label = " ".join(w[:1].upper() + w[1:].lower() for w in label.split())
    return label[:18].strip() or rid


def _semantic_ref_map_from_payload(job_dir: Path) -> dict[str, str]:
    payload_path = job_dir / "semantic_scholar_candidates.json"
    if not payload_path.exists():
        return {}
    try:
        payload = read_json(payload_path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    papers = payload.get("papers")
    if not isinstance(papers, list):
        return {}
    ref_map: dict[str, str] = {}
    for row in papers:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if not re.fullmatch(r"R\d+", rid):
            continue
        title = str(row.get("title") or "").strip()
        year = str(row.get("year") or "").strip()
        ref_map[rid] = _compact_ref_label_from_title(title=title, year=year, rid=rid)
    return ref_map


def _compact_technical_positioning_reference_labels(markdown_text: str, *, job_dir: Path) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = _TECHNICAL_POSITIONING_SECTION_PATTERN.search(text)
    if not sec:
        return text
    body = sec.group("body")
    lines = body.splitlines()

    ref_map: dict[str, str] = _semantic_ref_map_from_payload(job_dir)
    ref_line = re.compile(r"^\s*-\s*(R\d+)\s*:\s*(.+)$")
    for raw in lines:
        m = ref_line.match(raw.strip())
        if not m:
            continue
        rid = m.group(1).strip()
        rest = m.group(2).strip()
        ym = re.search(r"\((\d{4})[^)]*\)", rest)
        year = ym.group(1) if ym else ""
        title = re.sub(r"\(\d{4}[^)]*\)", "", rest).strip(" -")
        ref_map.setdefault(rid, _compact_ref_label_from_title(title=title, year=year, rid=rid))

    # Remove standalone legend block lines.
    cleaned: list[str] = []
    for raw in lines:
        s = raw.strip()
        if re.match(r"^\*\*Legend:\*\*\s*$", s):
            continue
        if re.match(r"^-\s*R\d+\s*:", s):
            continue
        cleaned.append(raw)

    # Replace table header R-columns with compact labels, without R-identifiers.
    for i, raw in enumerate(cleaned):
        s = raw.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].lower() != "research domain" or cells[1].lower() != "method":
            continue
        new_cells = cells[:2]
        for c in cells[2:]:
            rid_match = re.match(r"^\s*(R\d+)(?::\s*(.*))?$", c)
            if rid_match:
                rid = rid_match.group(1)
                explicit_label = str(rid_match.group(2) or "").strip()
                new_cells.append(explicit_label or ref_map.get(rid, rid))
            else:
                # Plain niche/capability labels are already the desired output.
                # Do not reinterpret them as R1/R2 columns from Semantic Scholar.
                new_cells.append(c)
        cleaned[i] = "| " + " | ".join(new_cells) + " |"
        break

    new_body = "\n".join(cleaned).strip("\n") + "\n\n"
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


def _extract_title_method_hint(markdown_text: str) -> str:
    text = str(markdown_text or "")
    m = re.search(r"(?im)^\s*-\s*(?:\*\*)?Title(?:\*\*)?\s*:\s*(.+?)\s*$", text)
    if not m:
        return ""
    title = str(m.group(1) or "").strip()
    # Prefer acronym-like token (e.g., COMPGCN).
    tokens = re.findall(r"[A-Za-z0-9\-]+", title)
    for t in tokens:
        alpha = re.sub(r"[^A-Za-z]", "", t)
        if len(alpha) >= 3 and alpha.isupper():
            return t
    # Fallback: first significant title token.
    for t in tokens:
        if len(t) >= 4:
            return t
    return ""


def _extract_paper_method_hint(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return ""

    def _is_valid_method_token(token: str) -> bool:
        t = str(token or "").strip()
        if len(t) < 3:
            return False
        low = t.lower()
        banned = {
            "title",
            "task",
            "code",
            "table",
            "figure",
            "dataset",
            "appendix",
            "method",
            "model",
            "approach",
            "architecture",
            "research",
            "domain",
            "this",
            "work",
            "paper",
            "main",
            "result",
            "ablation",
            "baseline",
            "train",
            "training",
            "test",
            "testing",
            "val",
            "validation",
            "mrr",
            "mr",
            "accuracy",
            "auc",
            "f1",
            "bleu",
            "rouge",
            "error",
            "loss",
            "imagenet",
            "mnist",
            "cifar",
            "pascal",
            "voc",
            "caltech",
            "jft",
            "coco",
            "flickr",
            "ucf",
            "tacos",
            "kinetics",
            "charades",
            "ava",
            "nuswide",
            "fb15k",
            "wn18",
            "wn18rr",
            "fb15k237",
            "for",
            "and",
            "or",
            "of",
            "with",
            "using",
            "based",
        }
        if low in banned:
            return False
        # Ignore retrieval-id like R1/R2.
        if re.fullmatch(r"R\d+", t, flags=re.IGNORECASE):
            return False
        return True

    # Strong cue patterns from manuscript/report body.
    cue_patterns = (
        r"(?i)\bwe\s+(?:propose|present|introduce|develop)\s+([A-Za-z][A-Za-z0-9\-]{2,})\b",
        r"(?i)\bour\s+(?:method|model|approach)\s+([A-Za-z][A-Za-z0-9\-]{2,})\b",
        r"(?i)\bcalled\s+([A-Za-z][A-Za-z0-9\-]{2,})\b",
    )
    for pattern in cue_patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        cand = str(m.group(1) or "").strip()
        if _is_valid_method_token(cand):
            return cand

    # Fallback: score frequent method-like tokens in body context.
    scores: dict[str, int] = {}
    method_like = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}\b")
    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        low = line.lower()
        context_bonus = 0
        if re.search(r"(?i)\b(propose|proposed|introduce|our method|our model|our approach)\b", low):
            context_bonus += 3
        if re.search(r"(?i)\b(method|model|approach|architecture)\b", low):
            context_bonus += 1
        for m in method_like.finditer(line):
            cand = str(m.group(0) or "").strip()
            if not _is_valid_method_token(cand):
                continue
            scores[cand] = int(scores.get(cand, 0)) + 1 + context_bonus

    if not scores:
        return ""
    ranked = sorted(scores.items(), key=lambda kv: (-int(kv[1]), len(kv[0]), kv[0].lower()))
    return str(ranked[0][0] or "").strip()


def _extract_report_title_text(markdown_text: str) -> str:
    text = str(markdown_text or "")
    m = re.search(r"(?im)^\s*-\s*(?:\*\*)?Title(?:\*\*)?\s*:\s*(.+?)\s*$", text)
    if m:
        return str(m.group(1) or "").strip()
    h = re.search(r"(?im)^\[(.+?)\]\s*$", text)
    if h:
        return str(h.group(1) or "").strip()
    return ""


def _normalize_title_tokens_local(title: str) -> list[str]:
    raw = str(title or "").strip().lower()
    if not raw:
        return []
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    stop = {"a", "an", "and", "for", "from", "in", "is", "of", "on", "the", "to", "with"}
    return [tok for tok in raw.split() if tok and tok not in stop]


def _looks_like_self_title_variant(*, report_title: str, candidate: str) -> bool:
    title_tokens = _normalize_title_tokens_local(report_title)
    cand_tokens = _normalize_title_tokens_local(candidate)
    if not title_tokens or not cand_tokens:
        return False
    t_norm = " ".join(title_tokens)
    c_norm = " ".join(cand_tokens)
    if t_norm == c_norm:
        return True
    if len(title_tokens) >= 4 and (t_norm in c_norm or c_norm in t_norm):
        return True
    t_set = set(title_tokens)
    c_set = set(cand_tokens)
    inter = len(t_set & c_set)
    union = len(t_set | c_set)
    if union == 0:
        return False
    return inter >= 4 and (inter / union) >= 0.8


def _extract_method_acronym_from_title(report_title: str, text: str) -> str:
    """Find the paper's method acronym by trying initial-letter subsequences of title words.

    Splits the title on spaces and hyphens, keeps only uppercase-starting non-stop words,
    then tries every consecutive subsequence of their initials (length 2-7) and returns
    the one that appears most frequently in ``text`` (requires >= 2 occurrences).
    """
    if not report_title or not text:
        return ""
    stop = {
        "a", "an", "and", "as", "at", "by", "end", "for", "from", "in", "into",
        "is", "its", "of", "on", "or", "the", "to", "using", "via", "with",
    }
    words = re.split(r"[\s\-]+", report_title)
    sig = [w for w in words if w and w[:1].isupper() and w.lower() not in stop and len(w) >= 2]
    if len(sig) < 2:
        return ""
    initials = [w[0].upper() for w in sig]
    best_cand, best_count = "", 0
    for start in range(len(initials)):
        for length in range(2, min(8, len(initials) - start + 1)):
            candidate = "".join(initials[start : start + length])
            count = len(re.findall(r"\b" + re.escape(candidate) + r"\b", text))
            if count > best_count or (count == best_count and len(candidate) > len(best_cand)):
                best_cand, best_count = candidate, count
    return best_cand if best_count >= 2 else ""


def _normalize_technical_positioning_layout(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    sec = _TECHNICAL_POSITIONING_SECTION_PATTERN.search(text)
    if not sec:
        return text

    method_hint_raw = _extract_paper_method_hint(text).strip()
    method_hint = method_hint_raw.lower()
    report_title = _extract_report_title_text(text).strip()
    body = sec.group("body")

    # Remove explicit gap line in section 2.
    body = re.sub(r"(?im)^\s*Gap\s*:\s*.*$", "", body)
    # Use empty alt text to avoid rendering label words near the image.
    body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", r"![](\1)", body)
    # Remove "Figure x:" prefix from caption line.
    body = re.sub(r"(?im)^\s*Figure\s*\d*\s*:\s*(.+?)\s*$", r"\1", body)
    # Remove "Figure caption:" prefix if model emits it.
    body = re.sub(r"(?im)^\s*Figure\s*caption\s*:\s*(.+?)\s*$", r"\1", body)
    # Normalize "Figure x shows ..." sentence without using Figure marker.
    body = re.sub(r"(?im)^\s*Figure\s*\d+\s*shows\s*", "This overview shows ", body)
    # Remove verbose overview explanation line; keep a single short caption only.
    body = re.sub(r"(?im)^\s*This overview shows.*$", "", body)

    lines = body.splitlines()
    # If section has an image, enforce one-line short caption directly under image.
    image_idx = -1
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("![") and "](" in s and s.endswith(")"):
            image_idx = idx
            break
    if image_idx >= 0:
        title_acronym = _extract_method_acronym_from_title(report_title, text)
        caption_method = title_acronym or method_hint_raw or "the proposed method"
        # Remove existing short overview caption variants around image.
        filtered: list[str] = []
        for i, raw in enumerate(lines):
            s = raw.strip()
            if i in {image_idx + 1, image_idx + 2, image_idx + 3} and re.match(
                r"(?im)^(overview of .+\.?|this overview shows.+)$", s
            ):
                continue
            filtered.append(raw)
        lines = filtered
        # Re-locate image index after filtering.
        for idx, raw in enumerate(lines):
            s = raw.strip()
            if s.startswith("![") and "](" in s and s.endswith(")"):
                image_idx = idx
                break
        if image_idx >= 0:
            # Keep image as a standalone markdown paragraph so PDF renderer can load it as an image,
            # not inline text fallback.
            lines.insert(image_idx + 1, "")
            lines.insert(image_idx + 2, f"Overview of {caption_method}.")
            lines.insert(image_idx + 3, "")

    table_start = -1
    table_end = -1
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("|") and s.endswith("|"):
            if table_start < 0:
                table_start = idx
            table_end = idx
        elif table_start >= 0 and table_end >= table_start:
            break
    if table_start >= 0 and table_end >= table_start and table_end - table_start >= 2:
        table_lines = lines[table_start : table_end + 1]
        rows: list[list[str]] = []
        for raw in table_lines:
            s = raw.strip()
            if not (s.startswith("|") and s.endswith("|")):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
                continue
            rows.append(cells)
        if len(rows) >= 2:
            header = rows[0]
            body_rows = rows[1:]
            if (
                len(header) >= 2
                and header[0].strip().lower() == "research domain"
                and header[1].strip().lower() == "method"
            ):
                header[0] = "Research domain"
                header[1] = "Method"
                header = header[:2] + [
                    re.sub(r"^\s*R\d+\s*:?\s*", "", str(c or "")).strip() or f"Niche dimension {idx}"
                    for idx, c in enumerate(header[2:], start=1)
                ]

                def _normalize_niche_mark(value: str) -> str:
                    raw = _strip_inline_formatting(value).strip().lower()
                    if raw in {"√", "✓", "yes", "y", "true", "1", "supported", "present"}:
                        return "√"
                    if raw in {"×", "✗", "x", "no", "n", "false", "0", "absent"}:
                        return "×"
                    return "×"

                normalized_body_rows: list[list[str]] = []
                for r in body_rows:
                    if len(r) < len(header):
                        r = r + ["×"] * (len(header) - len(r))
                    r = r[: len(header)]
                    r[2:] = [_normalize_niche_mark(c) for c in r[2:]]
                    normalized_body_rows.append(r)
                body_rows = normalized_body_rows

                def _is_this_work_row(row: list[str]) -> bool:
                    domain = str(row[0] if len(row) > 0 else "").strip().lower()
                    method = str(row[1] if len(row) > 1 else "").strip().lower()
                    tokens = (
                        "this work",
                        "this paper",
                        "our work",
                        "our method",
                        "our approach",
                        "this method",
                        "this model",
                        "proposed",
                        "proposed method",
                        "proposed approach",
                        "ours",
                    )
                    return any(tok in domain for tok in tokens) or any(tok in method for tok in tokens)

                def _is_self_paper_row(row: list[str]) -> bool:
                    domain = str(row[0] if len(row) > 0 else "").strip()
                    method = str(row[1] if len(row) > 1 else "").strip()
                    if _is_this_work_row(row):
                        return True
                    if report_title and (
                        _looks_like_self_title_variant(report_title=report_title, candidate=method)
                        or _looks_like_self_title_variant(report_title=report_title, candidate=domain)
                    ):
                        return True
                    return bool(method_hint and method_hint in method.lower())

                # Primary rule: collect any explicit self-paper rows so external rows are filtered.
                normal_rows: list[list[str]] = []
                this_work_rows: list[list[str]] = []
                for r in body_rows:
                    if _is_self_paper_row(r):
                        this_work_rows.append(r)
                    else:
                        normal_rows.append(r)

                # Fallback: if no explicit this-work marker exists, use legacy heuristic to pick one.
                if not this_work_rows:
                    pick_idx = -1
                    for i, r in enumerate(normal_rows):
                        method = str(r[1] if len(r) > 1 else "").strip().lower()
                        if method_hint and method_hint in method:
                            pick_idx = i
                            break
                    if pick_idx >= 0:
                        this_work_rows.append(normal_rows.pop(pick_idx))

                normalized_this_work_rows: list[list[str]] = []
                if this_work_rows:
                    source_row = this_work_rows[-1]
                else:
                    source_row = ["Current paper", "This Work"] + ["×"] * max(0, len(header) - 2)
                if len(source_row) < len(header):
                    source_row = source_row + ["×"] * (len(header) - len(source_row))
                source_row = source_row[: len(header)]
                inferred_method = str(source_row[1] if len(source_row) > 1 else "").strip()
                inferred_method_low = inferred_method.lower()
                source_domain = str(source_row[0] if len(source_row) > 0 else "").strip()
                if inferred_method_low in {
                    "this work",
                    "this paper",
                    "our work",
                    "our method",
                    "our approach",
                    "proposed method",
                    "proposed approach",
                    "ours",
                }:
                    source_domain_low = source_domain.lower()
                    if source_domain and source_domain_low not in {
                        "this work",
                        "this paper",
                        "our work",
                        "current paper",
                    }:
                        inferred_method = source_domain
                    else:
                        inferred_method = ""
                paper_method = inferred_method or method_hint_raw or "Not found in manuscript"
                # Product requirement: keep the final self row fixed to
                # Research domain = This Work, Method = paper method.
                source_row[0] = "This Work"
                source_row[1] = paper_method
                source_row[2:] = [_normalize_niche_mark(c) for c in source_row[2:]]
                normalized_this_work_rows.append(source_row)

                body_rows = normal_rows + normalized_this_work_rows

                new_table = _format_table(header, body_rows)
                lines = lines[:table_start] + new_table.splitlines() + lines[table_end + 1 :]

    new_body = "\n".join(lines)
    new_body = re.sub(r"\n{3,}", "\n\n", new_body).strip("\n") + "\n\n"
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


def _inject_technical_positioning_overview_image(
    *,
    markdown_text: str,
    job_dir: Path,
    content_list: list[dict[str, Any]] | None,
) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text

    match = _TECHNICAL_POSITIONING_SECTION_PATTERN.search(text)
    if not match:
        return text

    body = match.group("body")
    page_match = _TECHNICAL_POSITIONING_PAGE_PATTERN.search(body)
    if page_match:
        # page hint is currently advisory only; image selection comes from MinerU image assets
        _ = page_match.group(1)

    figure_paths, mineru_caption = _pick_overview_mineru_figure_bundle(
        content_list=content_list,
        job_dir=job_dir,
    )
    if not figure_paths:
        fallback = _pick_overview_mineru_image(content_list=content_list, job_dir=job_dir)
        if fallback is not None:
            figure_paths = [fallback]
        else:
            figure_paths = _fallback_overview_images_from_assets(job_dir=job_dir, max_images=2)
            if not figure_paths:
                return text

    canonical_image = _materialize_canonical_overview_image(
        image_paths=figure_paths,
        job_dir=job_dir,
    )
    if canonical_image is None:
        return text

    # Remove any existing markdown image lines in this section; we will inject one canonical image.
    body = re.sub(r"(?im)^\s*!\[[^\]]*\]\([^)]+\)\s*$\n?", "", body)
    caption_match = re.search(r"(?im)^\s*Figure\s*1\s*:\s*.*$", body)
    caption_line = ""
    if caption_match:
        caption_line = str(caption_match.group(0) or "").strip()
        body = body[: caption_match.start()] + body[caption_match.end() :]

    # Prefer model caption in section output; fallback to MinerU caption if missing.
    if caption_line:
        caption_line = _abbreviate_figure_caption(caption_line)
    elif mineru_caption and str(mineru_caption).strip():
        caption_line = _abbreviate_figure_caption(str(mineru_caption))
    else:
        caption_line = ""

    body = body.lstrip("\n")
    images_block = "![Overview](./overview_figure.jpg)"
    image_block = f"{images_block}\n\n"
    if caption_line:
        image_block += f"{caption_line}\n\n"
    new_body = image_block + body
    return text[: match.start("body")] + new_body + text[match.end("body") :]


def _sanitize_technical_positioning_markers(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.strip():
        return text
    match = _TECHNICAL_POSITIONING_SECTION_PATTERN.search(text)
    if not match:
        return text
    body = match.group("body")
    cleaned_lines = [
        line for line in body.splitlines() if not _TECHNICAL_POSITIONING_MARKER_PATTERN.match(line)
    ]
    cleaned_body = "\n".join(cleaned_lines).strip("\n")
    if cleaned_body:
        cleaned_body = cleaned_body + "\n\n"
    return text[: match.start("body")] + cleaned_body + text[match.end("body") :]


def _publish_outputs_to_output_dir(
    *,
    job_id: str,
    final_md_path: Path,
    report_pdf_path: Path,
) -> tuple[Path, Path]:
    artifacts = ensure_artifact_paths(job_id)
    job_latest_md = Path(artifacts["latest_output_md"])
    job_latest_pdf = Path(artifacts["latest_output_pdf"])
    job_latest_md.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_md_path, job_latest_md)
    # Re-home section-2 image beside job-level latest markdown for stable local preview rendering.
    try:
        md_text = job_latest_md.read_text(encoding="utf-8")
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", md_text)
        if m:
            src = Path(m.group(1)).expanduser()
            if src.exists() and src.is_file():
                dst = job_latest_md.parent / "overview_figure.jpg"
                shutil.copy2(src, dst)
                md_text = md_text[: m.start(1)] + "./overview_figure.jpg" + md_text[m.end(1) :]
                job_latest_md.write_text(md_text, encoding="utf-8")
    except Exception:
        pass
    if report_pdf_path.exists():
        shutil.copy2(report_pdf_path, job_latest_pdf)
    return job_latest_md, job_latest_pdf


def _persist_mineru_image_files(
    *,
    job_dir: Path,
    image_files: dict[str, bytes] | None,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not image_files:
        return mapping
    assets_root = job_dir / "mineru_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    for key, value in image_files.items():
        rel = str(key or "").strip().replace("\\", "/")
        if not rel:
            continue
        safe_rel = rel.lstrip("/")
        target = (assets_root / safe_rel).resolve()
        # ensure write stays inside assets_root
        if assets_root.resolve() not in target.parents and target != assets_root.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        mapping[rel] = target
    return mapping


def _render_report_pdf(
    *,
    job_id: str,
    job_title: str,
    source_pdf_name: str,
    final_md_path: Path,
    source_pdf_path: Path,
    report_pdf_path: Path,
    annotations: list[AnnotationItem] | list[dict[str, Any]],
    content_list: list[dict[str, Any]] | None,
    token_usage: dict[str, int],
    agent_model: str,
) -> dict[str, int]:
    final_report_markdown = final_md_path.read_text(encoding="utf-8")
    # Keep model-authored content as-is; only do minimal marker cleanup and image injection.
    final_report_markdown = _sanitize_technical_positioning_markers(final_report_markdown)
    final_report_markdown = _inject_technical_positioning_overview_image(
        markdown_text=final_report_markdown,
        job_dir=final_md_path.parent,
        content_list=content_list,
    )
    final_report_markdown = _compact_technical_positioning_reference_labels(
        final_report_markdown,
        job_dir=final_md_path.parent,
    )
    final_report_markdown = _normalize_technical_positioning_layout(final_report_markdown)
    final_report_markdown = _demote_experiment_child_headings(final_report_markdown)
    final_report_markdown = _hard_validate_experiment_tables(
        final_report_markdown,
        content_list=content_list,
    )
    final_report_markdown = _stabilize_experiment_section(final_report_markdown)
    final_report_markdown = _ensure_experiment_contract(final_report_markdown)
    final_report_markdown = _compress_experiment_note(final_report_markdown)
    # The execution stage now runs as a separate sub-stage after the agent
    # runtime job (``stages/fact_generation/execution``) and emits its own
    # outputs there. The two augment helpers below still produce useful work
    # against empty payloads — they normalise the Claims / Experiment table
    # columns and insert their colour legends — but they cannot mark
    # execution-aligned status until/unless the execution-stage outputs are
    # wired back into this in-process render pass.
    execution_summary: dict[str, Any] = {}
    execution_alignment: dict[str, Any] = {}
    final_report_markdown = _augment_claims_with_assessment_status(
        final_report_markdown,
        summary=execution_summary,
        alignment=execution_alignment,
    )
    final_report_markdown = _augment_experiment_with_eval_status(
        final_report_markdown,
        summary=execution_summary,
        alignment=execution_alignment,
    )
    # Re-assert experiment section contract after augmentation to avoid accidental section loss.
    final_report_markdown = _ensure_experiment_contract(final_report_markdown)
    final_report_markdown = _apply_hard_formatting_requirements(final_report_markdown)
    write_text_atomic(final_md_path, final_report_markdown)
    final_report_markdown = _inject_overview_figure_image(
        markdown_text=final_report_markdown,
        source_pdf_path=source_pdf_path,
        job_dir=final_md_path.parent,
    )
    source_pdf_bytes = source_pdf_path.read_bytes() if source_pdf_path.exists() else None
    source_annotations = build_source_annotations_for_export(
        annotations=annotations,
        content_list=content_list,
    )

    report_pdf_bytes = build_review_report_pdf(
        workspace_title=job_title,
        source_pdf_name=source_pdf_name,
        run_id=job_id,
        status="completed",
        decision=None,
        estimated_cost=0,
        actual_cost=None,
        exported_at=datetime.now(UTC),
        meta_review={},
        reviewers=[],
        raw_output=None,
        final_report_markdown=final_report_markdown,
        source_pdf_bytes=source_pdf_bytes,
        source_annotations=source_annotations,
        review_display_id=None,
        owner_email=None,
        token_usage=token_usage,
        agent_model=agent_model,
    )
    report_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    report_pdf_path.write_bytes(report_pdf_bytes)

    export_stats = {
        "source_annotations_input_count": len(annotations),
        "source_annotations_exported_count": len(source_annotations),
        "content_list_count": len(content_list or []),
        "report_pdf_size_bytes": len(report_pdf_bytes),
    }
    append_event(job_id, "pdf_export_rendered", **export_stats)
    return export_stats


def _run_final_report_audit(
    *,
    job_id: str,
    final_md_path: Path,
    source_markdown: str,
) -> dict[str, Any]:
    settings = get_settings()
    if not bool(settings.enable_final_report_audit):
        payload = {
            "enabled": False,
            "applied": False,
            "iterations_run": 0,
            "max_iterations": int(settings.final_report_audit_max_iterations),
            "stop_reason": "disabled_by_config",
        }
        append_event(job_id, "final_report_audit_skipped", **payload)
        return payload

    if not final_md_path.exists():
        payload = {
            "enabled": False,
            "applied": False,
            "iterations_run": 0,
            "max_iterations": int(settings.final_report_audit_max_iterations),
            "stop_reason": "final_markdown_missing",
        }
        append_event(job_id, "final_report_audit_skipped", **payload)
        return payload

    current_markdown = final_md_path.read_text(encoding="utf-8")
    result = audit_and_refine_final_report(
        final_markdown=current_markdown,
        source_markdown=source_markdown,
        max_iterations=int(settings.final_report_audit_max_iterations),
        max_source_chars=int(settings.final_report_audit_max_source_chars),
        max_review_chars=int(settings.final_report_audit_max_review_chars),
        model=_resolved_agent_model(),
        min_english_words=int(settings.min_english_words_for_final),
        min_chinese_chars=int(settings.min_chinese_chars_for_final),
        force_english_output=bool(settings.force_english_output),
    )
    audit_payload = result.to_dict()

    if result.applied and str(result.final_markdown or "").strip():
        write_text_atomic(final_md_path, result.final_markdown)

    append_event(
        job_id,
        "final_report_audit_completed",
        enabled=bool(audit_payload.get("enabled")),
        applied=bool(audit_payload.get("applied")),
        iterations_run=int(audit_payload.get("iterations_run") or 0),
        max_iterations=int(audit_payload.get("max_iterations") or 0),
        stop_reason=str(audit_payload.get("stop_reason") or "").strip(),
        llm_provider=str(audit_payload.get("llm_provider") or "").strip(),
        llm_model=str(audit_payload.get("llm_model") or "").strip(),
    )
    return audit_payload


def _complete_with_existing_final_report(job_id: str, *, warning: str) -> bool:
    state = load_job_state(job_id)
    if state is None:
        return False

    artifacts = ensure_artifact_paths(job_id)
    final_md_path = Path(state.artifacts.final_markdown_path or artifacts["final_markdown"])
    if not final_md_path.exists():
        return False

    metadata = state.metadata if isinstance(state.metadata, dict) else {}
    has_persist_marker = bool(
        state.final_report_ready
        or str(state.artifacts.final_markdown_path or "").strip()
        or str(metadata.get("final_report_source") or "").strip()
    )
    if not has_persist_marker:
        append_event(
            job_id,
            "completed_recovery_skipped",
            warning=warning,
            reason="final_markdown_exists_without_persist_marker",
        )
        return False

    report_pdf_path = Path(state.artifacts.report_pdf_path or artifacts["report_pdf"])
    final_report_audit_path = Path(state.artifacts.final_report_audit_path or artifacts["final_report_audit"])
    pdf_error: str | None = None
    if not report_pdf_path.exists():
        try:
            source_pdf_path = Path(state.artifacts.source_pdf_path or artifacts["source_pdf"])
            annotations_path = Path(state.artifacts.annotations_path or artifacts["annotations"])
            content_list_path = Path(
                state.artifacts.mineru_content_list_path or artifacts["mineru_content_list"]
            )
            annotations = _load_annotations_payload(annotations_path)
            content_list = _load_content_list(content_list_path)
            _render_report_pdf(
                job_id=job_id,
                job_title=state.title,
                source_pdf_name=state.source_pdf_name,
                final_md_path=final_md_path,
                source_pdf_path=source_pdf_path,
                report_pdf_path=report_pdf_path,
                annotations=annotations,
                content_list=content_list,
                token_usage=_token_usage_payload_from_state(state),
                agent_model=_resolved_agent_model(),
            )
        except Exception as exc:
            pdf_error = f"{type(exc).__name__}: {exc}"
    job_latest_md, job_latest_pdf = _publish_outputs_to_output_dir(
        job_id=job_id,
        final_md_path=final_md_path,
        report_pdf_path=report_pdf_path,
    )

    def apply_completed(state_obj):
        state_obj.status = JobStatus.completed
        state_obj.final_report_ready = True
        state_obj.pdf_ready = report_pdf_path.exists()
        state_obj.artifacts.final_markdown_path = str(final_md_path)
        state_obj.artifacts.final_report_audit_path = (
            str(final_report_audit_path) if final_report_audit_path.exists() else None
        )
        state_obj.artifacts.report_pdf_path = str(report_pdf_path) if report_pdf_path.exists() else None
        state_obj.artifacts.latest_output_md_path = str(job_latest_md)
        state_obj.artifacts.latest_output_pdf_path = str(job_latest_pdf) if job_latest_pdf.exists() else None
        state_obj.error = pdf_error
        state_obj.message = (
            "Review pipeline completed via recovery after post-write exception."
            if pdf_error is None
            else "Final report persisted, but PDF export failed during recovery."
        )
        metadata = dict(state_obj.metadata)
        metadata["post_exception_recovery"] = True
        metadata["post_exception_warning"] = warning
        if pdf_error:
            metadata["pdf_export_recovery_error"] = pdf_error
        state_obj.metadata = metadata

    mutate_job_state(job_id, apply_completed)
    append_event(
        job_id,
        "completed_recovered",
        warning=warning,
        pdf_ready=report_pdf_path.exists(),
        pdf_error=pdf_error,
    )
    return True


async def run_job_async(job_id: str) -> None:
    settings = get_settings()
    job = load_job_state(job_id)
    if job is None:
        raise FileNotFoundError(f"Job not found: {job_id}")

    api_mode = (
        "codex_responses"
        if _uses_codex_subscription_backend()
        else ("responses" if settings.openai_use_responses_api else "chat_completions")
    )
    append_event(
        job_id,
        "llm_api_mode_selected",
        api_mode=api_mode,
        model=_resolved_agent_model(),
    )

    def apply_llm_mode(state):
        metadata = dict(state.metadata)
        metadata["llm_api_mode"] = api_mode
        state.metadata = metadata

    mutate_job_state(job_id, apply_llm_mode)

    artifacts = ensure_artifact_paths(job_id)
    source_pdf = Path(artifacts["source_pdf"])
    if not source_pdf.exists():
        raise RuntimeError(f"Source PDF missing: {source_pdf}")
    file_size = int(source_pdf.stat().st_size)
    if file_size <= 0:
        raise RuntimeError("Source PDF is empty.")
    if file_size > int(settings.max_pdf_bytes):
        raise RuntimeError(
            f"Source PDF too large: {file_size} bytes, max allowed {int(settings.max_pdf_bytes)} bytes."
        )

    set_status(job_id, JobStatus.pdf_uploading_to_mineru, "Submitting PDF to MinerU and uploading file...")
    set_status(job_id, JobStatus.pdf_parsing, "Polling MinerU parse result and assembling markdown...")

    mineru = _build_mineru_adapter()
    parse_result = await mineru.parse_pdf(pdf_path=source_pdf, data_id=job_id)
    mineru_image_map = _persist_mineru_image_files(
        job_dir=source_pdf.parent,
        image_files=parse_result.image_files,
    )
    if parse_result.content_list is not None and mineru_image_map:
        normalized_map = {k.replace("\\", "/"): v for k, v in mineru_image_map.items()}
        for row in parse_result.content_list:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("img_path") or "").strip().replace("\\", "/")
            if ref and ref in normalized_map:
                row["img_path"] = str(normalized_map[ref])

    write_text_atomic(Path(artifacts["mineru_markdown"]), parse_result.markdown)
    if parse_result.content_list is not None:
        write_json_atomic(Path(artifacts["mineru_content_list"]), {"content_list": parse_result.content_list})
    if parse_result.raw_result is not None:
        write_json_atomic(Path(artifacts["raw_result"]), parse_result.raw_result)

    def apply_parsed(state):
        state.artifacts.mineru_markdown_path = str(artifacts["mineru_markdown"])
        state.artifacts.mineru_content_list_path = (
            str(artifacts["mineru_content_list"]) if Path(artifacts["mineru_content_list"]).exists() else None
        )
        state.artifacts.annotations_path = str(artifacts["annotations"])
        state.metadata["markdown_provider"] = parse_result.provider
        state.metadata["mineru_batch_id"] = parse_result.batch_id
        state.metadata["parse_warning"] = parse_result.warning

    mutate_job_state(job_id, apply_parsed)
    if parse_result.warning:
        append_event(
            job_id, "markdown_parse_warning", warning=parse_result.warning, provider=parse_result.provider
        )

    page_index = build_page_index(parse_result.markdown, parse_result.content_list)

    set_status(job_id, JobStatus.agent_running, "Running review agent with tool loop...")

    paper_adapter = _build_paper_adapter()
    paper_search_runtime_state = (await paper_adapter.get_search_runtime_state()).to_dict()
    append_event(
        job_id,
        "paper_search_runtime_state_resolved",
        enabled=paper_search_runtime_state.get("enabled"),
        started=paper_search_runtime_state.get("started"),
        availability=paper_search_runtime_state.get("availability"),
        base_url=paper_search_runtime_state.get("base_url"),
        health_url=paper_search_runtime_state.get("health_url"),
        error=paper_search_runtime_state.get("error"),
    )

    def apply_paper_search_state(state):
        metadata = dict(state.metadata)
        metadata["paper_search_runtime_state"] = dict(paper_search_runtime_state)
        state.metadata = metadata

    mutate_job_state(job_id, apply_paper_search_state)

    # Objective retrieval context for section-2 niche positioning matrix.
    semantic_adapter = _build_semantic_scholar_adapter()
    title_hint = _extract_title_hint(parse_result.markdown, job.source_pdf_name)
    semantic_payload = await semantic_adapter.search_related(query=title_hint)
    semantic_context = _format_semantic_scholar_context(semantic_payload)
    write_json_atomic(
        Path(artifacts["source_pdf"]).parent / "semantic_scholar_candidates.json", semantic_payload
    )

    prompt = build_review_agent_system_prompt(
        source_file_id=job_id,
        source_file_name=job.source_pdf_name,
        ui_language=settings.ui_language,
        paper_markdown=parse_result.markdown,
        use_meta_review=False,
        paper_search_runtime_state=paper_search_runtime_state,
        semantic_scholar_context=semantic_context,
    )
    write_text_atomic(Path(artifacts["prompt_snapshot"]), prompt)

    def apply_prompt(state):
        state.artifacts.prompt_snapshot_path = str(artifacts["prompt_snapshot"])

    mutate_job_state(job_id, apply_prompt)

    runtime = ReviewRuntimeContext(
        job_id=job_id,
        job_dir=Path(artifacts["source_pdf"]).parent,
        page_index=page_index,
        source_markdown=parse_result.markdown,
        paper_adapter=paper_adapter,
        paper_search_runtime_state=paper_search_runtime_state,
        settings=settings,
    )

    tools = build_review_tools(runtime)
    agent_model = _build_agent_model()
    agent = Agent(
        name="FactReviewAgent",
        instructions=prompt,
        tools=tools,
        model=agent_model,
        model_settings=_build_agent_model_settings(),
    )

    requested_attempts = int(settings.agent_resume_attempts)
    max_attempts = max(1, min(2, requested_attempts))
    if requested_attempts != max_attempts:
        append_event(
            job_id,
            "agent_resume_attempts_capped",
            requested=requested_attempts,
            applied=max_attempts,
            reason="hard_cap_2",
        )
    run_config = _build_run_config()
    # Use the exact same full review prompt as user input (parity requirement).
    next_input: str | list[Any] = prompt
    usage_totals = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    def _consume_run_result(run_result: Any, *, output_tag: str) -> str:
        usage = run_result.context_wrapper.usage
        usage_totals["requests"] += int(getattr(usage, "requests", 0) or 0)
        usage_totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        usage_totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        usage_totals["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        usage_payload = SimpleNamespace(**usage_totals)
        _sync_token_usage(job_id, usage_payload)
        runtime.sync_state_usage(usage_payload)

        final_output_text = str(run_result.final_output or "").strip()
        if final_output_text:
            write_text_atomic(Path(runtime.job_dir / "agent_final_output.txt"), final_output_text)
            write_text_atomic(
                Path(runtime.job_dir / f"agent_final_output_{output_tag}.txt"),
                final_output_text,
            )
        return final_output_text

    for attempt in range(1, max_attempts + 1):
        if runtime.final_markdown_text:
            append_event(
                job_id,
                "agent_run_skipped_after_final_write",
                attempt=attempt,
                reason="final_report_already_persisted",
            )
            break

        run_task = asyncio.create_task(
            _run_agent_once(
                agent,
                input_payload=next_input,
                context=runtime,
                max_turns=max(20, settings.agent_max_turns),
                run_config=run_config,
            )
        )
        run_result = None
        while True:
            done, _ = await asyncio.wait({run_task}, timeout=0.5)
            if run_task in done:
                try:
                    run_result = run_task.result()
                except Exception as exc:
                    if runtime.final_markdown_text:
                        append_event(
                            job_id,
                            "agent_run_post_final_exception_ignored",
                            attempt=attempt,
                            error=f"{type(exc).__name__}: {exc}",
                            reason="final_report_already_persisted",
                        )
                        run_result = None
                        break
                    raise
                break
            if runtime.final_markdown_text:
                run_task.cancel()
                append_event(
                    job_id,
                    "agent_run_cancelled_after_final_write",
                    attempt=attempt,
                    reason="final_report_already_persisted",
                )
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    append_event(
                        job_id,
                        "agent_run_cancel_post_final_exception_ignored",
                        attempt=attempt,
                        error=f"{type(exc).__name__}: {exc}",
                        reason="final_report_already_persisted",
                    )
                break

        if run_result is None and runtime.final_markdown_text:
            append_event(
                job_id,
                "agent_run_terminated_after_final_write",
                attempt=attempt,
                reason="final_report_already_persisted",
            )
            break

        _consume_run_result(run_result, output_tag=f"attempt_{attempt}")

        if runtime.final_markdown_text:
            break

        append_event(
            job_id,
            "agent_run_incomplete",
            attempt=attempt,
            max_attempts=max_attempts,
            reason="no_final_report_persisted",
        )

        if attempt >= max_attempts:
            append_event(
                job_id,
                "agent_forced_final_write_start",
                attempt=attempt,
                reason="max_attempt_reached_without_final_write",
            )
            forced_input = [
                *run_result.to_input_list(),
                {
                    "role": "user",
                    "content": (
                        "MANDATORY ACTION NOW: Call review_final_markdown_write in section mode immediately. "
                        "Submit exactly one required section per call using "
                        "review_final_markdown_write(section_id=<required_section_id>, section_content=<section_markdown>). "
                        "After each call, inspect completed_sections/missing_sections/next_required_section and "
                        "submit the next required section right away until status=ok. "
                        "Do not output plain-text final report. If the tool returns retry_required/error, "
                        "follow message/next_steps and retry review_final_markdown_write."
                    ),
                },
            ]
            forced_choices = ["review_final_markdown_write", "required"]
            for forced_choice in forced_choices:
                if runtime.final_markdown_text:
                    append_event(
                        job_id,
                        "agent_forced_final_write_skipped_after_success",
                        attempt=attempt,
                        tool_choice=forced_choice,
                        reason="final_report_already_persisted",
                    )
                    break
                try:
                    forced_agent = Agent(
                        name="FactReviewAgentFinalWriteEnforcer",
                        instructions=prompt,
                        tools=tools,
                        model=agent_model,
                        model_settings=_build_agent_model_settings(tool_choice=forced_choice),
                    )
                    forced_result = await _run_agent_once(
                        forced_agent,
                        input_payload=forced_input,
                        context=runtime,
                        max_turns=12,
                        run_config=run_config,
                    )
                except Exception as exc:
                    if runtime.final_markdown_text:
                        append_event(
                            job_id,
                            "agent_forced_final_write_post_success_exception_ignored",
                            attempt=attempt,
                            tool_choice=forced_choice,
                            error=f"{type(exc).__name__}: {exc}",
                            reason="final_report_already_persisted",
                        )
                        break
                    append_event(
                        job_id,
                        "agent_forced_final_write_error",
                        attempt=attempt,
                        tool_choice=forced_choice,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                forced_output_text = _consume_run_result(
                    forced_result,
                    output_tag=f"attempt_{attempt}_forced_final_write",
                )
                append_event(
                    job_id,
                    "agent_forced_final_write_result",
                    attempt=attempt,
                    tool_choice=forced_choice,
                    final_output_chars=len(forced_output_text),
                    final_write_persisted=bool(runtime.final_markdown_text),
                )
                if runtime.final_markdown_text:
                    break
                forced_input = [
                    *forced_result.to_input_list(),
                    {
                        "role": "user",
                        "content": (
                            "The final report is still not persisted. Continue section-mode submission now: "
                            "call review_final_markdown_write with section_id + section_content for the next required section."
                        ),
                    },
                ]

            break

        set_status(
            job_id,
            JobStatus.agent_running,
            (
                "Agent ended without final report write. "
                f"Resuming review runtime (attempt {attempt + 1}/{max_attempts})..."
            ),
        )
        usage = runtime.paper_search_usage
        continuation_instruction = (
            "Resume the same review job from current state. "
            "Do not restart Phase 1 planning unless a hard gate is still unmet.\n"
            f"Current state: annotations={runtime.annotation_count}, "
            f"paper_search_total_calls={usage.total_calls}, "
            f"distinct_queries={usage.distinct_queries}, "
            f"effective_paper_search_calls={usage.effective_calls}.\n"
            "If gates are met, go directly to final report assembly in section mode and call "
            "review_final_markdown_write(section_id=<required_section_id>, section_content=<section_markdown>) "
            "as soon as possible.\n"
            "Mandatory: your next substantive action must be a section-mode tool call "
            "`review_final_markdown_write(...)`; plain chat markdown is invalid.\n"
            "If a gate is unmet or the write tool returns an error, follow message/next_steps exactly, "
            "perform minimal remediation, then retry review_final_markdown_write.\n"
            "Never end this run without a successful review_final_markdown_write."
        )
        next_input = [
            *run_result.to_input_list(),
            {
                "role": "user",
                "content": continuation_instruction,
            },
        ]

    if not runtime.final_markdown_text:
        raise RuntimeError(
            "Agent finished without successful review_final_markdown_write. "
            "Final report gate was not satisfied."
        )

    set_status(job_id, JobStatus.pdf_exporting, "Rendering final markdown report into PDF...")

    final_md_path = Path(artifacts["final_markdown"])
    final_report_audit_path = Path(artifacts["final_report_audit"])
    report_pdf_path = Path(artifacts["report_pdf"])
    if not final_md_path.exists():
        raise RuntimeError(f"Final markdown not found: {final_md_path}")

    audit_payload = _run_final_report_audit(
        job_id=job_id,
        final_md_path=final_md_path,
        source_markdown=parse_result.markdown,
    )
    write_json_atomic(final_report_audit_path, audit_payload)

    def apply_audit(state):
        state.artifacts.final_report_audit_path = str(final_report_audit_path)
        metadata = dict(state.metadata)
        metadata["final_report_audit"] = audit_payload
        state.metadata = metadata

    mutate_job_state(job_id, apply_audit)

    state_token_usage = _token_usage_payload_from_state(load_job_state(job_id))
    token_usage_for_pdf = {
        "requests": max(int(usage_totals.get("requests", 0)), int(state_token_usage.get("requests", 0))),
        "input_tokens": max(
            int(usage_totals.get("input_tokens", 0)), int(state_token_usage.get("input_tokens", 0))
        ),
        "output_tokens": max(
            int(usage_totals.get("output_tokens", 0)), int(state_token_usage.get("output_tokens", 0))
        ),
        "total_tokens": max(
            int(usage_totals.get("total_tokens", 0)), int(state_token_usage.get("total_tokens", 0))
        ),
    }
    if token_usage_for_pdf["total_tokens"] <= 0:
        token_usage_for_pdf["total_tokens"] = int(token_usage_for_pdf["input_tokens"]) + int(
            token_usage_for_pdf["output_tokens"]
        )

    _render_report_pdf(
        job_id=job_id,
        job_title=job.title,
        source_pdf_name=job.source_pdf_name,
        final_md_path=final_md_path,
        source_pdf_path=source_pdf,
        report_pdf_path=report_pdf_path,
        annotations=list(runtime.annotations),
        content_list=parse_result.content_list,
        token_usage=token_usage_for_pdf,
        agent_model=_resolved_agent_model(),
    )
    job_latest_md, job_latest_pdf = _publish_outputs_to_output_dir(
        job_id=job_id,
        final_md_path=final_md_path,
        report_pdf_path=report_pdf_path,
    )

    def apply_completed(state):
        state.status = JobStatus.completed
        state.message = "Review pipeline completed."
        state.error = None
        state.final_report_ready = True
        state.pdf_ready = report_pdf_path.exists()
        state.artifacts.final_markdown_path = str(final_md_path)
        state.artifacts.final_report_audit_path = (
            str(final_report_audit_path) if final_report_audit_path.exists() else None
        )
        state.artifacts.report_pdf_path = str(report_pdf_path)
        state.artifacts.latest_output_md_path = str(job_latest_md)
        state.artifacts.latest_output_pdf_path = str(job_latest_pdf) if job_latest_pdf.exists() else None
        metadata = dict(state.metadata)
        metadata["final_report_audit"] = audit_payload
        state.metadata = metadata

    mutate_job_state(job_id, apply_completed)
    append_event(job_id, "completed", report_pdf_path=str(report_pdf_path))


def run_job(job_id: str) -> None:
    try:
        asyncio.run(run_job_async(job_id))
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        stack = traceback.format_exc()
        append_event(job_id, "pipeline_exception", error=detail, stack=stack)
        if _complete_with_existing_final_report(job_id, warning=detail):
            return
        fail_job(
            job_id,
            message="Review pipeline failed.",
            error=detail,
        )
