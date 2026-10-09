from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote_plus

import httpx

from preprocessing.parse.markdown_parser import parse_pdf_locally
from util.arxiv_requests import ARXIV_REQUESTS
from util.cutoff_date import CutoffDate, filter_papers


@dataclass
class PaperSearchConfig:
    enabled: bool
    provider: str
    base_url: str | None
    api_key: str | None
    endpoint: str
    timeout_seconds: int
    health_endpoint: str
    health_timeout_seconds: int
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: str | None = None
    openalex_base_url: str = "https://api.openalex.org"
    openalex_api_key: str | None = None


@dataclass
class PaperReadConfig:
    base_url: str | None
    api_key: str | None
    endpoint: str
    timeout_seconds: int


@dataclass
class PaperSearchRuntimeState:
    enabled: bool
    started: bool
    availability: str
    provider: str = "remote"
    base_url: str | None = None
    health_url: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "started": bool(self.started),
            "availability": str(self.availability or "").strip(),
            "provider": str(self.provider or "").strip() or "remote",
            "base_url": str(self.base_url or "").strip() or None,
            "health_url": str(self.health_url or "").strip() or None,
            "error": str(self.error or "").strip() or None,
        }


class PaperSearchAdapter:
    def __init__(self, search_cfg: PaperSearchConfig, read_cfg: PaperReadConfig):
        self.search_cfg = search_cfg
        self.read_cfg = read_cfg
        self._search_state_cache: PaperSearchRuntimeState | None = None

    @property
    def search_configured(self) -> bool:
        return bool(self.search_cfg.enabled and self._search_provider() != "remote") or bool(
            self.search_cfg.enabled and self.search_cfg.base_url
        )

    @property
    def read_configured(self) -> bool:
        return bool(self.read_cfg.base_url)

    async def lookup_metadata(self, *, identifier: str) -> dict:
        """Resolve an arXiv identifier without downloading or reading its paper."""
        requested = str(identifier or "").strip().removeprefix("arXiv:").removeprefix("arxiv:")
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?", requested, re.I):
            return {"success": False, "provider": "arxiv", "error": "unsupported_metadata_identifier"}
        paper = await self._arxiv_fetch_single(requested)
        returned = str((paper or {}).get("arxiv_id") or "").strip()
        # An unversioned request may resolve to the latest version of that paper.
        # An explicit version must resolve to precisely the version requested.
        same_id = (
            returned.lower() == requested.lower()
            if re.search(r"v\d+$", requested, re.I)
            else re.sub(r"v\d+$", "", returned, flags=re.I).lower() == requested.lower()
        )
        if not paper or not same_id:
            return {"success": False, "provider": "arxiv", "error": "metadata_identifier_mismatch"}
        return {"success": True, "provider": "arxiv", "requested_id": identifier, "paper": paper}

    async def search(
        self,
        *,
        query: str | None = None,
        question_list: list[str] | None = None,
        cutoff_date: CutoffDate | None = None,
    ) -> dict:
        state = await self.get_search_runtime_state()
        if not state.started:
            payload = self._search_not_started_payload(
                state=state,
                query=query,
                question_list=question_list,
            )
            if cutoff_date is not None:
                payload["cutoff_date"] = cutoff_date.to_metadata()
            return payload
        try:
            provider = self._search_provider()
            if provider == "remote":
                result = await self._search_remote(query=query, question_list=question_list)
            elif provider == "arxiv":
                result = await self._search_arxiv_fallback(query=query, question_list=question_list)
            elif provider == "semantic_scholar":
                result = await self._search_semantic_scholar(query=query, question_list=question_list)
            elif provider == "openalex":
                result = await self._search_openalex(query=query, question_list=question_list)
            else:
                result = {
                    "success": False,
                    "error": "unsupported_provider",
                    "provider": provider,
                    "papers": [],
                    "count": 0,
                    "question_results": [],
                }
        except Exception as exc:
            self._search_state_cache = PaperSearchRuntimeState(
                enabled=bool(self.search_cfg.enabled),
                started=False,
                availability="became_unavailable_during_run",
                provider=self._search_provider(),
                base_url=self.search_cfg.base_url,
                health_url=self._search_health_url(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        return _apply_cutoff_to_search_result(result, cutoff_date)

    async def read_papers(self, *, items: list[dict]) -> dict:
        if self.read_configured:
            return await self._read_remote(items)
        return await self._read_arxiv_fallback(items)

    async def get_search_runtime_state(
        self,
        *,
        force_refresh: bool = False,
    ) -> PaperSearchRuntimeState:
        if self._search_state_cache is not None and not force_refresh:
            return self._search_state_cache

        provider = self._search_provider()
        base_url = str(self.search_cfg.base_url or "").strip() or None
        health_url = self._search_health_url()
        if not bool(self.search_cfg.enabled):
            state = PaperSearchRuntimeState(
                enabled=False,
                started=False,
                availability="disabled_by_config",
                provider=provider,
                base_url=base_url,
                health_url=health_url,
            )
            self._search_state_cache = state
            return state

        if provider in {"arxiv", "semantic_scholar", "openalex"}:
            state = PaperSearchRuntimeState(
                enabled=True,
                started=True,
                availability="ready",
                provider=provider,
                base_url=self._provider_base_url(provider),
                health_url=None,
            )
            self._search_state_cache = state
            return state

        if provider != "remote":
            state = PaperSearchRuntimeState(
                enabled=True,
                started=False,
                availability="unsupported_provider",
                provider=provider,
                base_url=base_url,
                health_url=health_url,
                error=f"unsupported paper_search provider: {provider}",
            )
            self._search_state_cache = state
            return state

        if not base_url:
            state = PaperSearchRuntimeState(
                enabled=True,
                started=False,
                availability="missing_base_url",
                provider=provider,
                base_url=None,
                health_url=health_url,
            )
            self._search_state_cache = state
            return state

        if not str(self.search_cfg.health_endpoint or "").strip():
            state = PaperSearchRuntimeState(
                enabled=True,
                started=True,
                availability="ready",
                provider=provider,
                base_url=base_url,
                health_url=None,
            )
            self._search_state_cache = state
            return state

        headers: dict[str, str] = {}
        api_key = str(self.search_cfg.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(
                timeout=max(1, int(self.search_cfg.health_timeout_seconds)),
            ) as client:
                response = await client.get(health_url, headers=headers)
            response.raise_for_status()

            payload = None
            try:
                payload = response.json()
            except Exception:
                payload = None

            if isinstance(payload, dict):
                status = str(payload.get("status") or "").strip().lower()
                if status and status not in {"healthy", "ok", "ready"}:
                    raise RuntimeError(
                        str(payload.get("error") or payload.get("message") or f"health status={status}")
                    )
                if "models_loaded" in payload and not bool(payload.get("models_loaded")):
                    raise RuntimeError(
                        str(payload.get("error") or payload.get("message") or "models_loaded=false")
                    )

            state = PaperSearchRuntimeState(
                enabled=True,
                started=True,
                availability="ready",
                provider=provider,
                base_url=base_url,
                health_url=health_url,
            )
        except Exception as exc:
            state = PaperSearchRuntimeState(
                enabled=True,
                started=False,
                availability="health_check_failed",
                provider=provider,
                base_url=base_url,
                health_url=health_url,
                error=f"{type(exc).__name__}: {exc}",
            )

        self._search_state_cache = state
        return state

    def _search_provider(self) -> str:
        provider = str(self.search_cfg.provider or "").strip().lower().replace("-", "_")
        return provider or "remote"

    def _provider_base_url(self, provider: str) -> str | None:
        if provider == "arxiv":
            return "https://export.arxiv.org/api"
        if provider == "semantic_scholar":
            return str(
                self.search_cfg.semantic_scholar_base_url or "https://api.semanticscholar.org/graph/v1"
            ).strip()
        if provider == "openalex":
            return str(self.search_cfg.openalex_base_url or "https://api.openalex.org").strip()
        return str(self.search_cfg.base_url or "").strip() or None

    def _search_health_url(self) -> str | None:
        base_url = str(self.search_cfg.base_url or "").strip()
        health_endpoint = str(self.search_cfg.health_endpoint or "").strip()
        if not base_url or not health_endpoint:
            return None
        return f"{base_url.rstrip('/')}/{health_endpoint.lstrip('/')}"

    def _search_not_started_payload(
        self,
        *,
        state: PaperSearchRuntimeState,
        query: str | None,
        question_list: list[str] | None,
    ) -> dict:
        questions = [q for q in (question_list or []) if str(q or "").strip()]
        query_text = str(query or "").strip()
        if query_text and query_text not in questions:
            questions = [query_text, *questions]

        return {
            "status": "not_started",
            "success": False,
            "reason": "paper_search_not_started",
            "message": "External paper search was not started in this run.",
            "query": query_text,
            "questions": questions,
            "papers": [],
            "count": 0,
            "question_results": [],
            "retry_required": False,
            "next_action": "enter_retrieval_disabled_mode",
            "next_steps": [
                "Proceed without external literature search in this run.",
                "Mark novelty/comparison conclusions as deferred manual verification.",
                "If external literature search is required, start the retrieval service and rerun the job.",
            ],
            "paper_search_state": state.to_dict(),
        }

    async def _search_remote(
        self,
        *,
        query: str | None,
        question_list: list[str] | None,
    ) -> dict:
        assert self.search_cfg.base_url is not None

        url = f"{self.search_cfg.base_url.rstrip('/')}/{self.search_cfg.endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.search_cfg.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"query": query, "question_list": question_list}

        async with httpx.AsyncClient(timeout=max(20, int(self.search_cfg.timeout_seconds))) as client:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            papers = [self._normalize_remote_paper_item(item) for item in data if isinstance(item, dict)]
            papers = [row for row in papers if row]
            questions = [q for q in (question_list or []) if str(q or "").strip()]
            query_text = str(query or "").strip()
            if query_text and query_text not in questions:
                questions = [query_text, *questions]
            return {
                "success": True,
                "provider": "remote_list_adapted",
                "query": query_text,
                "questions": questions,
                "papers": papers,
                "count": len(papers),
                "question_results": [
                    {
                        "question": q,
                        "success": bool(papers),
                        "count": len(papers),
                        "papers": papers,
                    }
                    for q in (questions or ([query_text] if query_text else []))
                ],
            }
        return {
            "success": False,
            "error": "invalid_remote_payload",
            "papers": [],
            "count": 0,
        }

    async def _search_semantic_scholar(
        self,
        *,
        query: str | None,
        question_list: list[str] | None,
    ) -> dict:
        questions = [q for q in (question_list or []) if str(q or "").strip()]
        query_text = str(query or "").strip()
        if query_text and query_text not in questions:
            questions = [query_text, *questions]
        if not questions:
            return _empty_search_result(provider="semantic_scholar", error="empty_query")

        base_url = self._provider_base_url("semantic_scholar") or "https://api.semanticscholar.org/graph/v1"
        url = f"{base_url.rstrip('/')}/paper/search"
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "url",
                "year",
                "authors",
                "externalIds",
                "openAccessPdf",
                "citationCount",
                "venue",
                "publicationDate",
            ]
        )
        headers: dict[str, str] = {}
        api_key = str(self.search_cfg.semantic_scholar_api_key or "").strip()
        if api_key:
            headers["x-api-key"] = api_key

        all_papers: list[dict] = []
        seen: set[str] = set()
        question_results: list[dict] = []
        async with httpx.AsyncClient(timeout=max(20, int(self.search_cfg.timeout_seconds))) as client:
            for q in questions:
                response = await client.get(
                    url,
                    headers=headers,
                    params={"query": q, "limit": 8, "fields": fields},
                )
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("data") if isinstance(payload, dict) else []
                papers = [
                    self._normalize_semantic_scholar_item(item)
                    for item in (rows or [])
                    if isinstance(item, dict)
                ]
                papers = [paper for paper in papers if paper.get("title")]
                question_results.append(
                    {
                        "question": q,
                        "success": bool(papers),
                        "count": len(papers),
                        "papers": papers,
                    }
                )
                for paper in papers:
                    key = str(
                        paper.get("arxiv_id")
                        or paper.get("semantic_scholar_id")
                        or paper.get("url")
                        or paper.get("title")
                        or ""
                    )
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    all_papers.append(paper)

        return {
            "success": True,
            "query": questions[0],
            "questions": questions,
            "papers": all_papers,
            "count": len(all_papers),
            "question_results": question_results,
            "provider": "semantic_scholar",
        }

    async def _search_openalex(
        self,
        *,
        query: str | None,
        question_list: list[str] | None,
    ) -> dict:
        questions = [q for q in (question_list or []) if str(q or "").strip()]
        query_text = str(query or "").strip()
        if query_text and query_text not in questions:
            questions = [query_text, *questions]
        if not questions:
            return _empty_search_result(provider="openalex", error="empty_query")

        base_url = self._provider_base_url("openalex") or "https://api.openalex.org"
        url = f"{base_url.rstrip('/')}/works"
        api_key = str(self.search_cfg.openalex_api_key or "").strip()

        all_papers: list[dict] = []
        seen: set[str] = set()
        question_results: list[dict] = []
        async with httpx.AsyncClient(timeout=max(20, int(self.search_cfg.timeout_seconds))) as client:
            for q in questions:
                params = {
                    "search": q,
                    "per-page": 8,
                }
                if api_key:
                    params["api_key"] = api_key
                response = await client.get(
                    url,
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("results") if isinstance(payload, dict) else []
                papers = [
                    self._normalize_openalex_item(item) for item in (rows or []) if isinstance(item, dict)
                ]
                papers = [paper for paper in papers if paper.get("title")]
                question_results.append(
                    {
                        "question": q,
                        "success": bool(papers),
                        "count": len(papers),
                        "papers": papers,
                    }
                )
                for paper in papers:
                    key = str(
                        paper.get("arxiv_id")
                        or paper.get("openalex_id")
                        or paper.get("doi")
                        or paper.get("url")
                        or paper.get("title")
                        or ""
                    )
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    all_papers.append(paper)

        return {
            "success": True,
            "query": questions[0],
            "questions": questions,
            "papers": all_papers,
            "count": len(all_papers),
            "question_results": question_results,
            "provider": "openalex",
        }

    async def _read_remote(self, items: list[dict]) -> dict:
        assert self.read_cfg.base_url is not None

        url = f"{self.read_cfg.base_url.rstrip('/')}/{self.read_cfg.endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.read_cfg.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=max(20, int(self.read_cfg.timeout_seconds))) as client:
            response = await client.post(url, headers=headers, json={"items": items})
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict):
            return data
        return {
            "success": False,
            "error": "invalid_remote_payload",
            "items": [],
        }

    async def _search_arxiv_fallback(
        self,
        *,
        query: str | None,
        question_list: list[str] | None,
    ) -> dict:
        questions = [q for q in (question_list or []) if str(q or "").strip()]
        if not questions and query:
            questions = [query]
        if not questions:
            return {
                "success": False,
                "error": "empty_query",
                "papers": [],
                "count": 0,
                "question_results": [],
                "provider": "arxiv_fallback",
            }

        all_papers: list[dict] = []
        seen: set[str] = set()
        question_results: list[dict] = []

        for q in questions:
            papers = await self._arxiv_query(q, max_results=8)
            question_results.append(
                {
                    "question": q,
                    "success": bool(papers),
                    "count": len(papers),
                    "papers": papers,
                }
            )
            for paper in papers:
                key = str(paper.get("arxiv_id") or paper.get("url") or "")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                all_papers.append(paper)

        return {
            "success": True,
            "query": questions[0],
            "questions": questions,
            "papers": all_papers,
            "count": len(all_papers),
            "question_results": question_results,
            "provider": "arxiv_fallback",
        }

    async def _read_arxiv_fallback(self, items: list[dict]) -> dict:
        normalized = [item for item in items if isinstance(item, dict)]
        if not normalized:
            return {
                "success": False,
                "error": "empty_items",
                "items": [],
                "provider": "arxiv_fallback",
            }

        outputs: list[dict] = []
        for item in normalized[:8]:
            arxiv_id = str(item.get("id") or item.get("arxiv_id") or "").strip()
            question = str(item.get("question") or "").strip()
            title_hint = str(item.get("title") or "").strip()

            if not arxiv_id and title_hint:
                guessed = await self._arxiv_query(title_hint, max_results=1)
                if guessed:
                    arxiv_id = str(guessed[0].get("arxiv_id") or "").strip()

            if not arxiv_id:
                outputs.append(
                    {
                        "id": "",
                        "question": question,
                        "success": False,
                        "error": "missing_arxiv_id",
                    }
                )
                continue

            detail = await self._arxiv_fetch_single(arxiv_id)
            if not detail:
                outputs.append(
                    {
                        "id": arxiv_id,
                        "question": question,
                        "success": False,
                        "error": "paper_not_found",
                    }
                )
                continue

            full_text_payload = await self._try_read_arxiv_full_text(detail=detail, question=question)
            if full_text_payload:
                outputs.append(full_text_payload)
                continue

            answer = self._build_abstract_read_answer(detail=detail, question=question)
            outputs.append(
                {
                    "id": arxiv_id,
                    "question": question,
                    "success": True,
                    "paper": detail,
                    "answer": answer,
                    "reader_provider": "arxiv_abstract_fallback",
                }
            )

        return {
            "success": True,
            "items": outputs,
            "count": len(outputs),
            "provider": "arxiv_fallback",
        }

    async def _try_read_arxiv_full_text(self, *, detail: dict, question: str) -> dict | None:
        pdf_url = str(detail.get("pdf_url") or "").strip()
        arxiv_id = str(detail.get("arxiv_id") or detail.get("id") or "").strip()
        if not pdf_url:
            return None

        try:
            pdf_bytes = await self._download_pdf(pdf_url)
            parsed = parse_pdf_locally(pdf_bytes)
        except Exception:
            return None

        pages = [page.strip() for page in parsed.pages if str(page or "").strip()]
        if not pages:
            return None

        evidence = self._select_relevant_passages(pages=pages, question=question, max_items=5)
        if not evidence:
            return None

        answer = self._build_full_text_read_answer(
            detail=detail,
            question=question,
            evidence=evidence,
        )
        return {
            "id": arxiv_id,
            "question": question,
            "success": True,
            "paper": detail,
            "answer": answer,
            "evidence": evidence,
            "reader_provider": "arxiv_full_text_fallback",
        }

    async def _download_pdf(self, url: str) -> bytes:
        async with ARXIV_REQUESTS.slot(), httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(url, headers=self._arxiv_headers())
            ARXIV_REQUESTS.observe_retry_after(response.status_code, response.headers.get("Retry-After"))
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise RuntimeError("downloaded content is not a PDF")
        return content

    def _build_abstract_read_answer(self, *, detail: dict, question: str) -> str:
        title = str(detail.get("title") or "").strip()
        abstract = str(detail.get("abstract") or "").strip()
        if not abstract:
            abstract = "No abstract available."

        if not question:
            return f"Title: {title}\n\nAbstract:\n{abstract}"

        return (
            f"Question: {question}\n\n"
            f"From paper '{title}', available evidence (abstract-level) is:\n{abstract}\n\n"
            "Note: This fallback reader uses arXiv metadata/abstract, not full-text deep parsing."
        )

    def _build_full_text_read_answer(
        self,
        *,
        detail: dict,
        question: str,
        evidence: list[dict],
    ) -> str:
        title = str(detail.get("title") or "").strip()
        header = f"Question: {question}\n\n" if question else ""
        lines = [
            f"{header}From paper '{title}', available full-text evidence from arXiv PDF is:",
        ]
        for item in evidence:
            page = int(item.get("page") or 0)
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- Page {page}: {text}")
        lines.append(
            "\nNote: This fallback reader downloaded the arXiv PDF and used local text extraction; "
            "evidence quality depends on PDF text extractability."
        )
        return "\n".join(lines).strip()

    def _select_relevant_passages(
        self,
        *,
        pages: list[str],
        question: str,
        max_items: int,
    ) -> list[dict]:
        query_tokens = set(_normalize_text_tokens(question))
        scored: list[tuple[float, int, str]] = []

        for page_no, page_text in enumerate(pages, start=1):
            for passage in _split_passages(page_text):
                tokens = set(_normalize_text_tokens(passage))
                if not tokens:
                    continue
                overlap = len(query_tokens & tokens) if query_tokens else 0
                # Keep a little signal from informative text even when the
                # question is broad or empty, but prefer direct token overlap.
                score = float(overlap) + min(1.0, len(tokens) / 80.0)
                if score <= 0:
                    continue
                scored.append((score, page_no, passage))

        if not scored:
            return []

        ranked = sorted(scored, key=lambda row: (-row[0], row[1], len(row[2])))
        evidence: list[dict] = []
        seen: set[str] = set()
        for score, page_no, passage in ranked:
            key = " ".join(passage.lower().split())
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "page": page_no,
                    "text": _truncate_text(passage, 900),
                    "score": round(float(score), 3),
                }
            )
            if len(evidence) >= max(1, int(max_items or 1)):
                break
        return evidence

    async def _arxiv_query(self, question: str, *, max_results: int) -> list[dict]:
        tokens = self._question_to_arxiv_query(question)
        query = quote_plus(tokens)
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{query}&start=0&max_results={max(1, min(16, max_results))}"
        )

        async with ARXIV_REQUESTS.slot(), httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url, headers=self._arxiv_headers())
            ARXIV_REQUESTS.observe_retry_after(response.status_code, response.headers.get("Retry-After"))
        response.raise_for_status()

        return self._parse_arxiv_feed(response.text)

    async def _arxiv_fetch_single(self, arxiv_id: str) -> dict | None:
        clean = arxiv_id.strip()
        if not clean:
            return None

        query = quote_plus(f"id:{clean}")
        url = f"https://export.arxiv.org/api/query?search_query={query}&start=0&max_results=1"

        async with ARXIV_REQUESTS.slot(), httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url, headers=self._arxiv_headers())
            ARXIV_REQUESTS.observe_retry_after(response.status_code, response.headers.get("Retry-After"))
        response.raise_for_status()

        papers = self._parse_arxiv_feed(response.text)
        return papers[0] if papers else None

    def _arxiv_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "FactReview/0.1 (https://github.com/DEFENSE-SEU/FactReview; arxiv paper search)",
        }

    def _question_to_arxiv_query(self, question: str) -> str:
        text = re.sub(r"\s+", " ", str(question or "").strip().lower())
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        tokens = [tok for tok in text.split(" ") if tok]
        stop = {
            "what",
            "which",
            "how",
            "are",
            "is",
            "the",
            "for",
            "of",
            "to",
            "in",
            "and",
            "on",
            "with",
            "recent",
            "papers",
            "methods",
            "paper",
            "about",
            "does",
            "can",
            "be",
            "used",
            "that",
        }
        kept = [tok for tok in tokens if tok not in stop]
        return " ".join(kept[:10]) or text

    def _normalize_remote_paper_item(self, item: dict) -> dict:
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("abstract") or "").strip()
        link = str(item.get("link") or item.get("url") or "").strip()
        raw_id = str(item.get("id") or item.get("arxiv_id") or "").strip()

        # Common PASA list response uses "link" as arXiv identifier.
        arxiv_id = raw_id
        if not arxiv_id and link and "http" not in link:
            arxiv_id = link
        if arxiv_id.startswith("arXiv:"):
            arxiv_id = arxiv_id.split(":", 1)[1].strip()

        abs_url = ""
        pdf_url = ""
        if arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif link.startswith("http://") or link.startswith("https://"):
            abs_url = link

        return {
            "id": arxiv_id or link,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": snippet,
            "url": abs_url or link,
            "abs_url": abs_url or link,
            "pdf_url": pdf_url,
            "source": "remote",
        }

    def _normalize_semantic_scholar_item(self, item: dict) -> dict:
        external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        arxiv_id = str(external.get("ArXiv") or external.get("arXiv") or "").strip()
        if arxiv_id.startswith("arXiv:"):
            arxiv_id = arxiv_id.split(":", 1)[1].strip()
        open_pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
        pdf_url = str(open_pdf.get("url") or "").strip()
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        authors_raw = item.get("authors") if isinstance(item.get("authors"), list) else []
        authors = [
            str(author.get("name") or "").strip()
            for author in authors_raw
            if isinstance(author, dict) and str(author.get("name") or "").strip()
        ]
        paper_id = str(item.get("paperId") or "").strip()
        url = str(item.get("url") or "").strip()
        return {
            "id": arxiv_id or paper_id or url,
            "arxiv_id": arxiv_id,
            "semantic_scholar_id": paper_id,
            "title": str(item.get("title") or "").strip(),
            "abstract": str(item.get("abstract") or "").strip(),
            "authors": authors,
            "year": item.get("year"),
            "published": str(item.get("publicationDate") or "").strip(),
            "venue": str(item.get("venue") or "").strip(),
            "citation_count": int(item.get("citationCount") or 0),
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
        }

    def _normalize_openalex_item(self, item: dict) -> dict:
        ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
        primary_location = (
            item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        )
        best_oa_location = (
            item.get("best_oa_location") if isinstance(item.get("best_oa_location"), dict) else {}
        )
        openalex_id = str(item.get("id") or ids.get("openalex") or "").strip()
        doi = str(item.get("doi") or ids.get("doi") or "").strip()
        url = str(ids.get("openalex") or openalex_id or doi or "").strip()
        pdf_url = str(best_oa_location.get("pdf_url") or primary_location.get("pdf_url") or "").strip()
        landing_url = str(
            best_oa_location.get("landing_page_url") or primary_location.get("landing_page_url") or ""
        ).strip()
        arxiv_id = _extract_arxiv_id_from_text(" ".join([url, doi, pdf_url, landing_url]))
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
        authors: list[str] = []
        for row in authorships:
            if not isinstance(row, dict):
                continue
            author = row.get("author") if isinstance(row.get("author"), dict) else {}
            name = str(author.get("display_name") or "").strip()
            if name:
                authors.append(name)

        return {
            "id": arxiv_id or openalex_id or doi or url,
            "arxiv_id": arxiv_id,
            "openalex_id": openalex_id,
            "doi": doi,
            "title": str(item.get("display_name") or item.get("title") or "").strip(),
            "abstract": _openalex_abstract_text(item.get("abstract_inverted_index")),
            "authors": authors,
            "year": item.get("publication_year"),
            "published": str(item.get("publication_date") or "").strip(),
            "venue": _openalex_venue(item),
            "citation_count": int(item.get("cited_by_count") or 0),
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (landing_url or url),
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (landing_url or url),
            "pdf_url": pdf_url,
            "source": "openalex",
        }

    def _parse_arxiv_feed(self, xml_text: str) -> list[dict]:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[dict] = []

        for entry in root.findall("atom:entry", ns):
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            published = entry.findtext("atom:published", default="", namespaces=ns).strip()
            updated = entry.findtext("atom:updated", default="", namespaces=ns).strip()

            authors: list[str] = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", default="", namespaces=ns).strip()
                if name:
                    authors.append(name)

            # Legacy IDs include a subject archive (e.g. hep-th/9901001).
            # Preserve it so lookup_metadata can verify the requested identity.
            arxiv_id = entry_id.split("/abs/", 1)[-1] if "/abs/" in entry_id else ""
            abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

            papers.append(
                {
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "published": published,
                    "updated": updated,
                    "arxiv_id": arxiv_id,
                    "url": abs_url,
                    "abs_url": abs_url,
                    "pdf_url": pdf_url,
                    "source": "arxiv",
                }
            )

        return papers


def _apply_cutoff_to_search_result(result: dict, cutoff: CutoffDate | None) -> dict:
    """Filter ``papers`` and ``question_results`` by ``cutoff`` (client-side).

    The remote paper-search service has no documented year filter, so the
    cutoff is enforced here as a final safety net before the result reaches
    the agent. Returns the same dict (mutated) for convenience.
    """
    if not isinstance(result, dict):
        return result
    if cutoff is None:
        return result

    papers_raw = result.get("papers") if isinstance(result.get("papers"), list) else []
    kept, dropped = filter_papers(papers_raw, cutoff)
    result["papers"] = kept
    result["count"] = len(kept)
    result["filtered_out_count"] = len(dropped)
    result["cutoff_date"] = cutoff.to_metadata()

    grouped = result.get("question_results")
    if isinstance(grouped, list):
        rebuilt: list[dict] = []
        for row in grouped:
            if not isinstance(row, dict):
                continue
            sub_papers = row.get("papers") if isinstance(row.get("papers"), list) else []
            sub_kept, sub_dropped = filter_papers(sub_papers, cutoff)
            rebuilt.append(
                {
                    **row,
                    "papers": sub_kept,
                    "count": len(sub_kept),
                    "filtered_out_count": len(sub_dropped),
                }
            )
        result["question_results"] = rebuilt
    return result


def _empty_search_result(*, provider: str, error: str) -> dict:
    return {
        "success": False,
        "error": error,
        "papers": [],
        "count": 0,
        "question_results": [],
        "provider": provider,
    }


def _openalex_abstract_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, raw_indexes in value.items():
        if not isinstance(raw_indexes, list):
            continue
        for raw_idx in raw_indexes:
            try:
                positions.append((int(raw_idx), str(token)))
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    return " ".join(token for _, token in sorted(positions, key=lambda row: row[0]))


def _openalex_venue(item: dict) -> str:
    primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    return str(source.get("display_name") or item.get("host_venue") or "").strip()


def _extract_arxiv_id_from_text(text: str) -> str:
    match = re.search(
        r"(?i)(?:arxiv(?:\.org)?/(?:abs|pdf)/|arxiv:)([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z\-]+/[0-9]{7}(?:v[0-9]+)?)",
        str(text or ""),
    )
    if not match:
        return ""
    token = match.group(1).strip()
    if token.lower().endswith(".pdf"):
        token = token[:-4]
    return token


_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}


def _normalize_text_tokens(text: str) -> list[str]:
    raw = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return [token for token in raw.split() if len(token) >= 3 and token not in _TEXT_STOPWORDS]


def _split_passages(page_text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", str(page_text or "")).strip()
    if not text:
        return []

    raw_parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(raw_parts) <= 1:
        raw_parts = [part.strip() for part in text.splitlines() if part.strip()]

    passages: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for part in raw_parts:
        words = part.split()
        if not words:
            continue
        if buffer and buffer_words + len(words) > 180:
            passages.append(" ".join(buffer).strip())
            buffer = []
            buffer_words = 0
        buffer.append(part)
        buffer_words += len(words)
        if buffer_words >= 80:
            passages.append(" ".join(buffer).strip())
            buffer = []
            buffer_words = 0
    if buffer:
        passages.append(" ".join(buffer).strip())

    return [_truncate_text(passage, 1200) for passage in passages if len(passage.split()) >= 6]


def _truncate_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def normalize_question_list(raw: object) -> list[str]:
    raw_items: list[str] = []
    if isinstance(raw, list):
        raw_items.extend(str(item).strip() for item in raw if str(item).strip())

    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw_items.extend(str(item).strip() for item in parsed if str(item).strip())
            else:
                raw_items.extend(line.strip("-• \t") for line in text.splitlines() if line.strip("-• \t"))

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = " ".join(item.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned[:3]
