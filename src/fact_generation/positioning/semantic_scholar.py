from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SemanticScholarConfig:
    enabled: bool
    base_url: str
    api_key: str | None
    timeout_seconds: int
    top_k: int


class SemanticScholarAdapter:
    def __init__(self, cfg: SemanticScholarConfig):
        self.cfg = cfg

    async def search_related(self, *, query: str, top_k: int | None = None) -> dict[str, Any]:
        q = str(query or "").strip()
        if not self.cfg.enabled or not q:
            return {
                "enabled": bool(self.cfg.enabled),
                "query": q,
                "success": False,
                "papers": [],
                "message": "semantic_scholar_disabled_or_empty_query",
            }

        k = max(5, min(10, int(top_k or self.cfg.top_k or 8)))
        fetch_limit = max(k * 3, 20)
        url = f"{self.cfg.base_url.rstrip('/')}/paper/search"
        params = {
            "query": q,
            "limit": str(fetch_limit),
            "fields": "title,year,citationCount,venue,url,authors,externalIds",
        }
        headers: dict[str, str] = {}
        api_key = str(self.cfg.api_key or "").strip()
        if api_key:
            headers["x-api-key"] = api_key

        try:
            async with httpx.AsyncClient(timeout=max(10, int(self.cfg.timeout_seconds))) as client:
                resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            return {
                "enabled": True,
                "query": q,
                "success": False,
                "papers": [],
                "message": f"{type(exc).__name__}: {exc}",
            }

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            data = []

        norm_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            if _is_self_paper_title(query=q, candidate=title):
                continue
            citation_count = int(row.get("citationCount") or 0)
            year = int(row.get("year") or 0) if str(row.get("year") or "").strip() else None
            venue = str(row.get("venue") or "").strip()
            url_item = str(row.get("url") or "").strip()
            authors = row.get("authors") if isinstance(row.get("authors"), list) else []
            first_author = ""
            if authors and isinstance(authors[0], dict):
                first_author = str(authors[0].get("name") or "").strip()
            # Hybrid ranking: keep relevance bias but prioritize citation impact.
            rel_bonus = max(0.0, (fetch_limit - idx) / fetch_limit)
            score = rel_bonus + math.log10(max(1, citation_count))
            norm_rows.append(
                {
                    "title": title,
                    "year": year,
                    "citationCount": citation_count,
                    "venue": venue,
                    "url": url_item,
                    "firstAuthor": first_author,
                    "score": score,
                }
            )

        dedup: dict[str, dict[str, Any]] = {}
        for row in norm_rows:
            key = str(row.get("title") or "").strip().lower()
            prev = dedup.get(key)
            if prev is None or float(row.get("score") or 0.0) > float(prev.get("score") or 0.0):
                dedup[key] = row

        ranked = sorted(dedup.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)[:k]
        papers: list[dict[str, Any]] = []
        for i, row in enumerate(ranked, start=1):
            papers.append(
                {
                    "id": f"R{i}",
                    "title": str(row.get("title") or "").strip(),
                    "year": row.get("year"),
                    "citationCount": int(row.get("citationCount") or 0),
                    "venue": str(row.get("venue") or "").strip(),
                    "url": str(row.get("url") or "").strip(),
                    "firstAuthor": str(row.get("firstAuthor") or "").strip(),
                }
            )

        return {
            "enabled": True,
            "query": q,
            "success": True,
            "papers": papers,
            "count": len(papers),
            "message": None,
        }


_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _normalize_title_tokens(title: str) -> list[str]:
    raw = str(title or "").strip().lower()
    if not raw:
        return []
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = [tok for tok in raw.split() if tok and tok not in _TITLE_STOPWORDS]
    return tokens


def _is_self_paper_title(*, query: str, candidate: str) -> bool:
    q_tokens = _normalize_title_tokens(query)
    c_tokens = _normalize_title_tokens(candidate)
    if not q_tokens or not c_tokens:
        return False

    q_norm = " ".join(q_tokens)
    c_norm = " ".join(c_tokens)
    if q_norm == c_norm:
        return True

    # High-confidence near-duplicate title variants.
    if len(q_tokens) >= 4 and (q_norm in c_norm or c_norm in q_norm):
        return True

    q_set = set(q_tokens)
    c_set = set(c_tokens)
    inter = len(q_set & c_set)
    union = len(q_set | c_set)
    if union == 0:
        return False
    jaccard = inter / union
    return bool(inter >= 4 and jaccard >= 0.8)
