"""Title-to-BibTeX lookup via Semantic Scholar Graph API.

Library-level API used by the positioning stage to resolve paper titles to
BibTeX records without shelling out to an external CLI.

Usage::

    from fact_generation.positioning.bibtex import lookup_bibtex

    result = lookup_bibtex("Attention Is All You Need")
    # result -> {"matched_title": "...", "bibtex": "@article{...}", "exact": True}

    results = lookup_bibtex_batch(["Title A", "Title B"])

Environment:
    SEMANTIC_SCHOLAR_API_KEY  (preferred) or S2_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-\u2010\u2011\u2012\u2013\u2014\u2015_:;,.!?()\[\]{}\"'`~]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(s: str) -> list[str]:
    s = _norm_title(s)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def title_similarity(a: str, b: str) -> float:
    """Return similarity in [0,1].  Mix of char-level and token Jaccard."""
    a_n = _norm_title(a)
    b_n = _norm_title(b)
    if not a_n or not b_n:
        return 0.0
    char = SequenceMatcher(None, a_n, b_n).ratio()
    ta = set(_tokenize(a_n))
    tb = set(_tokenize(b_n))
    jac = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
    return 0.65 * char + 0.35 * jac


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get_json(url: str, headers: dict[str, str], timeout_s: int = 20, retries: int = 4) -> dict:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (2**i))
    if last_err:
        raise last_err
    raise RuntimeError("request failed")


def _make_headers(api_key: str) -> dict[str, str]:
    return {
        "accept": "application/json",
        "user-agent": "factreview-bibtex/1.0",
        "x-api-key": api_key,
    }


def _resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY") or ""
    if not key:
        raise OSError(
            "Semantic Scholar API key not found. Set SEMANTIC_SCHOLAR_API_KEY or pass api_key= explicitly."
        )
    return key


# ---------------------------------------------------------------------------
# Core lookup
# ---------------------------------------------------------------------------


def _fetch_bibtex_by_paper_id(paper_id: str, headers: dict[str, str]) -> str:
    fields = urllib.parse.quote("citationStyles")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    try:
        paper = _http_get_json(url, headers=headers)
    except Exception:
        return ""
    return (((paper.get("citationStyles") or {}).get("bibtex")) or "").strip()


def lookup_bibtex(title: str, *, api_key: str | None = None) -> dict[str, str]:
    """Look up a single title and return its BibTeX from Semantic Scholar.

    Returns a dict with keys:
        matched_title  – the title that was matched (may differ if fuzzy)
        bibtex         – the BibTeX string, or "" if not found
        exact          – True if the match was exact (normalised equality)
    """
    title = (title or "").strip()
    if not title:
        return {"matched_title": "", "bibtex": "", "exact": False}

    key = _resolve_api_key(api_key)
    headers = _make_headers(key)

    q = urllib.parse.quote(title)
    fields = urllib.parse.quote("title,paperId")
    search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=25&fields={fields}"

    try:
        search = _http_get_json(search_url, headers=headers)
    except Exception:
        return {"matched_title": "", "bibtex": "", "exact": False}

    candidates = search.get("data") or []
    if not isinstance(candidates, list) or not candidates:
        return {"matched_title": "", "bibtex": "", "exact": False}

    target_norm = _norm_title(title)

    # 1) Exact normalised match
    for c in candidates:
        t = (c.get("title") or "").strip()
        pid = c.get("paperId")
        if pid and t and _norm_title(t) == target_norm:
            bib = _fetch_bibtex_by_paper_id(pid, headers=headers)
            if bib:
                return {"matched_title": t, "bibtex": bib, "exact": True}
            return {"matched_title": "", "bibtex": "", "exact": False}

    # 2) Fuzzy fallback
    best: tuple[str, str] | None = None
    best_score = -1.0
    for c in candidates:
        t = (c.get("title") or "").strip()
        pid = c.get("paperId")
        if not (pid and t):
            continue
        s = title_similarity(title, t)
        if s > best_score:
            best_score = s
            best = (t, pid)

    if not best:
        return {"matched_title": "", "bibtex": "", "exact": False}

    matched_title, pid = best
    bib = _fetch_bibtex_by_paper_id(pid, headers=headers)
    if not bib:
        return {"matched_title": "", "bibtex": "", "exact": False}

    return {"matched_title": matched_title, "bibtex": bib, "exact": False}


def lookup_bibtex_batch(titles: list[str], *, api_key: str | None = None) -> list[dict[str, str]]:
    """Look up multiple titles.  Returns a list aligned with *titles*."""
    return [lookup_bibtex(t, api_key=api_key) for t in titles]


# ---------------------------------------------------------------------------
# CLI entry-point  (python -m src.tools.bibtex ...)
# ---------------------------------------------------------------------------


def _cli_main(argv: list[str]) -> int:
    mode = "single"
    file_path = None
    args = argv[1:]
    if args[:1] == ["--stdin"]:
        mode = "stdin"
        args = args[1:]
    elif args[:1] == ["--file"] and len(args) >= 2:
        mode = "file"
        file_path = args[1]
        args = args[2:]

    try:
        api_key = _resolve_api_key()
    except OSError:
        print("ERROR: SEMANTIC_SCHOLAR_API_KEY not set", file=sys.stderr)
        return 2

    titles: list[str] = []
    if mode == "stdin":
        titles = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    elif mode == "file":
        try:
            with open(file_path, encoding="utf-8") as f:
                titles = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        except Exception:
            return 0
    else:
        if len(argv) < 2:
            return 0
        title_in = " ".join(argv[1:]).strip()
        if not title_in:
            return 0
        titles = [title_in]

    emit_not_found = mode != "single"
    first = True
    for t in titles:
        r = lookup_bibtex(t, api_key=api_key)
        bib = r["bibtex"]
        matched = r["matched_title"]
        if not bib:
            if emit_not_found:
                if not first:
                    sys.stdout.write("\n")
                sys.stdout.write(f"% NOT_FOUND: {t}\n")
                first = False
            continue
        if not first:
            sys.stdout.write("\n")
        if matched and _norm_title(matched) != _norm_title(t):
            sys.stdout.write(f"% MATCHED_TITLE: {matched}\n")
        sys.stdout.write(bib)
        if not bib.endswith("\n"):
            sys.stdout.write("\n")
        first = False

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv))
