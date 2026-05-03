"""Title similarity, author overlap, OCR-garbled detection."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.fuzz import ratio

from refcopilot.verify.thresholds import (
    LOWERCASE_HEAD_STOPWORDS,
    MAX_AUTHORS_TO_COMPARE,
)


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def title_similarity(a: str | None, b: str | None) -> float:
    """Token-level Jaccard with a character-level tiebreaker.

    Returns 1.0 when the titles share all content tokens; near 0 when they share
    none. The character-level component (rapidfuzz ratio) is used only to break
    ties between titles that share most tokens.
    """
    if not a or not b:
        return 0.0
    tokens_a = _content_tokens(a)
    tokens_b = _content_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0

    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    jaccard = len(inter) / len(union)

    # Length-aware bonus: if the shorter set is fully contained, treat as exact.
    smaller = min(len(tokens_a), len(tokens_b))
    if smaller and len(inter) == smaller:
        # All tokens of the shorter side are in the longer side.
        return max(jaccard, 0.85)

    # Character-level tiebreaker on a normalized form (gives a small boost when
    # token sets disagree only on stems / hyphenation).
    char_ratio = ratio(_normalize_for_match(a), _normalize_for_match(b)) / 100.0
    return max(jaccard, jaccard * 0.7 + char_ratio * 0.3) if jaccard > 0 else 0.0


def _content_tokens(text: str) -> set[str]:
    return {t for t in _normalize_for_match(text).split() if t and t not in _STOPWORDS and len(t) > 1}


def _normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Author overlap
# ---------------------------------------------------------------------------


def author_overlap(cited: list[str], retrieved: list[str]) -> float:
    """Overlap of normalized author names; tolerant of First-vs-Initial.

    Strategy:
      1. Drop "et al." sentinels.
      2. Cap both lists to ``MAX_AUTHORS_TO_COMPARE`` so a long author list
         doesn't dominate the score.
      3. For each cited author, find the first unused retrieved author whose
         (last_name, first_initial) is compatible.
      4. Return matches / max(len(cited), len(retrieved)).
    """
    if not cited or not retrieved:
        return 0.0

    c_parsed = [_parse_author(a) for a in cited[:MAX_AUTHORS_TO_COMPARE] if a]
    r_parsed = [_parse_author(a) for a in retrieved[:MAX_AUTHORS_TO_COMPARE] if a]
    c_parsed = [x for x in c_parsed if x[0] or x[1]]
    r_parsed = [x for x in r_parsed if x[0] or x[1]]
    if not c_parsed or not r_parsed:
        return 0.0

    matches = 0
    used: set[int] = set()
    for ca in c_parsed:
        for j, ra in enumerate(r_parsed):
            if j in used:
                continue
            if _author_match(ca, ra):
                matches += 1
                used.add(j)
                break

    denom = max(len(c_parsed), len(r_parsed))
    return matches / denom if denom else 0.0


def _author_match(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """a, b are (first_norm, last_norm) tuples."""
    a_first, a_last = a
    b_first, b_last = b
    a_full = (a_first + a_last).strip()
    b_full = (b_first + b_last).strip()

    # Team / corporate-style "single token" name (e.g. "deepseekai") matches if
    # it appears as a substring on the other side.
    if not a_first and a_last and len(a_last) >= 4 and a_last in b_full:
        return True
    if not b_first and b_last and len(b_last) >= 4 and b_last in a_full:
        return True

    # Both have a last name — last names must match.
    if not a_last or not b_last:
        if a_full == b_full and a_full:
            return True
        return False

    if a_last != b_last or len(a_last) < 2:
        return False

    # Last names match. If either side has no first-name info, accept.
    if not a_first or not b_first:
        return True

    # Both have first-name info. Initials must be compatible.
    return a_first[0] == b_first[0]


def _parse_author(name: str) -> tuple[str, str]:
    """Returns (first_norm, last_norm). Either may be empty."""
    s = unicodedata.normalize("NFKC", name or "").strip()
    if not s:
        return "", ""

    # "Last, First" → "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    s = re.sub(r"[^\w\s\-]+", " ", s, flags=re.UNICODE).lower()
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not tokens:
        return "", ""

    if len(tokens) == 1:
        return "", tokens[0].replace("-", "")

    last = tokens[-1].replace("-", "")
    first = "".join(tokens[:-1]).replace("-", "")
    return first, last


# ---------------------------------------------------------------------------
# Garbled / OCR-noise detection
# ---------------------------------------------------------------------------


def is_garbled_title(title: str | None, raw_text: str | None = None) -> bool:
    """True if the title looks like an OCR artifact rather than a fabrication.

    Heuristic:
      - An empty title is garbled.
      - If ``raw_text`` begins with ``#`` (no author field) AND the title has
        at least five words AND the first word is lowercase, length ≤4, and
        not a common short stopword, the title likely starts mid-word — a
        typical PDF extraction artifact.
    """
    if not title or not title.strip():
        return True

    words = title.strip().split()
    if not words:
        return True

    if (raw_text or "").lstrip().startswith("#") and len(words) >= 5:
        first = words[0].strip(".,;:!?")
        if first and first.islower() and len(first) <= 4 and first not in LOWERCASE_HEAD_STOPWORDS:
            return True

    return False
