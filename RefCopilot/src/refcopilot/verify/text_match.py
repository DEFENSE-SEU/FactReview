"""Title similarity, author overlap, OCR-garbled detection."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.fuzz import ratio

from refcopilot.verify.thresholds import (
    ET_AL_VARIANTS,
    LOWERCASE_HEAD_STOPWORDS,
    MAX_AUTHORS_TO_COMPARE,
)


# ---------------------------------------------------------------------------
# Strict citation-vs-record comparison
# ---------------------------------------------------------------------------


def titles_match(cited: str | None, retrieved: str | None) -> bool:
    """Exact title equality, ignoring only case, accents, punctuation, spacing
    and LaTeX styling commands."""
    a = _strict_title_key(cited)
    b = _strict_title_key(retrieved)
    return bool(a) and a == b


def authors_match(cited: list[str], retrieved: list[str]) -> bool:
    """True if the cited author list is the retrieved one, name by name, in order.

    If the citation truncates its list ("others" / "et al." / "etc."), the
    authors it does list must be the leading authors of the retrieved list.
    A citation listing no authors has nothing to contradict and matches.
    """
    truncated = False
    listed: list[str] = []
    for name in cited:
        if is_et_al(name):
            truncated = True
            break
        if name and name.strip():
            listed.append(name)

    if not listed:
        return True
    if truncated:
        if len(listed) > len(retrieved):
            return False
    elif len(listed) != len(retrieved):
        return False
    return all(_same_author(c, r) for c, r in zip(listed, retrieved, strict=False))


def is_et_al(name: str | None) -> bool:
    s = re.sub(r"\s+", " ", (name or "").strip().lower()).rstrip(".").strip()
    return s in ET_AL_VARIANTS


def _same_author(cited: str, retrieved: str) -> bool:
    c_given, c_last = _name_parts(cited)
    r_given, r_last = _name_parts(retrieved)
    if not c_last or not r_last:
        return False
    # A single-token name on either side (surname-only, or a team like
    # "DeepSeek-AI") must equal the other's surname or its whole name.
    if not c_given or not r_given:
        return c_last == r_last or "".join(c_given) + c_last == "".join(r_given) + r_last
    if c_last != r_last:
        return False
    a, b = c_given[0], r_given[0]
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    # Full given names on both sides: allow short forms (Sam / Samuel) only.
    return a.startswith(b) or b.startswith(a)


_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _name_parts(name: str) -> tuple[list[str], str]:
    """``(given_name_tokens, surname)``, folded to plain lowercase ASCII-ish."""
    s = unicodedata.normalize("NFKC", name or "").strip()
    if "," in s:
        last, _, first = s.partition(",")
        if last.strip() and first.strip():
            s = f"{first} {last}"
    tokens = [t.replace("-", "") for t in re.split(r"[^\w\-]+", _fold(s))]
    tokens = [t for t in tokens if t]
    while len(tokens) > 1 and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    if not tokens:
        return [], ""
    return tokens[:-1], tokens[-1]


# LaTeX commands that only style their argument. The argument is title text;
# the command name never is, but it is alphanumeric and would survive the
# isalnum filter (DBLP writes a superscript as ``S\({}^{\mbox{3}}\)``).
_LATEX_STYLE_COMMAND = re.compile(
    r"\\(?:mbox|hbox|ensuremath|emph"
    r"|text(?:rm|sf|tt|bf|it|sl|sc|up|normal)?"
    r"|math(?:rm|sf|tt|bf|it|normal|cal|bb|frak|scr)"
    r"|rm|sf|tt|bf|it|sl|sc|em)(?![a-zA-Z])"
)


def _strict_title_key(text: str | None) -> str:
    text = _LATEX_STYLE_COMMAND.sub("", text or "")
    return "".join(c for c in _fold(text) if c.isalnum())


# Letters NFKD does not decompose into a base letter + accent.
_FOLD_TABLE = str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ß": "ss", "æ": "ae", "œ": "oe", "ı": "i"})


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower().translate(_FOLD_TABLE)


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
# Identifier normalization
# ---------------------------------------------------------------------------


_DOI_URL_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def normalize_doi(doi: str | None) -> str | None:
    """Strip a doi.org URL / ``doi:`` prefix and lowercase, for comparison."""
    if not doi:
        return None
    s = doi.strip()
    for prefix in _DOI_URL_PREFIXES:
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
            break
    return s.strip().lower() or None


def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    """Lowercase and drop a trailing ``vN`` version suffix, for comparison."""
    if not arxiv_id:
        return None
    s = arxiv_id.strip().lower()
    if "v" in s:
        head, _, tail = s.rpartition("v")
        if tail.isdigit() and head:
            s = head
    return s or None


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
