"""Threshold constants used by the verification heuristics."""

from __future__ import annotations

# Whether a citation matches a candidate is NOT threshold-based: it requires an
# exact title and an exact ordered author list (text_match.titles_match /
# authors_match). The thresholds below only govern recall and clustering.

# "Same real-world paper" clustering thresholds (verify/matching.py). These
# decide whether two already-retrieved CANDIDATE RECORDS (clean metadata,
# compared to each other) represent the same paper when they don't share a
# doi/arxiv_id — a different question from SEARCH_RESULT_MIN_TITLE_SIM
# (backend recall gate below). Starting point only; expect to tune
# empirically once we see real clustering behavior (some papers' versions
# genuinely differ in title more than this allows).
SAME_PAPER_TITLE_SIM_THRESHOLD = 0.75
SAME_PAPER_AUTHOR_OVERLAP_MIN = 0.5

# Backends rank title searches by relevance, so unrelated papers that share
# a few topic words (or even none, when authors prompt-engineer the cited
# title) can rank near the top — Semantic Scholar's relevance fallback and
# OpenReview's /notes/search both exhibit this. Drop candidates whose title
# shares too few content tokens with the query before they reach the merger.
# Tuned to keep typo / casing variants (Math-arena → MathArena ≈ 0.80) while
# dropping topical-overlap-only noise (workshop title vs unrelated paper
# that shares one topic word, ≤ 0.36).
SEARCH_RESULT_MIN_TITLE_SIM = 0.40

# Cap on author-list comparison so a long author list does not dominate scoring.
MAX_AUTHORS_TO_COMPARE = 10

# Stop-words for the lowercase-short-word "garbled" heuristic.
LOWERCASE_HEAD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "toward",
        "towards",
        "using",
        "via",
        "with",
    }
)

# Venues that should be treated as arXiv aliases (not "real" published venues).
ARXIV_VENUE_ALIASES = frozenset({"arxiv", "arxiv.org", "preprint", "corr", "arxiv preprint"})

# Truncated-author sentinels, lowercased with any trailing "." removed.
ET_AL_VARIANTS = frozenset({"et al", "et. al", "and et al", "others", "and others", "etc"})
