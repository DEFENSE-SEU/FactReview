"""Threshold constants used by the verification heuristics."""

from __future__ import annotations

# Author-overlap fraction below which a citation is treated as a fake.
AUTHOR_FAKE_THRESHOLD = 0.10

# Title-similarity thresholds: at or above SIMILARITY → real; below FAKE → fake.
TITLE_SIMILARITY_THRESHOLD = 0.75
TITLE_FAKE_THRESHOLD = 0.25

# Lower bound for the "real paper, but cited title differs from canonical"
# warning. Set below TITLE_SIMILARITY_THRESHOLD so it covers the whole
# matched-but-typo range; the check additionally requires author overlap.
TITLE_MISMATCH_MIN_SIM = 0.50

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

# Truncated-author signals
ET_AL_VARIANTS = ("et al.", "et al", "and others")
