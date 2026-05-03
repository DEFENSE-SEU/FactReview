"""Threshold constants used by the verification heuristics."""

from __future__ import annotations

# Author-overlap fraction below which a citation is treated as a fake.
AUTHOR_FAKE_THRESHOLD = 0.10

# Title-similarity thresholds: at or above SIMILARITY → real; below FAKE → fake.
TITLE_SIMILARITY_THRESHOLD = 0.75
TITLE_FAKE_THRESHOLD = 0.25

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
