"""Centralized threshold constants (referenced from refchecker tuned defaults)."""

from __future__ import annotations

# Author overlap thresholds
AUTHOR_MATCH_THRESHOLD = 0.6
AUTHOR_VERIFIED_THRESHOLD = 0.4
AUTHOR_FAKE_THRESHOLD = 0.10

# Title similarity thresholds
TITLE_SIMILARITY_THRESHOLD = 0.75
TITLE_FAKE_THRESHOLD = 0.25

# Author list comparison
MAX_AUTHORS_TO_COMPARE = 10

# Year tolerance (off-by-one ignored)
YEAR_TOLERANCE = 1

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
