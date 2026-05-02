"""Regex-based claim detection — the no-LLM fallback for §3.1b.

The extractor uses these heuristics when the LLM route is disabled or
unavailable. They over-generate by design: the decomposer and the §3.4
synthesiser are expected to drop or merge claims that end up without
supporting evidence.

The patterns are intentionally domain-generic — no CompGCN-specific
literals. Dataset / baseline / metric names are discovered from the
text itself, not hard-coded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.claim import Claim, ClaimLocation, ClaimType
from schemas.paper import Paper, Section

# ---------------------------------------------------------------------------
# Trigger phrases, mapped to a :class:`ClaimType`
# ---------------------------------------------------------------------------

_EMPIRICAL_TRIGGERS = [
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bsurpass(?:es|ed|ing)?\b",
    r"\bbest(?:\s+results?|\s+performance)\b",
    r"\bstate[-\s]of[-\s]the[-\s]art\b",
    r"\bSOTA\b",
    r"\bachieve(?:s|d|ing)?\b",
    r"\bimprove(?:s|d|ment|ments)?\s+(?:by|of|over)\b",
    r"\b(?:absolute|relative)\s+(?:improvement|gain|gains)\b",
    r"\b(?:higher|lower|better)\s+than\b",
    r"\bcompared?\s+(?:to|with|against)\b",
]

_METHOD_TRIGGERS = [
    r"\bwe\s+propose\b",
    r"\bwe\s+introduce\b",
    r"\bwe\s+present\b",
    r"\bwe\s+design\b",
    r"\bwe\s+develop\b",
    r"\bour\s+(?:method|approach|framework|model|architecture)\b",
    r"\bnovel\s+(?:method|approach|framework|model|architecture)\b",
    r"\bcontribution[s]?\s+of\s+(?:this|our)\b",
]

_THEORY_TRIGGERS = [
    r"\bwe\s+prove\b",
    r"\b(?:theorem|lemma|corollary|proposition)\b",
    r"\bgeneral(?:ize|izes|izing|ization)\b",
    r"\b(?:is|are)\s+equivalent\s+to\b",
    r"\b(?:reduces?|reduction)\s+to\b",
    r"\bspecial\s+case\s+of\b",
]

_REPRO_TRIGGERS = [
    r"\bsource\s+code\b",
    r"\bcode\s+(?:is|will\s+be|has\s+been)\s+(?:made\s+)?(?:publicly\s+)?available\b",
    r"\b(?:released|release)\s+(?:the\s+)?(?:code|implementation)\b",
    r"github\.com/[\w\-./]+",
    r"\breproducib(?:le|ility)\b",
]


# Compile at module import for speed; case-insensitive on free-text.
_TRIG_BY_TYPE: dict[ClaimType, list[re.Pattern[str]]] = {
    ClaimType.EMPIRICAL: [re.compile(p, re.IGNORECASE) for p in _EMPIRICAL_TRIGGERS],
    ClaimType.METHODOLOGICAL: [re.compile(p, re.IGNORECASE) for p in _METHOD_TRIGGERS],
    ClaimType.THEORETICAL: [re.compile(p, re.IGNORECASE) for p in _THEORY_TRIGGERS],
    ClaimType.REPRODUCIBILITY: [re.compile(p, re.IGNORECASE) for p in _REPRO_TRIGGERS],
}


# Numeric claim detector: "MRR of 0.355", "87.2% accuracy", "+2.3 Hits@10".
_METRIC_TOKENS = (
    r"MRR|MR|Hits?@\d+|Accuracy|Acc\.?|F1|BLEU|ROUGE|AUC|AUROC|AUPRC|"
    r"Precision|Recall|EM|Perplexity|PPL|Rouge-?\d+|mAP"
)
_NUMERIC_CLAIM_RE = re.compile(
    rf"""
    (?:                                      # number first, metric after
        (?P<v1>[+-]?\d+(?:\.\d+)?)\s*%?\s*
        (?:\w+\s+)?                          # optional adjective ("absolute")
        (?P<m1>{_METRIC_TOKENS})
    )
    |
    (?:                                      # metric first, number after
        (?P<m2>{_METRIC_TOKENS})\s*(?:of|=|:)?\s*
        (?P<v2>[+-]?\d+(?:\.\d+)?)\s*%?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Sentence-level scanning
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter — good enough for paper body text."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def _classify_sentence(sentence: str) -> ClaimType | None:
    """Return the most likely :class:`ClaimType` or ``None``."""
    has_numeric = bool(_NUMERIC_CLAIM_RE.search(sentence))

    empirical_hit = any(p.search(sentence) for p in _TRIG_BY_TYPE[ClaimType.EMPIRICAL])
    if empirical_hit or has_numeric:
        return ClaimType.EMPIRICAL

    for ctype in (ClaimType.THEORETICAL, ClaimType.METHODOLOGICAL, ClaimType.REPRODUCIBILITY):
        if any(p.search(sentence) for p in _TRIG_BY_TYPE[ctype]):
            return ctype
    return None


@dataclass(frozen=True)
class _Mention:
    sentence: str
    ctype: ClaimType
    section: Section
    char_offset_in_section: int


def _scan_section(section: Section) -> list[_Mention]:
    out: list[_Mention] = []
    text = section.text or ""
    # Walk sentences while tracking each one's offset in the section body so
    # we can round-trip to a :class:`ClaimLocation` later.
    cursor = 0
    for sentence in _split_sentences(text):
        idx = text.find(sentence, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(sentence)
        ctype = _classify_sentence(sentence)
        if ctype is None:
            continue
        out.append(_Mention(sentence=sentence, ctype=ctype, section=section, char_offset_in_section=idx))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _is_body_section(section: Section) -> bool:
    """Skip acknowledgement / references / appendix-only sections."""
    title = (section.title or "").strip().lower()
    if not title:
        return True
    skip = ("references", "acknowledg", "appendix-only")
    return not any(title.startswith(s) for s in skip)


def extract_claims_heuristic(paper: Paper, *, max_claims: int = 60) -> list[Claim]:
    """Regex-based claim extraction.

    Parameters
    ----------
    paper
        The structured paper produced by §3.1a ingestion.
    max_claims
        Upper bound on the number of claims returned. The scanner is
        deliberately over-generative; the cap prevents downstream stages
        from being swamped on pathologically long papers.
    """
    mentions: list[_Mention] = []
    for section in paper.sections:
        if not _is_body_section(section):
            continue
        mentions.extend(_scan_section(section))

    # Stable ordering: preserve document order by section char_start, then
    # offset within section.
    mentions.sort(key=lambda m: (m.section.char_start, m.char_offset_in_section))

    claims: list[Claim] = []
    for i, m in enumerate(mentions[:max_claims], start=1):
        sec_char_start = m.section.char_start or 0
        claims.append(
            Claim(
                id=f"claim_{i:02d}",
                text=m.sentence,
                type=m.ctype,
                scope=_infer_scope(m.sentence, m.ctype),
                datasets=_extract_datasets(m.sentence),
                baselines=[],  # filled in by the decomposer / LLM pass
                metrics=_extract_metrics(m.sentence),
                location=ClaimLocation(
                    section_id=m.section.id,
                    char_start=sec_char_start + m.char_offset_in_section,
                    char_end=sec_char_start + m.char_offset_in_section + len(m.sentence),
                ),
            )
        )
    return claims


# ---------------------------------------------------------------------------
# Helpers exposed for the decomposer / tests
# ---------------------------------------------------------------------------


def _infer_scope(sentence: str, ctype: ClaimType) -> str:
    """Mark claims that span multiple tasks/datasets as ``broad``."""
    # A claim is broad when it lists ≥2 datasets/tasks or uses an enumeration
    # like "across X, Y, and Z". Single-dataset numeric claims stay local —
    # the decomposer does not need to split them.
    del ctype  # reserved for future type-specific rules
    s = sentence.lower()
    if re.search(r"\bacross\b|\b(?:multiple|several|various)\b", s):
        return "broad"
    if len(_extract_datasets(sentence)) >= 2:
        return "broad"
    return "local"


# Known benchmarks — extend as new papers are added. The *detector* is
# generic (uppercase tokens + digits); this list only boosts recall on
# well-known corpora.
_KNOWN_BENCHMARKS = (
    "FB15k-237",
    "FB15k",
    "WN18RR",
    "WN18",
    "YAGO3-10",
    "NELL-995",
    "ImageNet",
    "CIFAR-10",
    "CIFAR-100",
    "MNIST",
    "MS-COCO",
    "COCO",
    "MUTAG",
    "PTC",
    "PROTEINS",
    "NCI1",
    "IMDB",
    "Cora",
    "Citeseer",
    "Pubmed",
    "GLUE",
    "SuperGLUE",
    "SQuAD",
    "WMT",
    "CoNLL",
)

_DATASET_RE = re.compile(r"\b(?:" + "|".join(re.escape(d) for d in _KNOWN_BENCHMARKS) + r")\b")
# Fallback: uppercase token with digits (e.g. "WN18", "FB15k-237") or
# hyphenated capitalised benchmark names.
_DATASET_FALLBACK_RE = re.compile(r"\b[A-Z][A-Za-z]*\d+[A-Za-z0-9\-]*\b")


def _extract_datasets(sentence: str) -> list[str]:
    seen: list[str] = []
    for m in _DATASET_RE.finditer(sentence):
        tok = m.group(0)
        if tok not in seen:
            seen.append(tok)
    if not seen:
        for m in _DATASET_FALLBACK_RE.finditer(sentence):
            tok = m.group(0)
            # Filter obvious non-datasets: single-digit tokens, equation labels.
            if len(tok) <= 2 or tok.lower() in {"gpu", "cpu", "id"}:
                continue
            if tok not in seen:
                seen.append(tok)
    return seen


_METRIC_TOKEN_RE = re.compile(_METRIC_TOKENS, re.IGNORECASE)


def _extract_metrics(sentence: str) -> list[str]:
    seen: list[str] = []
    for m in _METRIC_TOKEN_RE.finditer(sentence):
        tok = m.group(0)
        # Canonicalise common forms.
        canon = tok.upper()
        canon = canon.replace("HITS@", "Hits@")
        if canon not in seen:
            seen.append(canon)
    return seen
