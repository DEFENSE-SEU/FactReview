"""Limit publication warnings to locally identity-bound metadata observations."""

import re
import unicodedata

from screening.reference_corrections import _same_identity, bound_record


def _normal(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _fragment(text):
    return bool(re.fullmatch(r"(?:volume|vol|issue|number|no|part|supplement)\s+\w+", _normal(text)))


def workshop_publication_metadata(row, block, index, result_path, records_path, correction):
    """Return a neutral metadata observation or an explicit unavailable reason.

    Public Report has no workshop-to-full-version relationship field. Even a
    complete same-identity publication record therefore cannot prove promotion.
    """
    try:
        checked, _ = bound_record(row, block, index, result_path, records_path)
        if correction is None or correction.state != "metadata_candidate":
            raise ValueError("Complete same-work/version metadata provenance is unavailable")
        merged = checked["merged"]
        venue = merged.get("venue") or ""
        if (
            _fragment(venue)
            or re.fullmatch(
                r"(?:(?:international|annual|the|proceedings|of|on)\s+)*"
                r"(?:conference|journal|transactions|symposium|annals|review|letters)(?:\s+(?:of|on))?",
                _normal(venue),
            )
            or re.search(r"\b(?:workshops?|arxiv|preprint)\b", venue, re.I)
            or not re.search(
                r"\b(?:conference|proceedings|journal|transactions|symposium|annals|review|letters)\b",
                venue,
                re.I,
            )
        ):
            raise ValueError("Retrieved venue does not establish a complete publication container")
        # A merger may prefer a short publication_venue over a contradicting
        # full journal field. Keep every same-identity source visible to this gate.
        sources = [s for s in merged.get("sources", []) if _same_identity(s, correction.identity_identifier)]
        if not sources:
            raise ValueError("Publication container lacks a same-work/version source")
        for source in sources:
            for field in ("publication_venue", "venue", "journal"):
                value = source.get(field)
                if not value or _fragment(value):
                    continue
                if re.search(r"\b(?:workshops?|arxiv|preprint)\b", value, re.I):
                    raise ValueError("Same-identity publication fields still describe a workshop or preprint")
                if _normal(value) != _normal(venue):
                    raise ValueError("Same-identity publication container fields disagree")
        return (
            f"Identity-bound retrieved metadata lists publication venue {venue!r}. "
            "This is a metadata candidate; workshop promotion and a need to replace the original citation remain unconfirmed.",
            None,
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
