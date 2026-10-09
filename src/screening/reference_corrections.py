"""Local, fail-closed binding of public reference records to metadata suggestions."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

from fact_generation.refcheck.refcheck import record_digest, reference_records
from schemas.reference import ReferenceCorrection, ReferenceFieldSource


def bibliographic_ids(text: str) -> set[str]:
    text = unquote(text)
    identifiers = set()
    for match in re.finditer(r"\b10\.\d{4,9}/[^\s<>\"?#]+", text, re.IGNORECASE):
        doi = match.group().rstrip(".,;:")
        for closing, opening in ((")", "("), ("]", "["), ("}", "{")):
            while doi.endswith(closing) and doi.count(closing) > doi.count(opening):
                doi = doi[:-1].rstrip(".,;:")
        identifiers.add("doi:" + doi.casefold())
    for match in re.finditer(
        r"(?:arxiv\s*:\s*|\b(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v[1-9]\d*)?"
        r"|[a-z][a-z.-]*/\d{7}(?:v[1-9]\d*)?)(?:\.pdf)?(?=$|[\s.,;:)\]}>])",
        text,
        re.IGNORECASE,
    ):
        identifiers.add("arxiv:" + match[1].casefold())
    return identifiers


def url_identifiers(value):
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return set()
        path = unquote(parsed.path)
        if parsed.hostname in {"doi.org", "dx.doi.org"} and re.match(r"^/10\.\d{4,9}/", path):
            return {item for item in bibliographic_ids(path) if item.startswith("doi:")}
        if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and path.startswith(
            ("/abs/", "/pdf/")
        ):
            return {item for item in bibliographic_ids(path) if item.startswith("arxiv:")}
    except ValueError:
        pass
    return set()


def bound_record(row, block, index, result_path, records_path):
    """Reload and regenerate the same public export/sidecar before using a row."""
    from refcopilot import Report

    if records_path is None:
        raise ValueError("Legacy checker output has no full Report metadata/provenance")
    payload = json.loads(Path(result_path).read_text("utf-8"))
    records = json.loads(Path(records_path).read_text("utf-8"))
    original_input = (Path(result_path).parent / "bibliography.txt").read_text("utf-8-sig")
    expected = reference_records(
        Report.model_validate(records["report"]), payload, hashlib.sha256(original_input.encode()).hexdigest()
    )
    if records != expected:
        raise ValueError("Reference Report/export/input provenance changed or is inconsistent")
    if type(index) is not int or index < 0 or index >= len(records["bindings"]):
        raise ValueError("Reference issue index is outside its exported records")
    binding = records["bindings"][index]
    if payload["issues"][index] != row or binding["exported_issue"] != row:
        raise ValueError("Reference issue does not match its original export index")
    checked = records["report"]["checked"][binding["reference_index"]]
    if checked["reference"]["raw"] != block.text:
        raise ValueError("Reference raw text does not exactly bind the located bibliography entry")
    return checked, binding


def _record_ids(record):
    ids = url_identifiers(record.get("url") or "")
    if record.get("doi"):
        ids |= bibliographic_ids(str(record["doi"]))
    if record.get("arxiv_id"):
        explicit = bibliographic_ids("arxiv:" + str(record["arxiv_id"]))
        # An unversioned eprint is an alias of an explicitly versioned source URL.
        # A versions list/latest-version number alone does not bind the source text.
        for item in explicit:
            if not any(re.sub(r"v[1-9]\d*$", "", known) == item for known in ids):
                ids.add(item)
    return ids


def _same_identity(record, identifier):
    ids = _record_ids(record)
    namespace = identifier.split(":", 1)[0] + ":"
    return {x for x in ids if x.startswith(namespace)} == {identifier}


def _fields(merged, identifier):
    result = []
    sources = merged.get("sources") or []
    for field in ("title", "authors", "year", "venue", "doi", "arxiv_id", "url"):
        value = merged.get(field)
        if value in (None, "", []):
            continue
        backend = merged["provenance"].get(field)
        if not backend and field != "url":
            raise ValueError(f"No recorded backend provenance for {field}")
        matches = []
        for index, source in enumerate(sources):
            if backend and source["backend"] != backend:
                continue
            if not _same_identity(source, identifier):
                continue
            keys = ("publication_venue", "venue", "journal") if field == "venue" else (field,)
            # The public merger chooses the first nonempty venue representation.
            source_field = next((key for key in keys if source.get(key) not in (None, "", [])), None)
            if source_field and source[source_field] == value:
                matches.append((index, source, source_field))
        if len(matches) != 1:
            raise ValueError(f"Field {field} lacks one uniquely identity-bound source record")
        index, source, source_field = matches[0]
        result.append(
            ReferenceFieldSource(
                field=field,
                value=value,
                source_field=source_field,
                source_record_index=index,
                backend=source["backend"],
                record_id=source["record_id"],
                url=source["url"],
                record_sha256=record_digest(source),
            )
        )
    return result


def _check_bibtex(text, merged):
    import bibtexparser

    if hasattr(bibtexparser, "parse_string"):
        parsed = bibtexparser.parse_string(text)
        if parsed.failed_blocks or parsed.strings or parsed.preambles:
            raise ValueError("Corrected BibTeX has malformed or executable/indirect blocks")
        entries = []
        for entry in parsed.entries:
            fields = {field.key.lower(): field.value for field in entry.fields}
            if len(fields) != len(entry.fields):
                raise ValueError("Corrected BibTeX has duplicate fields")
            entries.append({"ENTRYTYPE": entry.entry_type, "ID": entry.key, **fields})
    else:
        entries = bibtexparser.loads(text).entries
    if len(entries) != 1:
        raise ValueError("Corrected BibTeX must contain exactly one complete entry")
    fields = entries[0]
    allowed = {
        "ENTRYTYPE",
        "ID",
        "title",
        "author",
        "year",
        "journal",
        "booktitle",
        "doi",
        "eprint",
        "archiveprefix",
        "url",
    }
    if set(fields) - allowed:
        raise ValueError("Corrected BibTeX contains fields absent from the public formatter")
    expected = {
        "title": " ".join(merged["title"].split()),
        "author": " and ".join(" ".join(a.split()) for a in merged["authors"]),
        "year": str(merged["year"]),
    }
    for key, source_key in (("doi", "doi"), ("eprint", "arxiv_id"), ("url", "url")):
        if merged.get(source_key):
            expected[key] = merged[source_key]
    if merged.get("venue"):
        expected["booktitle" if "booktitle" in fields else "journal"] = " ".join(merged["venue"].split())
    if merged.get("arxiv_id"):
        expected["archiveprefix"] = "arXiv"
    if {k: v for k, v in fields.items() if k not in {"ID", "ENTRYTYPE"}} != expected:
        raise ValueError("Corrected BibTeX fields disagree with the bound merged metadata")


def build_correction(row, block, index, result_path, records_path):
    base = dict(
        raw_reference=block.text,
        raw_result_pointer=f"{Path(result_path).resolve()}#issues.{index}",
        verified_url=str(row.get("verified_url") or ""),
    )
    try:
        checked, binding = bound_record(row, block, index, result_path, records_path)
        code = str(row.get("type") or "").casefold()
        merged = checked.get("merged")
        if row.get("severity") != "warning" or any(
            x in code for x in ("retract", "unverified", "hallucination")
        ):
            raise ValueError("This reference issue cannot receive a general metadata replacement")
        if "closest match:" in str(row.get("details") or "").casefold():
            raise ValueError("Closest-match search candidates do not establish citation identity")
        if not merged or merged.get("is_retracted"):
            raise ValueError("No non-retracted merged record is available")
        original = bibliographic_ids(block.text)
        if len(original) != 1:
            raise ValueError("Original citation must contain one unambiguous DOI or explicit arXiv version")
        identifier = next(iter(original))
        if identifier.startswith("arxiv:") and not re.search(r"v[1-9]\d*$", identifier):
            raise ValueError("Original arXiv citation has no explicit version")
        if not _same_identity(merged, identifier):
            raise ValueError("Merged metadata does not have the original work/version identity")
        verified_ids = url_identifiers(base["verified_url"])
        if verified_ids and identifier not in verified_ids:
            raise ValueError("Verified URL conflicts with the original citation identity")
        if not all(merged.get(field) for field in ("title", "authors", "year")):
            raise ValueError("Insufficient metadata for a complete citation entry")
        fields = _fields(merged, identifier)
        full = binding["corrected_bibtex"]
        if not full:
            raise ValueError("No complete corrected BibTeX is available")
        _check_bibtex(full, merged)
        return ReferenceCorrection(
            **base,
            state="metadata_candidate",
            corrected_bibtex=full,
            identity_identifier=identifier,
            records_pointer=f"{Path(records_path).resolve()}#report.checked.{binding['reference_index']}",
            records_sha256=hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
            fields=fields,
            reason="Same-work/version retrieved metadata candidate; original-PDF error confirmation and metadata truth are not established by this suggestion.",
        )
    except Exception as exc:
        return ReferenceCorrection(**base, state="unavailable", reason=f"{type(exc).__name__}: {exc}")


def checked_correction(correction: ReferenceCorrection) -> ReferenceCorrection:
    """Rebuild saved suggestions from their original local records before rendering."""
    try:
        saved = ReferenceCorrection.model_validate(correction.model_dump(mode="json"))
        if saved.state == "unavailable":
            return saved.model_copy(deep=True)
        raw_match = re.fullmatch(r"(.+)#issues\.(\d+)", saved.raw_result_pointer)
        records_match = re.fullmatch(r"(.+)#report\.checked\.(\d+)", saved.records_pointer or "")
        if raw_match is None or records_match is None:
            raise ValueError("Reference correction has an invalid audit pointer")
        raw_path, records_path = Path(raw_match[1]), Path(records_match[1])
        if raw_path.parent.resolve() != records_path.parent.resolve():
            raise ValueError("Reference export and full Report must share the original run directory")
        payload = json.loads(raw_path.read_text("utf-8"))
        index = int(raw_match[2])
        rebuilt = build_correction(
            payload["issues"][index], SimpleNamespace(text=saved.raw_reference), index, raw_path, records_path
        )
        if rebuilt.model_dump(mode="json") != saved.model_dump(mode="json"):
            raise ValueError(
                "Stored reference correction differs from its original identity/provenance records"
            )
        return rebuilt
    except Exception as exc:
        return correction.model_copy(
            deep=True,
            update={
                "state": "unavailable",
                "corrected_bibtex": "",
                "fields": [],
                "reason": f"Stored reference correction unavailable: {type(exc).__name__}: {exc}",
            },
        )


def printed_publication_venue(row, block, checked=None):
    """Locate an already printed venue; no new lookup or title-only identity grant."""
    venues = []
    if checked:
        venues.append(checked["reference"].get("venue"))
    match = re.search(r"published at venue ['\"]([^'\"]+)['\"]", str(row.get("details") or ""), re.I)
    if match:
        venues.append(match[1])
    title = str(row.get("reference_title") or "")
    # A venue word in a paper title or author list cannot establish publication.
    tail = block.text.split(title, 1)[1] if title and block.text.count(title) == 1 else ""

    def normal(value):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    for venue in venues:
        if venue and not re.search(r"\b(?:arxiv|preprint)\b", venue, re.I) and normal(venue) in normal(tail):
            return venue
    return None
