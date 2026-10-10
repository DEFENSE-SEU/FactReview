"""Observed paging facts; no scientific search-adequacy decision is made here."""

import hashlib
import xml.etree.ElementTree as ET


class SearchRows(list):
    """Keep the historical arXiv list API while exposing the original page facts."""

    def __init__(self, rows, wire, *, wire_bytes=None):
        super().__init__(rows)
        self.metadata = ET.fromstring(wire)
        self.response_sha256 = hashlib.sha256(wire.encode("utf-8") if wire_bytes is None else wire_bytes).hexdigest()


def _number(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid_paging_metadata")
    return value


def paging_facts(provider, payload, *, offset, cursor, limit, returned):
    """Return only observed/derived page facts and a private continuation value."""
    if _number(returned) > _number(limit) or limit == 0:
        raise ValueError("invalid_paging_metadata")
    total, next_value, exhausted = None, None, None
    if provider == "arxiv":
        namespace = "{http://a9.com/-/spec/opensearch/1.1/}"
        values = [payload.findtext(namespace + key) for key in ("totalResults", "startIndex", "itemsPerPage")] if payload is not None else []
        if values and all(value is not None for value in values):
            if not all(value.strip().isdigit() for value in values):
                raise ValueError("invalid_paging_metadata")
            total, start, size = map(int, values)
            if start != offset or size != limit or offset + returned > total:
                raise ValueError("invalid_paging_metadata")
            exhausted = offset + returned == total
            next_value = offset + returned if returned and not exhausted else None
    elif provider == "semantic_scholar":
        if "total" in payload and "offset" in payload:
            total, start = _number(payload["total"]), _number(payload["offset"])
            if start != offset or offset + returned > total:
                raise ValueError("invalid_paging_metadata")
            exhausted = offset + returned == total
            observed_next = payload.get("next")
            if observed_next is not None:
                observed_next = _number(observed_next)
                if exhausted or observed_next != offset + returned or observed_next <= offset:
                    raise ValueError("invalid_paging_metadata")
            next_value = observed_next if observed_next is not None else (offset + returned if returned and not exhausted else None)
    elif provider == "openalex":
        meta = payload.get("meta")
        if isinstance(meta, dict):
            total = _number(meta["count"]) if "count" in meta else None
            if total is not None and offset + returned > total:
                raise ValueError("invalid_paging_metadata")
            if "next_cursor" in meta:
                next_value = meta["next_cursor"]
                if next_value is not None and (not isinstance(next_value, str) or not next_value or next_value == cursor):
                    raise ValueError("invalid_paging_metadata")
                if total is not None and next_value is None and offset + returned != total:
                    raise ValueError("invalid_paging_metadata")
                exhausted = offset + returned == total if total is not None else None
                if exhausted:
                    next_value = None
    record = {
        "provider_total": total, "exhausted": exhausted,
        "next_offset": next_value if type(next_value) is int else None,
        "next_cursor_sha256": hashlib.sha256(next_value.encode()).hexdigest() if isinstance(next_value, str) else None,
    }
    return record, next_value
