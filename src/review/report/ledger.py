"""Lossless layered ledger view: operational facts inline, exact raw JSON locators."""
from __future__ import annotations

import hashlib
import json
import re


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _fence(value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    longest = max((len(s) for s in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + "json", text, fence, ""]


_BINDING = re.compile(
    r"^/ledger/\d+/(?:plan/target_bindings/[^/]+|attempts/\d+/request/plan/target_bindings/[^/]+"
    r"|alignment/\d+/paper_target_binding|paper_target_validation/\d+/bindings/[^/]+)$"
)


class LedgerRendering:
    """Render without modifying records or reading any external artifact."""

    def __init__(self, navigation, ledger):
        self.nav = navigation
        self.ledger = ledger
        self.records = []

    def __call__(self, entry, index):
        path = f"/ledger/{index}"
        bindings, first, records = [], {}, []

        def reference(value, pointer, kind, destination):
            row = {"json_pointer": pointer, "sha256": _digest(value),
                   "canonical_artifact": "final_review.json", "kind": kind,
                   "raw_value_printed": False, "display_location": "raw JSON locator",
                   "first_expanded_pointer": destination}
            records.append(row)
            return {"raw_json_artifact": row["canonical_artifact"], "json_pointer": pointer,
                    "sha256": row["sha256"], "displayed_binding_pointer": destination}

        def project(value, pointer, *, in_binding=False):
            if not in_binding and isinstance(value, dict) and _BINDING.fullmatch(pointer):
                key = _digest(value)
                previous = first.get(key)
                if previous is not None and previous[1] == value:
                    return reference(value, pointer, "binding", previous[0])
                first[key] = (pointer, value)
                # Traverse once; only the known machine registry is deferred.
                body = project(value, pointer, in_binding=pointer)
                bindings.append((pointer, body))
                return reference(value, pointer, "binding", pointer)
            if in_binding and pointer == in_binding + "/projection/registry_snapshot" and isinstance(value, (dict, list)):
                return reference(value, pointer, "registry_snapshot", None)
            if isinstance(value, dict):
                return {key: project(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"),
                                     in_binding=in_binding) for key, child in value.items()}
            if isinstance(value, list):
                return [project(child, pointer + f"/{i}", in_binding=in_binding)
                        for i, child in enumerate(value)]
            return value

        projected = project(entry, path)
        lines = ["Displayed execution facts; complete original ledger: "
                 f"final_review.json, JSON pointer `{path}`, SHA-256 `{_digest(entry)}`.", "",
                 *_fence(projected)]
        for pointer, body in bindings:
            lines += ["#### Target binding " + self.nav.marker(pointer, path), "", *_fence(body)]
        for row in records:
            pointer = row["json_pointer"]
            # First binding already has a target; every other raw location gets
            # a real paragraph explaining exactly which canonical JSON to read.
            marker = "" if row["kind"] == "binding" and row["first_expanded_pointer"] == pointer else self.nav.marker(pointer, path)
            target = row["first_expanded_pointer"]
            link = (" [First expanded binding](technical_appendix.md#" + self.nav.appendix[target] + ").") if target else ""
            lines += [f"Raw JSON locator: final_review.json; `{pointer}`; "
                      f"SHA-256 `{row['sha256']}`. " + marker + link, ""]
            row["appendix_anchor"] = self.nav.appendix[pointer]
        self.records.extend(records)
        return lines
