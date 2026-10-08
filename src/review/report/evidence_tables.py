"""Render original table text without interpreting manuscript HTML as markup."""

from __future__ import annotations

import html
import re

from verification.experiment_catalog import TableGrid

_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
_INLINE = {"b", "strong", "i", "em", "span", "sup", "sub"}
_GROUPS = {"thead", "tbody", "tfoot"}


class _DisplayTable(TableGrid):
    """Reuse span expansion, with stricter structure and lossless inline text."""

    def __init__(self, source):
        self.stack = []
        self.caption = []
        self.input_rows = 0
        self.header_rows = 0
        self.row_tags = []
        super().__init__(source)
        if self.stack or len(self.tables) != 1 or not self.tables[0]:
            raise ValueError("Incomplete table structure")
        grid = self.tables[0]
        rows = max(row for row, _ in grid) + 1
        columns = max(column for _, column in grid) + 1
        if rows * columns > 10000:
            raise ValueError("Table is too large for a reliable inline view")
        if rows != self.input_rows or len(grid) != rows * columns:
            raise ValueError("Uneven table rows have ambiguous column associations")

    def handle_starttag(self, tag, attrs):
        names = [name for name, _ in attrs]
        if len(names) != len(set(names)):
            raise ValueError("Repeated HTML attributes are ambiguous")
        parent = self.stack[-1] if self.stack else None
        if tag == "br" and (self.cell is not None or "caption" in self.stack):
            self.handle_data(" ")
            return
        if tag == "table":
            allowed = parent is None
        elif tag in _GROUPS or tag == "caption":
            allowed = parent == "table"
        elif tag == "tr":
            allowed = parent == "table" or parent in _GROUPS
        elif tag in {"th", "td"}:
            allowed = parent == "tr"
        elif tag in _INLINE:
            allowed = self.cell is not None or "caption" in self.stack
        else:
            allowed = False
        if not allowed:
            raise ValueError("Unsupported or malformed table markup")
        self.stack.append(tag)
        if tag == "tr":
            self.row_tags = []
        elif tag in {"th", "td"}:
            self.row_tags.append(tag)
        super().handle_starttag(tag, attrs)
        if tag in {"sup", "sub"}:
            # 224<sup>2</sup> must remain an exponent, never the integer 2242.
            self.handle_data("^(" if tag == "sup" else "_(")

    def handle_startendtag(self, tag, attrs):
        if tag != "br":
            raise ValueError("Unsupported self-closing table element")
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            raise ValueError("Mismatched table elements")
        if tag in {"sup", "sub"}:
            self.handle_data(")")
        if tag == "tr":
            if self.header_rows == self.input_rows and (
                "thead" in self.stack or (self.row_tags and set(self.row_tags) == {"th"})
            ):
                self.header_rows += 1
            self.input_rows += 1
        super().handle_endtag(tag)
        self.stack.pop()

    def handle_data(self, data):
        if self.cell is not None:
            super().handle_data(data)
        elif "caption" in self.stack:
            self.caption.append(data)
        elif data.strip():
            raise ValueError("Text outside a table cell would be lost")

    def handle_comment(self, data):
        raise ValueError("Table comments require the original source view")

    def handle_decl(self, decl):
        raise ValueError("Table declarations require the original source view")

    def unknown_decl(self, data):
        raise ValueError("Table declarations require the original source view")

    def handle_pi(self, data):
        raise ValueError("Table processing instructions require the original source view")


def _cell_text(value):
    # Escape Markdown before generating HTML entities, so a literal pipe stays
    # inside its cell and entities do not turn into visible escaped source.
    value = " ".join(value.split())
    value = re.sub(r"([\\`*_\[\]{}()#+.!~$-])", r"\\\1", value)
    return html.escape(value, quote=False).replace("|", "&#124;")


def table_passage_lines(quote, escape_text):
    """Return safe nested-list Markdown, or None to preserve an original fallback."""
    fragment = not re.search(r"<table\b", quote, re.I) and bool(
        re.fullmatch(r"(?:\s*<tr\b[^>]*>.*?</tr\s*>)+\s*", quote, re.I | re.S)
    )
    if fragment:
        quote = f"<table>{quote}</table>"
    matches = list(_TABLE.finditer(quote))
    if not matches or len(matches) != len(re.findall(r"<table\b", quote, re.I)):
        return None
    lines = ["  - Passage:", ""]
    cursor = 0
    try:
        for match in matches:
            surrounding = quote[cursor : match.start()].strip()
            if surrounding:
                lines += [f"    {escape_text(surrounding)}", ""]
            table = _DisplayTable(match.group())
            if table.caption:
                lines += [f"    {_cell_text(''.join(table.caption))}", ""]
            grid = table.tables[0]
            row_count = max(row for row, _ in grid) + 1
            column_count = max(column for _, column in grid) + 1
            rows = [
                "    | " + " | ".join(_cell_text(grid[row, column]) for column in range(column_count)) + " |"
                for row in range(row_count)
            ]
            if fragment:
                lines += [
                    "    Original table-row excerpt; column headings are not included in this passage.",
                    "",
                ]
                rows.insert(
                    0,
                    "    | " + " | ".join(f"Column {column + 1}" for column in range(column_count)) + " |",
                )
            elif table.header_rows > 1:
                # GFM/PDF repeats one header row. Derive that row from explicit
                # th/thead axes and retain every original row below it.
                headings = []
                for column in range(column_count):
                    levels = dict.fromkeys(grid[row, column] for row in range(table.header_rows))
                    headings.append(_cell_text(" / ".join(value for value in levels if value)))
                rows.insert(0, "    | " + " | ".join(headings) + " |")
            lines += [
                rows[0],
                "    | " + " | ".join("---" for _ in range(column_count)) + " |",
                *rows[1:],
                "",
            ]
            cursor = match.end()
    except (ValueError, IndexError, AssertionError):
        return None
    trailing = quote[cursor:].strip()
    if trailing:
        lines += [f"    {escape_text(trailing)}", ""]
    return lines
