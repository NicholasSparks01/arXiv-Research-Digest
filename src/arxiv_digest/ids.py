"""Canonical arXiv identifiers and version handling."""

from __future__ import annotations

import re

# New-style (0704.0001 onwards: YYMM.NNNN / YYMM.NNNNN) and old-style (cs/0112017, math.GT/0309136).
_NEW = r"(?P<new>\d{4}\.\d{4,5})"
_OLD = r"(?P<old>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})"
ID_RE = re.compile(rf"(?:{_NEW}|{_OLD})(?:v(?P<ver>\d+))?")
# Used when scanning free text (Markdown, comments): anchored on boundaries / URL forms.
TEXT_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|arXiv:\s*|(?<![\w./]))"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
    r"(?![\w])",
    re.IGNORECASE,
)


def split_id(raw: str) -> tuple[str, int | None]:
    """'2511.01234v2' -> ('2511.01234', 2); 'http://arxiv.org/abs/2511.01234v1' -> ('2511.01234', 1)."""
    s = raw.strip()
    s = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf|html)/", "", s)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.IGNORECASE)
    s = s.removesuffix(".pdf")
    m = ID_RE.fullmatch(s)
    if not m:
        raise ValueError(f"not an arXiv identifier: {raw!r}")
    base = m.group("new") or m.group("old")
    ver = int(m.group("ver")) if m.group("ver") else None
    return base, ver


def is_valid_new_style(base: str) -> bool:
    """Reject obviously bogus new-style ids (month 00/13+), which are common false positives in text."""
    m = re.fullmatch(r"(\d{2})(\d{2})\.\d{4,5}", base)
    if not m:
        return True
    return 1 <= int(m.group(2)) <= 12


def find_ids(text: str) -> list[tuple[str, int | None]]:
    """All distinct arXiv ids mentioned in free text, in order of first appearance."""
    seen: dict[str, int | None] = {}
    for m in TEXT_ID_RE.finditer(text):
        try:
            base, ver = split_id(m.group("id"))
        except ValueError:
            continue
        if not is_valid_new_style(base):
            continue
        if base not in seen or (ver or 0) > (seen[base] or 0):
            seen[base] = ver
    return list(seen.items())


def abs_url(base: str) -> str:
    return f"https://arxiv.org/abs/{base}"


def pdf_url(base: str, version: int | None = None) -> str:
    return f"https://arxiv.org/pdf/{base}{f'v{version}' if version else ''}"
