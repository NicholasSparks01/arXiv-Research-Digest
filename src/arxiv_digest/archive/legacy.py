"""Import and reconcile the pre-existing Markdown archive (Claude/Knowra era) into SQLite.

The parser is deliberately tolerant: it understands Markdown tables (using header names such as
"Title", "Score", "Date" when present) and falls back to scanning any line for arXiv identifiers.
Every legacy row is preserved verbatim in `legacy_records` for provenance.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.discovery.arxiv_client import ArxivClient
from arxiv_digest.ids import find_ids

LEGACY_MODEL = "legacy-claude-knowra"
_SCORE_RE = re.compile(r"(?<![\d.])(\d{1,3})(?:\s*/\s*100)?(?![\d.])")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")


@dataclass
class LegacyEntry:
    source_file: str
    line_no: int
    arxiv_id: str
    version: int | None
    title: str | None
    score: int | None
    date: str | None
    raw: str


@dataclass
class ReconciliationResult:
    files: list[str] = field(default_factory=list)
    entries: list[LegacyEntry] = field(default_factory=list)
    unique_ids: list[str] = field(default_factory=list)
    duplicates: dict[str, int] = field(default_factory=dict)
    imported: list[str] = field(default_factory=list)  # new to the archive
    matched: list[str] = field(default_factory=list)  # already present
    not_found: list[str] = field(default_factory=list)  # arXiv returned nothing (placeholder kept)
    title_conflicts: list[tuple[str, str, str]] = field(default_factory=list)
    score_conflicts: dict[str, list[int]] = field(default_factory=dict)
    scores_imported: int = 0
    expected: int | None = None
    lookup_error: str | None = None

    @property
    def count_matches_expected(self) -> bool:
        return self.expected is None or self.expected == len(self.unique_ids)


def _clean_cell(cell: str) -> str:
    cell = _WIKILINK_RE.sub(r"\1", cell)
    cell = _LINK_RE.sub(r"\1", cell)
    return re.sub(r"[*_`]", "", cell).strip()


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown(text: str, source_file: str) -> list[LegacyEntry]:
    entries: list[LegacyEntry] = []
    header: list[str] | None = None
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        is_table = stripped.startswith("|") and stripped.count("|") >= 2
        if is_table and re.fullmatch(r"\|?[\s:\-|]+\|?", stripped):
            continue  # separator row
        ids = find_ids(line)
        if is_table and not ids:
            # potential header row (next line is a separator)
            nxt = lines[i].strip() if i < len(lines) else ""
            if re.fullmatch(r"\|?[\s:\-|]+\|?", nxt) and "-" in nxt:
                header = [_clean_cell(c).lower() for c in _split_row(stripped)]
            continue
        if not is_table:
            header = None if not stripped else header
        if not ids:
            continue
        title = score = date = None
        if is_table:
            cells = [_clean_cell(c) for c in _split_row(stripped)]
            cols = dict(zip(header, cells)) if header and len(header) == len(cells) else {}
            for key, val in cols.items():
                if title is None and "title" in key:
                    title = val or None
                elif score is None and ("score" in key or "relevance" in key):
                    m = _SCORE_RE.search(val)
                    score = int(m.group(1)) if m and int(m.group(1)) <= 100 else None
                elif date is None and ("date" in key or "added" in key or "week" in key):
                    m = _DATE_RE.search(val)
                    date = m.group(1) if m else (val or None)
            if title is None:
                textual = [c for c in cells if not find_ids(c) and len(c) > 15 and not _SCORE_RE.fullmatch(c)]
                title = max(textual, key=len) if textual else None
        else:
            m = _LINK_RE.search(line)
            title = m.group(1) if m and not find_ids(m.group(1)) else None
        if date is None:
            m = _DATE_RE.search(line)
            date = m.group(1) if m else None
        for base, ver in ids:
            entries.append(LegacyEntry(source_file, i, base, ver, title, score, date, line.rstrip()))
    return entries


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.md") if p.is_file())


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


def reconcile(
    cfg: Config,
    archive: Archive,
    client: ArxivClient | None,
    path: Path,
    run_id: int,
    expected: int | None = None,
    dry_run: bool = False,
) -> ReconciliationResult:
    res = ReconciliationResult(expected=expected)
    files = collect_files(path)
    res.files = [str(f) for f in files]
    for f in files:
        res.entries.extend(parse_markdown(f.read_text(encoding="utf-8", errors="replace"), str(f)))

    by_id: dict[str, list[LegacyEntry]] = {}
    for e in res.entries:
        by_id.setdefault(e.arxiv_id, []).append(e)
    res.unique_ids = sorted(by_id)
    res.duplicates = {k: len(v) for k, v in by_id.items() if len(v) > 1}
    for k, v in by_id.items():
        scores = sorted({e.score for e in v if e.score is not None})
        if len(scores) > 1:
            res.score_conflicts[k] = scores

    if dry_run:
        return res

    fetched = {}
    if client is not None and res.unique_ids:
        search_id = archive.start_search(run_id, "legacy_lookup", f"legacy archive lookup ({len(res.unique_ids)} ids)", "id_list", None, None)
        qr = client.fetch_ids(res.unique_ids)
        fetched = {p.arxiv_id: p for p in qr.papers}
        res.lookup_error = qr.error
        archive.finish_search(
            search_id, total=len(res.unique_ids), fetched=len(fetched), new=0, revised=0, pages=qr.pages,
            status="ok" if qr.error is None else ("incomplete" if fetched else "failed"), error=qr.error,
        )
    else:
        search_id = None

    with archive.transaction():
        for arxiv_id in res.unique_ids:
            group = by_id[arxiv_id]
            first = group[0]
            existed = archive.has_paper(arxiv_id)
            rec = fetched.get(arxiv_id)
            status, note = ("matched" if existed else "imported"), None
            if rec is not None:
                out = archive.upsert_paper(rec, run_id, origin="legacy")
                if search_id is not None:
                    archive.link_run_paper(run_id, arxiv_id, search_id, rec.version, out.is_new, out.is_revised)
                legacy_title = next((e.title for e in group if e.title), None)
                if legacy_title:
                    ratio = difflib.SequenceMatcher(None, _norm_title(legacy_title), _norm_title(rec.title)).ratio()
                    if ratio < 0.6 and _norm_title(legacy_title) not in _norm_title(rec.title):
                        status, note = "conflict", f"title mismatch (similarity {ratio:.2f}): arXiv says {rec.title!r}"
                        res.title_conflicts.append((arxiv_id, legacy_title, rec.title))
            else:
                if not existed:
                    archive.insert_placeholder(arxiv_id, first.title or arxiv_id, run_id)
                status, note = "not_found", res.lookup_error or "arXiv id_list lookup returned no entry"
                res.not_found.append(arxiv_id)
            (res.matched if existed else res.imported).append(arxiv_id)

            # Legacy score: keep the most recent non-null one as a 'legacy' stage score (idempotent).
            scored = [e for e in group if e.score is not None]
            if scored:
                legacy_score = scored[-1].score
                dup = archive.conn.execute(
                    "SELECT 1 FROM paper_scores WHERE arxiv_id=? AND stage='legacy' AND total=?", (arxiv_id, legacy_score)
                ).fetchone()
                if not dup:
                    archive.add_score(
                        arxiv_id, version=first.version, run_id=run_id, stage="legacy", model=LEGACY_MODEL, components={},
                        total=legacy_score, band=cfg.band_for(legacy_score), confidence=None, evidence_level="legacy",
                        adjustments=["imported from legacy Markdown archive"],
                    )
                    res.scores_imported += 1

            for e in group:
                row_status = status if e is first else "duplicate"
                archive.conn.execute(
                    """INSERT INTO legacy_records(import_run_id, source_file, line_no, arxiv_id, raw_title, raw_score, raw_date, raw_row, status, note)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, e.source_file, e.line_no, e.arxiv_id, e.title, e.score, e.date, e.raw, row_status, note),
                )
    return res
