"""Master archive: a Markdown table generated from SQLite (every qualifying paper exactly once)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.ids import abs_url
from arxiv_digest.report.digest import EVIDENCE_LABEL
from arxiv_digest.report.files import commit_replace, restore_backup, sha256_file, write_temp
from arxiv_digest.timeutil import utcnow_iso

ROW_ID_RE = re.compile(r"^\| \d+ \| \[([^\]]+)\]\(https://arxiv\.org/abs/", re.M)


def _cell(s: str | None) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_archive(cfg: Config, archive: Archive) -> tuple[str, list[str]]:
    threshold = int(cfg.section("scoring")["archive_threshold"])
    ids = archive.qualifying_ids(threshold)
    scores = archive.current_scores()
    lines = [
        "---",
        'title: "arXiv Research Digest — Master Archive"',
        f"generated: {utcnow_iso()}",
        f"qualifying_papers: {len(ids)}",
        f"threshold: {threshold}",
        "tags: [arxiv-digest, archive]",
        "---",
        "",
        "# Master Archive",
        "",
        f"Every paper with a current score ≥ {threshold}. Generated from the SQLite archive — edit the database, not this file.",
        "",
        "| # | arXiv | Title | Score | Band | Evidence | Confidence | Topics | Category | Submitted | First seen |",
        "|---:|---|---|---:|---|---|---|---|---|---|---|",
    ]
    for n, pid in enumerate(ids, start=1):
        p = archive.paper(pid)
        s = scores[pid]
        topics = ", ".join(archive.current_topics(pid))
        lines.append(
            f"| {n} | [{pid}]({abs_url(pid)}) | {_cell(p['title'])} | {s.total} | {s.band} | "
            f"{EVIDENCE_LABEL.get(s.evidence_level, s.evidence_level)} | {s.confidence or '—'} | {_cell(topics) or '—'} | "
            f"{p['primary_category'] or '—'} | {(p['published'] or '')[:10] or '—'} | {(p['first_seen_at'] or '')[:10]} |"
        )
    lines.append("")
    return "\n".join(lines), ids


def verify_archive_text(text: str, expected_ids: list[str]) -> list[str]:
    found = ROW_ID_RE.findall(text)
    errors = []
    dupes = sorted({i for i in found if found.count(i) > 1})
    if dupes:
        errors.append(f"duplicate archive rows: {dupes}")
    if sorted(found) != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(found))
        extra = sorted(set(found) - set(expected_ids))
        errors.append(f"archive rows do not match qualifying papers (missing={missing[:10]}, unexpected={extra[:10]})")
    return errors


@dataclass
class ArchiveExport:
    path: Path
    sha256: str
    paper_ids: list[str]
    backup: Path | None


def export_archive(cfg: Config, archive: Archive) -> ArchiveExport:
    text, ids = render_archive(cfg, archive)
    errors = verify_archive_text(text, ids)
    if errors:
        raise RuntimeError("master archive failed validation before commit: " + "; ".join(errors))
    final = cfg.archive_path
    tmp = write_temp(final, text)
    backup = commit_replace(tmp, final)
    errors = verify_archive_text(final.read_text(encoding="utf-8"), ids)
    if errors:
        restore_backup(final, backup)
        raise RuntimeError("master archive failed post-write verification: " + "; ".join(errors))
    return ArchiveExport(final, sha256_file(final), ids, backup)
