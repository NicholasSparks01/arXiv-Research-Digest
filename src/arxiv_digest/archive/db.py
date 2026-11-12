"""SQLite research archive — the authoritative record. Markdown is only a presentation layer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from arxiv_digest.discovery.arxiv_client import PaperRecord
from arxiv_digest.timeutil import utcnow_iso

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    mode             TEXT NOT NULL,              -- run | collect | reconcile
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    window_start     TEXT,
    window_end       TEXT,
    research_week    TEXT,
    state            TEXT NOT NULL,              -- STARTED, DISCOVERED, ANALYZED, REPORT_GENERATED, REPORT_VALIDATED, ARCHIVE_COMMITTED, SUCCESS
    status           TEXT NOT NULL,              -- RUNNING, SUCCESS, PARTIAL, FAILED
    error            TEXT,
    manifest_path    TEXT,
    config_hash      TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    at      TEXT NOT NULL,
    state   TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,          -- canonical id without version
    latest_version    INTEGER NOT NULL,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL DEFAULT '',
    authors           TEXT NOT NULL DEFAULT '[]', -- JSON array
    primary_category  TEXT,
    categories        TEXT NOT NULL DEFAULT '[]', -- JSON array
    published         TEXT,                      -- v1 submission
    updated           TEXT,                      -- latest version timestamp
    comment           TEXT,
    journal_ref       TEXT,
    doi               TEXT,
    origin            TEXT NOT NULL,             -- arxiv | legacy
    metadata_complete INTEGER NOT NULL DEFAULT 1,
    first_seen_run    INTEGER REFERENCES runs(id),
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_versions (
    arxiv_id       TEXT NOT NULL REFERENCES papers(arxiv_id),
    version        INTEGER NOT NULL,
    updated        TEXT,
    title          TEXT,
    abstract_hash  TEXT,
    first_seen_run INTEGER REFERENCES runs(id),
    first_seen_at  TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, version)
);

CREATE TABLE IF NOT EXISTS paper_topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
    topic_id   TEXT NOT NULL,
    tier       INTEGER,
    source     TEXT NOT NULL,                    -- prefilter | screening | investigation | legacy
    run_id     INTEGER REFERENCES runs(id),
    rationale  TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id),
    version         INTEGER,
    run_id          INTEGER REFERENCES runs(id),
    stage           TEXT NOT NULL,               -- legacy | triage | screening | investigation | verification
    model           TEXT,
    alignment       INTEGER,
    applicability   INTEGER,
    cross_domain    INTEGER,
    novelty         INTEGER,
    reproducibility INTEGER,
    total           INTEGER NOT NULL,
    band            TEXT NOT NULL,
    confidence      TEXT,
    evidence_level  TEXT NOT NULL,               -- legacy | abstract_only | full_text_reviewed | independently_corroborated
    adjustments     TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_analysis (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id       TEXT NOT NULL REFERENCES papers(arxiv_id),
    version        INTEGER,
    run_id         INTEGER REFERENCES runs(id),
    stage          TEXT NOT NULL,                -- triage | screening | investigation | verification | github | legacy
    model          TEXT,
    prompt_version TEXT,
    input_hash     TEXT,
    payload        TEXT NOT NULL,                -- validated JSON
    flags          TEXT NOT NULL DEFAULT '[]',   -- validation / integrity flags
    metrics        TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    kind          TEXT NOT NULL,                 -- category | topic | adaptive | id_refresh | legacy_lookup
    name          TEXT NOT NULL,
    query         TEXT NOT NULL,
    window_start  TEXT,
    window_end    TEXT,
    total_results INTEGER,
    fetched       INTEGER NOT NULL DEFAULT 0,
    new_papers    INTEGER NOT NULL DEFAULT 0,
    revised       INTEGER NOT NULL DEFAULT 0,
    pages         INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,                 -- ok | saturated | failed | incomplete
    error         TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS run_papers (
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    arxiv_id       TEXT NOT NULL REFERENCES papers(arxiv_id),
    search_run_id  INTEGER NOT NULL REFERENCES search_runs(id),
    version_seen   INTEGER,
    is_new         INTEGER NOT NULL DEFAULT 0,
    is_revised     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, arxiv_id, search_run_id)
);

CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER REFERENCES runs(id),
    kind         TEXT NOT NULL,                  -- digest | collection | master_archive | reconciliation | benchmark
    path         TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    paper_count  INTEGER NOT NULL DEFAULT 0,
    paper_ids    TEXT NOT NULL DEFAULT '[]',
    validated    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS legacy_records (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    import_run_id         INTEGER REFERENCES runs(id),
    source_file           TEXT NOT NULL,
    line_no               INTEGER,
    arxiv_id              TEXT,
    raw_title             TEXT,
    raw_score             INTEGER,
    raw_date              TEXT,
    raw_row               TEXT,
    status                TEXT NOT NULL,          -- imported | matched | duplicate | not_found | conflict
    note                  TEXT
);

CREATE INDEX IF NOT EXISTS ix_scores_paper ON paper_scores(arxiv_id, stage, created_at);
CREATE INDEX IF NOT EXISTS ix_analysis_paper ON paper_analysis(arxiv_id, stage, version);
CREATE INDEX IF NOT EXISTS ix_topics_paper ON paper_topics(arxiv_id);
CREATE INDEX IF NOT EXISTS ix_run_papers_paper ON run_papers(arxiv_id);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(arxiv_id UNINDEXED, title, abstract);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract) VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE OF title, abstract ON papers BEGIN
    DELETE FROM papers_fts WHERE rowid = old.rowid;
    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract) VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;
"""

# Higher rank wins when choosing a paper's current score.
# A triage score ranks papers for the shortlist but never qualifies one for the archive or the digest. It ranks below
# every other stage so that it never displaces an existing (e.g. legacy) score as a paper's current score.
STAGE_RANK = {"triage": -1, "legacy": 0, "screening": 1, "investigation": 2, "verification": 3}
RANKING_ONLY_STAGES = frozenset({"triage"})

TERMINAL_OK = "SUCCESS"
RUN_STATES = ["STARTED", "DISCOVERED", "ANALYZED", "REPORT_GENERATED", "REPORT_VALIDATED", "ARCHIVE_COMMITTED", "SUCCESS"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class UpsertOutcome:
    is_new: bool
    is_revised: bool
    version: int


@dataclass
class CurrentScore:
    arxiv_id: str
    total: int
    band: str
    stage: str
    confidence: str | None
    evidence_level: str
    components: dict[str, int | None]
    model: str | None


class Archive:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit; explicit transactions
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- meta ------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # -- runs ------------------------------------------------------------------
    def create_run(self, mode: str, window_start: str | None, window_end: str | None, research_week: str | None, config_hash: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(mode, started_at, window_start, window_end, research_week, state, status, config_hash) VALUES (?,?,?,?,?,?,?,?)",
            (mode, utcnow_iso(), window_start, window_end, research_week, "STARTED", "RUNNING", config_hash),
        )
        run_id = int(cur.lastrowid)
        self.add_event(run_id, "STARTED", f"mode={mode}")
        return run_id

    def add_event(self, run_id: int, state: str, message: str | None = None) -> None:
        self.conn.execute("INSERT INTO run_events(run_id, at, state, message) VALUES (?,?,?,?)", (run_id, utcnow_iso(), state, message))

    def set_run_state(self, run_id: int, state: str, message: str | None = None) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"unknown run state {state}")
        self.conn.execute("UPDATE runs SET state = ? WHERE id = ?", (state, run_id))
        self.add_event(run_id, state, message)

    def finish_run(self, run_id: int, status: str, error: str | None = None, manifest_path: str | None = None) -> None:
        if status not in ("SUCCESS", "PARTIAL", "FAILED"):
            raise ValueError(status)
        self.conn.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = ?, manifest_path = COALESCE(?, manifest_path) WHERE id = ?",
            (status, error, utcnow_iso(), manifest_path, run_id),
        )
        self.add_event(run_id, status, error)

    def run(self, run_id: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def recent_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def mark_success(self, run_id: int, window_end: str, mode: str) -> None:
        """The ONLY place last_successful_* advances. Called inside the final commit transaction."""
        key = "last_successful_window_end" if mode == "run" else f"last_successful_{mode}_window_end"
        self.set_meta(key, window_end)
        self.set_meta(f"last_successful_{mode}_run_id", str(run_id))
        self.conn.execute("UPDATE runs SET state = 'SUCCESS' WHERE id = ?", (run_id,))
        self.add_event(run_id, "SUCCESS", f"{key} advanced to {window_end}")

    # -- papers ----------------------------------------------------------------
    def upsert_paper(self, rec: PaperRecord, run_id: int | None, origin: str = "arxiv") -> UpsertOutcome:
        now = utcnow_iso()
        row = self.conn.execute("SELECT latest_version, metadata_complete FROM papers WHERE arxiv_id = ?", (rec.arxiv_id,)).fetchone()
        is_new = row is None
        is_revised = False
        if is_new:
            self.conn.execute(
                """INSERT INTO papers(arxiv_id, latest_version, title, abstract, authors, primary_category, categories,
                   published, updated, comment, journal_ref, doi, origin, metadata_complete, first_seen_run, first_seen_at, last_seen_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)""",
                (
                    rec.arxiv_id, rec.version, rec.title, rec.abstract, json.dumps(rec.authors), rec.primary_category,
                    json.dumps(rec.categories), rec.published, rec.updated, rec.comment, rec.journal_ref, rec.doi,
                    origin, run_id, now, now,
                ),
            )
        else:
            prev_version = int(row["latest_version"])
            if rec.version > prev_version or not row["metadata_complete"]:
                is_revised = rec.version > prev_version and bool(row["metadata_complete"])
                self.conn.execute(
                    """UPDATE papers SET latest_version = MAX(latest_version, ?), title = ?, abstract = ?, authors = ?,
                       primary_category = ?, categories = ?, published = COALESCE(?, published), updated = ?, comment = ?,
                       journal_ref = ?, doi = ?, metadata_complete = 1, last_seen_at = ? WHERE arxiv_id = ?""",
                    (
                        rec.version, rec.title, rec.abstract, json.dumps(rec.authors), rec.primary_category,
                        json.dumps(rec.categories), rec.published or None, rec.updated, rec.comment, rec.journal_ref,
                        rec.doi, now, rec.arxiv_id,
                    ),
                )
            else:
                self.conn.execute("UPDATE papers SET last_seen_at = ? WHERE arxiv_id = ?", (now, rec.arxiv_id))
        self.conn.execute(
            """INSERT OR IGNORE INTO paper_versions(arxiv_id, version, updated, title, abstract_hash, first_seen_run, first_seen_at)
               VALUES (?,?,?,?,?,?,?)""",
            (rec.arxiv_id, rec.version, rec.updated, rec.title, sha256_text(rec.abstract), run_id, now),
        )
        return UpsertOutcome(is_new=is_new, is_revised=is_revised, version=rec.version)

    def insert_placeholder(self, arxiv_id: str, title: str, run_id: int | None) -> bool:
        """Legacy paper whose metadata could not be fetched. Returns True if inserted."""
        now = utcnow_iso()
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO papers(arxiv_id, latest_version, title, origin, metadata_complete, first_seen_run, first_seen_at, last_seen_at)
               VALUES (?, 0, ?, 'legacy', 0, ?, ?, ?)""",
            (arxiv_id, title or arxiv_id, run_id, now, now),
        )
        return cur.rowcount > 0

    def paper(self, arxiv_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()

    def has_paper(self, arxiv_id: str) -> bool:
        return self.paper(arxiv_id) is not None

    def all_paper_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT arxiv_id FROM papers")}

    # -- search runs -----------------------------------------------------------
    def start_search(self, run_id: int, kind: str, name: str, query: str, window_start: str | None, window_end: str | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO search_runs(run_id, kind, name, query, window_start, window_end, status, started_at) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, kind, name, query, window_start, window_end, "running", utcnow_iso()),
        )
        return int(cur.lastrowid)

    def finish_search(self, search_id: int, *, total: int | None, fetched: int, new: int, revised: int, pages: int, status: str, error: str | None) -> None:
        self.conn.execute(
            "UPDATE search_runs SET total_results=?, fetched=?, new_papers=?, revised=?, pages=?, status=?, error=?, finished_at=? WHERE id=?",
            (total, fetched, new, revised, pages, status, error, utcnow_iso(), search_id),
        )

    def link_run_paper(self, run_id: int, arxiv_id: str, search_id: int, version: int, is_new: bool, is_revised: bool) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO run_papers(run_id, arxiv_id, search_run_id, version_seen, is_new, is_revised) VALUES (?,?,?,?,?,?)",
            (run_id, arxiv_id, search_id, version, int(is_new), int(is_revised)),
        )

    def searches_for_run(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM search_runs WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()

    def run_paper_ids(self, run_id: int, kinds: tuple[str, ...] | None = None) -> list[str]:
        sql = "SELECT DISTINCT rp.arxiv_id FROM run_papers rp JOIN search_runs s ON s.id = rp.search_run_id WHERE rp.run_id = ?"
        params: list[Any] = [run_id]
        if kinds:
            sql += f" AND s.kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return [r[0] for r in self.conn.execute(sql + " ORDER BY rp.arxiv_id", params)]

    def run_paper_sources(self, run_id: int) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for r in self.conn.execute(
            "SELECT rp.arxiv_id, s.kind, s.name FROM run_papers rp JOIN search_runs s ON s.id = rp.search_run_id WHERE rp.run_id = ?",
            (run_id,),
        ):
            out.setdefault(r["arxiv_id"], set()).add(f"{r['kind']}:{r['name']}")
        return out

    # -- topics / scores / analysis -------------------------------------------
    def add_topics(self, arxiv_id: str, topics: list[tuple[str, int | None, str | None]], source: str, run_id: int | None) -> None:
        now = utcnow_iso()
        self.conn.executemany(
            "INSERT INTO paper_topics(arxiv_id, topic_id, tier, source, run_id, rationale, created_at) VALUES (?,?,?,?,?,?,?)",
            [(arxiv_id, t, tier, source, run_id, why, now) for t, tier, why in topics],
        )

    def current_topics(self, arxiv_id: str) -> list[str]:
        """Topics from the highest-ranked stage that produced any."""
        rows = self.conn.execute(
            "SELECT topic_id, source, run_id, created_at, id FROM paper_topics WHERE arxiv_id = ? ORDER BY id", (arxiv_id,)
        ).fetchall()
        if not rows:
            return []
        rank = {"prefilter": -1, "legacy": 0, "screening": 1, "investigation": 2}
        anchor = max(rows, key=lambda r: (rank.get(r["source"], 0), r["id"]))
        batch = [
            r["topic_id"]
            for r in rows
            if r["source"] == anchor["source"] and r["run_id"] == anchor["run_id"] and r["created_at"] == anchor["created_at"]
        ]
        return list(dict.fromkeys(batch))

    def add_score(
        self,
        arxiv_id: str,
        *,
        version: int | None,
        run_id: int | None,
        stage: str,
        model: str | None,
        components: dict[str, int | None],
        total: int,
        band: str,
        confidence: str | None,
        evidence_level: str,
        adjustments: list[str] | None = None,
    ) -> None:
        if stage not in STAGE_RANK:
            raise ValueError(stage)
        if not 0 <= total <= 100:
            raise ValueError(f"total out of range: {total}")
        self.conn.execute(
            """INSERT INTO paper_scores(arxiv_id, version, run_id, stage, model, alignment, applicability, cross_domain, novelty,
               reproducibility, total, band, confidence, evidence_level, adjustments, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                arxiv_id, version, run_id, stage, model, components.get("alignment"), components.get("applicability"),
                components.get("cross_domain"), components.get("novelty"), components.get("reproducibility"), total, band,
                confidence, evidence_level, json.dumps(adjustments or []), utcnow_iso(),
            ),
        )

    def current_score(self, arxiv_id: str) -> CurrentScore | None:
        rows = self.conn.execute("SELECT * FROM paper_scores WHERE arxiv_id = ? ORDER BY id", (arxiv_id,)).fetchall()
        if not rows:
            return None
        best = max(rows, key=lambda r: (STAGE_RANK[r["stage"]], r["id"]))
        return CurrentScore(
            arxiv_id=arxiv_id,
            total=int(best["total"]),
            band=best["band"],
            stage=best["stage"],
            confidence=best["confidence"],
            evidence_level=best["evidence_level"],
            components={k: best[k] for k in ("alignment", "applicability", "cross_domain", "novelty", "reproducibility")},
            model=best["model"],
        )

    def current_scores(self) -> dict[str, CurrentScore]:
        ids = [r[0] for r in self.conn.execute("SELECT DISTINCT arxiv_id FROM paper_scores")]
        return {i: s for i in ids if (s := self.current_score(i)) is not None}

    def qualifying_ids(self, threshold: int) -> list[str]:
        scores = self.current_scores()
        return sorted((i for i, s in scores.items() if s.total >= threshold and s.stage not in RANKING_ONLY_STAGES),
                      key=lambda i: (-scores[i].total, i))

    def add_analysis(
        self,
        arxiv_id: str,
        *,
        version: int | None,
        run_id: int | None,
        stage: str,
        model: str | None,
        prompt_version: str | None,
        input_hash: str | None,
        payload: dict,
        flags: list[str] | None = None,
        metrics: dict | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO paper_analysis(arxiv_id, version, run_id, stage, model, prompt_version, input_hash, payload, flags, metrics, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                arxiv_id, version, run_id, stage, model, prompt_version, input_hash, json.dumps(payload, ensure_ascii=False),
                json.dumps(flags or []), json.dumps(metrics or {}), utcnow_iso(),
            ),
        )
        return int(cur.lastrowid)

    def latest_analysis(self, arxiv_id: str, stage: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM paper_analysis WHERE arxiv_id = ? AND stage = ? ORDER BY id DESC LIMIT 1", (arxiv_id, stage)
        ).fetchone()

    def find_cached_analysis(self, arxiv_id: str, stage: str, model: str, prompt_version: str, input_hash: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM paper_analysis WHERE arxiv_id=? AND stage=? AND model=? AND prompt_version=? AND input_hash=?
               ORDER BY id DESC LIMIT 1""",
            (arxiv_id, stage, model, prompt_version, input_hash),
        ).fetchone()

    # -- reports ---------------------------------------------------------------
    def add_report(self, run_id: int | None, kind: str, path: str, sha256: str, paper_ids: list[str], validated: bool) -> int:
        cur = self.conn.execute(
            "INSERT INTO reports(run_id, kind, path, sha256, paper_count, paper_ids, validated, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, kind, path, sha256, len(paper_ids), json.dumps(paper_ids), int(validated), utcnow_iso()),
        )
        return int(cur.lastrowid)

    # -- search ----------------------------------------------------------------
    def fts_search(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT arxiv_id, title, bm25(papers_fts) AS rank FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
