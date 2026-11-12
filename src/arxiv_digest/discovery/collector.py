"""Discovery engine: category baseline + focused topic searches + adaptive searches, deduplicated into the archive."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.discovery.arxiv_client import ArxivClient, QueryResult
from arxiv_digest.timeutil import Window

log = logging.getLogger(__name__)


def category_clause(categories: list[str]) -> str:
    return "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"


def build_query(topic_query: str | None, categories: list[str], window: Window) -> str:
    parts = []
    if topic_query:
        parts.append(f"({topic_query})")
    parts.append(category_clause(categories))
    parts.append(window.arxiv_range())
    return " AND ".join(parts)


@dataclass
class SearchSummary:
    search_id: int
    kind: str
    name: str
    query: str
    total: int | None
    fetched: int
    new: int
    revised: int
    status: str
    error: str | None
    paper_ids: list[str] = field(default_factory=list)


@dataclass
class DiscoverySummary:
    searches: list[SearchSummary] = field(default_factory=list)

    @property
    def failed(self) -> list[SearchSummary]:
        return [s for s in self.searches if s.status in ("failed", "incomplete")]

    @property
    def saturated(self) -> list[SearchSummary]:
        return [s for s in self.searches if s.status == "saturated"]

    def unique_ids(self, kinds: tuple[str, ...] | None = None) -> set[str]:
        return {p for s in self.searches if kinds is None or s.kind in kinds for p in s.paper_ids}


class Collector:
    def __init__(self, cfg: Config, archive: Archive, client: ArxivClient):
        self.cfg = cfg
        self.archive = archive
        self.client = client
        self.acfg = cfg.section("arxiv")

    def _record(self, run_id: int, kind: str, name: str, query: str, window: Window | None, result: QueryResult) -> SearchSummary:
        ws = window.start.isoformat() if window else None
        we = window.end.isoformat() if window else None
        search_id = self.archive.start_search(run_id, kind, name, query, ws, we)
        new = revised = 0
        ids: list[str] = []
        with self.archive.transaction():
            for rec in result.papers:
                out = self.archive.upsert_paper(rec, run_id)
                new += out.is_new
                revised += out.is_revised
                self.archive.link_run_paper(run_id, rec.arxiv_id, search_id, rec.version, out.is_new, out.is_revised)
                ids.append(rec.arxiv_id)
        if result.error and not result.papers:
            status = "failed"
        elif result.error:
            status = "incomplete"
        elif result.saturated:
            status = "saturated"
        else:
            status = "ok"
        self.archive.finish_search(
            search_id, total=result.total_results, fetched=result.fetched, new=new, revised=revised, pages=result.pages,
            status=status, error=result.error,
        )
        log.info("%-8s %-60s total=%s fetched=%d new=%d revised=%d %s", kind, name[:60], result.total_results, result.fetched, new, revised, status)
        return SearchSummary(search_id, kind, name, query, result.total_results, result.fetched, new, revised, status, result.error, ids)

    def collect_categories(self, run_id: int, window: Window) -> list[SearchSummary]:
        out = []
        cap = int(self.acfg.get("category_max_results", 20000))
        for cat in self.cfg.categories:
            query = build_query(None, [cat], window)
            out.append(self._record(run_id, "category", cat, query, window, self.client.search(query, cap)))
        return out

    def collect_topics(self, run_id: int, window: Window) -> list[SearchSummary]:
        out = []
        cap = int(self.acfg.get("topic_max_results", 600))
        for t in self.cfg.topic_searches:
            query = build_query(t.query, self.cfg.categories, window)
            out.append(self._record(run_id, "topic", f"{t.id} {t.name}", query, window, self.client.search(query, cap)))
        return out

    def collect_adaptive(self, run_id: int, window: Window, searches: list[tuple[str, str]]) -> list[SearchSummary]:
        out = []
        cap = int(self.acfg.get("topic_max_results", 600))
        for name, topic_query in searches:
            query = build_query(topic_query, self.cfg.categories, window)
            out.append(self._record(run_id, "adaptive", name, query, window, self.client.search(query, cap)))
        return out

    def refresh_versions(self, run_id: int, arxiv_ids: list[str]) -> SearchSummary | None:
        """Re-fetch archived papers to detect new arXiv versions (submittedDate queries never return revisions)."""
        ids = [i for i in arxiv_ids if i]
        if not ids:
            return None
        result = self.client.fetch_ids(ids)
        return self._record(run_id, "id_refresh", f"version refresh ({len(ids)} archived papers)", result.query, None, result)

    def discover(self, run_id: int, window: Window) -> DiscoverySummary:
        summary = DiscoverySummary()
        summary.searches.extend(self.collect_categories(run_id, window))
        summary.searches.extend(self.collect_topics(run_id, window))
        return summary
