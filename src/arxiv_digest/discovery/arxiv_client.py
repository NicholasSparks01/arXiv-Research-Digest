"""Minimal, polite arXiv API client (Atom feed), with pagination, retries and rate limiting."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Iterable

import httpx

from arxiv_digest.ids import split_id

log = logging.getLogger(__name__)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class PaperRecord:
    arxiv_id: str  # canonical base id, no version
    version: int
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published: str  # v1 submission timestamp (ISO, UTC)
    updated: str  # this version's timestamp (ISO, UTC)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    links: dict[str, str] = field(default_factory=dict)


@dataclass
class QueryResult:
    query: str
    total_results: int | None = None
    papers: list[PaperRecord] = field(default_factory=list)
    pages: int = 0
    error: str | None = None
    saturated: bool = False

    @property
    def fetched(self) -> int:
        return len(self.papers)

    @property
    def complete(self) -> bool:
        return self.error is None and (self.total_results is None or self.fetched >= self.total_results or self.saturated)


class ArxivAPIError(RuntimeError):
    pass


def _text(el: ET.Element | None) -> str:
    return " ".join((el.text or "").split()) if el is not None else ""


def parse_feed(xml_text: str) -> tuple[int | None, list[PaperRecord]]:
    root = ET.fromstring(xml_text)
    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else None
    papers: list[PaperRecord] = []
    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry.find("atom:id", NS))
        title = _text(entry.find("atom:title", NS))
        # arXiv reports API errors as a single entry titled "Error"
        if title == "Error" and "api/errors" in raw_id:
            raise ArxivAPIError(_text(entry.find("atom:summary", NS)) or "arXiv API error")
        try:
            base, ver = split_id(raw_id)
        except ValueError:
            log.warning("skipping entry with unparseable id %r", raw_id)
            continue
        cats = [c.get("term", "") for c in entry.findall("atom:category", NS) if c.get("term")]
        prim = entry.find("arxiv:primary_category", NS)
        links = {}
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                links["pdf"] = link.get("href", "")
            elif link.get("rel") == "alternate":
                links["abs"] = link.get("href", "")
        papers.append(
            PaperRecord(
                arxiv_id=base,
                version=ver or 1,
                title=title,
                abstract=_text(entry.find("atom:summary", NS)),
                authors=[_text(a.find("atom:name", NS)) for a in entry.findall("atom:author", NS)],
                primary_category=prim.get("term", "") if prim is not None else (cats[0] if cats else ""),
                categories=cats,
                published=_text(entry.find("atom:published", NS)),
                updated=_text(entry.find("atom:updated", NS)),
                comment=_text(entry.find("arxiv:comment", NS)),
                journal_ref=_text(entry.find("arxiv:journal_ref", NS)),
                doi=_text(entry.find("arxiv:doi", NS)),
                links=links,
            )
        )
    return total, papers


class ArxivClient:
    def __init__(
        self,
        api_url: str,
        user_agent: str,
        min_interval_s: float = 3.1,
        page_size: int = 500,
        max_retries: int = 4,
        timeout_s: float = 90,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_url = api_url
        self.min_interval_s = min_interval_s
        self.page_size = page_size
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_request = 0.0
        self._http = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout_s, transport=transport, follow_redirects=True
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ArxivClient":
        return cls(
            api_url=cfg["api_url"],
            user_agent=cfg["user_agent"],
            min_interval_s=float(cfg.get("min_request_interval_s", 3.1)),
            page_size=int(cfg.get("page_size", 500)),
            max_retries=int(cfg.get("max_retries", 4)),
            timeout_s=float(cfg.get("timeout_s", 90)),
        )

    def close(self) -> None:
        self._http.close()

    # -- low level -------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, params: dict) -> tuple[int | None, list[PaperRecord]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._http.get(self.api_url, params=params)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                return parse_feed(resp.text)
            except (httpx.HTTPError, ET.ParseError) as exc:
                last_exc = exc
                backoff = min(60.0, self.min_interval_s * (2**attempt))
                log.warning("arXiv request failed (attempt %d/%d): %s; retrying in %.0fs", attempt + 1, self.max_retries + 1, exc, backoff)
                self._sleep(backoff)
        raise ArxivAPIError(f"arXiv request failed after {self.max_retries + 1} attempts: {last_exc}")

    # -- high level ------------------------------------------------------------
    def search(self, query: str, max_results: int, sort_by: str = "submittedDate") -> QueryResult:
        """Page through a search query. Detects saturation (total > cap) and short reads."""
        result = QueryResult(query=query)
        start = 0
        empty_retries = 0
        try:
            while start < max_results:
                size = min(self.page_size, max_results - start)
                total, page = self._get(
                    {"search_query": query, "start": start, "max_results": size, "sortBy": sort_by, "sortOrder": "descending"}
                )
                result.pages += 1
                if total is not None:
                    result.total_results = total
                if not page:
                    # The arXiv API intermittently returns empty pages mid-pagination; retry a few times.
                    if result.total_results is not None and start < result.total_results and empty_retries < self.max_retries:
                        empty_retries += 1
                        log.warning("empty page at offset %d/%s; retrying", start, result.total_results)
                        continue
                    break
                empty_retries = 0
                result.papers.extend(page)
                start += len(page)
                if result.total_results is not None and start >= result.total_results:
                    break
        except ArxivAPIError as exc:
            result.error = str(exc)
        if result.total_results is not None and result.total_results > max_results:
            result.saturated = True
        if result.error is None and result.total_results is not None and not result.saturated and result.fetched < result.total_results:
            result.error = f"incomplete: fetched {result.fetched} of {result.total_results}"
        # de-duplicate within a query (paging over a moving result set can repeat entries)
        seen: dict[str, PaperRecord] = {}
        for p in result.papers:
            if p.arxiv_id not in seen or p.version > seen[p.arxiv_id].version:
                seen[p.arxiv_id] = p
        result.papers = list(seen.values())
        return result

    def fetch_ids(self, ids: Iterable[str], batch: int = 100) -> QueryResult:
        """Fetch current metadata for explicit ids (latest version)."""
        ids = list(dict.fromkeys(ids))
        result = QueryResult(query=f"id_list[{len(ids)}]", total_results=len(ids))
        for i in range(0, len(ids), batch):
            chunk = ids[i : i + batch]
            try:
                _, page = self._get({"id_list": ",".join(chunk), "max_results": len(chunk)})
            except ArxivAPIError as exc:
                result.error = str(exc)
                break
            result.pages += 1
            result.papers.extend(page)
        return result
