"""Deterministic keyword pre-filter: decides which category-baseline papers reach LLM screening.

High-volume categories (cs.LG / cs.CL / cs.AI) produce thousands of papers per month. Papers found by a topic or
adaptive search, or listed in a full-screen category, are always screened; the rest need a keyword signal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config


@dataclass
class PrefilterHit:
    arxiv_id: str
    score: int
    topics: list[str]
    reason: str


@lru_cache(maxsize=512)
def _kw_re(kw: str) -> re.Pattern:
    # Word-start boundary; keywords are stems ("decompil", "quantiz") so the end is open.
    return re.compile(r"(?<![a-z0-9])" + re.escape(kw))


def keyword_score(cfg: Config, title: str, abstract: str) -> tuple[int, list[str]]:
    t, a = title.lower(), abstract.lower()
    score = 0
    topics = []
    for topic in cfg.taxonomy:
        hit = 0
        for kw in topic.keywords:
            if _kw_re(kw).search(t):
                hit += 3
            elif _kw_re(kw).search(a):
                hit += 1
        if hit:
            weight = {1: 2, 2: 2, 3: 1}[topic.tier]
            score += hit * weight
            topics.append(topic.id)
    return score, topics


def select_for_screening(cfg: Config, archive: Archive, run_id: int) -> tuple[list[PrefilterHit], dict[str, int]]:
    scfg = cfg.section("screening")
    full = set(scfg.get("full_screen_categories", []))
    min_score = int(scfg.get("prefilter_min_score", 2))
    cap = int(scfg.get("max_papers_per_run", 4000))

    sources = archive.run_paper_sources(run_id)
    selected: list[tuple[int, PrefilterHit]] = []
    stats = {"considered": 0, "searched": 0, "full_category": 0, "keyword": 0, "skipped": 0}
    for arxiv_id, srcs in sources.items():
        kinds = {s.split(":", 1)[0] for s in srcs}
        if kinds <= {"id_refresh", "legacy_lookup"}:
            # archived paper seen only via version refresh: re-screen only if revised in this run
            row = archive.conn.execute(
                "SELECT MAX(is_revised) FROM run_papers WHERE run_id=? AND arxiv_id=?", (run_id, arxiv_id)
            ).fetchone()
            if not row or not row[0]:
                continue
        paper = archive.paper(arxiv_id)
        if paper is None or not paper["metadata_complete"]:
            continue
        stats["considered"] += 1
        cats = set(json.loads(paper["categories"] or "[]"))
        kw, topics = keyword_score(cfg, paper["title"], paper["abstract"])
        if kinds & {"topic", "adaptive"}:
            reason, priority = "topic/adaptive search hit", 2
            stats["searched"] += 1
        elif cats & full:
            reason, priority = f"full-screen category ({', '.join(sorted(cats & full))})", 1
            stats["full_category"] += 1
        elif kw >= min_score:
            reason, priority = f"keyword pre-filter score {kw}", 0
            stats["keyword"] += 1
        else:
            stats["skipped"] += 1
            continue
        selected.append((priority, PrefilterHit(arxiv_id, kw, topics, reason)))
    selected.sort(key=lambda x: (-x[0], -x[1].score, x[1].arxiv_id))
    if len(selected) > cap:
        stats["over_cap"] = len(selected) - cap
        selected = selected[:cap]
    return [h for _, h in selected], stats
