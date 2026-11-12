"""Stage 1b — comparative re-ranking of the fully screened shortlist.

Absolute scores from a small local model saturate: on the 2026-11-11 run, 65 of 300 shortlisted papers scored exactly 81
and most relevant papers got the same five components, so "top 10 by score" was decided by arXiv id. Models are much
better at *relative* judgements, so this stage shows the model ~10 papers at a time and asks for an order.

1. Tournament rounds: papers are dealt round-robin into groups of `group_size` (each group spans the whole score range),
   each group is ranked, and the top `keep_fraction` of every group advances, until at most `final_pool` remain.
2. Sliding window (as in RankGPT): a window of `window` papers moves from the bottom of the pool to the top in steps
   of `step`; each pass carries the best papers upward, and ceil(top_n / step) passes settle the top `top_n`.

The order decides which papers get full-text investigation and which make the digest; scores are still shown and still
gate qualification. A failed call keeps the incoming order for that group (fail-safe, counted in the stats).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field

from pydantic import Field

from arxiv_digest.analysis.schemas import _Strict
from arxiv_digest.analysis.scoring import projects_text, taxonomy_text
from arxiv_digest.analysis.untrusted import UNTRUSTED_PREAMBLE, fence, sanitize
from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config, ModelProfile
from arxiv_digest.llm.provider import LLMError, LLMProvider, LLMUnavailable, SemanticError, run_structured

log = logging.getLogger(__name__)
PROMPT_VERSION = "rerank-v2"


class RerankOutput(_Strict):
    ranking: list[str] = Field(max_length=30)


def system_prompt(cfg: Config) -> str:
    return f"""You compare new arXiv papers for a security researcher who works on reverse engineering, binary
analysis, vulnerability research and AI-assisted security tooling.

Their interests (Tier 1 matters most, then Tier 2, then Tier 3):
{taxonomy_text(cfg)}

Their projects:
{projects_text(cfg)}

Rank the papers you are given from most to least useful to this researcher. Judge relevance to Tier 1 and Tier 2
interests first. Among equally relevant papers, prefer work with concrete evaluated results (new methods, tools,
benchmarks, attacks, or empirical studies of how well humans, tools or AI agents do these tasks) over surveys,
position papers and incremental variations. A paper whose only link is generic AI/LLM engineering ranks below any paper
with a real security or reverse-engineering contribution.

Return ONLY the JSON object {{"ranking": [...]}} listing every paper label exactly once, best first.

{UNTRUSTED_PREAMBLE}"""


def paper_brief(archive: Archive, arxiv_id: str) -> dict:
    p = archive.paper(arxiv_id)
    scr = archive.latest_analysis(arxiv_id, "screening")
    d = json.loads(scr["payload"]) if scr else {}
    return {
        "title": p["title"],
        "problem": d.get("research_problem") or "",
        "solution": d.get("proposed_solution") or "",
        "topics": [t.get("topic_id") for t in d.get("topics", [])],
        "abstract": p["abstract"] or "",
    }


def user_prompt(labels: list[str], briefs: list[dict]) -> str:
    blocks = []
    for label, b in zip(labels, briefs):
        if b["problem"] or b["solution"]:
            body = f"Problem: {sanitize(b['problem'], 350)}\nApproach: {sanitize(b['solution'], 350)}"
        else:
            body = f"Abstract: {sanitize(b['abstract'], 700)}"
        topics = ", ".join(t for t in b["topics"] if t) or "none"
        blocks.append(f"[{label}] {sanitize(b['title'], 250)}\n{body}\nTopics: {topics}")
    return (f"Rank these {len(labels)} papers, best first. Labels: {', '.join(labels)}.\n\n"
            + fence("PAPER SUMMARIES", "\n\n".join(blocks)))


def check_permutation(labels: list[str], out: RerankOutput) -> None:
    got = [x.strip().strip("[]") for x in out.ranking]
    missing = [x for x in labels if x not in got]
    dupes = sorted({x for x in got if got.count(x) > 1})
    unknown = [x for x in got if x not in labels]
    if missing or dupes or unknown:
        raise SemanticError(f"ranking must list each of {', '.join(labels)} exactly once "
                            f"(missing: {missing or 'none'}; repeated: {dupes or 'none'}; unknown: {unknown or 'none'})")


def repair(labels: list[str], ranking: list[str]) -> list[str]:
    seen: list[str] = []
    for x in (r.strip().strip("[]") for r in ranking):
        if x in labels and x not in seen:
            seen.append(x)
    return seen + [x for x in labels if x not in seen]


@dataclass
class RerankStats:
    pool: int = 0
    ranked_top: int = 0
    rounds: int = 0
    calls: int = 0
    failed_calls: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)


def rank_group(cfg: Config, provider: LLMProvider, profile: ModelProfile, ids: list[str], briefs: dict[str, dict],
               stats: RerankStats) -> list[str]:
    """Order `ids` best first with one model call; on failure keep the incoming order."""
    if len(ids) < 2:
        return list(ids)
    labels = [f"P{i}" for i in range(1, len(ids) + 1)]
    stats.calls += 1
    try:
        res = run_structured(provider, profile, system_prompt(cfg), user_prompt(labels, [briefs[i] for i in ids]),
                             RerankOutput, max_retries=int(cfg.section("llm").get("max_validation_retries", 1)),
                             post_validate=lambda o: check_permutation(labels, o))
        order = repair(labels, res.value.ranking)
    except LLMUnavailable:
        raise
    except LLMError as exc:
        # still not a permutation after the corrective retry (or overflow): keep the incoming order for this group
        stats.failed_calls += 1
        stats.failures.append(str(exc)[:200])
        return list(ids)
    by_label = dict(zip(labels, ids))
    return [by_label[x] for x in order]


def rerank(cfg: Config, archive: Archive, provider: LLMProvider, run_id: int, ids: list[str]) -> tuple[list[str], RerankStats]:
    """Return `ids` re-ordered best first (the top `top_n` settled by comparison) and stats; records each rank."""
    rcfg = cfg.section("rerank")
    group = max(2, int(rcfg.get("group_size", 10)))
    keep = float(rcfg.get("keep_fraction", 0.5))
    final_pool = max(group, int(rcfg.get("final_pool", 40)))
    window, step = max(2, int(rcfg.get("window", 10))), max(1, int(rcfg.get("step", 5)))
    top_n = int(rcfg.get("top_n", 15))
    profile = cfg.profile("reranker")
    stats = RerankStats(pool=len(ids))
    t0 = time.monotonic()
    briefs = {i: paper_brief(archive, i) for i in ids}

    survivors = list(ids)
    eliminated: list[list[str]] = []  # per round, best first; later rounds rank above earlier ones
    while len(survivors) > final_pool:
        stats.rounds += 1
        n_groups = math.ceil(len(survivors) / group)
        groups = [survivors[g::n_groups] for g in range(n_groups)]
        ranked = [rank_group(cfg, provider, profile, g, briefs, stats) for g in groups]
        kept = [r[: max(1, math.ceil(len(r) * keep))] for r in ranked]
        dropped = [r[len(k):] for r, k in zip(ranked, kept)]
        survivors = _interleave(kept)
        eliminated.append(_interleave(dropped))
        log.info("rerank round %d: %d groups, %d papers advance", stats.rounds, n_groups, len(survivors))

    passes = math.ceil(min(top_n, len(survivors)) / step)
    for p in range(passes):
        floor = p * step
        if len(survivors) - floor < 2:
            break
        start = max(floor, len(survivors) - window)
        while True:
            seg = survivors[start: start + window]
            survivors[start: start + window] = rank_group(cfg, provider, profile, seg, briefs, stats)
            if start == floor:
                break
            start = max(floor, start - step)
    stats.ranked_top = min(top_n, len(survivors))

    order = survivors + [i for rnd in reversed(eliminated) for i in rnd]
    with archive.transaction():
        for rank, arxiv_id in enumerate(order, start=1):
            archive.add_analysis(arxiv_id, version=archive.paper(arxiv_id)["latest_version"], run_id=run_id, stage="rerank",
                                 model=profile.model, prompt_version=PROMPT_VERSION, input_hash=None,
                                 payload={"rank": rank, "of": len(order), "settled": rank <= stats.ranked_top})
    stats.seconds = round(time.monotonic() - t0, 1)
    log.info("rerank done: %d papers, %d calls (%d failed) in %.0f s", len(ids), stats.calls, stats.failed_calls, stats.seconds)
    return order, stats


def build_pool(cfg: Config, archive: Archive, ids: list[str], exclude: set[str]) -> list[str]:
    """Fully screened papers worth ranking, ordered by screening (or deeper) score plus triage score as a starting order."""
    min_score = int(cfg.section("rerank").get("min_score", 60))
    rows = []
    for arxiv_id in ids:
        if arxiv_id in exclude:
            continue
        s = archive.current_score(arxiv_id)
        if not s or s.stage not in ("screening", "investigation", "verification") or s.total < min_score:
            continue
        t = archive.latest_analysis(arxiv_id, "triage")
        tri = json.loads(t["payload"]).get("total") or 0 if t else 0
        rows.append((-(s.total + tri), -s.total, arxiv_id))
    return [r[2] for r in sorted(rows)]


def _interleave(lists: list[list[str]]) -> list[str]:
    out: list[str] = []
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x))
    return out


def run_ranks(archive: Archive, run_id: int) -> dict[str, int]:
    """Ranks recorded by the latest re-ranking in `run_id` (empty if none ran)."""
    ranks: dict[str, int] = {}
    for r in archive.conn.execute("SELECT arxiv_id, payload FROM paper_analysis WHERE stage='rerank' AND run_id=? ORDER BY id",
                                  (run_id,)):
        ranks[r["arxiv_id"]] = int(json.loads(r["payload"])["rank"])
    return ranks
