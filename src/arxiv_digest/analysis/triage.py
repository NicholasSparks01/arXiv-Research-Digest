"""Stage 0 — compact triage of every selected abstract.

Full screening spends ~680 output tokens per paper on prose (problem, solution, rationales) that is only useful for the
few papers that pass. Triage asks the same model for topic ids, the security connection, the five rubric components and
a recommendation (~100 output tokens, ~1 s/paper) and Python applies the same scoring rules. The top papers by triage
score (a shortlist sized per research week) go on to full screening; papers whose triage failed are screened anyway.

Triage scores rank papers; they never qualify a paper for the archive or the digest on their own.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from arxiv_digest.analysis import screening as scr_mod
from arxiv_digest.analysis.schemas import TopicAssignment, TriageOutput
from arxiv_digest.analysis.scoring import apply_rules, rubric_text, taxonomy_text
from arxiv_digest.analysis.untrusted import UNTRUSTED_PREAMBLE
from arxiv_digest.archive.db import sha256_text
from arxiv_digest.config import Config, ModelProfile
from arxiv_digest.llm.provider import LLMError, LLMProvider, LLMUnavailable, SemanticError, run_structured

log = logging.getLogger(__name__)
PROMPT_VERSION = "triage-v1"


def system_prompt(cfg: Config) -> str:
    return f"""You are triaging new arXiv papers for a security researcher who works on reverse engineering, binary
analysis, vulnerability research and AI-assisted security tooling. Judge only from the title and abstract.
Return ONLY the JSON object: topic ids, security connection, the five integer scores and a recommendation.
Do not write explanations.

Interest taxonomy (use these exact topic ids; list 0-5 that genuinely apply, an empty list if none):
{taxonomy_text(cfg)}

{rubric_text(cfg)}

recommendation: "candidate" (worth deeper reading), "archive" (relevant but minor) or "reject" (unrelated).

{UNTRUSTED_PREAMBLE}"""


def check_triage(cfg: Config, out: TriageOutput) -> None:
    if out.recommendation != "reject" and not [t for t in out.topic_ids if t in cfg.topic_ids]:
        raise SemanticError(
            "a non-rejected paper must list at least one exact taxonomy topic id "
            f"(valid ids: {', '.join(sorted(cfg.topic_ids))})"
        )


@dataclass
class TriageResult:
    arxiv_id: str
    ok: bool
    total: int | None = None
    band: str | None = None
    topics: list[str] = field(default_factory=list)
    security_connection: str | None = None
    recommendation: str | None = None
    components: dict[str, int] = field(default_factory=dict)
    adjustments: list[str] = field(default_factory=list)
    first_attempt_valid: bool = False
    wall_s: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None


def triage_one(cfg: Config, provider: LLMProvider, profile: ModelProfile, paper) -> TriageResult:
    """Triage one paper row (dict-like with arxiv_id, latest_version, categories, title, abstract, comment). No DB writes."""
    try:
        res = run_structured(
            provider, profile, system_prompt(cfg), scr_mod.user_prompt(paper), TriageOutput,
            max_retries=int(cfg.section("llm").get("max_validation_retries", 1)),
            post_validate=lambda o: check_triage(cfg, o),
        )
    except LLMUnavailable:
        raise
    except LLMError as exc:
        return TriageResult(paper["arxiv_id"], False, error=str(exc))
    out: TriageOutput = res.value
    sc = apply_rules(cfg, out.scores, [TopicAssignment(topic_id=t, rationale="") for t in out.topic_ids],
                     out.security_connection, [])
    m = res.metrics
    return TriageResult(
        paper["arxiv_id"], True, sc.total, sc.band, [t.topic_id for t in sc.topics], out.security_connection,
        out.recommendation, sc.components, sc.adjustments, res.first_attempt_valid, m.get("sum_wall_s"),
        m.get("output_tokens"), m.get("prompt_tokens"),
    )


def run_concurrently(fn: Callable[[dict], object], papers: Sequence, workers: int = 1, label: str = "triage",
                     progress_every: int = 25) -> tuple[list, float]:
    """Apply `fn` to every paper with up to `workers` requests in flight; results keep input order.

    Ollama serves concurrent requests only when the server runs with OLLAMA_NUM_PARALLEL >= workers; otherwise they queue
    (still correct, just no faster).
    """
    t0 = time.monotonic()
    results: list = [None] * len(papers)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, p): i for i, p in enumerate(papers)}
        for fut in futures:  # iterate in submission order; each .result() blocks until that one is done
            try:
                results[futures[fut]] = fut.result()
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)  # e.g. Ollama went down: don't queue hundreds more calls
                raise
            done += 1
            if done % progress_every == 0 or done == len(papers):
                el = time.monotonic() - t0
                log.info("%s %d/%d (%.1f papers/min, eta %.1f min)", label, done, len(papers), done / el * 60,
                         (len(papers) - done) * el / done / 60)
    return results, round(time.monotonic() - t0, 2)


def passes(result: TriageResult, pass_score: int) -> bool:
    return result.ok and result.total is not None and result.total >= pass_score


# --- pipeline stage -----------------------------------------------------------------
_REC_RANK = {"candidate": 0, "archive": 1, "reject": 2}


@dataclass
class TriageStats:
    attempted: int = 0
    triaged: int = 0
    cached: int = 0
    failed: int = 0
    rechecked: int = 0
    raised_by_recheck: int = 0
    shortlist_size: int = 0
    shortlisted: int = 0
    failed_passed_through: int = 0
    cutoff_score: int | None = None
    min_score: int = 0
    first_attempt_valid: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def _input_hash(profile: ModelProfile, prompt: str) -> str:
    return sha256_text(PROMPT_VERSION + profile.model + prompt)


def _from_payload(arxiv_id: str, payload: dict) -> TriageResult:
    known = {k: payload[k] for k in TriageResult.__dataclass_fields__ if k in payload}
    known["arxiv_id"] = arxiv_id
    return TriageResult(**known)


def _store(cfg: Config, archive, run_id: int, paper, profile: ModelProfile, input_hash: str, r: TriageResult,
           extra: dict | None = None) -> None:
    payload = asdict(r) | (extra or {})
    with archive.transaction():
        if not r.ok:
            archive.add_analysis(r.arxiv_id, version=paper["latest_version"], run_id=run_id, stage="triage_failed",
                                 model=profile.model, prompt_version=PROMPT_VERSION, input_hash=input_hash,
                                 payload={"error": r.error})
            return
        archive.add_analysis(r.arxiv_id, version=paper["latest_version"], run_id=run_id, stage="triage", model=profile.model,
                             prompt_version=PROMPT_VERSION, input_hash=input_hash, payload=payload,
                             metrics={"wall_s": r.wall_s, "output_tokens": r.output_tokens, "prompt_tokens": r.prompt_tokens})
        archive.add_score(r.arxiv_id, version=paper["latest_version"], run_id=run_id, stage="triage", model=profile.model,
                          components=r.components, total=int(r.total), band=r.band, confidence=None,
                          evidence_level="abstract_only", adjustments=r.adjustments)


def triage_stored(cfg: Config, archive, provider: LLMProvider, run_id: int, arxiv_id: str, stats: TriageStats,
                  recheck_of: TriageResult | None = None) -> TriageResult:
    """Triage one archived paper, reusing a cached result for identical input; records analysis + score.

    With `recheck_of`, the cache is bypassed and the paper is triaged again; the higher of the two totals is kept
    (triage repeats differ by ~4 points on average, so a second look keeps borderline papers from being dropped).
    """
    profile = cfg.profile("triage")
    paper = archive.paper(arxiv_id)
    prompt = scr_mod.user_prompt(paper)
    h = _input_hash(profile, prompt)
    if recheck_of is None:
        stats.attempted += 1
        cached = archive.find_cached_analysis(arxiv_id, "triage", profile.model, PROMPT_VERSION, h)
        if cached:
            stats.cached += 1
            return _from_payload(arxiv_id, json.loads(cached["payload"]))
    r = triage_one(cfg, provider, profile, paper)
    if recheck_of is not None:
        stats.rechecked += 1
        if not r.ok or (r.total or 0) <= (recheck_of.total or 0):
            return recheck_of
        stats.raised_by_recheck += 1
        _store(cfg, archive, run_id, paper, profile, h, r, {"rechecked": True, "previous_total": recheck_of.total})
        return r
    _store(cfg, archive, run_id, paper, profile, h, r)
    if r.ok:
        stats.triaged += 1
        stats.first_attempt_valid += int(r.first_attempt_valid)
    else:
        stats.failed += 1
        stats.failures.append(f"{arxiv_id}: {r.error}")
    return r


def _rank_key(r: TriageResult):
    return (-(r.total or 0), -r.components.get("alignment", 0), _REC_RANK.get(r.recommendation or "reject", 2), r.arxiv_id)


def shortlist(results: list[TriageResult], size: int, min_score: int) -> tuple[list[str], int | None, list[str]]:
    """Top `size` successful triage results scoring ≥ `min_score`, best first; ties broken by alignment, then
    recommendation. Returns (shortlisted ids, cut-off score, failed ids). Failed triage is passed through separately
    (fail-open) so a model hiccup cannot silently drop a paper."""
    ok = sorted((r for r in results if r.ok and r.total is not None and r.total >= min_score), key=_rank_key)
    chosen = ok[: max(0, size)]
    cutoff = chosen[-1].total if len(chosen) >= size and chosen else None
    return [r.arxiv_id for r in chosen], cutoff, [r.arxiv_id for r in results if not r.ok]


def shortlist_size(cfg: Config, window_days: float) -> int:
    tcfg = cfg.section("triage")
    weeks = max(1, round(window_days / 7))
    return min(int(tcfg.get("shortlist_per_week", 40)) * weeks, int(tcfg.get("max_shortlist", 200)))


def triage_papers(cfg: Config, archive, provider: LLMProvider, run_id: int, arxiv_ids: list[str], size: int,
                  fixed_cutoff: int | None = None) -> tuple[TriageStats, list[str]]:
    """Triage every paper, re-check those just below the cut-off, and return (stats, ids to fully screen).

    `fixed_cutoff` (used for adaptive-search additions) admits every paper scoring at least that much, up to `size`,
    instead of taking a top-N.
    """
    tcfg = cfg.section("triage")
    min_score = int(tcfg.get("min_score", 40))
    margin = int(tcfg.get("recheck_margin", 10))
    recheck_max = int(tcfg.get("recheck_max", 60))
    stats = TriageStats(min_score=min_score, shortlist_size=size)
    t0 = time.monotonic()
    results: dict[str, TriageResult] = {}
    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        results[arxiv_id] = triage_stored(cfg, archive, provider, run_id, arxiv_id, stats)
        if i % 50 == 0 or i == len(arxiv_ids):
            el = time.monotonic() - t0
            log.info("triage %d/%d (new=%d cached=%d failed=%d) %.1f papers/min", i, len(arxiv_ids), stats.triaged,
                     stats.cached, stats.failed, i / el * 60 if el else 0)

    def pick() -> tuple[list[str], int | None, list[str]]:
        if fixed_cutoff is None:
            return shortlist(list(results.values()), size, min_score)
        floor = max(min_score, fixed_cutoff)
        ids, _, failed = shortlist([r for r in results.values() if not r.ok or (r.total or 0) >= floor], size, floor)
        return ids, floor, failed

    chosen, cutoff, failed = pick()
    if margin > 0 and results:
        bar = cutoff if cutoff is not None else min_score
        chosen_set = set(chosen)
        near = sorted((r for r in results.values() if r.ok and r.arxiv_id not in chosen_set and r.total is not None
                       and bar - margin <= r.total <= bar), key=_rank_key)[:recheck_max]
        for r in near:
            results[r.arxiv_id] = triage_stored(cfg, archive, provider, run_id, r.arxiv_id, stats, recheck_of=r)
        if near:
            chosen, cutoff, failed = pick()
    stats.shortlisted = len(chosen)
    stats.cutoff_score = cutoff
    stats.failed_passed_through = len(failed)
    stats.seconds = round(time.monotonic() - t0, 1)
    log.info("triage done: %d papers, shortlist %d (cut-off %s), %d re-checked (%d raised), %d failed → screened anyway",
             len(arxiv_ids), len(chosen), cutoff, stats.rechecked, stats.raised_by_recheck, len(failed))
    return stats, chosen + failed
