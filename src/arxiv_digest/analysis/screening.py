"""Stage 1 — abstract screening of every selected paper (high-volume, smaller model)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from arxiv_digest.analysis.schemas import ScreeningOutput
from arxiv_digest.analysis.scoring import apply_rules, projects_text, rubric_text, taxonomy_text
from arxiv_digest.analysis.untrusted import UNTRUSTED_PREAMBLE, detect_injection, fence, sanitize
from arxiv_digest.archive.db import Archive, sha256_text
from arxiv_digest.config import Config
from arxiv_digest.llm.provider import LLMOutputError, LLMProvider, LLMUnavailable, SemanticError, run_structured

log = logging.getLogger(__name__)
PROMPT_VERSION = "screen-v1"


def system_prompt(cfg: Config) -> str:
    return f"""You are a security research analyst screening new arXiv papers for a researcher who works on
reverse engineering, binary analysis, vulnerability research and AI-assisted security tooling.
You only see the title and abstract, so your evidence level is abstract-only: describe what the abstract
claims, never invent results, numbers, datasets or code that the abstract does not mention.

Interest taxonomy (use these exact topic ids; assign 0-5 that genuinely apply):
{taxonomy_text(cfg)}

The researcher's projects (use these exact project ids for applicable_projects):
{projects_text(cfg)}

{rubric_text(cfg)}

recommendation: "candidate" (worth deeper reading), "archive" (relevant but minor) or "reject" (unrelated; explain in rejection_reason).
artifacts_mentioned: only URLs or artifact names literally present in the input.

{UNTRUSTED_PREAMBLE}"""


def user_prompt(paper) -> str:
    body = (
        f"arXiv id: {paper['arxiv_id']}v{paper['latest_version']}\n"
        f"Categories: {', '.join(json.loads(paper['categories'] or '[]'))}\n"
        f"Title: {sanitize(paper['title'])}\n\n"
        f"Abstract:\n{sanitize(paper['abstract'], 6000)}\n"
    )
    if paper["comment"]:
        body += f"\nAuthor comment: {sanitize(paper['comment'], 500)}\n"
    return "Screen this paper and return the JSON object.\n\n" + fence("PAPER METADATA", body)


def check_screening(cfg: Config, out: ScreeningOutput) -> None:
    """Semantic checks that trigger one corrective retry (structure is already schema-valid)."""
    valid = [t for t in out.topics if t.topic_id in cfg.topic_ids]
    if out.recommendation != "reject" and not valid:
        raise SemanticError(
            "a non-rejected paper must have at least one topic using an exact taxonomy id "
            f"(valid ids: {', '.join(sorted(cfg.topic_ids))})"
        )


@dataclass
class ScreeningStats:
    attempted: int = 0
    screened: int = 0
    cached: int = 0
    failed: int = 0
    qualifying: int = 0
    rejected: int = 0
    first_attempt_valid: int = 0
    injection_flags: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def screen_one(cfg: Config, archive: Archive, provider: LLMProvider, run_id: int, arxiv_id: str, stats: ScreeningStats) -> None:
    profile = cfg.profile("screener")
    paper = archive.paper(arxiv_id)
    prompt = user_prompt(paper)
    input_hash = sha256_text(PROMPT_VERSION + profile.model + prompt)
    stats.attempted += 1
    if archive.find_cached_analysis(arxiv_id, "screening", profile.model, PROMPT_VERSION, input_hash):
        stats.cached += 1
        return
    try:
        res = run_structured(
            provider, profile, system_prompt(cfg), prompt, ScreeningOutput,
            max_retries=int(cfg.section("llm").get("max_validation_retries", 1)),
            post_validate=lambda o: check_screening(cfg, o),
        )
    except LLMUnavailable:
        raise
    except LLMOutputError as exc:
        stats.failed += 1
        stats.failures.append(f"{arxiv_id}: {exc}")
        archive.add_analysis(
            arxiv_id, version=paper["latest_version"], run_id=run_id, stage="screening_failed", model=profile.model,
            prompt_version=PROMPT_VERSION, input_hash=input_hash, payload={"error": str(exc)}, metrics={"attempts": exc.attempts},
        )
        return

    out: ScreeningOutput = res.value
    scored = apply_rules(cfg, out.scores, out.topics, out.security_connection, out.applicable_projects)
    flags = list(scored.flags)

    source_text = f"{paper['title']} {paper['abstract']} {paper['comment'] or ''}".lower()
    kept_artifacts = [a for a in out.artifacts_mentioned if a and a.lower().rstrip("/.") in source_text]
    if len(kept_artifacts) < len(out.artifacts_mentioned):
        flags.append(f"dropped {len(out.artifacts_mentioned) - len(kept_artifacts)} artifact(s) not present in the abstract")
    threshold = int(cfg.section("scoring")["archive_threshold"])
    if out.recommendation == "reject" and scored.total >= threshold:
        flags.append(f"inconsistent: model recommended reject but components total {scored.total}")
    if out.recommendation != "reject" and scored.total < threshold and not out.rejection_reason:
        flags.append("below threshold without rejection reason")
    injections = detect_injection(paper["abstract"])
    if injections:
        flags.append("possible prompt-injection text in abstract")
        stats.injection_flags += 1

    payload = out.model_dump()
    payload.update(
        topics=[t.model_dump() for t in scored.topics],
        applicable_projects=[p.model_dump() for p in scored.projects],
        artifacts_mentioned=kept_artifacts,
        computed={"total": scored.total, "band": scored.band, "components": scored.components, "adjustments": scored.adjustments},
        injection_snippets=injections,
    )
    with archive.transaction():
        archive.add_analysis(
            arxiv_id, version=paper["latest_version"], run_id=run_id, stage="screening", model=res.model,
            prompt_version=PROMPT_VERSION, input_hash=input_hash, payload=payload, flags=flags, metrics=res.metrics,
        )
        archive.add_topics(arxiv_id, [(t.topic_id, cfg.topic(t.topic_id).tier, t.rationale) for t in scored.topics], "screening", run_id)
        archive.add_score(
            arxiv_id, version=paper["latest_version"], run_id=run_id, stage="screening", model=res.model,
            components=scored.components, total=scored.total, band=scored.band, confidence=out.confidence,
            evidence_level="abstract_only", adjustments=scored.adjustments,
        )
    stats.screened += 1
    stats.first_attempt_valid += int(res.first_attempt_valid)
    if scored.total >= threshold:
        stats.qualifying += 1
    if out.recommendation == "reject":
        stats.rejected += 1


def screen_papers(cfg: Config, archive: Archive, provider: LLMProvider, run_id: int, arxiv_ids: list[str]) -> ScreeningStats:
    stats = ScreeningStats()
    t0 = time.monotonic()
    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        screen_one(cfg, archive, provider, run_id, arxiv_id, stats)
        if i % 25 == 0 or i == len(arxiv_ids):
            elapsed = time.monotonic() - t0
            done_new = max(1, stats.screened + stats.failed)
            eta = (elapsed / done_new) * (len(arxiv_ids) - i) if stats.screened else 0
            log.info("screening %d/%d (new=%d cached=%d failed=%d qualifying=%d) eta=%.0f min",
                     i, len(arxiv_ids), stats.screened, stats.cached, stats.failed, stats.qualifying, eta / 60)
    stats.seconds = round(time.monotonic() - t0, 1)
    return stats
