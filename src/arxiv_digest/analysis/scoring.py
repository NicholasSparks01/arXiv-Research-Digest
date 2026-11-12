"""Deterministic scoring rules applied on top of model output: maxima, totals, bands and relevance guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field

from arxiv_digest.analysis.schemas import ProjectRelevance, ScoreComponents, TopicAssignment
from arxiv_digest.config import Config

COMPONENTS = ("alignment", "applicability", "cross_domain", "novelty", "reproducibility")


@dataclass
class ScoredResult:
    components: dict[str, int]
    total: int
    band: str
    topics: list[TopicAssignment]
    projects: list[ProjectRelevance]
    adjustments: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def apply_rules(
    cfg: Config,
    scores: ScoreComponents,
    topics: list[TopicAssignment],
    security_connection: str,
    projects: list[ProjectRelevance],
) -> ScoredResult:
    scfg = cfg.section("scoring")
    g = cfg.section("guardrails")
    flags: list[str] = []
    adjustments: list[str] = []

    components: dict[str, int] = {}
    for name in COMPONENTS:
        value = getattr(scores, name)
        cap = int(scfg[f"{name}_max"])
        if value > cap:
            flags.append(f"{name} {value} exceeded max {cap}; clipped")
            value = cap
        components[name] = value

    # Topics must exist in the taxonomy; de-duplicate.
    clean_topics: list[TopicAssignment] = []
    seen: set[str] = set()
    for t in topics:
        topic = cfg.topic(t.topic_id)
        if topic is None:
            flags.append(f"unknown topic id dropped: {t.topic_id!r}")
            continue
        if t.topic_id in seen:
            continue
        if g.get("security_only_tier2", True) and topic.tier == 2 and security_connection == "none":
            flags.append(f"tier-2 topic {t.topic_id} dropped: model reported no security connection")
            continue
        seen.add(t.topic_id)
        clean_topics.append(t)

    clean_projects = []
    for p in projects:
        if p.project_id in cfg.project_ids:
            clean_projects.append(p)
        else:
            flags.append(f"unknown project id dropped: {p.project_id!r}")

    total = sum(components.values())

    # Guardrail 1: no security / RE connection → cannot be High/Critical.
    if security_connection == "none":
        cap = int(g.get("no_security_connection_cap", 69))
        if total > cap:
            adjustments.append(f"capped {total}→{cap}: no demonstrated connection to security / RE research")
            total = cap

    # Guardrail 2: generic agent papers (tool calling, orchestration) need a security connection.
    generic = set(g.get("generic_agent_topics", []))
    topic_ids = {t.topic_id for t in clean_topics}
    if topic_ids and topic_ids <= generic and security_connection not in ("direct", "ai_for_security"):
        cap = int(g.get("generic_agent_cap", 69))
        if total > cap:
            adjustments.append(f"capped {total}→{cap}: generic agent paper without security / RE application")
            total = cap

    if not clean_topics and total >= int(scfg["archive_threshold"]):
        flags.append("qualifying score but no valid taxonomy topic")

    return ScoredResult(
        components=components, total=total, band=cfg.band_for(total), topics=clean_topics, projects=clean_projects,
        adjustments=adjustments, flags=flags,
    )


def rubric_text(cfg: Config) -> str:
    s = cfg.section("scoring")
    bands = ", ".join(f"{name} ≥ {thr}" for name, thr in cfg.bands())
    return f"""Score each component as an integer within its range (Python computes the total; do not add them up yourself):
- alignment (0-{s['alignment_max']}): fit with the research taxonomy. Tier-1 topics score highest, then tier 2, then tier 3.
- applicability (0-{s['applicability_max']}): could this be used in one of the listed projects soon? Concrete techniques, tools or data score high.
- cross_domain (0-{s['cross_domain_max']}): bridges security / reverse engineering with ML, systems or hardware in a useful way.
- novelty (0-{s['novelty_max']}): new technique AND convincing evaluation. Incremental work or weak evaluation scores low.
- reproducibility (0-{s['reproducibility_max']}): released code, models, datasets or enough detail to reimplement. Unknown → low.
Bands (for context): {bands}; below {s['archive_threshold']} is not retained in the digest.
IMPORTANT: a paper about generic LLM agents, tool calling or multi-agent orchestration is NOT highly relevant just because
it discusses tools. It needs a concrete connection to security research, vulnerability analysis, or reverse engineering.
Set security_connection to "none" when no such connection exists."""


def taxonomy_text(cfg: Config) -> str:
    lines = []
    for tier in (1, 2, 3):
        label = {1: "Tier 1 - direct research interests", 2: "Tier 2 - AI-assisted security research", 3: "Tier 3 - foundational technologies"}[tier]
        lines.append(label + ":")
        lines.extend(f"  - {t.id}: {t.name}" for t in cfg.taxonomy if t.tier == tier)
    return "\n".join(lines)


def projects_text(cfg: Config) -> str:
    return "\n".join(f"  - {p.id}: {p.name}" for p in cfg.projects)
