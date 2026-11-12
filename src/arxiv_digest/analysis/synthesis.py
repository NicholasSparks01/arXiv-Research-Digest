"""Stage 3 — research synthesis across the week's top papers, plus adaptive-search proposals and optional verification."""

from __future__ import annotations

import json
import logging
import re

from arxiv_digest.analysis.schemas import AdaptiveOutput, SynthesisOutput, VerificationOutput
from arxiv_digest.analysis.scoring import projects_text, taxonomy_text
from arxiv_digest.analysis.untrusted import UNTRUSTED_PREAMBLE, fence, sanitize
from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.llm.provider import LLMOutputError, LLMProvider, SemanticError, run_structured

log = logging.getLogger(__name__)
SYNTH_PROMPT_VERSION = "synth-v2"
ADAPTIVE_PROMPT_VERSION = "adaptive-v1"
VERIFY_PROMPT_VERSION = "verify-v1"
_PHRASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+\-]{1,40}$")


def paper_brief(archive: Archive, arxiv_id: str) -> dict:
    """Compact, validated record of a paper for cross-paper reasoning (no raw PDF text)."""
    p = archive.paper(arxiv_id)
    s = archive.current_score(arxiv_id)
    brief = {
        "arxiv_id": arxiv_id,
        "title": p["title"],
        "score": s.total if s else None,
        "band": s.band if s else None,
        "evidence_level": s.evidence_level if s else None,
        "topics": archive.current_topics(arxiv_id),
    }
    inv = archive.latest_analysis(arxiv_id, "investigation")
    scr = archive.latest_analysis(arxiv_id, "screening")
    if inv:
        d = json.loads(inv["payload"])
        brief.update(
            summary=d.get("summary"),
            methodology=d.get("methodology"),
            demonstrated=[c["statement"] for c in d.get("claims", []) if c.get("status") == "demonstrated"][:6],
            claimed_only=[c["statement"] for c in d.get("claims", []) if c.get("status") != "demonstrated"][:4],
            limitations=[lim["text"] for lim in d.get("limitations", [])][:5],
            artifacts=d.get("reproducibility", {}).get("artifact_urls", [])[:4],
            implementation_opportunity=d.get("implementation_opportunity"),
        )
    elif scr:
        d = json.loads(scr["payload"])
        brief.update(summary=f"{d.get('research_problem')} {d.get('proposed_solution')}", techniques=d.get("techniques", []))
    else:
        brief["summary"] = p["abstract"][:1200]
    return brief


def synthesize(cfg: Config, provider: LLMProvider, archive: Archive, arxiv_ids: list[str]) -> tuple[dict, list[str], dict]:
    profile = cfg.profile("synthesizer")
    rcfg = cfg.section("report")
    briefs = [paper_brief(archive, i) for i in arxiv_ids]
    allowed = set(arxiv_ids)
    system = f"""You are the lead analyst writing the weekly synthesis of a security research digest.
You receive structured, already-validated records of this week's top papers. Work only from these records.
- executive_summary: {min(3, len(arxiv_ids))} to {rcfg.get('max_exec_summary', 5)} items, one per genuinely important development, each
  citing the arxiv_ids it is about.
- trends / comparisons / research_gaps: grounded in the papers; cite arxiv_ids from the input only.
- experiments: at most {rcfg.get('max_experiments', 3)} practical experiments the researcher could run, each with integration
  points, complexity, and concrete first validation steps. integration_points holds bare project ids from the list
  below (e.g. ["ghidra_mcp"]), never explanations; put the how-to in first_validation_steps.
Never invent results that are not in the records; prefer 'demonstrated' items over 'claimed_only' items.

Projects:
{projects_text(cfg)}

{UNTRUSTED_PREAMBLE}"""
    user = "Synthesize this week's papers.\n\n" + fence("PAPER RECORDS", sanitize(json.dumps(briefs, ensure_ascii=False, indent=1)))

    def check(out: SynthesisOutput) -> None:
        if not out.executive_summary:
            raise SemanticError("executive_summary must not be empty")

    res = run_structured(provider, profile, system, user, SynthesisOutput, int(cfg.section("llm").get("max_validation_retries", 1)), check)
    out: SynthesisOutput = res.value
    flags: list[str] = []
    data = out.model_dump()

    def clean_ids(items: list[dict], key: str = "arxiv_ids") -> list[dict]:
        kept = []
        for it in items:
            bad = [i for i in it.get(key, []) if i not in allowed]
            if bad:
                flags.append(f"removed unknown arxiv ids {bad} from synthesis item {it.get('headline') or it.get('trend') or it.get('title') or it.get('gap') or it.get('theme')!r}")
            it[key] = [i for i in it.get(key, []) if i in allowed]
            kept.append(it)
        return kept

    for section in ("executive_summary", "trends", "comparisons", "research_gaps", "experiments"):
        data[section] = clean_ids(data[section])
    if len(data["executive_summary"]) < min(3, len(arxiv_ids)):
        flags.append(f"executive summary has only {len(data['executive_summary'])} item(s)")
    for exp in data["experiments"]:
        # models often write "project_id: explanation"; keep the id when the prefix is a valid one
        exp["integration_points"] = list(dict.fromkeys(
            q if q in cfg.project_ids else q.split(":", 1)[0].strip() for q in exp["integration_points"]))
        bad = [p for p in exp["integration_points"] if p not in cfg.project_ids]
        if bad:
            flags.append(f"removed unknown project ids {bad} from experiment {exp['title']!r}")
        exp["integration_points"] = [p for p in exp["integration_points"] if p in cfg.project_ids]
    data["executive_summary"] = data["executive_summary"][: int(rcfg.get("max_exec_summary", 5))]
    data["experiments"] = data["experiments"][: int(rcfg.get("max_experiments", 3))]
    return data, flags, {"model": res.model, "prompt_version": SYNTH_PROMPT_VERSION, **res.metrics}


def _phrase_ok(p: str) -> bool:
    return bool(_PHRASE_RE.match(p.strip()))


def build_adaptive_query(must: list[str], any_of: list[str]) -> str | None:
    must = [m.strip() for m in must if _phrase_ok(m)]
    any_of = [a.strip() for a in any_of if _phrase_ok(a)]
    if not must:
        return None
    q = " AND ".join(f'all:"{m}"' for m in must)
    if any_of:
        q = f"({q}) AND (" + " OR ".join(f'all:"{a}"' for a in any_of) + ")"
    return q


def propose_adaptive(cfg: Config, provider: LLMProvider, archive: Archive, seed_ids: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """LLM proposes up to N follow-up searches from promising papers; Python builds and sanitises the arXiv query."""
    acfg = cfg.section("adaptive")
    profile = cfg.profile("screener")
    briefs = [{"arxiv_id": i, "title": archive.paper(i)["title"], "topics": archive.current_topics(i)} for i in seed_ids]
    core = "\n".join(f"  - {t.name}" for t in cfg.topic_searches)
    system = f"""You design follow-up literature searches for a security research digest.
Given this week's most promising papers, propose up to {acfg.get('max_searches', 4)} NEW searches that could find related work the
core searches below would miss (specific techniques, tools, benchmarks or problem names - not generic terms).
Each search: 1-2 must_include phrases and 0-4 any_of phrases; each phrase 1-4 plain words (letters, digits, hyphens).

Core searches already run:
{core}

Taxonomy:
{taxonomy_text(cfg)}

{UNTRUSTED_PREAMBLE}"""
    user = "Propose adaptive searches.\n\n" + fence("SEED PAPERS", sanitize(json.dumps(briefs, ensure_ascii=False)))
    flags: list[str] = []
    try:
        res = run_structured(provider, profile, system, user, AdaptiveOutput, int(cfg.section("llm").get("max_validation_retries", 1)))
    except LLMOutputError as exc:
        return [], [f"adaptive search proposal failed: {exc}"]
    out: list[tuple[str, str]] = []
    seen = set()
    for i, s in enumerate(res.value.searches[: int(acfg.get("max_searches", 4))], start=1):
        q = build_adaptive_query(s.must_include, s.any_of)
        if q is None:
            flags.append(f"adaptive search {i} rejected (invalid phrases): {s.must_include} / {s.any_of}")
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append((f"A{i:02d} {' + '.join(s.must_include)}"[:80], q))
    return out, flags


def verify_system_prompt() -> str:
    return f"""You are a skeptical reviewer. Challenge the analysis of a security paper: identify claims that are not
adequately supported, overlooked limitations, competing approaches, and assess novelty. Suggest a score adjustment
between -15 and +5. Cite pages from the analysis where possible. {UNTRUSTED_PREAMBLE}"""


def verify_user_prompt(title: str, analysis: dict) -> str:
    trimmed = {k: analysis.get(k) for k in ("summary", "methodology", "claims", "experiments", "limitations", "reproducibility", "computed")}
    return f"Paper: {title}\n\n" + fence("ANALYSIS", sanitize(json.dumps(trimmed, ensure_ascii=False)))


def verify(cfg: Config, provider: LLMProvider, archive: Archive, run_id: int, arxiv_id: str) -> tuple[dict, list[str]] | None:
    """Optional challenge pass (e.g. DeepSeek-R1) over an existing investigation."""
    inv = archive.latest_analysis(arxiv_id, "investigation")
    if inv is None:
        return None
    profile = cfg.profile("verifier")
    system = verify_system_prompt()
    user = verify_user_prompt(archive.paper(arxiv_id)["title"], json.loads(inv["payload"]))
    try:
        res = run_structured(provider, profile, system, user, VerificationOutput, int(cfg.section("llm").get("max_validation_retries", 1)))
    except LLMOutputError as exc:
        return {"error": str(exc)}, [f"verification failed: {exc}"]
    out: VerificationOutput = res.value
    flags = []
    adj = max(-15, min(5, out.score_adjustment))
    if adj != out.score_adjustment:
        flags.append(f"score adjustment {out.score_adjustment} clamped to {adj}")
    base = archive.current_score(arxiv_id)
    total = max(0, min(100, base.total + adj))
    github = archive.latest_analysis(arxiv_id, "github")
    repo_ok = bool(github and any(r.get("status") == "verified" for r in json.loads(github["payload"]).get("repos", [])))
    major = any(c.severity == "major" for c in out.challenges)
    evidence = "independently_corroborated" if out.agrees_with_analysis and not major and repo_ok else "full_text_reviewed"
    payload = {**out.model_dump(), "applied_adjustment": adj, "evidence_level": evidence}
    with archive.transaction():
        archive.add_analysis(arxiv_id, version=archive.paper(arxiv_id)["latest_version"], run_id=run_id, stage="verification",
                             model=res.model, prompt_version=VERIFY_PROMPT_VERSION, input_hash=None, payload=payload, flags=flags,
                             metrics=res.metrics)
        archive.add_score(arxiv_id, version=archive.paper(arxiv_id)["latest_version"], run_id=run_id, stage="verification",
                          model=res.model, components=base.components, total=total, band=cfg.band_for(total),
                          confidence=out.confidence, evidence_level=evidence,
                          adjustments=[f"verifier adjustment {adj:+d}"] + ([f"{len(out.challenges)} challenge(s)"] if out.challenges else []))
    return payload, flags
