"""Weekly digest: paper selection, Markdown rendering (Obsidian-friendly) and integrity validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import timedelta

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.ids import abs_url, pdf_url
from arxiv_digest.timeutil import Window, utcnow_iso

SECTIONS = [
    "01 — Executive summary",
    "02 — Research coverage",
    "03 — Top research papers",
    "04 — Implementation opportunities",
    "05 — Emerging trends and research gaps",
    "06 — Open-source resources",
    "07 — Collection integrity",
]
EVIDENCE_LABEL = {
    "abstract_only": "Abstract screened",
    "full_text_reviewed": "Full-text reviewed",
    "independently_corroborated": "Independently corroborated",
    "legacy": "Legacy archive",
}
MARKER_RE = re.compile(r"<!-- paper:(\S+?) -->")
DISCOVERY_KINDS = ("category", "topic", "adaptive")


def previously_featured(archive: Archive) -> set[str]:
    rows = archive.conn.execute(
        "SELECT r.paper_ids FROM reports r JOIN runs u ON u.id = r.run_id WHERE r.kind = 'digest' AND r.validated = 1 AND u.status = 'SUCCESS'"
    ).fetchall()
    return {i for r in rows for i in json.loads(r["paper_ids"])}


def select_digest_papers(cfg: Config, archive: Archive, run_id: int) -> list[str]:
    """Top 5-10: every High/Critical paper up to max, topped up to min with the best remaining qualifying papers.

    "Best" is the comparative ranking recorded for this run when there is one (scores from a small model tie heavily),
    otherwise the score.
    """
    from arxiv_digest.analysis.rerank import run_ranks

    ranks = run_ranks(archive, run_id)
    rcfg = cfg.section("report")
    lo, hi = int(rcfg.get("min_papers", 5)), int(rcfg.get("max_papers", 10))
    min_score, high = int(rcfg.get("min_digest_score", 40)), int(rcfg.get("high_score", 70))
    featured = previously_featured(archive)
    pool = []
    for arxiv_id in archive.run_paper_ids(run_id, DISCOVERY_KINDS):
        if arxiv_id in featured:
            continue
        s = archive.current_score(arxiv_id)
        if s and s.stage not in ("legacy", "triage") and s.total >= min_score:
            pool.append((s.total, arxiv_id))
    pool.sort(key=lambda x: (ranks.get(x[1], 10**6), -x[0], x[1]))
    chosen = [i for t, i in pool if t >= high][:hi]
    for t, i in pool:
        if len(chosen) >= lo:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen


@dataclass
class RunFacts:
    """Everything the report needs that is not in the per-paper records."""

    run_id: int
    window: Window
    status: str  # SUCCESS | PARTIAL (provisional, set before validation)
    searches: list = field(default_factory=list)
    screening: dict = field(default_factory=dict)
    triage: dict = field(default_factory=dict)
    rerank: dict = field(default_factory=dict)
    investigation: dict = field(default_factory=dict)
    prefilter: dict = field(default_factory=dict)
    synthesis: dict | None = None
    synthesis_flags: list[str] = field(default_factory=list)
    adaptive_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reconciliation: dict | None = None


def _esc(s: str | None) -> str:
    return (s or "").replace("\n", " ").strip()


def _ids_link(ids: list[str], numbering: dict[str, int]) -> str:
    refs = [f"Paper {numbering[i]}" if i in numbering else f"[{i}]({abs_url(i)})" for i in ids]
    return f" ({', '.join(refs)})" if refs else ""


def render_paper(cfg: Config, archive: Archive, n: int, arxiv_id: str) -> tuple[str, list[dict]]:
    p = archive.paper(arxiv_id)
    s = archive.current_score(arxiv_id)
    inv = archive.latest_analysis(arxiv_id, "investigation")
    scr = archive.latest_analysis(arxiv_id, "screening")
    gh = archive.latest_analysis(arxiv_id, "github")
    ver = archive.latest_analysis(arxiv_id, "verification")
    repos = json.loads(gh["payload"]).get("repos", []) if gh else []
    lines = [f"### Paper {n}: {_esc(p['title'])}", f"<!-- paper:{arxiv_id} -->", ""]
    conf = (s.confidence or "unknown").capitalize()
    lines.append(f"**{s.band} · {s.total}**  ")
    lines.append(f"{p['primary_category']} · {EVIDENCE_LABEL.get(s.evidence_level, s.evidence_level)} · {conf} confidence")
    topics = archive.current_topics(arxiv_id)
    if topics:
        lines.append(f"Topics: {', '.join(cfg.topic(t).name if cfg.topic(t) else t for t in topics)}")
    lines.append("")
    if inv:
        d = json.loads(inv["payload"])
        demonstrated = [c for c in d.get("claims", []) if c.get("status") == "demonstrated"]
        claimed = [c for c in d.get("claims", []) if c.get("status") != "demonstrated"]
        exps = [e for e in d.get("experiments", []) if e.get("reported_result")]
        lines.append(f"- **Research problem and methodology:** {_esc(d.get('summary'))} {_esc(d.get('methodology'))}")
        lines.append(f"- **What is technically interesting:** {_esc(d.get('technically_interesting'))}")
        ev = []
        for e in exps[:4]:
            pg = f" (p. {', '.join(map(str, e['pages']))})" if e.get("pages") else ""
            ev.append(f"{_esc(e.get('dataset_or_benchmark') or e.get('description'))}: {_esc(e.get('metric'))} {_esc(e['reported_result'])}{pg}")
        repro = d.get("reproducibility", {})
        lines.append(f"- **Evaluation and reproducibility:** " + ("; ".join(ev) + ". " if ev else "")
                     + f"Code available: {repro.get('code_available', 'unclear')}. {_esc(repro.get('notes'))}")
        if demonstrated:
            lines.append("  - Demonstrated: " + "; ".join(
                f"{_esc(c['statement'])}" + (f" (p. {', '.join(map(str, c['pages']))})" if c.get("pages") else "") for c in demonstrated[:4]))
        if claimed:
            lines.append("  - Claimed, not demonstrated: " + "; ".join(_esc(c["statement"]) for c in claimed[:3]))
        lims = d.get("limitations", [])
        if lims:
            lines.append("  - Limitations: " + "; ".join(
                f"{_esc(l['text'])}{' (authors)' if l.get('source') == 'stated_by_authors' else ' (analyst)'}" for l in lims[:3]))
        rel = d.get("project_relevance", [])
        why = "; ".join(f"{cfg_project(cfg, r['project_id'])}: {_esc(r['explanation'])}" for r in rel[:3]) or _esc(d.get("security_connection_explanation"))
        lines.append(f"- **Why it matters to your research:** {why}")
        if ver:
            vd = json.loads(ver["payload"])
            if vd.get("challenges"):
                lines.append("  - Reviewer challenges: " + "; ".join(f"[{c['severity']}] {_esc(c['concern'])}" for c in vd["challenges"][:3]))
    elif scr:
        d = json.loads(scr["payload"])
        lines.append(f"- **Research problem:** {_esc(d.get('research_problem'))}")
        lines.append(f"- **Proposed approach:** {_esc(d.get('proposed_solution'))}")
        if d.get("techniques"):
            lines.append(f"- **Techniques:** {', '.join(_esc(t) for t in d['techniques'][:8])}")
        rel = d.get("applicable_projects", [])
        why = "; ".join(f"{cfg_project(cfg, r['project_id'])}: {_esc(r['explanation'])}" for r in rel[:3]) or _esc(d.get("security_connection_explanation"))
        lines.append(f"- **Why it matters to your research:** {why}")
        lines.append("- *Evidence: abstract only — results and artifacts not yet verified against the full text.*")
    links = [f"[arXiv]({abs_url(arxiv_id)})", f"[PDF]({pdf_url(arxiv_id)})"]
    annotated = cfg.reports_dir / "Annotated" / f"{arxiv_id.replace('/', '_')}-annotated.pdf"
    if annotated.exists():
        links.append(f"[Annotated PDF](Annotated/{annotated.name})")
    for r in repos:
        if r.get("status") == "verified":
            links.append(f"[{r['owner']}/{r['repo']}]({r['url']}) (verified, ★{r.get('stars', '?')})")
        elif r.get("status") == "not_found":
            links.append(f"~~{r['owner']}/{r['repo']}~~ (repository not found)")
    lines.append(f"- **Links:** {' · '.join(links)}")
    lines.append("")
    return "\n".join(lines), repos


def cfg_project(cfg: Config, pid: str) -> str:
    return next((p.name for p in cfg.projects if p.id == pid), pid)


def _triage_line(t: dict, prefilter: dict) -> str:
    cut = f", cut-off score {t['cutoff_score']}" if t.get("cutoff_score") is not None else f", every paper ≥{t.get('min_score')}"
    return (f"- **Triaged by local model:** {t.get('triaged', 0)} new + {t.get('cached', 0)} cached of {t.get('attempted', 0)} selected"
            f" (pre-filter skipped {prefilter.get('skipped', 0)}); {t.get('failed', 0)} failed."
            f" Shortlist for full screening: {t.get('shortlisted', 0)} of {t.get('shortlist_size', 0)} places{cut};"
            f" {t.get('rechecked', 0)} borderline paper(s) triaged twice")


def render_digest(cfg: Config, archive: Archive, facts: RunFacts, paper_ids: list[str]) -> str:
    w = facts.window
    week_of = w.research_week_end + timedelta(days=1)
    numbering = {pid: n for n, pid in enumerate(paper_ids, start=1)}
    out: list[str] = [
        "---",
        f'title: "arXiv Research Digest — Week of {week_of:%B} {week_of.day}, {week_of.year}"',
        f"run_id: {facts.run_id}",
        f"status: {facts.status}",
        f"research_week: {w.research_week_start.isoformat()} to {w.research_week_end.isoformat()}",
        f"collection_window: {w.start.isoformat()} to {w.end.isoformat()}",
        f"generated: {utcnow_iso()}",
        f"papers: {len(paper_ids)}",
        "tags: [arxiv-digest, research-digest]",
        "---",
        "",
        "# arXiv Research Digest",
        "",
        f"**Week of {week_of:%B} {week_of.day}, {week_of.year}** · Research window · {w.label()}",
        "",
    ]
    if facts.status != "SUCCESS":
        out += [f"> [!warning] Run status: {facts.status}", "> " + " ".join(facts.warnings[:5]) if facts.warnings else "> See section 07.", ""]

    syn = facts.synthesis or {}
    # 01
    out += [f"## {SECTIONS[0]}", ""]
    if syn.get("executive_summary"):
        for item in syn["executive_summary"]:
            out.append(f"- **{_esc(item['headline'])}** — {_esc(item['detail'])}{_ids_link(item.get('arxiv_ids', []), numbering)}")
    elif paper_ids:
        out.append("*Synthesis unavailable; top papers by score:*")
        for pid in paper_ids[:5]:
            s = archive.current_score(pid)
            out.append(f"- Paper {numbering[pid]}: {_esc(archive.paper(pid)['title'])} ({s.band} · {s.total})")
    else:
        out.append("*No papers met the digest threshold this week.*")
    out.append("")

    # 02
    out += [f"## {SECTIONS[1]}", ""]
    disc = [s for s in facts.searches if s["kind"] in DISCOVERY_KINDS]
    unique = len(archive.run_paper_ids(facts.run_id, DISCOVERY_KINDS))
    new = archive.conn.execute(
        "SELECT COUNT(DISTINCT arxiv_id) FROM run_papers WHERE run_id = ? AND is_new = 1", (facts.run_id,)
    ).fetchone()[0]
    sc = facts.screening
    out += [
        f"- **Collection window:** {w.start:%Y-%m-%d %H:%M %Z} → {w.end:%Y-%m-%d %H:%M %Z} ({w.reason})",
        f"- **Categories searched:** {', '.join(cfg.categories)}",
        f"- **Papers discovered:** {unique} unique ({new} new to the archive)",
        *([_triage_line(facts.triage, facts.prefilter)] if facts.triage else []),
        f"- **Screened by local model:** {sc.get('screened', 0)} new + {sc.get('cached', 0)} cached of {sc.get('attempted', 0)}"
        f" {'shortlisted' if facts.triage else 'selected'}"
        + ("" if facts.triage else f" (pre-filter skipped {facts.prefilter.get('skipped', 0)})") + f"; {sc.get('failed', 0)} failed",
        *([f"- **Comparative ranking:** {facts.rerank.get('pool', 0)} screened papers ordered by the local model in"
            f" {facts.rerank.get('calls', 0)} comparisons ({facts.rerank.get('failed_calls', 0)} failed); this order, not the score,"
            f" picks the full-text candidates and the papers below"] if facts.rerank else []),
        f"- **Full-text investigations:** {facts.investigation.get('investigated', 0)} new + {facts.investigation.get('cached', 0)} cached"
        f" of {facts.investigation.get('candidates', 0)} candidates",
        "",
        "| Search | Kind | Total | Fetched | New | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s in disc:
        out.append(f"| {_esc(s['name'])} | {s['kind']} | {s['total_results'] if s['total_results'] is not None else '—'} | {s['fetched']} | {s['new_papers']} | {s['status']} |")
    out.append("")

    # 03
    out += [f"## {SECTIONS[2]}", ""]
    all_repos: dict[str, list[dict]] = {}
    if not paper_ids:
        out += ["*No qualifying papers.*", ""]
    for pid in paper_ids:
        block, repos = render_paper(cfg, archive, numbering[pid], pid)
        out.append(block)
        all_repos[pid] = repos
    out.append(f"*The complete qualifying-paper index is in the master archive ([[{cfg.archive_path.stem}]]).*")
    out.append("")

    # 04
    out += [f"## {SECTIONS[3]}", ""]
    exps = syn.get("experiments") or []
    if exps:
        for i, e in enumerate(exps, start=1):
            pts = ", ".join(cfg_project(cfg, p) for p in e.get("integration_points", [])) or "—"
            out += [f"### Experiment {i}: {_esc(e['title'])}", "",
                    f"- **Objective:** {_esc(e['objective'])}{_ids_link(e.get('arxiv_ids', []), numbering)}",
                    f"- **Integration points:** {pts}",
                    f"- **Complexity:** {e['complexity']}",
                    "- **First validation steps:**"]
            out += [f"  {k}. {_esc(step)}" for k, step in enumerate(e.get("first_validation_steps", []), start=1)]
            out.append("")
    else:
        out += ["*No experiments proposed (synthesis unavailable or no suitable papers).*", ""]

    # 05
    out += [f"## {SECTIONS[4]}", ""]
    if syn.get("trends") or syn.get("research_gaps") or syn.get("comparisons"):
        if syn.get("trends"):
            out.append("**Trends**")
            out += [f"- **{_esc(t['trend'])}** — {_esc(t['explanation'])}{_ids_link(t.get('arxiv_ids', []), numbering)}" for t in syn["trends"]]
            out.append("")
        if syn.get("comparisons"):
            out.append("**Competing approaches**")
            out += [f"- **{_esc(c['theme'])}** — {_esc(c['comparison'])}{_ids_link(c.get('arxiv_ids', []), numbering)}" for c in syn["comparisons"]]
            out.append("")
        if syn.get("research_gaps"):
            out.append("**Research gaps**")
            out += [f"- **{_esc(g['gap'])}** — {_esc(g['rationale'])}{_ids_link(g.get('arxiv_ids', []), numbering)}" for g in syn["research_gaps"]]
            out.append("")
    else:
        out += ["*Synthesis unavailable.*", ""]

    # 06
    out += [f"## {SECTIONS[5]}", ""]
    verified = [(pid, r) for pid, rs in all_repos.items() for r in rs if r.get("status") == "verified"]
    if verified:
        out += ["| Repository | Paper | Stars | Last push | License | Notes |", "|---|---|---:|---|---|---|"]
        for pid, r in verified:
            notes = ", ".join(x for x in ("archived" if r.get("archived") else "", "fork" if r.get("fork") else "") if x) or _esc(r.get("description"))[:80]
            out.append(f"| [{r['owner']}/{r['repo']}]({r['url']}) | Paper {numbering[pid]} | {r.get('stars', '—')} | {(r.get('pushed_at') or '—')[:10]} | {r.get('license') or '—'} | {notes} |")
    else:
        out.append("*No verified repositories among this week's top papers.*")
    unverified = [(pid, r) for pid, rs in all_repos.items() for r in rs if r.get("status") != "verified"]
    if unverified:
        out.append("")
        out.append("Unverified references: " + "; ".join(f"{r['owner']}/{r['repo']} ({r['status']})" for _, r in unverified))
    out.append("")

    # 07
    out += [f"## {SECTIONS[6]}", ""]
    revised = archive.conn.execute(
        "SELECT DISTINCT rp.arxiv_id, rp.version_seen FROM run_papers rp WHERE rp.run_id = ? AND rp.is_revised = 1 ORDER BY rp.arxiv_id",
        (facts.run_id,),
    ).fetchall()
    problems = [s for s in facts.searches if s["status"] in ("failed", "incomplete", "saturated")]
    out.append(f"- **Run status (pre-commit):** {facts.status}")
    if facts.reconciliation:
        r = facts.reconciliation
        out.append(f"- **Archive reconciliation:** {r.get('summary')}")
    out.append(f"- **New papers:** {new} · **Revised versions detected:** {len(revised)}")
    for row in revised[:20]:
        out.append(f"  - [{row['arxiv_id']}]({abs_url(row['arxiv_id'])}) → v{row['version_seen']} — {_esc(archive.paper(row['arxiv_id'])['title'])}")
    if problems:
        out.append("- **Incomplete, failed or saturated searches:**")
        out += [f"  - {_esc(s['name'])}: {s['status']}" + (f" — {_esc(s['error'])}" if s["error"] else f" (total {s['total_results']}, fetched {s['fetched']})") for s in problems]
    else:
        out.append("- **Searches:** all completed without saturation.")
    limits = []
    if facts.rerank.get("failed_calls"):
        limits.append(f"{facts.rerank['failed_calls']} ranking comparison(s) failed (those groups kept their score order)")
    if facts.triage.get("failed"):
        limits.append(f"{facts.triage['failed']} triage failure(s) (those papers were fully screened anyway)")
    if sc.get("failed"):
        limits.append(f"{sc['failed']} screening failure(s)")
    if facts.prefilter.get("over_cap"):
        limits.append(f"{facts.prefilter['over_cap']} papers beyond the screening cap were not screened")
    inv = facts.investigation
    if inv.get("pdf_failures"):
        limits.append(f"PDF unavailable for {len(inv['pdf_failures'])} candidate(s)")
    if inv.get("llm_failures"):
        limits.append(f"{len(inv['llm_failures'])} investigation failure(s)")
    if inv.get("truncated"):
        limits.append(f"{len(inv['truncated'])} paper(s) exceeded the context budget (body pages truncated)")
    if inv.get("injection_flags") or sc.get("injection_flags"):
        limits.append("possible prompt-injection text detected in: " + ", ".join(inv.get("injection_flags", [])) if inv.get("injection_flags") else f"{sc.get('injection_flags')} abstract(s) with injection-like text")
    limits += facts.synthesis_flags[:5] + facts.adaptive_flags[:4] + facts.warnings
    out.append("- **Analysis limitations:** " + ("; ".join(limits) if limits else "none recorded"))
    abstract_only = [pid for pid in paper_ids if archive.current_score(pid).evidence_level == "abstract_only"]
    if abstract_only:
        out.append(f"- **Abstract-only entries:** {', '.join(f'Paper {numbering[p]}' for p in abstract_only)} (not full-text reviewed)")
    out.append("")
    out.append(f"*Generated locally by arxiv-research-digest · run {facts.run_id} · scores and evidence levels are recorded in SQLite.*")
    out.append("")
    return "\n".join(out)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    paper_ids: list[str]


def validate_digest(text: str, expected_ids: list[str], archive: Archive, min_score: int, run_id: int) -> ValidationResult:
    errors: list[str] = []
    if not text.startswith("---\n") or f"\nrun_id: {run_id}\n" not in text:
        errors.append("frontmatter missing or run_id mismatch")
    pos = -1
    for heading in SECTIONS:
        idx = text.find(f"\n## {heading}\n")
        if idx < 0:
            errors.append(f"missing section: {heading}")
        elif idx < pos:
            errors.append(f"section out of order: {heading}")
        else:
            pos = idx
    # Non-empty sections
    parts = re.split(r"\n## ", text)
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        if not body.strip():
            errors.append(f"empty section: {title}")
    markers = MARKER_RE.findall(text)
    dupes = sorted({m for m in markers if markers.count(m) > 1})
    if dupes:
        errors.append(f"papers appear more than once: {dupes}")
    if markers != expected_ids:
        missing = [i for i in expected_ids if i not in markers]
        extra = [i for i in markers if i not in expected_ids]
        errors.append(f"paper entries do not match selection (missing={missing}, unexpected={extra})")
    headings = re.findall(r"^### Paper (\d+): ", text, flags=re.M)
    if len(headings) != len(expected_ids) or [int(h) for h in headings] != list(range(1, len(expected_ids) + 1)):
        errors.append(f"paper headings {headings} do not match {len(expected_ids)} selected papers")
    for pid in set(markers):
        s = archive.current_score(pid)
        if s is None or s.total < min_score or s.stage == "triage":
            errors.append(f"{pid} is not a qualifying paper in the archive")
    return ValidationResult(ok=not errors, errors=errors, paper_ids=markers)
