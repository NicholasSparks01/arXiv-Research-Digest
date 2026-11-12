"""Collection report (Phase 1-2): exactly what discovery found, before any LLM analysis. Also the reconciliation report."""

from __future__ import annotations

from collections import Counter

from arxiv_digest.archive.db import Archive
from arxiv_digest.archive.legacy import ReconciliationResult
from arxiv_digest.config import Config
from arxiv_digest.ids import abs_url
from arxiv_digest.timeutil import Window, utcnow_iso

DISCOVERY_KINDS = ("category", "topic", "adaptive")


def _esc(s: str | None) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_collection(cfg: Config, archive: Archive, run_id: int, window: Window, status: str, notes: list[str]) -> tuple[str, dict]:
    searches = archive.searches_for_run(run_id)
    unique = archive.run_paper_ids(run_id, DISCOVERY_KINDS)
    rows = archive.conn.execute(
        "SELECT arxiv_id, MAX(is_new) AS is_new, MAX(is_revised) AS is_revised, MAX(version_seen) AS v FROM run_papers WHERE run_id=? GROUP BY arxiv_id",
        (run_id,),
    ).fetchall()
    flags = {r["arxiv_id"]: r for r in rows}
    new = [i for i in unique if flags[i]["is_new"]]
    revised = [r["arxiv_id"] for r in rows if r["is_revised"]]
    prim = Counter(archive.paper(i)["primary_category"] for i in unique)
    sources = archive.run_paper_sources(run_id)
    topic_hits = sorted(i for i, s in sources.items() if any(x.startswith(("topic:", "adaptive:")) for x in s))
    in_scope_cats = set(cfg.categories)

    out = [
        "---",
        f'title: "arXiv Collection Report — run {run_id}"',
        f"run_id: {run_id}",
        f"status: {status}",
        f"collection_window: {window.start.isoformat()} to {window.end.isoformat()}",
        f"generated: {utcnow_iso()}",
        f"unique_papers: {len(unique)}",
        "tags: [arxiv-digest, collection]",
        "---",
        "",
        f"# Collection report — run {run_id}",
        "",
        f"- **Window:** {window.start:%Y-%m-%d %H:%M %Z} → {window.end:%Y-%m-%d %H:%M %Z} ({window.reason})",
        f"- **Research week:** {window.label()}",
        f"- **Status:** {status}",
        f"- **Unique papers discovered:** {len(unique)}",
        f"- **New to the archive:** {len(new)} · **Already known:** {len(unique) - len(new)} · **Revised versions:** {len(revised)}",
        f"- **Found by topic/adaptive searches:** {len(topic_hits)}",
        "",
        "## Searches",
        "",
        "| # | Kind | Search | Total | Fetched | New | Revised | Pages | Status |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for n, s in enumerate(searches, start=1):
        out.append(f"| {n} | {s['kind']} | {_esc(s['name'])} | {s['total_results'] if s['total_results'] is not None else '—'} | "
                   f"{s['fetched']} | {s['new_papers']} | {s['revised']} | {s['pages']} | {s['status']}{' — ' + _esc(s['error']) if s['error'] else ''} |")
    out += ["", "## Primary-category breakdown", "", "| Primary category | Papers | In discovery scope |", "|---|---:|---|"]
    for cat, c in prim.most_common():
        out.append(f"| {cat} | {c} | {'yes' if cat in in_scope_cats else 'cross-listed'} |")
    out += ["", "## Topic-search discoveries", ""]
    by_search: dict[str, list[str]] = {}
    for i in topic_hits:
        for src in sorted(sources[i]):
            if src.startswith(("topic:", "adaptive:")):
                by_search.setdefault(src.split(":", 1)[1], []).append(i)
    for name in sorted(by_search):
        ids = by_search[name]
        out += [f"### {_esc(name)} ({len(ids)})", ""]
        for i in sorted(ids):
            p = archive.paper(i)
            mark = " **(new)**" if flags[i]["is_new"] else ""
            out.append(f"- [{i}]({abs_url(i)}) — {_esc(p['title'])} · {p['primary_category']}{mark}")
        out.append("")
    if revised:
        out += ["## Revised versions", ""]
        for i in revised:
            out.append(f"- [{i}]({abs_url(i)}) → v{flags[i]['v']} — {_esc(archive.paper(i)['title'])}")
        out.append("")
    problems = [s for s in searches if s["status"] in ("failed", "incomplete", "saturated")]
    out += ["## Integrity", ""]
    out.append("- Searches with problems: " + (", ".join(f"{_esc(s['name'])} ({s['status']})" for s in problems) if problems else "none"))
    out += [f"- {n}" for n in notes]
    out.append("")
    summary = {
        "unique": len(unique), "new": len(new), "revised": len(revised), "topic_hits": len(topic_hits),
        "searches": len(searches), "problem_searches": len(problems), "paper_ids": unique,
    }
    return "\n".join(out), summary


def validate_collection(text: str, summary: dict, run_id: int, searches_count: int) -> list[str]:
    errors = []
    if f"\nrun_id: {run_id}\n" not in text:
        errors.append("run_id missing")
    if f"\nunique_papers: {summary['unique']}\n" not in text:
        errors.append("unique paper count mismatch")
    table_rows = [ln for ln in text.split("## Searches", 1)[-1].split("## Primary", 1)[0].splitlines() if ln.startswith("| ") and not ln.startswith("| #")]
    if len(table_rows) != searches_count:
        errors.append(f"search table has {len(table_rows)} rows, expected {searches_count}")
    return errors


def render_reconciliation(res: ReconciliationResult, run_id: int, source: str) -> str:
    out = [
        "---",
        f'title: "Legacy archive reconciliation — run {run_id}"',
        f"run_id: {run_id}",
        f"generated: {utcnow_iso()}",
        f"legacy_unique_papers: {len(res.unique_ids)}",
        "tags: [arxiv-digest, reconciliation]",
        "---",
        "",
        f"# Legacy archive reconciliation — run {run_id}",
        "",
        f"- **Source:** `{source}` ({len(res.files)} Markdown file(s))",
        f"- **Rows referencing arXiv papers:** {len(res.entries)}",
        f"- **Unique papers:** {len(res.unique_ids)}" + (
            f" (expected {res.expected}: {'✔ match' if res.count_matches_expected else '✘ MISMATCH'})" if res.expected is not None else ""),
        f"- **Imported (new to SQLite):** {len(res.imported)} · **Already present:** {len(res.matched)}",
        f"- **Legacy scores imported:** {res.scores_imported}",
        f"- **Not found on arXiv (placeholder kept):** {len(res.not_found)}",
        f"- **Duplicate legacy entries:** {sum(v - 1 for v in res.duplicates.values())} across {len(res.duplicates)} paper(s)",
        f"- **Title conflicts:** {len(res.title_conflicts)} · **Score conflicts:** {len(res.score_conflicts)}",
    ]
    if res.lookup_error:
        out.append(f"- **arXiv lookup error:** {res.lookup_error}")
    out.append("")
    if res.not_found:
        out += ["## Not found on arXiv", ""] + [f"- {i}" for i in res.not_found] + [""]
    if res.duplicates:
        out += ["## Duplicates in the legacy archive", ""] + [f"- {i}: {n} entries" for i, n in sorted(res.duplicates.items())] + [""]
    if res.title_conflicts:
        out += ["## Title conflicts", "", "| arXiv | Legacy title | arXiv title |", "|---|---|---|"]
        out += [f"| {i} | {_esc(a)} | {_esc(b)} |" for i, a, b in res.title_conflicts] + [""]
    if res.score_conflicts:
        out += ["## Score conflicts (latest kept)", ""] + [f"- {i}: {v}" for i, v in sorted(res.score_conflicts.items())] + [""]
    out += ["## Papers", "", "| arXiv | Status | Legacy title | Legacy score |", "|---|---|---|---:|"]
    seen = set()
    for e in res.entries:
        if e.arxiv_id in seen:
            continue
        seen.add(e.arxiv_id)
        status = "not found" if e.arxiv_id in res.not_found else ("imported" if e.arxiv_id in res.imported else "already present")
        out.append(f"| [{e.arxiv_id}]({abs_url(e.arxiv_id)}) | {status} | {_esc(e.title)} | {e.score if e.score is not None else '—'} |")
    out.append("")
    return "\n".join(out)


def summary_line(res: ReconciliationResult) -> str:
    return (f"{len(res.unique_ids)} legacy papers ({len(res.imported)} imported, {len(res.matched)} already present, "
            f"{len(res.not_found)} not found, {len(res.title_conflicts)} title conflicts)")
