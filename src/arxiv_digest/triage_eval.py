"""Measure compact triage against full screening on a stratified sample of an already-collected run.

Answers: how much faster is triage (serial and concurrent), and which full-screening High / investigation-grade papers
would a given triage pass threshold lose? Read-only with respect to the archive; results go to the Benchmarks folder.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from arxiv_digest.analysis import screening as scr_mod
from arxiv_digest.analysis import triage as tri_mod
from arxiv_digest.analysis.prefilter import select_for_screening
from arxiv_digest.analysis.schemas import ScreeningOutput
from arxiv_digest.analysis.scoring import apply_rules
from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config, ModelProfile
from arxiv_digest.llm.provider import LLMError, LLMProvider, LLMUnavailable, OllamaProvider, run_structured
from arxiv_digest.report.files import non_clobbering

log = logging.getLogger(__name__)

STRATA = ("searched", "full_category", "keyword")
DEFAULT_ALLOCATION = {"searched": 0.4, "full_category": 0.4, "keyword": 0.2}
THRESHOLDS = (20, 25, 30, 35, 40, 45, 50, 55, 60)
POSITIVE_LEVELS = (70, 60, 40)  # High+ / investigation minimum / digest minimum (full-screening total)
TODAY_S_PER_PAPER = 7.0  # serial full screening on qwen3.5:9b, model benchmark 2026-11-11


def stratum_of(reason: str) -> str:
    if reason.startswith("topic/adaptive"):
        return "searched"
    if reason.startswith("full-screen category"):
        return "full_category"
    return "keyword"


@dataclass
class EvalPaper:
    arxiv_id: str
    stratum: str
    weight: float
    title: str
    row: dict


def draw_sample(cfg: Config, archive: Archive, run_id: int, n: int, seed: int,
                allocation: dict[str, float] | None = None) -> tuple[list[EvalPaper], dict[str, int]]:
    """Stratified random sample of the papers a run would screen; weight = stratum population / stratum sample."""
    hits, _ = select_for_screening(cfg, archive, run_id)
    pop: dict[str, list[str]] = {s: [] for s in STRATA}
    for h in hits:
        pop[stratum_of(h.reason)].append(h.arxiv_id)
    rng = random.Random(seed)
    alloc = allocation or DEFAULT_ALLOCATION
    sample: list[EvalPaper] = []
    for s in STRATA:
        ids = sorted(pop[s])
        k = min(len(ids), round(n * alloc.get(s, 0)))
        if not k:
            continue
        for aid in rng.sample(ids, k):
            r = archive.paper(aid)
            row = {k2: r[k2] for k2 in ("arxiv_id", "latest_version", "categories", "title", "abstract", "comment")}
            sample.append(EvalPaper(aid, s, len(ids) / k, r["title"], row))
    return sample, {s: len(v) for s, v in pop.items()}


def reference_one(cfg: Config, provider: LLMProvider, profile: ModelProfile, paper: dict) -> tri_mod.TriageResult:
    """Today's full screening (same prompt, schema and rules as Stage 1), without DB writes."""
    try:
        res = run_structured(provider, profile, scr_mod.system_prompt(cfg), scr_mod.user_prompt(paper), ScreeningOutput,
                             max_retries=int(cfg.section("llm").get("max_validation_retries", 1)),
                             post_validate=lambda o: scr_mod.check_screening(cfg, o))
    except LLMUnavailable:
        raise
    except LLMError as exc:
        return tri_mod.TriageResult(paper["arxiv_id"], False, error=str(exc))
    out: ScreeningOutput = res.value
    sc = apply_rules(cfg, out.scores, out.topics, out.security_connection, out.applicable_projects)
    m = res.metrics
    return tri_mod.TriageResult(paper["arxiv_id"], True, sc.total, sc.band, [t.topic_id for t in sc.topics],
                                out.security_connection, out.recommendation, sc.components, sc.adjustments,
                                res.first_attempt_valid, m.get("sum_wall_s"), m.get("output_tokens"), m.get("prompt_tokens"))


# --- metrics (pure) ---------------------------------------------------------------
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def threshold_table(rows: list[tuple[float, int | None, int | None]], thresholds=THRESHOLDS,
                    levels=POSITIVE_LEVELS) -> list[dict]:
    """rows = (weight, triage_total or None if triage failed, reference_total or None if reference failed).

    A failed triage counts as a pass (fail-open: an LLM glitch must not silently drop a paper).
    Recall is population-weighted; papers whose reference failed are excluded from recall.
    """
    total_w = sum(w for w, _, _ in rows) or 1.0
    out = []
    for t in thresholds:
        passed = [(w, tri, ref) for w, tri, ref in rows if tri is None or tri >= t]
        row = {"threshold": t, "pass_fraction": round(sum(w for w, _, _ in passed) / total_w, 4)}
        for lvl in levels:
            pos = [(w, tri) for w, tri, ref in rows if ref is not None and ref >= lvl]
            hit = [(w, tri) for w, tri in pos if tri is None or tri >= t]
            pw = sum(w for w, _ in pos)
            row[f"recall_ge{lvl}"] = round(sum(w for w, _ in hit) / pw, 4) if pw else None
            row[f"missed_ge{lvl}"] = len(pos) - len(hit)
            row[f"positives_ge{lvl}"] = len(pos)
        out.append(row)
    return out


def suggest_threshold(table: list[dict], min_recall60: float = 0.95) -> int | None:
    """Highest threshold that keeps every sampled High+ paper and >= min_recall60 of investigation-grade (>=60) papers."""
    ok = [r["threshold"] for r in table
          if r.get("missed_ge70") == 0 and (r.get("recall_ge60") is None or r["recall_ge60"] >= min_recall60)]
    return max(ok) if ok else None


# --- runner -------------------------------------------------------------------------
def _residency(provider: OllamaProvider, model: str) -> dict | None:
    try:
        for m in provider.running():
            if model in (m.get("name"), m.get("model")):
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2),
                        "gpu_fraction": round(vram / size, 3) if size else None, "context_length": m.get("context_length")}
    except Exception:  # noqa: BLE001
        return None
    return None


def repeat_phase_name(workers: int) -> str:
    return f"triage_x{workers}" if workers > 1 else "triage_repeat"


def _phase_summary(name: str, workers: int, results: list[tri_mod.TriageResult], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    return {
        "phase": name, "workers": workers, "papers": len(results), "ok": len(ok),
        "first_attempt_valid": sum(r.first_attempt_valid for r in ok), "wall_clock_s": wall,
        "s_per_paper": round(wall / len(results), 2) if results else None,
        "papers_per_min": round(len(results) / wall * 60, 1) if wall else None,
        "mean_output_tokens": round(sum(r.output_tokens or 0 for r in ok) / len(ok), 1) if ok else None,
        "mean_prompt_tokens": round(sum(r.prompt_tokens or 0 for r in ok) / len(ok), 1) if ok else None,
    }


def run_triage_eval(cfg: Config, archive: Archive, run_id: int | None = None, n: int = 300, seed: int = 7, workers: int = 1,
                    skip_reference: bool = False, skip_serial: bool = False) -> Path:
    provider = OllamaProvider.from_config(cfg.section("llm"))
    provider.ensure_available()
    tri_prof, ref_prof = cfg.profile("triage"), cfg.profile("screener")
    missing = sorted({tri_prof.model, ref_prof.model} - set(provider.list_models()))
    if missing:
        raise RuntimeError(f"models not installed in Ollama: {missing}")
    if run_id is None:
        run_id = archive.conn.execute("SELECT MAX(run_id) FROM run_papers").fetchone()[0]
        if run_id is None:
            raise RuntimeError("no collected run in the archive; run `digest collect` first")
    sample, population = draw_sample(cfg, archive, run_id, n, seed)
    log.info("triage eval: run %d, sample %d of %d (%s)", run_id, len(sample), sum(population.values()), population)

    now = datetime.now()
    out_dir = cfg.benchmark_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = non_clobbering(out_dir / f"{now:%Y-%m-%d} Triage Evaluation.md", f"{now:%H%M%S}")
    json_path = md_path.with_suffix(".json")
    data: dict = {"run_id": run_id, "seed": seed, "population": population, "workers": workers,
                  "sample": [{"arxiv_id": p.arxiv_id, "stratum": p.stratum, "weight": p.weight, "title": p.title} for p in sample],
                  "phases": {}, "results": {}, "residency": {}}

    def save():
        json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    rows = [p.row for p in sample]
    tri_fn = partial(tri_mod.triage_one, cfg, provider, tri_prof)
    # Warm-up (not timed): load the model so the first phase doesn't pay the load time.
    tri_mod.triage_one(cfg, provider, tri_prof, rows[0])
    phases = []
    if not skip_serial:
        phases.append(("triage_serial", tri_fn, 1))
    # Second triage pass: run-to-run consistency (and concurrency speed-up when workers > 1).
    phases.append((repeat_phase_name(workers), tri_fn, workers))
    if not skip_reference:
        phases.append(("full_screening_reference", partial(reference_one, cfg, provider, ref_prof), workers))
    for name, fn, w in phases:
        results, wall = tri_mod.run_concurrently(fn, rows, w, label=name)
        data["phases"][name] = _phase_summary(name, w, results, wall)
        data["results"][name] = [asdict(r) for r in results]
        data["residency"][name] = _residency(provider, tri_prof.model)
        save()
        log.info("%s: %s", name, data["phases"][name])

    _write_report(cfg, data, md_path, sample)
    save()
    return md_path


def _write_report(cfg: Config, data: dict, md_path: Path, sample: list[EvalPaper]) -> None:
    phases = data["phases"]
    rep_name = repeat_phase_name(data["workers"])
    tri_name = next((n for n in ("triage_serial", rep_name) if n in data["results"]), None)
    ref = {r["arxiv_id"]: r for r in data["results"].get("full_screening_reference", [])}
    tri = {r["arxiv_id"]: r for r in data["results"].get(tri_name, [])} if tri_name else {}
    lines = ["---", f'title: "Triage Evaluation — {datetime.now():%Y-%m-%d %H:%M}"', "tags: [arxiv-digest, benchmark, triage]", "---", "",
             f"# Triage evaluation — run {data['run_id']}", "",
             f"Stratified sample of {len(sample)} papers (seed {data['seed']}) from the {sum(data['population'].values())} papers "
             f"run {data['run_id']} would screen. Recall figures are weighted back to the population.", "",
             "| Stratum | Population | Sampled | Weight |", "|---|---:|---:|---:|"]
    for s in STRATA:
        k = sum(1 for p in sample if p.stratum == s)
        lines.append(f"| {s} | {data['population'].get(s, 0)} | {k} | {round(data['population'].get(s, 0) / k, 2) if k else '—'} |")
    lines += ["", "## Speed", "", "| Phase | Concurrent requests | OK | First-attempt valid | Wall clock (s) | s / paper | Papers / min | Output tokens | Prompt tokens |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ph in phases.values():
        lines.append(f"| {ph['phase']} | {ph['workers']} | {ph['ok']}/{ph['papers']} | {ph['first_attempt_valid']} | {ph['wall_clock_s']} | "
                     f"{ph['s_per_paper']} | {ph['papers_per_min']} | {ph['mean_output_tokens']} | {ph['mean_prompt_tokens']} |")
    lines += ["", "GPU residency: " + "; ".join(f"{k}: {v['vram_gb']}/{v['size_gb']} GB, ctx {v['context_length']}"
                                               for k, v in data["residency"].items() if v), ""]

    if "triage_serial" in data["results"] and rep_name in data["results"]:
        a = {r["arxiv_id"]: r for r in data["results"]["triage_serial"]}
        b = {r["arxiv_id"]: r for r in data["results"][rep_name]}
        both = [(a[i]["total"], b[i]["total"]) for i in a if a[i]["ok"] and b.get(i, {}).get("ok")]
        if both:
            diffs = sorted(abs(x - y) for x, y in both)
            data["triage_repeat_consistency"] = {"mean_abs_diff": round(sum(diffs) / len(diffs), 2),
                                                 "p90_abs_diff": diffs[int(0.9 * (len(diffs) - 1))], "max_abs_diff": diffs[-1]}
            c = data["triage_repeat_consistency"]
            lines += [f"Triage run-to-run consistency (two passes over the same {len(both)} papers): mean |Δscore| "
                      f"{c['mean_abs_diff']}, 90th percentile {c['p90_abs_diff']}, max {c['max_abs_diff']}.", ""]

    if ref and tri:
        weights = {p.arxiv_id: p.weight for p in sample}
        rows = [(weights[i], tri[i]["total"] if tri[i]["ok"] else None, ref[i]["total"] if ref.get(i, {}).get("ok") else None)
                for i in tri]
        table = threshold_table(rows)
        sugg = suggest_threshold(table)
        pairs = [(tri[i]["total"], ref[i]["total"]) for i in tri if tri[i]["ok"] and ref.get(i, {}).get("ok")]
        rho = spearman([float(x) for x, _ in pairs], [float(y) for _, y in pairs])
        data["threshold_table"], data["suggested_threshold"], data["spearman"] = table, sugg, rho
        ph_t, ph_r = phases.get(tri_name, {}), phases.get("full_screening_reference", {})
        lines += ["## Agreement with full screening", "",
                  f"Spearman rank correlation (triage total vs full-screening total): **{rho}** over {len(pairs)} papers. "
                  f"Sampled positives: {table[0]['positives_ge70']} at ≥70, {table[0]['positives_ge60']} at ≥60, "
                  f"{table[0]['positives_ge40']} at ≥40.", "",
                  "A failed triage counts as a pass. *Projected / week* assumes ~900 screened papers; *initial* is this run's population.", "",
                  "| Triage pass ≥ | Pass rate | Projected / week | Projected initial | Recall ≥70 (missed) | Recall ≥60 (missed) | Recall ≥40 (missed) |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        pop_total = sum(data["population"].values())
        for r in table:
            def rc(lvl):
                v = r[f"recall_ge{lvl}"]
                return f"{v:.2f} ({r[f'missed_ge{lvl}']})" if v is not None else "—"
            mark = " ← suggested" if r["threshold"] == sugg else ""
            lines.append(f"| {r['threshold']}{mark} | {r['pass_fraction']:.1%} | {round(r['pass_fraction'] * 900)} | "
                         f"{round(r['pass_fraction'] * pop_total)} | {rc(70)} | {rc(60)} | {rc(40)} |")
        if sugg is not None and ph_t and ph_r:
            pass_frac = next(r["pass_fraction"] for r in table if r["threshold"] == sugg)
            for label, n_papers in (("Weekly (~900 papers)", 900), (f"Initial run ({pop_total} papers)", pop_total)):
                t_tri = n_papers * ph_t["s_per_paper"] / 60
                t_full = n_papers * pass_frac * ph_r["s_per_paper"] / 60
                t_now = n_papers * TODAY_S_PER_PAPER / 60
                lines.append("")
                lines.append(f"**{label}** at threshold {sugg}: triage ≈ {t_tri:.0f} min + full screening of "
                             f"{round(n_papers * pass_frac)} papers ≈ {t_full:.0f} min = **{t_tri + t_full:.0f} min** "
                             f"at the measured rates above (today: full screening of every paper, one at a time, "
                             f"≈ {t_now:.0f} min at {TODAY_S_PER_PAPER} s/paper).")
        cut = sugg if sugg is not None else 40
        missed = sorted(((ref[i]["total"], tri[i]["total"], i) for i in tri
                         if ref.get(i, {}).get("ok") and ref[i]["total"] >= 40 and tri[i]["ok"] and tri[i]["total"] < cut), reverse=True)
        titles = {p.arxiv_id: p.title for p in sample}
        lines += ["", f"### Papers full screening scored ≥40 that triage put below {cut}", ""]
        lines += [f"- {i} — {titles[i]} (full {r}, triage {t})" for r, t, i in missed] or ["- none"]
        over = sorted(((tri[i]["total"], ref[i]["total"], i) for i in tri
                       if ref.get(i, {}).get("ok") and tri[i]["ok"] and tri[i]["total"] >= 70 and ref[i]["total"] < 50), reverse=True)
        lines += ["", "### Triage ≥70 but full screening <50 (over-scored by triage)", ""]
        lines += [f"- {i} — {titles[i]} (triage {t}, full {r})" for t, r, i in over] or ["- none"]
    failures = [(name, r) for name, rs in data["results"].items() for r in rs if not r["ok"]]
    if failures:
        lines += ["", "## Failures", ""] + [f"- {name} · {r['arxiv_id']}: {r['error']}" for name, r in failures[:30]]
    lines += ["", f"Raw results: `{md_path.with_suffix('.json').name}`", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
