"""Repeatable model benchmark on representative papers (screening + full-text investigation).

Measures what matters for this pipeline: structured-output reliability, score stability, band/topic agreement with
provisional gold labels, guardrail behaviour on a negative control, evidence-validation quality, speed, and whether the
model stays resident in GPU memory at the configured context size.
"""

from __future__ import annotations

import json
import logging
import statistics
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from arxiv_digest.analysis import investigation as inv_mod
from arxiv_digest.analysis import screening as scr_mod
from arxiv_digest.analysis import synthesis as syn_mod
from arxiv_digest.analysis.schemas import InvestigationOutput, ScreeningOutput, VerificationOutput
from arxiv_digest.analysis.scoring import apply_rules
from arxiv_digest.config import Config, ModelProfile
from arxiv_digest.discovery.arxiv_client import ArxivClient
from arxiv_digest.llm.provider import LLMError, OllamaProvider, run_structured
from arxiv_digest.pdf import PDFError, build_context, download_pdf, extract
from arxiv_digest.report.files import non_clobbering

log = logging.getLogger(__name__)

STAGES = ("screening", "fulltext", "verification")


@dataclass
class Trial:
    model: str
    task: str
    arxiv_id: str
    ok: bool
    first_attempt_valid: bool = False
    total: int | None = None
    band: str | None = None
    topics: list[str] = field(default_factory=list)
    security_connection: str | None = None
    wall_s: float | None = None
    output_tok_per_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_s: float | None = None
    evidence: dict | None = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    payload: dict | None = None


def _profile_for(cfg: Config, bcfg: dict, base: str, model: str) -> ModelProfile:
    prof = cfg.profile(base)
    opts = bcfg.get("model_options", {}).get(model, {})
    think = opts.get("think", prof.think if prof.model == model else None)
    return replace(prof, model=model, think=think, num_ctx=int(opts.get("num_ctx", prof.num_ctx)))


def _residency(provider: OllamaProvider, model: str) -> dict | None:
    try:
        for m in provider.running():
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2),
                        "gpu_fraction": round(vram / size, 3) if size else None, "context_length": m.get("context_length")}
    except Exception:  # noqa: BLE001
        return None
    return None


def run_benchmark(cfg: Config, bench_path: Path, models: list[str] | None = None, skip_fulltext: bool = False,
                  repeats: int | None = None, stages: list[str] | None = None, verifiers: list[str] | None = None) -> Path:
    stages = list(stages or STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown benchmark stage(s) {unknown}; choose from {list(STAGES)}")
    if skip_fulltext:
        stages = [s for s in stages if s == "screening"]
    if "verification" in stages and "fulltext" not in stages:
        raise ValueError("the verification stage reviews full-text analyses, so it needs the fulltext stage too")
    with open(bench_path, "rb") as fh:
        bcfg = tomllib.load(fh)
    provider = OllamaProvider.from_config(cfg.section("llm"))
    provider.ensure_available()
    installed = set(provider.list_models())
    repeats = repeats or int(bcfg.get("repeats", 3))
    papers_cfg = bcfg["papers"]

    client = ArxivClient.from_config(cfg.section("arxiv"))
    qr = client.fetch_ids([p["arxiv_id"] for p in papers_cfg])
    meta = {p.arxiv_id: p for p in qr.papers}
    missing_meta = [p["arxiv_id"] for p in papers_cfg if p["arxiv_id"] not in meta]
    if missing_meta:
        raise RuntimeError(f"benchmark papers not found on arXiv: {missing_meta} ({qr.error or 'no error'})")

    def row(rec):
        return {"arxiv_id": rec.arxiv_id, "latest_version": rec.version, "categories": json.dumps(rec.categories),
                "title": rec.title, "abstract": rec.abstract, "comment": rec.comment}

    screen_models = models or bcfg.get("screening_models", [])
    analyst_models = models or bcfg.get("analyst_models", [])
    verifier_models = verifiers or bcfg.get("verifier_models", [cfg.profile("verifier").model])
    if "screening" not in stages:
        screen_models = []
    if "fulltext" not in stages:
        analyst_models = []
    if "verification" not in stages:
        verifier_models = []
    trials: list[Trial] = []
    residency: dict[str, dict | None] = {}
    skipped = sorted({m for m in screen_models + analyst_models + verifier_models if m not in installed})
    for m in skipped:
        log.warning("model %s is not installed in Ollama; skipping (ollama pull %s)", m, m)

    # --- screening ------------------------------------------------------------
    for model in [m for m in screen_models if m in installed]:
        profile = _profile_for(cfg, bcfg, "screener", model)
        system = scr_mod.system_prompt(cfg)
        for pc in papers_cfg:
            rec = meta[pc["arxiv_id"]]
            for r in range(repeats):
                log.info("screening %s on %s (%d/%d)", model, pc["label"], r + 1, repeats)
                try:
                    res = run_structured(provider, profile, system, scr_mod.user_prompt(row(rec)), ScreeningOutput, max_retries=1,
                                         post_validate=lambda o: scr_mod.check_screening(cfg, o))
                except LLMError as exc:
                    trials.append(Trial(model, "screening", rec.arxiv_id, False, error=str(exc)))
                    continue
                out: ScreeningOutput = res.value
                sc = apply_rules(cfg, out.scores, out.topics, out.security_connection, out.applicable_projects)
                m = res.metrics
                trials.append(Trial(model, "screening", rec.arxiv_id, True, res.first_attempt_valid, sc.total, sc.band,
                                    [t.topic_id for t in sc.topics], out.security_connection, m.get("sum_wall_s"),
                                    m.get("output_tok_per_s"), m.get("prompt_tokens"), m.get("output_tokens"), m.get("load_s"),
                                    flags=sc.flags + sc.adjustments))
            residency.setdefault(f"{model} (screening ctx {profile.num_ctx})", _residency(provider, model))

    # --- full-text investigation ---------------------------------------------
    if analyst_models:
        docs = {}
        for aid in bcfg.get("investigate_papers", []):
            try:
                path = download_pdf(aid, meta[aid].version, cfg.pdf_dir, cfg.section("arxiv")["user_agent"])
                docs[aid] = extract(path)
            except (PDFError, OSError, KeyError) as exc:
                log.warning("benchmark PDF unavailable for %s: %s", aid, exc)
        for model in [m for m in analyst_models if m in installed]:
            profile = _profile_for(cfg, bcfg, "analyst", model)
            system = inv_mod.system_prompt(cfg)
            for aid, doc in docs.items():
                ctx = build_context(doc, int(cfg.section("investigation").get("max_text_chars", 160000)))
                prompt = inv_mod.user_prompt(row(meta[aid]), ctx.text, doc.sections, doc.page_count)
                log.info("investigating %s with %s (%d chars)", aid, model, ctx.chars)
                try:
                    res = run_structured(provider, profile, system, prompt, InvestigationOutput, max_retries=1,
                                         post_validate=lambda o: inv_mod.check_investigation(cfg, o))
                except LLMError as exc:
                    trials.append(Trial(model, "investigation", aid, False, error=str(exc)))
                    continue
                out: InvestigationOutput = res.value
                payload, flags, evidence = inv_mod.validate_investigation(out, doc)
                sc = apply_rules(cfg, out.scores, out.topics, out.security_connection, out.project_relevance)
                m = res.metrics
                trials.append(Trial(model, "investigation", aid, True, res.first_attempt_valid, sc.total, sc.band,
                                    [t.topic_id for t in sc.topics], out.security_connection, m.get("sum_wall_s"),
                                    m.get("output_tok_per_s"), m.get("prompt_tokens"), m.get("output_tokens"), m.get("load_s"),
                                    evidence=evidence | {"claims": len(out.claims), "experiments": len(out.experiments),
                                                         "page_count": doc.page_count},
                                    flags=flags, payload=payload))
                residency[f"{model} (analyst ctx {profile.num_ctx})"] = _residency(provider, model)

    # --- verification (challenge pass over each analyst's full-text analysis) ---------------
    analyses = [t for t in trials if t.task == "investigation" and t.ok and t.payload]
    for model in [m for m in verifier_models if m in installed]:
        profile = _profile_for(cfg, bcfg, "verifier", model)
        system = syn_mod.verify_system_prompt()
        for inv in analyses:
            log.info("verifying %s's analysis of %s with %s", inv.model, inv.arxiv_id, model)
            try:
                res = run_structured(provider, profile, system, syn_mod.verify_user_prompt(meta[inv.arxiv_id].title, inv.payload),
                                     VerificationOutput, max_retries=1)
            except LLMError as exc:
                trials.append(Trial(model, "verification", inv.arxiv_id, False, error=f"reviewing {inv.model}: {exc}"))
                continue
            v: VerificationOutput = res.value
            pages = (inv.evidence or {}).get("page_count") or 0
            bad_pages = sum(1 for c in v.challenges for pg in c.pages if not 1 <= pg <= pages)
            m = res.metrics
            trials.append(Trial(model, "verification", inv.arxiv_id, True, res.first_attempt_valid, wall_s=m.get("sum_wall_s"),
                                output_tok_per_s=m.get("output_tok_per_s"), prompt_tokens=m.get("prompt_tokens"),
                                output_tokens=m.get("output_tokens"), load_s=m.get("load_s"),
                                evidence={"reviewed_model": inv.model, "agrees": v.agrees_with_analysis,
                                          "challenges": len(v.challenges),
                                          "major_challenges": sum(c.severity == "major" for c in v.challenges),
                                          "adjustment": max(-15, min(5, v.score_adjustment)),
                                          "adjustment_out_of_range": not -15 <= v.score_adjustment <= 5,
                                          "invalid_challenge_pages": bad_pages},
                                payload=v.model_dump()))
            residency[f"{model} (verifier ctx {profile.num_ctx})"] = _residency(provider, model)

    return _write_report(cfg, bcfg, trials, residency, skipped, meta, repeats)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def summarize(bcfg: dict, trials: list[Trial]) -> dict[str, dict]:
    labels = {p["arxiv_id"]: p for p in bcfg["papers"]}
    out: dict[str, dict] = {}
    for model in sorted({t.model for t in trials}):
        s = [t for t in trials if t.model == model and t.task == "screening"]
        ok = [t for t in s if t.ok]
        per_paper_sd = []
        band_hits = topic_recall = 0.0
        neg_ok = None
        for aid, p in labels.items():
            tt = [t for t in ok if t.arxiv_id == aid]
            if len(tt) > 1:
                per_paper_sd.append(statistics.pstdev([t.total for t in tt]))
            if tt:
                band_hits += sum(t.band in p.get("expected_bands", []) for t in tt) / len(tt)
                exp = set(p.get("expected_topics", []))
                topic_recall += sum(len(exp & set(t.topics)) / len(exp) for t in tt) / len(tt) if exp else 0
                if p.get("negative_control"):
                    neg_ok = all(t.total < 70 for t in tt)
        n_papers = len({t.arxiv_id for t in ok}) or 1
        i = [t for t in trials if t.model == model and t.task == "investigation"]
        iok = [t for t in i if t.ok]
        out[model] = {
            "screening_trials": len(s),
            "screening_success": f"{len(ok)}/{len(s)}" if s else "—",
            "first_attempt_json_valid": f"{sum(t.first_attempt_valid for t in ok)}/{len(s)}" if s else "—",
            "mean_latency_s": _mean([t.wall_s for t in ok]),
            "mean_tok_per_s": _mean([t.output_tok_per_s for t in ok]),
            "score_stdev": _mean(per_paper_sd),
            "band_agreement": round(band_hits / n_papers, 2) if ok else None,
            "topic_recall": round(topic_recall / n_papers, 2) if ok else None,
            "negative_control_held": neg_ok,
            "investigation_success": f"{len(iok)}/{len(i)}" if i else "—",
            "investigation_latency_s": _mean([t.wall_s for t in iok]),
            "investigation_tok_per_s": _mean([t.output_tok_per_s for t in iok]),
            "demonstrated_claims_downgraded": sum((t.evidence or {}).get("claims_downgraded", 0) for t in iok) if iok else None,
            "experiments_verified": sum((t.evidence or {}).get("experiments_verified", 0) for t in iok) if iok else None,
            "experiments_dropped_unverifiable": sum((t.evidence or {}).get("experiments_dropped", 0) for t in iok) if iok else None,
            "invalid_page_refs": sum((t.evidence or {}).get("invalid_page_refs", 0) for t in iok) if iok else None,
        }
        v = [t for t in trials if t.model == model and t.task == "verification"]
        vok = [t for t in v if t.ok]
        ev = [t.evidence or {} for t in vok]
        out[model] |= {
            "verification_success": f"{len(vok)}/{len(v)}" if v else "—",
            "verification_first_attempt_valid": f"{sum(t.first_attempt_valid for t in vok)}/{len(v)}" if v else "—",
            "verification_latency_s": _mean([t.wall_s for t in vok]),
            "verification_tok_per_s": _mean([t.output_tok_per_s for t in vok]),
            "verification_agree_rate": round(sum(e["agrees"] for e in ev) / len(ev), 2) if ev else None,
            "verification_mean_challenges": _mean([e["challenges"] for e in ev]),
            "verification_major_challenges": sum(e["major_challenges"] for e in ev) if ev else None,
            "verification_mean_adjustment": _mean([e["adjustment"] for e in ev]),
            "verification_out_of_range_adjustments": sum(e["adjustment_out_of_range"] for e in ev) if ev else None,
            "verification_invalid_pages": sum(e["invalid_challenge_pages"] for e in ev) if ev else None,
        }
    return out


def _write_report(cfg, bcfg, trials, residency, skipped, meta, repeats) -> Path:
    summary = summarize(bcfg, trials)
    now = datetime.now()
    out_dir = cfg.benchmark_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = non_clobbering(out_dir / f"{now:%Y-%m-%d} Model Benchmark.md", f"{now:%H%M%S}")
    json_path = md_path.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": summary, "residency": residency, "skipped_models": skipped,
                                     "trials": [t.__dict__ for t in trials]}, indent=2, default=str), encoding="utf-8")
    keys = [
        ("screening_success", "Screening success"), ("first_attempt_json_valid", "First-attempt valid JSON"),
        ("mean_latency_s", "Screening latency (s)"), ("mean_tok_per_s", "Output tok/s"), ("score_stdev", "Score stdev (repeats)"),
        ("band_agreement", "Band agreement"), ("topic_recall", "Topic recall"), ("negative_control_held", "Negative control held (<70)"),
        ("investigation_success", "Full-text success"), ("investigation_latency_s", "Full-text latency (s)"),
        ("investigation_tok_per_s", "Full-text tok/s"), ("experiments_verified", "Experiments verified verbatim"),
        ("experiments_dropped_unverifiable", "Experiments dropped (unverifiable numbers)"),
        ("demonstrated_claims_downgraded", "'Demonstrated' claims downgraded"), ("invalid_page_refs", "Invalid page refs"),
        ("verification_success", "Verifier success"), ("verification_first_attempt_valid", "Verifier first-attempt valid"),
        ("verification_latency_s", "Verifier latency (s)"), ("verification_tok_per_s", "Verifier tok/s"),
        ("verification_agree_rate", "Verifier agrees with analysis"), ("verification_mean_challenges", "Verifier challenges / analysis"),
        ("verification_major_challenges", "Verifier major challenges"), ("verification_mean_adjustment", "Verifier mean adjustment"),
        ("verification_out_of_range_adjustments", "Verifier adjustments outside -15..+5"),
        ("verification_invalid_pages", "Verifier invalid page cites"),
    ]
    models = list(summary)
    lines = [
        "---", f'title: "Model Benchmark — {now:%Y-%m-%d %H:%M}"', "tags: [arxiv-digest, benchmark]", "---", "",
        f"# Model benchmark — {now:%Y-%m-%d %H:%M}", "",
        f"Papers: {len(bcfg['papers'])} · screening repeats: {repeats} · full-text subset: {', '.join(bcfg.get('investigate_papers', []))}", "",
    ]
    if skipped:
        lines += [f"> [!note] Not installed (skipped, nothing was pulled): {', '.join(skipped)}", ""]
    if models:
        lines += ["| Metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for key, label in keys:
            lines.append(f"| {label} | " + " | ".join(str(summary[m].get(key) if summary[m].get(key) is not None else "—") for m in models) + " |")
        lines.append("")
    lines += ["## GPU residency", "", "| Model / profile | Size (GB) | In VRAM (GB) | GPU fraction | Context |", "|---|---:|---:|---:|---:|"]
    for k, r in residency.items():
        lines.append(f"| {k} | {r['size_gb'] if r else '—'} | {r['vram_gb'] if r else '—'} | {r['gpu_fraction'] if r else '—'} | {r.get('context_length') if r else '—'} |")
    lines += ["", "A GPU fraction below 1.0 means part of the model spilled to system RAM (expect a large slowdown).", "",
              "## Per-paper screening scores", "", "| Paper | Expected | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for p in bcfg["papers"]:
        cells = []
        for m in models:
            tt = [t for t in trials if t.model == m and t.task == "screening" and t.arxiv_id == p["arxiv_id"] and t.ok]
            cells.append(", ".join(f"{t.total} {t.band}" for t in tt) or "—")
        lines.append(f"| {p['label']} ({p['arxiv_id']}) | {'/'.join(p.get('expected_bands', []))} | " + " | ".join(cells) + " |")
    vt = [t for t in trials if t.task == "verification" and t.ok]
    if vt:
        lines += ["", "## Verifier reviews", "", "| Verifier | Reviewed analyst | Paper | Agrees | Challenges (major) | Adjustment |",
                  "|---|---|---|---|---|---:|"]
        for t in vt:
            e = t.evidence or {}
            lines.append(f"| {t.model} | {e.get('reviewed_model')} | {t.arxiv_id} | {'yes' if e.get('agrees') else 'no'} | "
                         f"{e.get('challenges')} ({e.get('major_challenges')}) | {e.get('adjustment'):+d} |")
    failures = [t for t in trials if not t.ok]
    if failures:
        lines += ["", "## Failures", ""] + [f"- {t.model} · {t.task} · {t.arxiv_id}: {t.error}" for t in failures]
    lines += ["", f"Raw trials: `{json_path.name}`", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
