"""Run orchestration with explicit, recorded run states.

    STARTED → DISCOVERED → ANALYZED → REPORT_GENERATED → REPORT_VALIDATED → ARCHIVE_COMMITTED → SUCCESS

`last_successful_window_end` advances only inside the final commit transaction, and only when no required
stage reported a problem. Anything else ends as PARTIAL (report kept, window not advanced) or FAILED.
"""

from __future__ import annotations

import json
import logging
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from arxiv_digest import __version__
from arxiv_digest.analysis.investigation import investigate, select_candidates
from arxiv_digest.analysis.prefilter import select_for_screening
from arxiv_digest.analysis.screening import screen_papers
from arxiv_digest.analysis.synthesis import propose_adaptive, synthesize, verify
from arxiv_digest.analysis.rerank import build_pool, rerank
from arxiv_digest.analysis.triage import shortlist_size, triage_papers
from arxiv_digest.archive.db import Archive
from arxiv_digest.archive.legacy import reconcile
from arxiv_digest.config import Config
from arxiv_digest.discovery.arxiv_client import ArxivClient
from arxiv_digest.discovery.collector import Collector
from arxiv_digest.llm.provider import LLMError, LLMProvider, LLMUnavailable, OllamaProvider
from arxiv_digest.report.archive_export import export_archive
from arxiv_digest.report.collection import render_collection, render_reconciliation, summary_line, validate_collection
from arxiv_digest.report.digest import (DISCOVERY_KINDS, RunFacts, previously_featured, render_digest, select_digest_papers,
                                        validate_digest)
from arxiv_digest.report.files import commit_new, non_clobbering, sha256_file, write_temp
from arxiv_digest.timeutil import Window, compute_window, now_in, parse_iso, utcnow_iso

log = logging.getLogger(__name__)

EXIT_SUCCESS, EXIT_FAILED, EXIT_PARTIAL, EXIT_NOTHING = 0, 1, 2, 0


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class Manifest:
    def __init__(self, path: Path, base: dict):
        self.path = path
        self.data = base
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save()

    def update(self, **kw) -> None:
        self.data.update(kw)
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def _searches_json(archive: Archive, run_id: int) -> list[dict]:
    return [dict(r) for r in archive.searches_for_run(run_id)]


def _events(archive: Archive, run_id: int) -> list[dict]:
    return [dict(r) for r in archive.conn.execute("SELECT at, state, message FROM run_events WHERE run_id=? ORDER BY id", (run_id,))]


def resolve_window(cfg: Config, archive: Archive, now: datetime | None, mode: str = "run") -> Window:
    sched = cfg.section("schedule")
    tz = sched.get("timezone", "America/New_York")
    last = parse_iso(archive.get_meta("last_successful_window_end"))
    return compute_window(now or now_in(tz), tz, last, int(sched.get("initial_lookback_days", 30)), int(sched.get("overlap_days", 3)))


@dataclass
class RunOutcome:
    run_id: int | None
    status: str
    exit_code: int
    report_path: str | None = None
    manifest_path: str | None = None
    message: str = ""


class Pipeline:
    def __init__(self, cfg: Config, archive: Archive | None = None, client: ArxivClient | None = None,
                 provider: LLMProvider | None = None):
        self.cfg = cfg
        self.archive = archive or Archive(cfg.db_path)
        self.client = client or ArxivClient.from_config(cfg.section("arxiv"))
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = OllamaProvider.from_config(self.cfg.section("llm"))
        return self._provider

    def _start(self, mode: str, window: Window | None) -> tuple[int, Manifest]:
        run_id = self.archive.create_run(
            mode, window.start.isoformat() if window else None, window.end.isoformat() if window else None,
            f"{window.research_week_start}..{window.research_week_end}" if window else None, self.cfg.fingerprint(),
        )
        manifest = Manifest(
            self.cfg.runs_dir / f"run-{run_id:04d}" / "manifest.json",
            {
                "run_id": run_id, "mode": mode, "started_at": utcnow_iso(), "package_version": __version__,
                "python": platform.python_version(), "host_os": platform.platform(), "config_fingerprint": self.cfg.fingerprint(),
                "window": None if window is None else {
                    "start": window.start.isoformat(), "end": window.end.isoformat(), "reason": window.reason,
                    "research_week": [str(window.research_week_start), str(window.research_week_end)],
                },
                "stages": {}, "problems": [], "status": "RUNNING",
            },
        )
        self.archive.conn.execute("UPDATE runs SET manifest_path=? WHERE id=?", (str(manifest.path), run_id))
        return run_id, manifest

    def _finish(self, run_id: int, manifest: Manifest, status: str, error: str | None, report: str | None, t0: float) -> RunOutcome:
        if self.archive.run(run_id)["status"] == "RUNNING":
            self.archive.finish_run(run_id, status, error, str(manifest.path))
        manifest.update(status=status, error=error, finished_at=utcnow_iso(), duration_s=round(time.monotonic() - t0, 1),
                        searches=_searches_json(self.archive, run_id), events=_events(self.archive, run_id),
                        final_state=self.archive.run(run_id)["state"])
        code = {"SUCCESS": EXIT_SUCCESS, "PARTIAL": EXIT_PARTIAL}.get(status, EXIT_FAILED)
        log.info("run %d finished: %s%s", run_id, status, f" ({error})" if error else "")
        return RunOutcome(run_id, status, code, report, str(manifest.path), error or "")

    # ------------------------------------------------------------------ legacy
    def _reconcile(self, run_id: int, manifest: Manifest, legacy_path: Path, expected: int | None) -> tuple[dict, Path]:
        res = reconcile(self.cfg, self.archive, self.client, legacy_path, run_id, expected=expected)
        text = render_reconciliation(res, run_id, str(legacy_path))
        final = non_clobbering(self.cfg.collection_dir / f"{datetime.now():%Y-%m-%d} Legacy Reconciliation.md", f"run {run_id}")
        tmp = write_temp(final, text)
        commit_new(tmp, final)
        self.archive.add_report(run_id, "reconciliation", str(final), sha256_file(final), res.unique_ids, True)
        info = {
            "summary": summary_line(res), "files": res.files, "unique": len(res.unique_ids), "imported": len(res.imported),
            "matched": len(res.matched), "not_found": res.not_found, "duplicates": res.duplicates,
            "title_conflicts": len(res.title_conflicts), "scores_imported": res.scores_imported, "expected": expected,
            "count_matches_expected": res.count_matches_expected, "lookup_error": res.lookup_error, "report": str(final),
        }
        manifest.data["stages"]["reconciliation"] = info
        manifest.save()
        problems = []
        if not res.count_matches_expected:
            problems.append(f"legacy archive has {len(res.unique_ids)} unique papers, expected {expected}")
        if res.lookup_error:
            problems.append(f"legacy metadata lookup incomplete: {res.lookup_error}")
        info["problems"] = problems
        return info, final

    def import_legacy(self, legacy_path: Path, expected: int | None) -> RunOutcome:
        t0 = time.monotonic()
        run_id, manifest = self._start("reconcile", None)
        try:
            info, report = self._reconcile(run_id, manifest, legacy_path, expected)
            self.archive.set_run_state(run_id, "DISCOVERED", info["summary"])
            self.archive.set_run_state(run_id, "REPORT_GENERATED")
            self.archive.set_run_state(run_id, "REPORT_VALIDATED")
            export = export_archive(self.cfg, self.archive)
            status = "SUCCESS" if not info["problems"] else "PARTIAL"
            with self.archive.transaction():
                self.archive.add_report(run_id, "master_archive", str(export.path), export.sha256, export.paper_ids, True)
                self.archive.set_run_state(run_id, "ARCHIVE_COMMITTED", f"{len(export.paper_ids)} qualifying papers")
                if status == "SUCCESS":
                    self.archive.mark_success(run_id, utcnow_iso(), "reconcile")
                self.archive.finish_run(run_id, status, "; ".join(info["problems"]) or None, str(manifest.path))
            manifest.data["problems"] = info["problems"]
            return self._finish(run_id, manifest, status, "; ".join(info["problems"]) or None, str(report), t0)
        except Exception as exc:  # noqa: BLE001
            manifest.update(traceback=traceback.format_exc())
            return self._finish(run_id, manifest, "FAILED", str(exc), None, t0)

    # ------------------------------------------------------------------ collect
    def _discover(self, run_id: int, manifest: Manifest, window: Window, problems: list[str]) -> None:
        collector = Collector(self.cfg, self.archive, self.client)
        t = time.monotonic()
        disc = collector.discover(run_id, window)
        threshold = int(self.cfg.section("scoring")["archive_threshold"])
        archived = self.archive.qualifying_ids(threshold)
        refresh = collector.refresh_versions(run_id, archived) if archived else None
        discovered = self.archive.run_paper_ids(run_id, DISCOVERY_KINDS)
        failed = disc.failed + ([refresh] if refresh and refresh.status in ("failed", "incomplete") else [])
        manifest.data["stages"]["discovery"] = {
            "seconds": round(time.monotonic() - t, 1), "unique_papers": len(discovered),
            "failed_searches": [f"{s.name}: {s.error}" for s in failed],
            "saturated_searches": [s.name for s in disc.saturated],
            "version_refresh": None if refresh is None else {"checked": len(archived), "revised": refresh.revised, "status": refresh.status},
        }
        manifest.save()
        if not discovered:
            raise StageFailure("DISCOVERED", "discovery returned no papers for the window (arXiv outage or query error)")
        if all(s.status == "failed" for s in disc.searches if s.kind == "category"):
            raise StageFailure("DISCOVERED", "every category query failed")
        if failed and self.cfg.section("arxiv").get("fail_run_on_query_error", True):
            problems.append(f"{len(failed)} search(es) failed or incomplete: " + ", ".join(s.name for s in failed))
        self.archive.set_run_state(run_id, "DISCOVERED", f"{len(discovered)} unique papers")

    def collect(self, now: datetime | None = None, legacy: Path | None = None, expected: int | None = None) -> RunOutcome:
        """Phase 1-2: discovery + archive + collection report, no LLM."""
        t0 = time.monotonic()
        window = resolve_window(self.cfg, self.archive, now)
        run_id, manifest = self._start("collect", window)
        problems: list[str] = []
        try:
            notes = []
            if legacy:
                info, _ = self._reconcile(run_id, manifest, legacy, expected)
                problems += info["problems"]
                notes.append(f"Legacy reconciliation: {info['summary']}")
            self._discover(run_id, manifest, window, problems)
            status = "SUCCESS" if not problems else "PARTIAL"
            text, summary = render_collection(self.cfg, self.archive, run_id, window, status, notes + [f"Problem: {p}" for p in problems])
            final = non_clobbering(self.cfg.collection_dir / f"{window.end.date():%Y-%m-%d} Collection Report.md", f"run {run_id}")
            tmp = write_temp(final, text)
            self.archive.set_run_state(run_id, "REPORT_GENERATED", str(final))
            errors = validate_collection(tmp.read_text(encoding="utf-8"), summary, run_id, len(self.archive.searches_for_run(run_id)))
            if errors:
                raise StageFailure("REPORT_VALIDATED", "; ".join(errors))
            commit_new(tmp, final)
            self.archive.set_run_state(run_id, "REPORT_VALIDATED")
            export = export_archive(self.cfg, self.archive)
            with self.archive.transaction():
                self.archive.add_report(run_id, "collection", str(final), sha256_file(final), summary["paper_ids"], True)
                self.archive.add_report(run_id, "master_archive", str(export.path), export.sha256, export.paper_ids, True)
                self.archive.set_run_state(run_id, "ARCHIVE_COMMITTED")
                if status == "SUCCESS":
                    self.archive.mark_success(run_id, window.end.isoformat(), "collect")
                self.archive.finish_run(run_id, status, "; ".join(problems) or None, str(manifest.path))
            manifest.data["problems"] = problems
            manifest.data["stages"]["collection_report"] = {k: v for k, v in summary.items() if k != "paper_ids"} | {"path": str(final)}
            return self._finish(run_id, manifest, status, "; ".join(problems) or None, str(final), t0)
        except StageFailure as exc:
            manifest.update(problems=problems)
            return self._finish(run_id, manifest, "FAILED", str(exc), None, t0)
        except Exception as exc:  # noqa: BLE001
            manifest.update(traceback=traceback.format_exc(), problems=problems)
            return self._finish(run_id, manifest, "FAILED", f"{type(exc).__name__}: {exc}", None, t0)

    def _annotate(self, run_id: int, paper_ids: list[str]) -> list[dict]:
        """Annotated PDFs of the digest papers, copied into the vault. Optional: problems are recorded, never fatal."""
        from arxiv_digest.analysis.annotate import annotate_papers, copy_to

        try:
            results = annotate_papers(self.cfg, self.archive, self.provider, run_id, paper_ids)
        except Exception as exc:  # noqa: BLE001 - e.g. Ollama went away; the digest still goes out without them
            log.warning("annotation stage failed: %s", exc)
            return [{"error": f"{type(exc).__name__}: {exc}"}]
        out = []
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k != "annotations"}
            if r.path:
                row["vault_copy"] = str(copy_to(r.path, self.cfg.reports_dir / "Annotated"))
            out.append(row)
        return out

    # ------------------------------------------------------------------ full run
    def run(self, now: datetime | None = None, force: bool = False, legacy: Path | None = None, expected: int | None = None) -> RunOutcome:
        t0 = time.monotonic()
        window = resolve_window(self.cfg, self.archive, now)
        last = parse_iso(self.archive.get_meta("last_successful_window_end"))
        if last and last >= window.end and not force:
            msg = f"research week {window.label()} already completed successfully (use --force to re-run)"
            log.info(msg)
            return RunOutcome(None, "NOTHING_TO_DO", EXIT_NOTHING, message=msg)

        run_id, manifest = self._start("run", window)
        problems: list[str] = []
        stages = manifest.data["stages"]
        reconciliation = None
        try:
            if legacy:
                info, _ = self._reconcile(run_id, manifest, legacy, expected)
                problems += info["problems"]
                reconciliation = info

            # ---- 1. discovery
            self._discover(run_id, manifest, window, problems)

            # ---- 2. analysis
            try:
                self.provider.ensure_available()
                triage_on = bool(self.cfg.section("triage").get("enabled", True))
                rerank_on = bool(self.cfg.section("rerank").get("enabled", True))
                needed = ({"screener", "analyst", "synthesizer"} | ({"triage"} if triage_on else set())
                          | ({"reranker"} if rerank_on else set())
                          | ({"annotator"} if self.cfg.section("annotation").get("enabled", True) else set())
                          | ({"verifier"} if self.cfg.section("verification").get("enabled") else set()))
                installed = set(self.provider.list_models())
                missing = sorted({self.cfg.profile(n).model for n in needed} - installed)
                if missing:
                    raise LLMUnavailable(f"model(s) not installed in Ollama: {', '.join(missing)} — pull them or change the profiles in config")
            except LLMUnavailable as exc:
                raise StageFailure("ANALYZED", str(exc)) from exc
            stages["models"] = {name: asdict(self.cfg.profile(name))
                                for name in ("triage", "screener", "reranker", "analyst", "synthesizer", "verifier")}

            hits, pstats = select_for_screening(self.cfg, self.archive, run_id)
            stages["prefilter"] = pstats
            to_screen = [h.arxiv_id for h in hits]
            tstats = None
            try:
                if triage_on:
                    size = shortlist_size(self.cfg, (window.end - window.start).total_seconds() / 86400)
                    log.info("triaging %d papers for a shortlist of %d (pre-filter: %s)", len(hits), size, pstats)
                    tstats, to_screen = triage_papers(self.cfg, self.archive, self.provider, run_id, to_screen, size)
                    stages["triage"] = asdict(tstats)
                    manifest.save()
                    if tstats.attempted and tstats.failure_rate > 0.10:
                        problems.append(f"triage failure rate {tstats.failure_rate:.0%} ({tstats.failed}/{tstats.attempted})")
                log.info("full screening of %d papers", len(to_screen))
                sstats = screen_papers(self.cfg, self.archive, self.provider, run_id, to_screen)
            except LLMUnavailable as exc:
                raise StageFailure("ANALYZED", str(exc)) from exc
            stages["screening"] = asdict(sstats)
            manifest.save()
            if sstats.attempted and sstats.failure_rate > 0.10:
                problems.append(f"screening failure rate {sstats.failure_rate:.0%} ({sstats.failed}/{sstats.attempted})")

            adaptive_flags: list[str] = []
            acfg = self.cfg.section("adaptive")
            if acfg.get("enabled", True):
                discovered = self.archive.run_paper_ids(run_id, DISCOVERY_KINDS)
                seeds = sorted(
                    ((s.total, i) for i in discovered if (s := self.archive.current_score(i)) and s.stage != "triage"
                     and s.total >= int(acfg.get("seed_min_score", 70))),
                    reverse=True,
                )[: int(acfg.get("seed_papers", 8))]
                if seeds:
                    searches, adaptive_flags = propose_adaptive(self.cfg, self.provider, self.archive, [i for _, i in seeds])
                    adaptive = Collector(self.cfg, self.archive, self.client).collect_adaptive(run_id, window, searches)
                    hits2, _ = select_for_screening(self.cfg, self.archive, run_id)
                    already = {h.arxiv_id for h in hits}
                    extra = [h.arxiv_id for h in hits2 if h.arxiv_id not in already]
                    t2 = None
                    if extra and tstats is not None:
                        # adaptive additions must clear the main shortlist's bar (or the minimum score if it was not full)
                        bar = tstats.cutoff_score if tstats.cutoff_score is not None else tstats.min_score
                        t2, extra = triage_papers(self.cfg, self.archive, self.provider, run_id, extra,
                                                  int(self.cfg.section("triage").get("adaptive_max_extra", 15)), fixed_cutoff=bar)
                    s2 = screen_papers(self.cfg, self.archive, self.provider, run_id, extra) if extra else None
                    stages["adaptive"] = {"queries": searches, "flags": adaptive_flags,
                                          "results": [{"name": a.name, "total": a.total, "fetched": a.fetched, "status": a.status} for a in adaptive],
                                          "triaged_additional": asdict(t2) if t2 else None,
                                          "screened_additional": asdict(s2) if s2 else None}
                    if any(a.status in ("failed", "incomplete") for a in adaptive):
                        adaptive_flags.append("an adaptive search failed (non-fatal)")
                else:
                    stages["adaptive"] = {"queries": [], "note": "no seed papers above threshold"}
                manifest.save()

            discovered = self.archive.run_paper_ids(run_id, DISCOVERY_KINDS)
            ranked = None
            if rerank_on:
                pool = build_pool(self.cfg, self.archive, discovered, previously_featured(self.archive))
                log.info("comparative re-ranking of %d screened papers", len(pool))
                try:
                    ranked, kstats = rerank(self.cfg, self.archive, self.provider, run_id, pool)
                except LLMUnavailable as exc:
                    raise StageFailure("ANALYZED", str(exc)) from exc
                stages["rerank"] = asdict(kstats) | {"top": ranked[: kstats.ranked_top]}
                manifest.save()
                if kstats.calls and kstats.failed_calls > kstats.calls // 4:
                    problems.append(f"comparative re-ranking: {kstats.failed_calls} of {kstats.calls} calls failed")
            candidates = select_candidates(self.cfg, self.archive, run_id, discovered, order=ranked)
            try:
                istats = investigate(self.cfg, self.archive, self.provider, run_id, candidates)
            except LLMUnavailable as exc:
                raise StageFailure("ANALYZED", str(exc)) from exc
            stages["investigation"] = asdict(istats)
            if istats.candidates and len(istats.llm_failures) > istats.candidates // 2:
                problems.append(f"{len(istats.llm_failures)} of {istats.candidates} full-text investigations failed")

            vcfg = self.cfg.section("verification")
            if vcfg.get("enabled", False):
                verified = []
                for arxiv_id in candidates[: int(vcfg.get("max_papers", 3))]:
                    try:
                        r = verify(self.cfg, self.provider, self.archive, run_id, arxiv_id)
                    except LLMError as exc:
                        r = ({"error": str(exc)}, [str(exc)])
                    if r:
                        verified.append({"arxiv_id": arxiv_id, "flags": r[1]})
                stages["verification"] = verified
            self.archive.set_run_state(run_id, "ANALYZED", f"screened={sstats.screened} investigated={istats.investigated}")
            manifest.save()

            # ---- 3. report
            paper_ids = select_digest_papers(self.cfg, self.archive, run_id)
            rcfg = self.cfg.section("report")
            if len(paper_ids) < int(rcfg.get("min_report_papers", 1)):
                problems.append(f"only {len(paper_ids)} qualifying paper(s) for the digest")
            if paper_ids and self.cfg.section("annotation").get("enabled", True):
                stages["annotation"] = self._annotate(run_id, paper_ids)
                manifest.save()
            synthesis, synth_flags = None, []
            if paper_ids:
                try:
                    synthesis, synth_flags, smetrics = synthesize(self.cfg, self.provider, self.archive, paper_ids)
                    stages["synthesis"] = {"metrics": smetrics, "flags": synth_flags}
                    (manifest.path.parent / "synthesis.json").write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
                except LLMError as exc:
                    problems.append(f"synthesis failed: {exc}")
            status = "SUCCESS" if not problems else "PARTIAL"
            facts = RunFacts(
                run_id=run_id, window=window, status=status, searches=self.archive.searches_for_run(run_id),
                screening=stages.get("screening", {}), investigation=stages.get("investigation", {}), prefilter=pstats,
                triage=stages.get("triage", {}), rerank=stages.get("rerank", {}),
                synthesis=synthesis, synthesis_flags=synth_flags, adaptive_flags=adaptive_flags, warnings=list(problems),
                reconciliation=reconciliation,
            )
            text = render_digest(self.cfg, self.archive, facts, paper_ids)
            week_of = window.research_week_end + timedelta(days=1)
            name = f"{week_of:%Y-%m-%d} arXiv Research Digest{' (PARTIAL)' if status != 'SUCCESS' else ''}.md"
            final = non_clobbering(self.cfg.reports_dir / name, f"run {run_id}")
            tmp = write_temp(final, text)
            self.archive.set_run_state(run_id, "REPORT_GENERATED", str(final))

            v = validate_digest(tmp.read_text(encoding="utf-8"), paper_ids, self.archive, int(rcfg.get("min_digest_score", 40)), run_id)
            stages["report_validation"] = {"ok": v.ok, "errors": v.errors, "papers": v.paper_ids}
            if not v.ok:
                rejected = manifest.path.parent / "rejected-report.md"
                tmp.replace(rejected)
                raise StageFailure("REPORT_VALIDATED", "; ".join(v.errors))
            commit_new(tmp, final)
            self.archive.set_run_state(run_id, "REPORT_VALIDATED", f"{len(paper_ids)} papers")

            # ---- 4. archive commit + success (single transaction)
            export = export_archive(self.cfg, self.archive)
            with self.archive.transaction():
                self.archive.add_report(run_id, "digest", str(final), sha256_file(final), paper_ids, True)
                self.archive.add_report(run_id, "master_archive", str(export.path), export.sha256, export.paper_ids, True)
                self.archive.set_run_state(run_id, "ARCHIVE_COMMITTED", f"{len(export.paper_ids)} qualifying papers in master archive")
                if status == "SUCCESS":
                    self.archive.mark_success(run_id, window.end.isoformat(), "run")
                self.archive.finish_run(run_id, status, "; ".join(problems) or None, str(manifest.path))
            stages["archive"] = {"path": str(export.path), "qualifying": len(export.paper_ids)}
            manifest.data["problems"] = problems
            if status == "SUCCESS" and self.cfg.section("blog").get("enabled", False):
                # after the commit transaction: a blog problem is reported but never changes the run's outcome
                from arxiv_digest.report.blog import publish

                manifest.save()
                stages["blog"] = asdict(publish(self.cfg, self.archive, run_id))
            return self._finish(run_id, manifest, status, "; ".join(problems) or None, str(final), t0)
        except StageFailure as exc:
            manifest.update(problems=problems)
            return self._finish(run_id, manifest, "FAILED", str(exc), None, t0)
        except Exception as exc:  # noqa: BLE001
            manifest.update(traceback=traceback.format_exc(), problems=problems)
            return self._finish(run_id, manifest, "FAILED", f"{type(exc).__name__}: {exc}", None, t0)
