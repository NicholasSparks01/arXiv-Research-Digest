"""Command-line interface: `digest <command>` or `python -m arxiv_digest <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from arxiv_digest.config import PROJECT_ROOT, load_config


def _setup_logging(cfg, verbose: bool, command: str) -> Path:
    log_dir = cfg.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{command}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_file


def _parse_now(value: str | None):
    return datetime.fromisoformat(value) if value else None


def _print_outcome(outcome) -> int:
    print(f"\nstatus: {outcome.status}")
    if outcome.run_id is not None:
        print(f"run:    {outcome.run_id}")
    if outcome.report_path:
        print(f"report: {outcome.report_path}")
    if outcome.manifest_path:
        print(f"manifest: {outcome.manifest_path}")
    if outcome.message:
        print(f"note:   {outcome.message}")
    return outcome.exit_code


def cmd_init(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive

    for d in (cfg.data_dir, cfg.reports_dir, cfg.collection_dir, cfg.runs_dir, cfg.pdf_dir):
        d.mkdir(parents=True, exist_ok=True)
    Archive(cfg.db_path).close()
    print(f"database:   {cfg.db_path}")
    print(f"vault:      {cfg.vault_dir}")
    print(f"reports:    {cfg.reports_dir}")
    print(f"archive:    {cfg.archive_path}")
    local = PROJECT_ROOT / "config" / "local.toml"
    if not local.exists():
        print(f"\nTip: create {local} with\n  [paths]\n  vault_dir = 'C:/path/to/your/Obsidian/Vault'\nto write reports into your vault.")
    return 0


def cmd_window(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive
    from arxiv_digest.pipeline import resolve_window

    archive = Archive(cfg.db_path)
    w = resolve_window(cfg, archive, _parse_now(args.now))
    print(f"research week:     {w.label()}")
    print(f"collection window: {w.start:%Y-%m-%d %H:%M %Z} -> {w.end:%Y-%m-%d %H:%M %Z}")
    print(f"reason:            {w.reason}")
    print(f"arXiv filter:      {w.arxiv_range()}")
    return 0


def cmd_collect(cfg, args) -> int:
    from arxiv_digest.pipeline import Pipeline

    return _print_outcome(Pipeline(cfg).collect(_parse_now(args.now), Path(args.legacy) if args.legacy else None, args.expect))


def cmd_run(cfg, args) -> int:
    from arxiv_digest.pipeline import Pipeline

    return _print_outcome(Pipeline(cfg).run(_parse_now(args.now), args.force, Path(args.legacy) if args.legacy else None, args.expect))


def cmd_import_legacy(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive
    from arxiv_digest.archive.legacy import reconcile
    from arxiv_digest.pipeline import Pipeline

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    if args.dry_run:
        res = reconcile(cfg, Archive(":memory:"), None, path, run_id=0, expected=args.expect, dry_run=True)
        print(f"files: {len(res.files)}  rows with arXiv ids: {len(res.entries)}  unique papers: {len(res.unique_ids)}"
              + (f"  (expected {args.expect}: {'match' if res.count_matches_expected else 'MISMATCH'})" if args.expect else ""))
        print(f"duplicates: {res.duplicates or 'none'}")
        print(f"rows with legacy score: {sum(e.score is not None for e in res.entries)}  with title: {sum(bool(e.title) for e in res.entries)}")
        for e in res.entries[:10]:
            print(f"  {e.arxiv_id:<12} score={e.score!s:<4} {e.title or ''}"[:120])
        return 0
    return _print_outcome(Pipeline(cfg).import_legacy(path, args.expect))


def cmd_status(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive

    archive = Archive(cfg.db_path)
    print(f"last successful weekly window end: {archive.get_meta('last_successful_window_end') or '(none - next run is the initial 30-day run)'}")
    n_papers = archive.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    threshold = int(cfg.section("scoring")["archive_threshold"])
    print(f"papers in archive: {n_papers} · qualifying (≥{threshold}): {len(archive.qualifying_ids(threshold))}")
    print("\n id  mode       status   state              started              window")
    for r in archive.recent_runs(args.n):
        print(f"{r['id']:>3}  {r['mode']:<9}  {r['status']:<7}  {r['state']:<17}  {r['started_at']:<19}  "
              f"{(r['window_start'] or '')[:10]}..{(r['window_end'] or '')[:10]}  {r['error'] or ''}"[:200])
    return 0


def cmd_export(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive
    from arxiv_digest.report.archive_export import export_archive

    e = export_archive(cfg, Archive(cfg.db_path))
    print(f"wrote {e.path} ({len(e.paper_ids)} qualifying papers)")
    return 0


def cmd_search(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive

    archive = Archive(cfg.db_path)
    for r in archive.fts_search(args.query, args.n):
        s = archive.current_score(r["arxiv_id"])
        print(f"{r['arxiv_id']:<12} {(str(s.total) + ' ' + s.band) if s else 'unscored':<16} {r['title']}"[:160])
    return 0


def cmd_models(cfg, args) -> int:
    from arxiv_digest.llm.provider import LLMUnavailable, OllamaProvider

    p = OllamaProvider.from_config(cfg.section("llm"))
    try:
        p.ensure_available()
    except LLMUnavailable as exc:
        print(exc)
        return 1
    installed = set(p.list_models())
    print(f"Ollama {p.version()} at {p.base_url}")
    for name in ("triage", "screener", "analyst", "synthesizer", "verifier"):
        prof = cfg.profile(name)
        print(f"  {name:<12} {prof.model:<22} ctx={prof.num_ctx:<6} think={prof.think!s:<7} {'installed' if prof.model in installed else 'NOT INSTALLED'}")
    print("installed models: " + ", ".join(sorted(installed)) if installed else "no models installed")
    running = p.running()
    for m in running:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        print(f"  loaded: {m['name']} {vram / 1e9:.1f}/{size / 1e9:.1f} GB in VRAM, ctx {m.get('context_length', '?')}")
    return 0


def cmd_benchmark(cfg, args) -> int:
    from arxiv_digest.benchmark import run_benchmark

    path = run_benchmark(cfg, Path(args.file), args.models.split(",") if args.models else None, args.skip_fulltext, args.repeats,
                         args.stages.split(",") if args.stages else None, args.verifiers.split(",") if args.verifiers else None)
    print(f"benchmark report: {path}")
    return 0


def cmd_triage_eval(cfg, args) -> int:
    from arxiv_digest.archive.db import Archive
    from arxiv_digest.triage_eval import run_triage_eval

    path = run_triage_eval(cfg, Archive(cfg.db_path), args.run, args.sample, args.seed, args.workers, args.skip_reference,
                           args.skip_serial)
    print(f"triage evaluation report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="digest", description="Local-first arXiv research digest")
    ap.add_argument("--config", help="path to digest.toml (default: config/digest.toml, env ARXIV_DIGEST_CONFIG)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create data directories and the SQLite archive").set_defaults(fn=cmd_init)
    p = sub.add_parser("window", help="show the next collection window")
    p.add_argument("--now", help="pretend the current time is this ISO timestamp")
    p.set_defaults(fn=cmd_window)

    for name, fn, helptext in (("collect", cmd_collect, "Phase 1-2: discovery + archive + collection report (no LLM)"),
                               ("run", cmd_run, "full weekly pipeline (discovery, analysis, digest, archive)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--now", help="pretend the current time is this ISO timestamp")
        p.add_argument("--legacy", help="reconcile a legacy Markdown archive (file or folder) first")
        p.add_argument("--expect", type=int, help="expected number of unique legacy papers (e.g. 67)")
        if name == "run":
            p.add_argument("--force", action="store_true", help="re-run even if this research week already succeeded")
        p.set_defaults(fn=fn)

    p = sub.add_parser("import-legacy", help="import and reconcile the legacy Markdown archive")
    p.add_argument("path")
    p.add_argument("--expect", type=int)
    p.add_argument("--dry-run", action="store_true", help="parse only; no network, no database writes")
    p.set_defaults(fn=cmd_import_legacy)

    p = sub.add_parser("status", help="recent runs and archive state")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(fn=cmd_status)

    sub.add_parser("export-archive", help="regenerate the Markdown master archive from SQLite").set_defaults(fn=cmd_export)

    p = sub.add_parser("search", help="full-text search (SQLite FTS5) over archived titles and abstracts")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(fn=cmd_search)

    sub.add_parser("models", help="check Ollama and the configured model profiles").set_defaults(fn=cmd_models)

    p = sub.add_parser("benchmark", help="benchmark local models on representative papers")
    p.add_argument("--file", default=str(PROJECT_ROOT / "config" / "benchmark.toml"))
    p.add_argument("--models", help="comma-separated Ollama tags (overrides benchmark.toml)")
    p.add_argument("--repeats", type=int)
    p.add_argument("--skip-fulltext", action="store_true")
    p.add_argument("--stages", help="comma-separated subset of screening,fulltext,verification (default: all)")
    p.add_argument("--verifiers", help="comma-separated verifier models (default: benchmark.toml verifier_models, else the verifier profile)")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("triage-eval", help="measure compact triage against full screening on a sample of a collected run")
    p.add_argument("--run", type=int, help="run id to sample from (default: latest run with collected papers)")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent requests (needs OLLAMA_NUM_PARALLEL >= this AND a model architecture Ollama can batch)")
    p.add_argument("--skip-reference", action="store_true", help="only time triage; skip the full-screening reference")
    p.add_argument("--skip-serial", action="store_true", help="skip the one-at-a-time triage timing")
    p.set_defaults(fn=cmd_triage_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command not in ("window", "status", "search"):
        log_file = _setup_logging(cfg, args.verbose, args.command)
        logging.getLogger(__name__).debug("log file: %s; args: %s", log_file, json.dumps(vars(args), default=str))
    return int(args.fn(cfg, args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
