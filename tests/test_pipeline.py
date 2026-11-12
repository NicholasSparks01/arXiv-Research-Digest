"""End-to-end state-machine tests with fake arXiv and fake LLM (no network, no Ollama)."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from arxiv_digest.archive.db import Archive
from arxiv_digest.pipeline import Pipeline
from arxiv_digest.report.digest import MARKER_RE

from conftest import FakeArxiv, FakeLLM, paper

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 16, 8, 0, tzinfo=NY)  # Monday 8 AM

PAPERS = [
    paper("2511.00001", "Decompilation with LLM agents", cats=("cs.CR", "cs.SE")),
    paper("2511.00002", "Fuzzing firmware at scale", cats=("cs.CR",)),
    paper("2511.00003", "Side-channel leakage in RISC-V cores", cats=("cs.AR",)),
    paper("2511.00004", "Generic multi-agent planning", cats=("cs.AI",)),
    paper("2511.00005", "Image diffusion transformers", cats=("cs.LG",)),  # no keyword signal → not screened
]
SCORES = {"Decompilation with LLM agents": 91, "Fuzzing firmware at scale": 76, "Side-channel leakage in RISC-V cores": 58,
          "Generic multi-agent planning": 30}


def make(cfg, tmp_path, llm=None, arxiv=None):
    archive = Archive(cfg.db_path)
    return Pipeline(cfg, archive=archive, client=arxiv or FakeArxiv(PAPERS), provider=llm or FakeLLM(SCORES)), archive


def test_full_run_success_advances_window(cfg, tmp_path):
    cfg.raw["investigation"]["max_candidates"] = 0  # no PDFs in unit tests
    pipe, archive = make(cfg, tmp_path)
    out = pipe.run(now=NOW)
    assert out.status == "SUCCESS", out.message
    assert archive.get_meta("last_successful_window_end") == "2026-11-16T00:00:00-05:00"
    states = [r["state"] for r in archive.conn.execute("SELECT state FROM run_events WHERE run_id=? ORDER BY id", (out.run_id,))]
    for s in ("DISCOVERED", "ANALYZED", "REPORT_GENERATED", "REPORT_VALIDATED", "ARCHIVE_COMMITTED", "SUCCESS"):
        assert s in states
    assert states.index("REPORT_VALIDATED") < states.index("ARCHIVE_COMMITTED") < states.index("SUCCESS")

    text = open(out.report_path, encoding="utf-8").read()
    markers = MARKER_RE.findall(text)
    # 3 qualifying (≥40) papers; the 30-point generic agent paper is excluded; unscreened diffusion paper excluded
    assert markers == ["2511.00001", "2511.00002", "2511.00003"]
    assert "Week of November 16, 2026" in text and "November 9–15, 2026" in text
    for n in ("01", "02", "03", "04", "05", "06", "07"):
        assert f"## {n} — " in text
    master = cfg.archive_path.read_text(encoding="utf-8")
    assert master.count("2511.00001](") == 1 and "2511.00004](" not in master

    # same research week again → nothing to do
    assert pipe.run(now=NOW).status == "NOTHING_TO_DO"
    # forced re-run must not overwrite the previous report, and must not repeat featured papers
    out2 = pipe.run(now=NOW, force=True)
    assert out2.report_path != out.report_path
    assert MARKER_RE.findall(open(out2.report_path, encoding="utf-8").read()) == []
    assert out2.status == "PARTIAL"  # zero papers left → min_report_papers not met → window NOT advanced again
    prev = open(out.report_path, encoding="utf-8").read()
    assert prev == text


def test_ollama_down_fails_without_advancing(cfg, tmp_path):
    pipe, archive = make(cfg, tmp_path, llm=FakeLLM(SCORES, available=False))
    out = pipe.run(now=NOW)
    assert out.status == "FAILED" and out.exit_code == 1
    assert archive.run(out.run_id)["state"] == "DISCOVERED"
    assert archive.get_meta("last_successful_window_end") is None
    assert not list(cfg.reports_dir.glob("*.md")) if cfg.reports_dir.exists() else True


def test_failed_query_is_partial(cfg, tmp_path):
    cfg.raw["investigation"]["max_candidates"] = 0
    pipe, archive = make(cfg, tmp_path, arxiv=FakeArxiv(PAPERS, fail_queries=("cat:cs.AR)",)))
    out = pipe.run(now=NOW)
    assert out.status == "PARTIAL" and out.exit_code == 2
    assert archive.get_meta("last_successful_window_end") is None
    assert "(PARTIAL)" in out.report_path
    text = open(out.report_path, encoding="utf-8").read()
    assert "failed" in text.split("## 07")[1]


def test_empty_discovery_fails(cfg, tmp_path):
    pipe, archive = make(cfg, tmp_path, arxiv=FakeArxiv([]))
    out = pipe.run(now=NOW)
    assert out.status == "FAILED" and "no papers" in out.message
    assert archive.get_meta("last_successful_window_end") is None


def test_report_validation_failure_blocks_success(cfg, tmp_path, monkeypatch):
    cfg.raw["investigation"]["max_candidates"] = 0
    import arxiv_digest.pipeline as pl

    real = pl.render_digest
    monkeypatch.setattr(pl, "render_digest", lambda *a, **k: real(*a, **k).replace("<!-- paper:2511.00002 -->", ""))
    pipe, archive = make(cfg, tmp_path)
    out = pipe.run(now=NOW)
    assert out.status == "FAILED" and "REPORT_VALIDATED" in out.message
    assert archive.get_meta("last_successful_window_end") is None
    assert not list(cfg.reports_dir.glob("*.md"))
    assert (cfg.runs_dir / f"run-{out.run_id:04d}" / "rejected-report.md").exists()


def test_screening_failures_recorded(cfg, tmp_path):
    cfg.raw["investigation"]["max_candidates"] = 0
    pipe, archive = make(cfg, tmp_path, llm=FakeLLM(SCORES, bad_json_for={"Fuzzing firmware at scale"}))
    out = pipe.run(now=NOW)
    m = json.loads(open(out.manifest_path, encoding="utf-8").read())
    assert m["stages"]["screening"]["failed"] == 1
    assert out.status == "PARTIAL"  # 1 of 4 failed > 10% failure threshold


def test_collect_and_legacy_reconcile(cfg, tmp_path):
    legacy = tmp_path / "legacy.md"
    legacy.write_text(
        "| arXiv | Title | Score |\n|---|---|---|\n"
        "| 2511.00001 | Decompilation with LLM agents | 88 |\n"
        "| 2401.99999 | Paper that no longer resolves | 55 |\n"
        "| 2511.00002 | Totally different title | 72 |\n",
        encoding="utf-8",
    )
    pipe, archive = make(cfg, tmp_path)
    out = pipe.collect(now=NOW, legacy=legacy, expected=3)
    assert out.status == "SUCCESS", out.message
    assert archive.get_meta("last_successful_collect_window_end")
    assert archive.get_meta("last_successful_window_end") is None  # collect never advances the weekly digest window
    assert archive.paper("2401.99999")["metadata_complete"] == 0
    assert archive.current_score("2511.00001").stage == "legacy"
    statuses = {r["arxiv_id"]: r["status"] for r in archive.conn.execute("SELECT arxiv_id, status FROM legacy_records")}
    assert statuses == {"2511.00001": "imported", "2401.99999": "not_found", "2511.00002": "conflict"}
    report = open(out.report_path, encoding="utf-8").read()
    assert "unique_papers: 5" in report
    master = cfg.archive_path.read_text(encoding="utf-8")
    assert "2401.99999](" in master and "2511.00001](" in master

    # expected-count mismatch → PARTIAL
    out2 = pipe.import_legacy(legacy, expected=67)
    assert out2.status == "PARTIAL" and "expected 67" in out2.message


def test_missing_model_fails_fast(cfg, tmp_path):
    cfg.raw["llm"]["profiles"]["analyst"]["model"] = "not-installed:1b"
    llm = FakeLLM(SCORES)
    pipe, archive = make(cfg, tmp_path, llm=llm)
    out = pipe.run(now=NOW)
    assert out.status == "FAILED" and "not-installed:1b" in out.message
    assert llm.calls == 0  # no screening time wasted
