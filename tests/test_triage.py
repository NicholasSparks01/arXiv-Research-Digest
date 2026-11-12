"""Compact triage stage and its evaluation metrics."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from arxiv_digest.analysis import triage as tri_mod
from arxiv_digest.archive.db import Archive
from arxiv_digest.llm.provider import LLMResponse, LLMUnavailable
from arxiv_digest.pipeline import Pipeline
from arxiv_digest.report.digest import MARKER_RE
from arxiv_digest.triage_eval import draw_sample, spearman, suggest_threshold, threshold_table

from conftest import FakeArxiv, FakeLLM, as_triage, paper, screening_json


def triage_json(alignment=20, topics=("t1.reverse_engineering",), security="direct", rec="candidate"):
    return {"topic_ids": list(topics), "security_connection": security,
            "scores": {"alignment": alignment, "applicability": 10, "cross_domain": 5, "novelty": 5, "reproducibility": 5},
            "recommendation": rec}


class TriageLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, profile, messages, schema):
        self.calls += 1
        assert "topic_ids" in schema["properties"] and "research_problem" not in schema["properties"]
        return LLMResponse(json.dumps(self.replies.pop(0)), profile.model, metrics={"wall_s": 0.01, "output_tokens": 60})


ROW = {"arxiv_id": "2511.00001", "latest_version": 1, "categories": '["cs.CR"]', "title": "Decompiling firmware",
       "abstract": "We decompile.", "comment": None}


def test_triage_scores_with_python_rules(cfg):
    r = tri_mod.triage_one(cfg, TriageLLM([triage_json(alignment=30)]), cfg.profile("triage"), ROW)
    assert r.ok and r.total == 55 and r.band == "Medium" and r.topics == ["t1.reverse_engineering"]
    assert r.output_tokens == 60


def test_triage_guardrails_and_retry(cfg):
    # unknown topic on a non-rejected paper → corrective retry; no security connection → capped and tier-2 dropped
    llm = TriageLLM([triage_json(topics=("made.up",)),
                     triage_json(alignment=30, topics=("t2.llm_reverse_engineering", "t3.agent_orchestration"), security="none")])
    r = tri_mod.triage_one(cfg, llm, cfg.profile("triage"), ROW)
    assert llm.calls == 2 and r.ok and not r.first_attempt_valid
    assert r.topics == ["t3.agent_orchestration"]


def test_triage_failure_is_reported_not_raised(cfg):
    llm = FakeLLM({"Decompiling firmware": 50}, bad_json_for={"Decompiling firmware"})
    r = tri_mod.triage_one(cfg, llm, cfg.profile("triage"), ROW)
    assert not r.ok and "not valid JSON" in r.error and llm.calls == 2


def test_run_concurrently_keeps_order_and_overlaps():
    active, peak, lock = [0], [0], threading.Lock()

    def fn(x):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return x * 2

    out, _ = tri_mod.run_concurrently(fn, list(range(12)), workers=4)
    assert out == [x * 2 for x in range(12)] and peak[0] > 1


def test_run_concurrently_propagates_unavailable():
    def fn(x):
        if x == 3:
            raise LLMUnavailable("down")
        return x

    with pytest.raises(LLMUnavailable):
        tri_mod.run_concurrently(fn, list(range(50)), workers=2)


def test_threshold_table_weighted_and_fail_open():
    # (weight, triage, reference)
    rows = [(1.0, 80, 85), (1.0, 30, 72), (2.0, None, 65), (3.0, 10, 5), (1.0, 50, None)]
    t = {r["threshold"]: r for r in threshold_table(rows, thresholds=(25, 40), levels=(70, 60))}
    # at 25: everything except the 10-point paper passes (failed triage passes)
    assert t[25]["pass_fraction"] == round(5 / 8, 4) and t[25]["recall_ge70"] == 1.0 and t[25]["missed_ge70"] == 0
    # at 40: the 72-point paper is missed; ≥60 recall = (1 + 2) / (1 + 1 + 2)
    assert t[40]["missed_ge70"] == 1 and t[40]["recall_ge70"] == 0.5 and t[40]["recall_ge60"] == 0.75
    assert suggest_threshold(list(t.values())) == 25


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None


def test_draw_sample_is_stratified_and_weighted(cfg):
    papers = [paper(f"2511.{i:05d}", f"Decompilation study {i}", cats=("cs.CR",)) for i in range(1, 11)]
    papers += [paper(f"2511.{i:05d}", f"Language model {i} for binary reverse engineering", cats=("cs.CL",)) for i in range(11, 21)]
    archive = Archive(cfg.db_path)
    pipe = Pipeline(cfg, archive=archive, client=FakeArxiv(papers), provider=FakeLLM({}))
    out = pipe.collect(now=datetime(2026, 11, 16, 8, 0, tzinfo=ZoneInfo("America/New_York")))
    sample, pop = draw_sample(cfg, archive, out.run_id, n=10, seed=1)
    assert sum(pop.values()) == 20
    assert len({p.arxiv_id for p in sample}) == len(sample)
    for p in sample:
        k = sum(1 for q in sample if q.stratum == p.stratum)
        assert p.weight == pop[p.stratum] / k
    again, _ = draw_sample(cfg, archive, out.run_id, n=10, seed=1)
    assert [p.arxiv_id for p in again] == [p.arxiv_id for p in sample]


# --- pipeline stage -----------------------------------------------------------------
def _r(aid, total, alignment=10, rec="candidate", ok=True):
    return tri_mod.TriageResult(aid, ok, total if ok else None, components={"alignment": alignment} if ok else {},
                                recommendation=rec if ok else None, error=None if ok else "boom")


def test_shortlist_orders_cuts_and_passes_failures_through():
    rs = [_r("a", 70), _r("b", 55, alignment=20), _r("c", 55, alignment=10), _r("d", 39), _r("e", 0, ok=False), _r("f", 90)]
    ids, cutoff, failed = tri_mod.shortlist(rs, size=3, min_score=40)
    assert ids == ["f", "a", "b"] and cutoff == 55 and failed == ["e"]
    ids, cutoff, _ = tri_mod.shortlist(rs, size=10, min_score=40)
    assert ids == ["f", "a", "b", "c"] and cutoff is None  # not full: everything ≥ min_score, no cut-off


def test_shortlist_size_scales_with_window(cfg):
    cfg.raw["triage"].update(shortlist_per_week=40, max_shortlist=200)
    assert tri_mod.shortlist_size(cfg, 10) == 40  # weekly run: 7 days + 3 overlap
    assert tri_mod.shortlist_size(cfg, 30) == 160  # first run
    assert tri_mod.shortlist_size(cfg, 120) == 200  # capped


class SeqLLM:
    """Triage replies per title from a queue (last one repeats); counts calls per title."""

    def __init__(self, seq: dict[str, list[int]]):
        self.seq = {k: list(v) for k, v in seq.items()}
        self.calls: dict[str, int] = {}

    def chat(self, profile, messages, schema):
        user = messages[1]["content"]
        title = next(t for t in self.seq if f"Title: {t}" in user)
        self.calls[title] = self.calls.get(title, 0) + 1
        q = self.seq[title]
        total = q.pop(0) if len(q) > 1 else q[0]
        return LLMResponse(json.dumps(as_triage(screening_json(total))), profile.model, metrics={"wall_s": 0.01, "output_tokens": 50})


def _archive_with(cfg, titles):
    archive = Archive(cfg.db_path)
    run_id = archive.create_run("run", None, None, None, "x")
    for i, t in enumerate(titles, start=1):
        archive.upsert_paper(paper(f"2511.{i:05d}", t), run_id)
    return archive, run_id


def test_triage_caches_rechecks_borderline_and_records_scores(cfg):
    cfg.raw["triage"]["min_score"] = 40
    titles = ["Alpha binary lifting", "Beta firmware fuzzing", "Gamma kernel exploit", "Delta unrelated"]
    archive, run_id = _archive_with(cfg, titles)
    # shortlist of 2: Alpha 80, Beta 60 → cut-off 60. Gamma (55) is within the 10-point margin and scores 65 the second time.
    llm = SeqLLM({"Alpha binary lifting": [80], "Beta firmware fuzzing": [60], "Gamma kernel exploit": [55, 65], "Delta unrelated": [5]})
    ids = [f"2511.{i:05d}" for i in range(1, 5)]
    stats, chosen = tri_mod.triage_papers(cfg, archive, llm, run_id, ids, size=2)
    assert chosen == ["2511.00001", "2511.00003"]
    assert stats.rechecked == 1 and stats.raised_by_recheck == 1 and stats.cutoff_score == 65
    assert llm.calls == {"Alpha binary lifting": 1, "Beta firmware fuzzing": 1, "Gamma kernel exploit": 2, "Delta unrelated": 1}
    # triage scores rank, never qualify
    assert archive.current_score("2511.00001").stage == "triage" and archive.qualifying_ids(40) == []
    # the second run reuses cached results (including the raised score) and makes no new calls for unchanged papers
    stats2, chosen2 = tri_mod.triage_papers(cfg, archive, llm, run_id, ids, size=2)
    assert stats2.cached == 4 and chosen2 == chosen
    assert llm.calls["Alpha binary lifting"] == 1


def test_triage_score_never_displaces_a_real_score(cfg):
    archive, run_id = _archive_with(cfg, ["Old legacy paper"])
    kw = dict(version=1, run_id=run_id, model=None, components={}, confidence=None, evidence_level="abstract_only")
    archive.add_score("2511.00001", stage="legacy", total=56, band="Low", **kw)
    archive.add_score("2511.00001", stage="triage", total=10, band="Below threshold", **kw)
    assert archive.current_score("2511.00001").stage == "legacy" and archive.qualifying_ids(40) == ["2511.00001"]


def test_pipeline_screens_only_the_shortlist(cfg):
    from test_pipeline import NOW, PAPERS, SCORES

    cfg.raw["investigation"]["max_candidates"] = 0
    cfg.raw["triage"]["max_shortlist"] = 2  # first run spans 30 days; cap the shortlist at 2
    cfg.raw["triage"]["recheck_margin"] = 0
    archive = Archive(cfg.db_path)
    llm = FakeLLM(SCORES)
    out = Pipeline(cfg, archive=archive, client=FakeArxiv(PAPERS), provider=llm).run(now=NOW)
    assert out.status == "SUCCESS", out.message
    screened_titles = [u for kind, u in llm.prompts if kind == "screening"]
    assert len([k for k, _ in llm.prompts if k == "triage"]) == 4 and len(screened_titles) == 2
    assert all("Decompilation" in u or "Fuzzing firmware" in u for u in screened_titles)
    m = json.loads(open(out.manifest_path, encoding="utf-8").read())
    t = m["stages"]["triage"]
    assert t["attempted"] == 4 and t["shortlisted"] == 2 and t["cutoff_score"] == 76 and m["stages"]["screening"]["attempted"] == 2
    # the 58-point side-channel paper was triaged but not shortlisted: not in the digest, not in the master archive
    assert archive.current_score("2511.00003").stage == "triage"
    text = open(out.report_path, encoding="utf-8").read()
    assert MARKER_RE.findall(text) == ["2511.00001", "2511.00002"]
    assert "Triaged by local model:" in text and "cut-off score 76" in text
    assert "2511.00003](" not in cfg.archive_path.read_text(encoding="utf-8")
