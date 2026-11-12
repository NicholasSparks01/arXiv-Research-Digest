"""Comparative re-ranking: tournament + sliding window, failure handling, and its effect on digest selection."""

from __future__ import annotations

import json
import random
import re

from arxiv_digest.analysis.rerank import RerankOutput, check_permutation, rerank, run_ranks
from arxiv_digest.archive.db import Archive
from arxiv_digest.llm.provider import LLMResponse
from arxiv_digest.pipeline import Pipeline
from arxiv_digest.report.digest import MARKER_RE

from conftest import FakeArxiv, FakeLLM, paper


class Judge:
    """Ranks labelled papers by a hidden quality number in the title ("Paper q=NN"); optionally garbles some calls."""

    def __init__(self, garble_every: int = 0):
        self.calls = 0
        self.garble_every = garble_every
        self.sizes: list[int] = []

    def chat(self, profile, messages, schema):
        self.calls += 1
        found = re.findall(r"^\[(P\d+)\] .*q=(\d+)", messages[1]["content"], flags=re.M)
        self.sizes.append(len(found))
        order = [lbl for lbl, q in sorted(found, key=lambda x: -int(x[1]))]
        if self.garble_every and self.calls % self.garble_every == 0:
            order = order[:-1]  # not a permutation → corrective retry, which is garbled the same way
        return LLMResponse(json.dumps({"ranking": order}), profile.model, metrics={"wall_s": 0.01})


def _archive(cfg, n, seed=3):
    archive = Archive(cfg.db_path)
    run_id = archive.create_run("run", None, None, None, "x")
    qualities = list(range(1, n + 1))
    random.Random(seed).shuffle(qualities)
    ids = []
    for i, q in enumerate(qualities, start=1):
        aid = f"2511.{i:05d}"
        archive.upsert_paper(paper(aid, f"Paper q={q:03d}"), run_id)
        ids.append(aid)
    return archive, run_id, ids, {aid: q for aid, q in zip(ids, qualities)}


def test_rerank_finds_the_true_top_papers(cfg):
    archive, run_id, ids, q = _archive(cfg, 60)
    cfg.raw["rerank"].update(group_size=10, keep_fraction=0.6, final_pool=40, window=10, step=5, top_n=10)
    judge = Judge()
    order, stats = rerank(cfg, archive, judge, run_id, ids)
    assert sorted(order) == sorted(ids)
    assert [q[a] for a in order[:10]] == list(range(60, 50, -1))  # exact top 10 with a consistent judge
    assert stats.rounds == 1 and stats.failed_calls == 0 and max(judge.sizes) <= 10
    ranks = run_ranks(archive, run_id)
    assert ranks[order[0]] == 1 and ranks[order[-1]] == 60


def test_failed_comparison_keeps_incoming_order(cfg):
    archive, run_id, ids, q = _archive(cfg, 12)
    cfg.raw["rerank"].update(final_pool=40, window=10, step=5, top_n=5)
    judge = Judge(garble_every=1)  # every call fails validation, including the retry
    order, stats = rerank(cfg, archive, judge, run_id, ids)
    assert order == ids and stats.failed_calls == stats.calls > 0


def test_permutation_check():
    labels = ["P1", "P2", "P3"]
    check_permutation(labels, RerankOutput(ranking=["P2", "[P1]", "P3"]))
    for bad in (["P1", "P2"], ["P1", "P1", "P2", "P3"], ["P1", "P2", "P9"]):
        try:
            check_permutation(labels, RerankOutput(ranking=bad))
        except Exception as exc:  # noqa: BLE001
            assert "exactly once" in str(exc)
        else:
            raise AssertionError(bad)


def test_digest_follows_ranking_not_score(cfg):
    from test_pipeline import NOW, PAPERS, SCORES

    cfg.raw["investigation"]["max_candidates"] = 0

    class Contrarian(FakeLLM):
        """Scores as usual but ranks the lowest-scoring screened paper first."""

        def chat(self, profile, messages, schema):
            if "ranking" in (schema or {}).get("properties", {}):
                found = re.findall(r"^\[(P\d+)\] (.+)$", messages[1]["content"], flags=re.M)
                order = sorted(found, key=lambda lt: next((v for t, v in self.scores.items() if lt[1].startswith(t)), 0))
                return LLMResponse(json.dumps({"ranking": [lbl for lbl, _ in order]}), profile.model, metrics={"wall_s": 0.01})
            return super().chat(profile, messages, schema)

    archive = Archive(cfg.db_path)
    out = Pipeline(cfg, archive=archive, client=FakeArxiv(PAPERS), provider=Contrarian(SCORES)).run(now=NOW)
    assert out.status == "SUCCESS", out.message
    # 58-point paper ranked first, then 76, then 91 (all ≥ min_score 60? only 76 and 91 are ranked; 58 < rerank.min_score)
    assert MARKER_RE.findall(open(out.report_path, encoding="utf-8").read()) == ["2511.00002", "2511.00001", "2511.00003"]
    m = json.loads(open(out.manifest_path, encoding="utf-8").read())
    assert m["stages"]["rerank"]["pool"] == 2 and m["stages"]["rerank"]["top"] == ["2511.00002", "2511.00001"]
    assert "Comparative ranking:" in open(out.report_path, encoding="utf-8").read()


def test_synthesis_recovers_project_ids_and_flags_thin_summary(cfg):
    from arxiv_digest.analysis.synthesis import synthesize

    archive = Archive(cfg.db_path)
    run_id = archive.create_run("run", None, None, None, "x")
    for i in range(1, 4):
        archive.upsert_paper(paper(f"2511.0000{i}", f"Paper {i}"), run_id)
    synth = {
        "executive_summary": [{"headline": "One", "detail": "d", "arxiv_ids": ["2511.00001"]}],
        "trends": [], "comparisons": [], "research_gaps": [],
        "experiments": [{"title": "Try", "objective": "o", "complexity": "low", "first_validation_steps": ["a"], "arxiv_ids": [],
                         "integration_points": ["ghidra_mcp: wire it into the plugin", "made_up: nope", "ghidra_mcp"]}],
    }
    data, flags, _ = synthesize(cfg, FakeLLM({}, synthesis=synth), archive, ["2511.00001", "2511.00002", "2511.00003"])
    assert data["experiments"][0]["integration_points"] == ["ghidra_mcp"]
    assert any("only 1 item" in f for f in flags) and any("made_up" in f for f in flags)
