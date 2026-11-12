from pathlib import Path

from arxiv_digest.analysis.investigation import validate_investigation
from arxiv_digest.analysis.schemas import InvestigationOutput
from arxiv_digest.pdf import ExtractedDoc, artifact_urls, build_context

DOC = ExtractedDoc(
    path=Path("x.pdf"),
    pages=[
        "Title\nAbstract\nWe present NeuDec.",
        "4 Evaluation\nOn Decompile-Eval NeuDec reaches 63.2% re-executability versus 41.0% for the baseline.",
        "Code is at https://github.com/example/neudec\nReferences\n[1] ...",
    ],
    references_page=3,
)


def out(**kw) -> InvestigationOutput:
    base = dict(
        summary="s", methodology="m", technically_interesting="t", security_connection="direct", security_connection_explanation="e",
        scores={"alignment": 25, "applicability": 20, "cross_domain": 10, "novelty": 10, "reproducibility": 12},
        confidence="high", suspicious_instructions_detected=False,
        reproducibility={"code_available": "yes", "artifact_urls": ["https://github.com/example/neudec", "https://github.com/fake/hallucinated"]},
        claims=[
            {"statement": "Improves re-executability", "status": "demonstrated", "evidence": "Table", "pages": [2]},
            {"statement": "Works on all architectures", "status": "demonstrated", "evidence": "?", "pages": [9]},
        ],
        experiments=[
            {"description": "main", "dataset_or_benchmark": "Decompile-Eval", "metric": "re-exec", "reported_result": "63.2% vs 41.0%", "pages": [2]},
            {"description": "invented", "reported_result": "88.8%", "pages": [2]},
            {"description": "wrong page", "reported_result": "63.2%", "pages": [1]},
        ],
    )
    base.update(kw)
    return InvestigationOutput.model_validate(base)


def test_evidence_cross_checks():
    payload, flags, stats = validate_investigation(out(), DOC)
    claims = payload["claims"]
    assert claims[0]["status"] == "demonstrated"
    assert claims[1]["status"] == "claimed" and stats["claims_downgraded"] == 1  # page 9 does not exist
    exps = payload["experiments"]
    assert [e["description"] for e in exps] == ["main", "wrong page"]  # invented number dropped
    assert stats["experiments_dropped"] == 1 and stats["experiments_verified"] == 1 and stats["experiments_page_mismatch"] == 1
    assert payload["reproducibility"]["artifact_urls"] == ["https://github.com/example/neudec"]
    assert any("not present in the paper" in f for f in flags)


def test_context_budget_and_urls():
    ctx = build_context(DOC, max_chars=10_000)
    assert "[[PAGE 1]]" in ctx.text and "[[PAGE 2]]" in ctx.text and ctx.included_pages == [1, 2, 3]
    small = build_context(DOC, max_chars=120)
    assert small.truncated_pages or small.omitted_pages
    assert artifact_urls(DOC.full_text()) == ["https://github.com/example/neudec"]


def test_benchmark_summarizes_verifier_trials():
    from arxiv_digest.benchmark import Trial, summarize

    bcfg = {"papers": [{"arxiv_id": "2403.05286", "expected_bands": ["High"], "expected_topics": []}]}
    ev = lambda agrees, ch, major, adj, oor=False, bad=0: {  # noqa: E731
        "reviewed_model": "a", "agrees": agrees, "challenges": ch, "major_challenges": major, "adjustment": adj,
        "adjustment_out_of_range": oor, "invalid_challenge_pages": bad}
    trials = [
        Trial("v", "verification", "2403.05286", True, True, wall_s=10.0, evidence=ev(True, 2, 0, -3)),
        Trial("v", "verification", "2308.06782", True, False, wall_s=20.0, evidence=ev(False, 4, 1, -15, oor=True, bad=2)),
        Trial("v", "verification", "2308.06782", False, error="bad json"),
    ]
    s = summarize(bcfg, trials)["v"]
    assert s["verification_success"] == "2/3" and s["verification_first_attempt_valid"] == "1/3"
    assert s["verification_agree_rate"] == 0.5 and s["verification_mean_challenges"] == 3
    assert s["verification_major_challenges"] == 1 and s["verification_mean_adjustment"] == -9
    assert s["verification_out_of_range_adjustments"] == 1 and s["verification_invalid_pages"] == 2
    assert s["screening_success"] == "—"


def test_context_overflow_shrinks_text_then_fails_only_that_paper(cfg, monkeypatch):
    import json

    from arxiv_digest.analysis import investigation as inv
    from arxiv_digest.archive.db import Archive
    from arxiv_digest.llm.provider import LLMContextOverflow, LLMResponse

    from conftest import paper

    big = ExtractedDoc(path=Path("x.pdf"), pages=[f"4 Evaluation\nresult 63.2% on page {i}\n" + "x" * 5000 for i in range(1, 21)])
    monkeypatch.setattr(inv, "download_pdf", lambda *a, **k: Path("x.pdf"))
    monkeypatch.setattr(inv, "extract", lambda p: big)
    monkeypatch.setattr(inv, "check_investigation", lambda cfg, o: None)  # completeness checks are tested elsewhere
    archive = Archive(cfg.db_path)
    run_id = archive.create_run("run", None, None, None, "x")
    archive.upsert_paper(paper("2511.00001", "Decompiling firmware"), run_id)
    archive.upsert_paper(paper("2511.00002", "Fuzzing firmware"), run_id)

    class OverflowOnce:
        def __init__(self, always=False):
            self.lengths, self.always = [], always

        def chat(self, profile, messages, schema):
            self.lengths.append(len(messages[1]["content"]))
            if self.always or len(self.lengths) == 1:
                raise LLMContextOverflow("stopped at the context limit", prompt_tokens=30631, num_ctx=32768)
            body = out(claims=[], experiments=[], topics=[{"topic_id": "t1.reverse_engineering", "rationale": "r"}]).model_dump()
            return LLMResponse(json.dumps(body), profile.model, metrics={"wall_s": 0.01})

    cfg.raw["investigation"]["max_text_chars"] = 90000
    stats = inv.InvestigationStats(candidates=2)
    llm = OverflowOnce()
    inv.investigate_one(cfg, archive, llm, None, run_id, "2511.00001", stats)
    assert stats.investigated == 1 and not stats.llm_failures
    assert len(llm.lengths) == 2 and llm.lengths[1] < llm.lengths[0] * 0.85  # second attempt sent less text

    llm = OverflowOnce(always=True)
    inv.investigate_one(cfg, archive, llm, None, run_id, "2511.00002", stats)  # must not raise
    assert len(llm.lengths) == 2 and stats.llm_failures and "2511.00002" in stats.llm_failures[0]


def test_shrink_budget():
    from arxiv_digest.analysis.investigation import shrink_budget

    assert shrink_budget(90000, 30631, 32768, 6000) == int(90000 * 26768 / 30631 * 0.9)
    assert shrink_budget(90000, None, 32768, 6000) == 45000
    assert shrink_budget(4200, 30000, 32768, 6000) is None  # would drop below 4,000 chars
