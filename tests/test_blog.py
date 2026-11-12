"""Blog post rendering (escaping of untrusted text) and publishing into a Hugo repo (git, offline)."""

from __future__ import annotations

import json
import subprocess

import pytest

from arxiv_digest.archive.db import Archive
from arxiv_digest.pipeline import Pipeline
from arxiv_digest.report.blog import load_run, publish, render_post, text

from conftest import FakeArxiv, FakeLLM, paper
from test_pipeline import NOW, PAPERS, SCORES

EVIL = "Pwn <script>alert(1)</script> {{< include \"/etc/passwd\" >}} \\[x^2\\] $$y$$ [click](https://evil.example) *b* a|b"


def test_text_neutralises_html_shortcodes_math_and_links():
    out = text(EVIL + " see [[PAGE 15]]")
    for bad in ("<script", "{{<", ">}}", "\\[", "\\]", "$$", "[click]", "](", "https://", "*b*", "|"):
        assert bad not in out, bad
    assert "(p. 15)" in out and "&lt;script&gt;" in out


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def published_run(cfg, tmp_path):
    cfg.raw["investigation"]["max_candidates"] = 0
    archive = Archive(cfg.db_path)
    papers = PAPERS + [paper("2511.00009", EVIL, cats=("cs.CR",))]
    scores = dict(SCORES) | {EVIL: 88}
    out = Pipeline(cfg, archive=archive, client=FakeArxiv(papers), provider=FakeLLM(scores)).run(now=NOW)
    assert out.status == "SUCCESS", out.message
    return archive, out


def _blog_repo(tmp_path, with_remote=False):
    repo = tmp_path / "site"
    (repo / "content" / "blog").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "hugo.yaml").write_text("title: t\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    if with_remote:
        remote = tmp_path / "remote.git"
        _git(tmp_path, "init", "-q", "--bare", str(remote))
        _git(repo, "remote", "add", "origin", str(remote))
        _git(repo, "push", "-q", "origin", "main")
        return repo, remote
    return repo, None


def test_render_post_is_clean_and_linked(cfg, published_run):
    archive, out = published_run
    data = load_run(cfg, archive, out.run_id)
    slug, post = render_post(cfg, archive, data)
    assert slug == "weekly-research-digest-2026-11-16"
    head, body = post.split("\n---\n", 1)
    assert 'title: "Weekly Research Digest: Week of November 16, 2026"' in head and "draft: false" in head
    assert "<script" not in post and "{{< include" not in post and "\\[x" not in post and "](https://evil" not in post
    # every paper has a stable anchor, and the table links to it
    for n in range(1, len(data.paper_ids) + 1):
        assert f"{{#paper-{n}}}" in body and f"](#paper-{n})" in body
    assert "<!--more-->" in body
    order = ["{{< callout", "## Research overview", "## Ranked research index", "## Paper analyses", "## Methodology and coverage"]
    assert [body.index(h) for h in order] == sorted(body.index(h) for h in order)
    assert "PDF analyzed" in body or "Abstract only" in body
    assert "Relevance to my work" not in body
    assert body.count("{{< details") == body.count("{{< /details >}}")


def test_publish_write_commit_and_push(cfg, tmp_path, published_run):
    archive, out = published_run
    repo, remote = _blog_repo(tmp_path, with_remote=True)
    cfg.raw["blog"].update(repo=str(repo), hugo_check=False)
    # unrelated work in progress in the blog repo must not be swept into the commit
    (repo / "notes.md").write_text("draft\n", encoding="utf-8")
    _git(repo, "add", "notes.md")

    r = publish(cfg, archive, out.run_id, mode="write")
    assert r.written and not r.committed and r.error is None
    post = repo / "content" / "blog" / "weekly-research-digest-2026-11-16" / "index.md"
    assert post.exists()

    r = publish(cfg, archive, out.run_id, mode="push")  # same content as written, but never committed: publish it
    assert r.error is None and r.committed and r.pushed

    r = publish(cfg, archive, out.run_id, mode="commit")  # committed and unchanged: nothing to do
    assert r.error is None and not r.committed and "post unchanged" in r.notes

    post.write_text("stale\n", encoding="utf-8")  # an older version of the post is what's committed
    _git(repo, "commit", "-q", "-m", "older post", "--", "content/blog/weekly-research-digest-2026-11-16/index.md")
    r = publish(cfg, archive, out.run_id, mode="push")
    assert r.error is None and r.committed and r.pushed
    assert _git(repo, "log", "-1", "--format=%s").startswith("Update weekly research digest")
    files = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert files == ["content/blog/weekly-research-digest-2026-11-16/index.md"]
    assert "A  notes.md" in _git(repo, "status", "--porcelain")  # still staged, not committed
    assert _git(remote, "rev-parse", "main") == _git(repo, "rev-parse", "HEAD")


def test_publish_refuses_wrong_branch_and_reports_errors(cfg, tmp_path, published_run):
    archive, out = published_run
    repo, _ = _blog_repo(tmp_path)
    cfg.raw["blog"].update(repo=str(repo), hugo_check=False)
    _git(repo, "checkout", "-q", "-b", "feature")
    r = publish(cfg, archive, out.run_id, mode="commit")
    assert not r.committed and "branch" in r.error
    cfg.raw["blog"]["repo"] = str(tmp_path / "missing")
    assert "not found" in publish(cfg, archive, out.run_id).error


def test_pipeline_publishes_after_success_without_affecting_status(cfg, tmp_path):
    cfg.raw["investigation"]["max_candidates"] = 0
    repo, _ = _blog_repo(tmp_path)
    cfg.raw["blog"].update(enabled=True, repo=str(repo), hugo_check=False, mode="commit")
    archive = Archive(cfg.db_path)
    out = Pipeline(cfg, archive=archive, client=FakeArxiv(PAPERS), provider=FakeLLM(SCORES)).run(now=NOW)
    assert out.status == "SUCCESS"
    m = json.loads(open(out.manifest_path, encoding="utf-8").read())
    assert m["stages"]["blog"]["committed"] and m["stages"]["blog"]["error"] is None

    # a broken blog setup is reported in the manifest, the run still succeeds
    cfg.raw["blog"]["repo"] = str(tmp_path / "nowhere")
    out2 = Pipeline(cfg, archive=archive, client=FakeArxiv(PAPERS), provider=FakeLLM(SCORES)).run(now=NOW, force=True)
    m2 = json.loads(open(out2.manifest_path, encoding="utf-8").read())
    assert out2.status == "PARTIAL"  # nothing new to feature on a forced re-run (pre-existing rule), not a blog failure
    assert "blog" not in m2["stages"]  # PARTIAL runs are never published


# --- evidence helpers, corrections and the synthesis guard ------------------------------------------------------------
from arxiv_digest.report.blog import (METRICS, _Ctx, _metrics_note, apply_corrections, correct_synthesis, locate_pages,  # noqa: E402
                                      repo_role)


def test_locate_pages_prefers_cited_then_finds_the_real_page():
    pages = ["intro 88% and 20%", "table: 55.0% 90.0% $36.59", "appendix 55.0% 90.0% $36.59", "nothing"]
    assert locate_pages(pages, "55% / 90%, $36.59", [2]) == [2]  # cited page holds the numbers: kept
    assert locate_pages(pages, "55% / 90%, $36.59", [1]) == [2, 3]  # wrong citation: moved to where they appear
    assert locate_pages(pages, "15 vulnerabilities, 2 exploits", [4]) == [4]  # nothing distinctive found: unchanged
    assert locate_pages(None, "55%", [7]) == [7]


def test_repo_role_separates_own_code_from_cited_tools():
    text = "Our tool is available at https://github. com/JasonZhongZexin/APG-ICSE-Demo. We build on radareorg/radare2."
    assert repo_role("", "Rethinking Neural Decompilation: Candidate Selection", "UzL-ITS", "rethinking-neural-decompilation", "no") == "own"
    assert repo_role(text, "APG: Agentic Static-Analysis Infrastructure", "JasonZhongZexin", "APG-ICSE-Demo", "yes") == "own"
    assert repo_role("we use https://github.com/radareorg/radare2", "CyberBinGym: Can AI Agents", "radareorg", "radare2", "yes") == "related"
    assert repo_role("Our code is at https://github.com/x/y", "Some Paper", "x", "y", "no") == "related"  # authors say no code


def test_corrections_fix_items_and_report_misses():
    payload = {"claims": [{"statement": "Hardening reduces exploits", "pages": [9]}],
               "experiments": [{"reported_result": "No patch: 55%; With patch: 90%", "pages": [8]}]}
    notes, warnings = apply_corrections("1234.5678", payload, [
        {"arxiv_id": "1234.5678", "target": "claims", "match": "Hardening", "set_pages": [10], "note": "page"},
        {"arxiv_id": "1234.5678", "target": "experiments", "match": "No patch", "replace": [["No patch", "Vulnerable only"]], "note": "label"},
        {"arxiv_id": "1234.5678", "target": "claims", "match": "not there", "note": "x"},
        {"arxiv_id": "9999.0000", "target": "claims", "match": "Hardening", "note": "other paper"},
    ])
    assert payload["claims"][0]["pages"] == [10] and payload["claims"][0]["pages_fixed"]
    assert payload["experiments"][0]["reported_result"].startswith("Vulnerable only")
    assert notes == ["page", "label"] and len(warnings) == 1
    syn = {"trends": [{"trend": "T", "explanation": "X proves Y", "arxiv_ids": []}, {"trend": "Bad", "explanation": "e", "arxiv_ids": []}]}
    out, notes, warnings = correct_synthesis(syn, [
        {"target": "synthesis.trends", "match": "proves", "replace": [["proves", "suggests"]], "note": "n1"},
        {"target": "synthesis.trends", "match": "Bad", "drop": True, "note": "n2"}])
    assert [t["explanation"] for t in out["trends"]] == ["X suggests Y"] and syn["trends"][0]["explanation"] == "X proves Y"


def test_synthesis_guard_drops_numbers_not_in_cited_papers(cfg):
    class Doc:
        def __init__(self, t):
            self.t = t

        def full_text(self):
            return self.t

    ctx = _Ctx(cfg, None, None, {"2611.00001": 1}, [])
    ctx.docs = {"2611.00001": Doc("accuracy rose to 91.8% on x86-64")}
    items = [{"trend": "ok", "explanation": "reaches 91.8% on x86-64 (arXiv 2611.00001, 2026)", "arxiv_ids": ["2611.00001"]},
             {"trend": "made up", "explanation": "reaches 99.9%", "arxiv_ids": ["2611.00001"]},
             {"trend": "uncited", "explanation": "no paper", "arxiv_ids": ["2611.99999"]}]
    kept = ctx.cited(items)
    assert [k["trend"] for k in kept] == ["ok"] and any("99.9" in w for w in ctx.warnings)


def test_metric_definitions_appear_once(cfg):
    ctx = _Ctx(cfg, None, None, {}, [])
    first = _metrics_note(ctx, "re-executability and pass@1 and macro F1")
    assert "Re-executability" in first and "pass@k" in first and "Macro F1" in first and "**F1**" not in first
    assert _metrics_note(ctx, "re-executability again") is None
    assert {name for _, name, _ in METRICS} >= {"Re-executability", "Edit similarity", "Success@k"}


def test_post_written_earlier_is_still_committed_and_pushed(cfg, tmp_path, published_run):
    archive, out = published_run
    repo, remote = _blog_repo(tmp_path, with_remote=True)
    cfg.raw["blog"].update(repo=str(repo), hugo_check=False)
    assert publish(cfg, archive, out.run_id, mode="write").written
    r = publish(cfg, archive, out.run_id, mode="commit")  # unchanged on disk, but never committed
    assert r.committed and r.error is None
    assert "Add weekly research digest" in _git(repo, "log", "-1", "--format=%s")
    r = publish(cfg, archive, out.run_id, mode="push")  # committed earlier, not yet pushed
    assert r.pushed and not r.committed and _git(remote, "rev-parse", "main") == _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "other")
    r = publish(cfg, archive, out.run_id, mode="push")
    assert not r.pushed and "branch" in r.error  # never pushes another branch to main
