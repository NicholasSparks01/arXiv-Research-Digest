"""Annotated PDFs: chunking, quote anchoring with PyMuPDF, caching, failure handling, and blog bundling."""

from __future__ import annotations

import json

import pymupdf
import pytest

from arxiv_digest.analysis import annotate as ann
from arxiv_digest.archive.db import Archive
from arxiv_digest.llm.provider import LLMResponse

from conftest import paper

PAGES = [
    "We introduce TESTBENCH, a benchmark of 100 binary tasks.\nAgents receive stripped binaries only.",
    "Success requires a proof of concept that crashes the vulnerable build but not the patched build.\n"
    "The best agent recovers 88% of vulnerabilities and exploits 20% of them.",
    "Surprisingly, hardening removes most exploits but two chains survive.\nReferences follow.",
]


def make_pdf(path, pages=PAGES):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        y = 72
        for line in text.split("\n"):
            page.insert_text((72, y), line, fontsize=10)
            y += 16
    doc.save(str(path))
    doc.close()
    return path


def test_chunk_pages_keeps_pages_whole():
    chunks = ann.chunk_pages(["a" * 50, "b" * 50, "c" * 50], chunk_chars=130)
    assert [(f, l) for f, l, _ in chunks] == [(1, 2), (3, 3)]
    assert "[[PAGE 1]]" in chunks[0][2] and "[[PAGE 3]]" in chunks[1][2]


def test_write_annotated_places_only_quotes_found_on_the_page(tmp_path):
    src = make_pdf(tmp_path / "p.pdf")
    proposed = [
        {"page": 1, "quote": "We introduce TESTBENCH, a benchmark of 100 binary tasks.", "kind": "key_aspect", "note": "core"},
        {"page": 2, "quote": "Success requires a proof of   concept that crashes the vulnerable build", "kind": "win_condition",
         "note": "criterion"},  # extra whitespace
        {"page": 2, "quote": "Surprisingly, hardening removes most exploits but two chains survive.", "kind": "interesting",
         "note": "on the next page"},  # wrong page by one
        {"page": 1, "quote": "This sentence does not appear anywhere in the paper at all.", "kind": "interesting",
         "note": "made up"},
    ]
    dest = tmp_path / "out" / "p-annotated.pdf"
    placed, dropped = ann.write_annotated(src, dest, "2611.00001", "Test", "fake-model", proposed,
                                          "Success = crash on vulnerable build only")
    assert len(placed) == 3 and dropped == 1 and placed[2]["placed_page"] == 3
    doc = pymupdf.open(str(dest))
    highlights = [a for p in doc for a in p.annots() if a.type[1] == "Highlight"]
    assert [a.info["subject"] for a in highlights] == ["Key aspect", "Win condition", "Interesting"]
    assert highlights[1].info["content"].startswith("Win condition: criterion")
    summary = [a for a in doc[0].annots() if a.type[1] == "Text"][0].info["content"]
    assert "Formula for success: Success = crash" in summary and "machine-generated" in summary and "arxiv.org/abs/2611.00001" in summary
    # the paper text itself is unchanged
    assert [p.get_text() for p in doc] == [p.get_text() for p in pymupdf.open(str(src))]


class ChunkLLM:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def chat(self, profile, messages, schema):
        self.calls += 1
        assert "annotations" in schema["properties"]
        if self.fail:
            return LLMResponse("not json", profile.model, metrics={"wall_s": 0.01})
        body = {"annotations": [
            {"page": 1, "quote": "We introduce TESTBENCH, a benchmark of 100 binary tasks.", "kind": "key_aspect", "note": "n1"},
            {"page": 2, "quote": "The best agent recovers 88% of vulnerabilities and exploits 20% of them.",
             "kind": "win_condition", "note": "n2"},
            {"page": 9, "quote": "out of this chunk", "kind": "interesting", "note": "ignored"}],
            "success_formula": "Success = PoC crashes vulnerable build only."}
        return LLMResponse(json.dumps(body), profile.model, metrics={"wall_s": 0.01})


@pytest.fixture
def paper_setup(cfg, tmp_path, monkeypatch):
    archive = Archive(cfg.db_path)
    run_id = archive.create_run("run", None, None, None, "x")
    archive.upsert_paper(paper("2611.00001", "Test benchmark paper"), run_id)
    src = make_pdf(tmp_path / "src.pdf")
    monkeypatch.setattr(ann, "download_pdf", lambda *a, **k: src)
    return archive, run_id


def test_annotate_paper_places_caches_and_reuses(cfg, paper_setup):
    archive, run_id = paper_setup
    llm = ChunkLLM()
    r = ann.annotate_paper(cfg, archive, llm, run_id, "2611.00001")
    assert r.error is None and r.placed == 2 and r.proposed == 2 and r.by_kind["win_condition"] == 1
    assert r.success_formula.startswith("Success = PoC") and ann.annotated_path(cfg, "2611.00001").exists()
    again = ann.annotate_paper(cfg, archive, llm, run_id, "2611.00001")
    assert again.cached and again.placed == 2 and llm.calls == 1  # no new model calls


def test_annotate_paper_failure_writes_nothing(cfg, paper_setup):
    archive, run_id = paper_setup
    r = ann.annotate_paper(cfg, archive, ChunkLLM(fail=True), run_id, "2611.00001")
    assert r.error and r.path is None and not ann.annotated_path(cfg, "2611.00001").exists()
    assert archive.latest_analysis("2611.00001", "annotation") is None


def test_blog_bundles_every_annotated_pdf(cfg, tmp_path):
    from arxiv_digest.report import blog

    for pid in ("2611.00001", "2611.00002"):
        ann.annotated_path(cfg, pid).parent.mkdir(parents=True, exist_ok=True)
        make_pdf(ann.annotated_path(cfg, pid))
    post_dir = tmp_path / "post"
    post_dir.mkdir()
    (post_dir / "2611.00009-annotated.pdf").write_bytes(b"stale")  # from an earlier version of the post
    published = blog._bundle_annotations(cfg, ["2611.00001", "2611.00002", "2611.00003"], post_dir)
    assert published == {"2611.00001": "2611.00001-annotated.pdf", "2611.00002": "2611.00002-annotated.pdf"}
    assert sorted(p.name for p in post_dir.iterdir()) == ["2611.00001-annotated.pdf", "2611.00002-annotated.pdf"]
