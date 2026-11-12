from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from arxiv_digest.config import load_config
from arxiv_digest.discovery.arxiv_client import PaperRecord, QueryResult
from arxiv_digest.llm.provider import LLMResponse, LLMUnavailable


@pytest.fixture
def cfg(tmp_path: Path):
    c = load_config()
    c.raw["paths"]["data_dir"] = str(tmp_path / "data")
    c.raw["paths"]["vault_dir"] = str(tmp_path / "vault")
    c.raw["github"]["enabled"] = False
    c.raw["annotation"]["enabled"] = False  # needs real PDFs; covered by test_annotate.py
    c.raw["adaptive"]["enabled"] = False
    c.raw["arxiv"]["min_request_interval_s"] = 0
    c.raw["triage"]["min_score"] = 40  # state-machine tests use a 40-90 score spread
    # never touch a real blog repo configured in config/local.toml
    c.raw["blog"] = {**c.raw.get("blog", {}), "enabled": False, "repo": str(tmp_path / "no-blog"), "mode": "write"}
    return c


def paper(arxiv_id: str, title: str, abstract: str = "", cats=("cs.CR",), version: int = 1, published="2026-11-03T10:00:00Z") -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id, version=version, title=title, abstract=abstract or f"Abstract of {title}.", authors=["A. Author"],
        primary_category=cats[0], categories=list(cats), published=published, updated=published,
    )


class FakeArxiv:
    """Category queries return `by_category[cat]`; topic queries return papers whose title matches a keyword."""

    def __init__(self, papers: list[PaperRecord], fail_queries: tuple[str, ...] = (), id_overrides: dict | None = None):
        self.papers = papers
        self.fail_queries = fail_queries
        self.id_overrides = id_overrides or {}
        self.queries: list[str] = []

    def search(self, query: str, max_results: int, sort_by: str = "submittedDate") -> QueryResult:
        self.queries.append(query)
        if any(f in query for f in self.fail_queries):
            return QueryResult(query=query, error="simulated arXiv failure")
        if query.startswith("(cat:") and query.count("cat:") == 1:
            cat = query.split("cat:")[1].split(")")[0]
            hits = [p for p in self.papers if cat in p.categories]
        else:
            hits = [p for p in self.papers if any(w in query.lower() for w in p.title.lower().split()[:2])]
        return QueryResult(query=query, total_results=len(hits), papers=hits[:max_results], pages=1)

    def fetch_ids(self, ids, batch: int = 100) -> QueryResult:
        by_id = {p.arxiv_id: p for p in self.papers} | self.id_overrides
        found = [by_id[i] for i in ids if i in by_id]
        return QueryResult(query=f"id_list[{len(ids)}]", total_results=len(ids), papers=found, pages=1)


def screening_json(total_hint: int, topics=("t1.reverse_engineering",), security="direct", recommendation="candidate") -> dict:
    # spread total over components within maxima
    maxima = [30, 25, 15, 15, 15]
    comps, left = [], total_hint
    for m in maxima:
        v = min(m, left)
        comps.append(v)
        left -= v
    return {
        "research_problem": "Problem.", "proposed_solution": "Solution.", "techniques": ["decompilation"],
        "topics": [{"topic_id": t, "rationale": "fits"} for t in topics], "security_connection": security,
        "security_connection_explanation": "explained", "applicable_projects": [{"project_id": "ghidra_mcp", "explanation": "useful"}],
        "scores": dict(zip(["alignment", "applicability", "cross_domain", "novelty", "reproducibility"], comps)),
        "confidence": "medium", "recommendation": recommendation, "rejection_reason": "", "artifacts_mentioned": [],
    }


def as_triage(screening: dict) -> dict:
    return {"topic_ids": [t["topic_id"] for t in screening["topics"]], "security_connection": screening["security_connection"],
            "scores": screening["scores"], "recommendation": screening["recommendation"]}


SYNTHESIS = {
    "executive_summary": [{"headline": "LLM decompilers improve", "detail": "Details.", "arxiv_ids": []}],
    "trends": [{"trend": "Agentic RE", "explanation": "More agents.", "arxiv_ids": []}],
    "comparisons": [], "research_gaps": [{"gap": "Benchmarks", "rationale": "Few.", "arxiv_ids": []}],
    "experiments": [{"title": "Try it", "objective": "Evaluate.", "integration_points": ["ghidra_mcp"], "complexity": "low",
                     "first_validation_steps": ["Install", "Run"], "arxiv_ids": []}],
}


class FakeLLM:
    """Scores papers by a title → score map; returns a fixed synthesis. No investigation candidates by default."""

    name = "fake"

    def __init__(self, scores: dict[str, int], available: bool = True, synthesis: dict | None = None, bad_json_for: set[str] = frozenset()):
        self.scores = scores
        self.available = available
        self.synthesis = synthesis or SYNTHESIS
        self.bad_json_for = bad_json_for
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def ensure_available(self):
        if not self.available:
            raise LLMUnavailable("fake ollama down")

    def list_models(self):
        return ["qwen3.5:9b", "gpt-oss:20b", "deepseek-r1:14b"]

    def chat(self, profile, messages, schema):
        self.calls += 1
        user = messages[1]["content"]
        props = set((schema or {}).get("properties", {}))
        if "executive_summary" in props:
            return LLMResponse(json.dumps(self.synthesis), profile.model, metrics={"wall_s": 0.01})
        if "ranking" in props:  # comparative re-ranking: order the labelled papers by this fake's score map
            self.prompts.append(("rerank", user))
            found = re.findall(r"^\[(P\d+)\] (.+)$", user, flags=re.M)
            order = sorted(found, key=lambda lt: -next((v for t, v in self.scores.items() if lt[1].startswith(t)), 0))
            return LLMResponse(json.dumps({"ranking": [label for label, _ in order]}), profile.model, metrics={"wall_s": 0.01})
        shape = as_triage if "topic_ids" in props else (lambda d: d)
        self.prompts.append(("triage" if "topic_ids" in props else "screening", user))
        for title, score in self.scores.items():
            if f"Title: {title}" in user:
                if title in self.bad_json_for:
                    return LLMResponse("not json", profile.model, metrics={"wall_s": 0.01})
                return LLMResponse(json.dumps(shape(screening_json(score))), profile.model, metrics={"wall_s": 0.01})
        return LLMResponse(json.dumps(shape(screening_json(10, topics=(), security="none", recommendation="reject"))), profile.model,
                           metrics={"wall_s": 0.01})
