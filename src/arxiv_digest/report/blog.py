"""Public blog post for a Hugo (Hextra) site, rendered from the same validated records as the weekly digest.

The digest in the vault is written for the researcher; the blog post is written for readers and RSS subscribers:
highlights first, one section per paper with verified results, the cross-paper synthesis, and a collapsed note on how
the digest was made. Pipeline internals (search tables, cache counts, injection flags) stay in the vault report.

Every piece of paper- or model-derived text is escaped before it reaches the post. The site renders raw HTML
(goldmark `unsafe: true`) and Hugo shortcodes, so an unescaped title like `<script>…` or `{{< include … >}}` would
execute on the site. Links are only ever built by this module (arXiv ids, verified GitHub repositories).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.ids import abs_url, pdf_url
from arxiv_digest.report.digest import cfg_project

log = logging.getLogger(__name__)

# Characters with Markdown, math-passthrough (\[ \] \( \) $$) or shortcode meaning become numeric character references,
# which render as the same character but are never parsed as syntax. (Backslash escapes are not safe here: on this
# site "\[" opens a display-math block.)
_ENTITIES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\\": "&#92;", "`": "&#96;", "*": "&#42;", "_": "&#95;",
             "[": "&#91;", "]": "&#93;", "|": "&#124;", "$": "&#36;", "{": "&#123;", "}": "&#125;", "~": "&#126;"}
_ENTITY_RE = re.compile("|".join(re.escape(k) for k in _ENTITIES))
_PAGE_MARKER = re.compile(r"\[\[\s*PAGE\s+(\d+)\s*\]\]", re.IGNORECASE)


# --- escaping -----------------------------------------------------------------------
def text(s: str | None, max_chars: int | None = None) -> str:
    """Untrusted text → inert Markdown: no HTML, shortcodes, math, links, images, emphasis or auto-linked URLs."""
    s = _PAGE_MARKER.sub(r"(p. \1)", s or "")  # the model sometimes copies our context page markers
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        s = s[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    s = _ENTITY_RE.sub(lambda m: _ENTITIES[m.group(0)], s)
    return s.replace("://", ":&#47;&#47;")  # stop goldmark's linkify from turning model text into links


def yaml_str(s: str) -> str:
    """Front-matter string. Themes may render descriptions as Markdown/HTML, so markup characters are dropped."""
    s = re.sub(r"[<>{}\[\]`$\\*|]", "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pages(pages) -> str:
    return f" (p. {', '.join(str(p) for p in pages)})" if pages else ""


# --- data -----------------------------------------------------------------------------
@dataclass
class PostData:
    run_id: int
    week_of: date
    research_week: tuple[date, date]
    window: tuple[date, date]  # collection window, inclusive (the initial run covers 30 days, weekly runs ~10)
    generated: datetime
    paper_ids: list[str]
    synthesis: dict
    stages: dict


def load_run(cfg: Config, archive: Archive, run_id: int) -> PostData:
    run = archive.run(run_id)
    if run is None:
        raise ValueError(f"run {run_id} does not exist")
    if run["status"] != "SUCCESS":
        raise ValueError(f"run {run_id} is {run['status']}; only successful runs are published")
    rep = archive.conn.execute(
        "SELECT paper_ids, created_at FROM reports WHERE run_id=? AND kind='digest' AND validated=1 ORDER BY id DESC LIMIT 1",
        (run_id,)).fetchone()
    if rep is None:
        raise ValueError(f"run {run_id} has no validated digest")
    run_dir = Path(run["manifest_path"]).parent
    manifest = json.loads(Path(run["manifest_path"]).read_text(encoding="utf-8"))
    synth_path = run_dir / "synthesis.json"
    synthesis = json.loads(synth_path.read_text(encoding="utf-8")) if synth_path.exists() else {}
    start, end = (date.fromisoformat(x) for x in run["research_week"].split(".."))
    w_start = datetime.fromisoformat(run["window_start"]).date()
    w_end = datetime.fromisoformat(run["window_end"]).date() - timedelta(days=1)  # window end is exclusive (midnight)
    return PostData(run_id, end + timedelta(days=1), (start, end), (w_start, w_end), datetime.fromisoformat(rep["created_at"]),
                    json.loads(rep["paper_ids"]), synthesis, manifest.get("stages", {}))


def _fmt_day(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def _range(a: date, b: date) -> str:
    if a.year != b.year:
        return f"{_fmt_day(a)} to {_fmt_day(b)}"
    return f"{a:%B} {a.day} to {b.day}, {b.year}" if a.month == b.month else f"{a:%B} {a.day} to {b:%B} {b.day}, {b.year}"


def slug_for(week_of: date) -> str:
    return f"weekly-research-digest-{week_of:%Y-%m-%d}"


def _short_title(title: str) -> str:
    head = re.split(r":\s", title, maxsplit=1)[0]
    return head if 3 <= len(head) <= 40 else (title[:40].rsplit(" ", 1)[0] + "…" if len(title) > 40 else title)


# --- evidence helpers -----------------------------------------------------------------------
EVIDENCE = {"abstract_only": "Abstract only", "full_text_reviewed": "PDF analyzed",
            "independently_corroborated": "PDF analyzed, independently checked", "legacy": "Earlier archive"}

# Plain-language definitions shown the first time a metric appears in a post. (pattern, name, definition)
METRICS = [
    (r"re-?executab", "Re-executability", "share of outputs that recompile and pass the benchmark's original tests"),
    (r"re-?compilab", "Re-compilability", "share of outputs that compile without errors"),
    (r"edit[ -]?sim", "Edit similarity", "closeness to the reference output from normalised edit distance; 100% means identical"),
    (r"full[- ]match|cfg (?:similarity|accuracy)", "CFG similarity and full match",
     "how closely the recovered control-flow graph matches the reference; a full match means the two graphs are identical"),
    (r"pass@", "pass@k", "share of problems where at least one of the first k generated candidates passes the tests"),
    (r"success@", "Success@k", "share of tasks solved in at least one of k independent attempts"),
    (r"macro[ -]?f1", "Macro F1", "F1 averaged over classes with equal weight, so a rare class counts as much as a common one"),
    (r"(?<![a-z])f1(?![a-z])", "F1", "harmonic mean of precision and recall (1.0 is perfect)"),
    (r"codebleu", "CodeBLEU", "BLEU extended with syntax-tree and data-flow matching for code"),
    (r"(?<!code)bleu", "BLEU", "n-gram overlap between generated and reference text"),
    (r"rouge-?l", "ROUGE-L", "overlap based on the longest common subsequence with the reference"),
    (r"levenshtein", "Levenshtein distance", "number of single-character edits needed to turn the output into the reference"),
    (r"(?<![a-z])auc(?![a-z])|auroc", "AUC", "area under the ROC curve; 0.5 is chance level and 1.0 is perfect separation"),
    (r"perplexity", "Perplexity", "how unlikely the model finds a text; lower means more likely"),
]
_DISTINCT_NUM = re.compile(r"\d+\.\d+|\d{2,}")
_OWN_CODE_CONTEXT = re.compile(
    r"\b(?:our|we\s+(?:release|open[- ]?source|make|provide|publish|share))\b"
    r"|\b(?:code|implementation|artifacts?|artefacts?|prototype|dataset|benchmark|tool)s?\s+(?:is|are)\s+"
    r"(?:publicly\s+|freely\s+|openly\s+)?(?:available|released|accessible)",
    re.IGNORECASE)


def locate_pages(pages: list[str] | None, result: str | None, cited: list[int]) -> list[int]:
    """Pages for a quoted result: the cited pages that contain all of its distinctive numbers, otherwise the (at most
    two) pages where they all appear. Falls back to the cited pages when the numbers are too common to place."""
    nums = sorted(set(_DISTINCT_NUM.findall(result or "")))
    if not pages or not nums:
        return cited

    def has_all(p: int) -> bool:
        return 1 <= p <= len(pages) and all(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", pages[p - 1]) for n in nums)

    if cited and any(has_all(p) for p in cited):
        return [p for p in cited if has_all(p)]
    found = [p for p in range(1, len(pages) + 1) if has_all(p)]
    return found[:2] if 0 < len(found) <= 3 else cited


def repo_role(full_text: str, title: str, owner: str, repo: str, code_available: str) -> str:
    """"own" when the repository looks like the paper's implementation, otherwise "related"."""
    short = re.split(r":\s", title, maxsplit=1)[0].lower()
    name = repo.lower()
    title_words = {w for w in re.findall(r"[a-z0-9]{4,}", title.lower())}
    repo_words = {w for w in re.findall(r"[a-z0-9]{4,}", name.replace("_", "-"))}
    short_key = re.sub(r"[^a-z0-9]", "", short)
    first_token = re.split(r"[-_.]", name)[0]
    if (len(title_words & repo_words) >= 2 or (len(short_key) >= 4 and short_key == re.sub(r"[^a-z0-9]", "", name))
            or (3 <= len(short_key) <= 20 and first_token == short_key)):
        return "own"
    if code_available == "yes" and full_text:
        loose = r"[\s-]*".join(re.escape(c) for c in f"{owner}/{repo}")
        flat = re.sub(r"\s+", " ", full_text)
        # PDF text often breaks URLs across lines ("https://github. com/owner/repo")
        for m in re.finditer(r"github\.\s*com\s*/\s*" + loose, flat, flags=re.IGNORECASE):
            if _OWN_CODE_CONTEXT.search(flat[max(0, m.start() - 250): m.start()]):
                return "own"
    return "related"


def load_corrections(run_dir: Path) -> list[dict]:
    path = run_dir / "blog-corrections.toml"
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        return list(tomllib.load(fh).get("correction", []))


def apply_corrections(pid: str, payload: dict, corrections: list[dict]) -> tuple[list[str], list[str]]:
    """Apply editorial corrections for one paper in place. Returns (applied notes, warnings for unmatched ones)."""
    applied, warnings = [], []
    fields = {"claims": ("statement",), "experiments": ("reported_result", "dataset_or_benchmark", "description", "metric"),
              "limitations": ("text",)}
    for c in (c for c in corrections if c.get("arxiv_id") == pid):
        target, match = c.get("target", ""), c.get("match", "")
        hit = False
        if target in fields:
            for item in payload.get(target, []):
                if any(match and match in (item.get(f) or "") for f in fields[target]):
                    hit = True
                    for old, new in c.get("replace", []):
                        for f in fields[target]:
                            if isinstance(item.get(f), str):
                                item[f] = item[f].replace(old, new)
                    if "set_pages" in c:
                        item["pages"] = list(c["set_pages"])
                        item["pages_fixed"] = True
        elif isinstance(payload.get(target), str) and match in payload[target]:
            hit = True
            for old, new in c.get("replace", []):
                payload[target] = payload[target].replace(old, new)
        if hit:
            applied.append(c.get("note", "editorial correction"))
        else:
            warnings.append(f"{pid}: correction not applied (no {target!r} text contains {match!r})")
    return applied, warnings


_NOT_A_RESULT = re.compile(r"\b\d{4}\.\d{4,5}\b|x86[-_]?64|arm64|aarch64|-O\d|\b(?:19|20)\d{2}\b", re.IGNORECASE)
_SYNTH_TEXT = ("headline", "detail", "takeaway", "theme", "comparison", "method", "explanation", "tension", "trend", "gap",
               "rationale", "title", "objective")


def correct_synthesis(synthesis: dict, corrections: list[dict]) -> tuple[dict, list[str], list[str]]:
    """Apply `target = "synthesis.<section>"` corrections (replace text or drop items). Returns (copy, notes, warnings)."""
    syn = json.loads(json.dumps(synthesis))
    notes, warnings = [], []
    for c in corrections:
        target = c.get("target", "")
        if not target.startswith("synthesis."):
            continue
        section, match = target.split(".", 1)[1], c.get("match", "")
        items = syn.get(section) or []
        hits = [it for it in items if any(match and match in (it.get(f) or "") for f in _SYNTH_TEXT)]
        if not hits:
            warnings.append(f"synthesis correction not applied (no {section!r} item contains {match!r})")
            continue
        if c.get("drop"):
            syn[section] = [it for it in items if it not in hits]
        else:
            for it in hits:
                for old, new in c.get("replace", []):
                    for f in _SYNTH_TEXT:
                        if isinstance(it.get(f), str):
                            it[f] = it[f].replace(old, new)
        notes.append(c.get("note", "editorial correction"))
    return syn, notes, warnings


# --- rendering -----------------------------------------------------------------------------
@dataclass
class _Ctx:
    cfg: Config
    archive: Archive
    data: PostData
    numbering: dict[str, int]
    corrections: list[dict]
    defined_metrics: set[str] = field(default_factory=set)
    synthesis_notes: list[str] = field(default_factory=list)
    annotated: dict[str, str] = field(default_factory=dict)  # arxiv id -> annotated PDF file name in the post
    warnings: list[str] = field(default_factory=list)
    docs: dict = field(default_factory=dict)

    def doc(self, pid: str):
        if pid not in self.docs:
            p = self.archive.paper(pid)
            path = self.cfg.pdf_dir / f"{pid}v{p['latest_version']}.pdf"
            try:
                from arxiv_digest.pdf import extract

                self.docs[pid] = extract(path) if path.exists() else None
            except Exception:  # noqa: BLE001 - a missing or unreadable PDF only disables page relocation
                self.docs[pid] = None
        return self.docs[pid]

    def ref(self, ids: list[str]) -> str:
        out = []
        for i in ids:
            if i in self.numbering:
                out.append(f"[#{self.numbering[i]}](#paper-{self.numbering[i]})")
            elif re.fullmatch(r"\d{4}\.\d{4,5}", i or ""):
                out.append(f"[arXiv:{i}]({abs_url(i)})")
        return f" ({', '.join(out)})" if out else ""

    def cited(self, items: list[dict]) -> list[dict]:
        """Synthesis items are published only if they cite a paper of this digest and every result-like number in them
        appears in the PDF of a paper they cite (a guard against numbers the synthesis model made up)."""
        kept = []
        for it in items:
            ids = [i for i in it.get("arxiv_ids") or ([it["arxiv_id"]] if it.get("arxiv_id") else []) if i in self.numbering]
            if not ids:
                continue
            blob = _NOT_A_RESULT.sub(" ", " ".join(str(it.get(f) or "") for f in _SYNTH_TEXT))
            nums = sorted(set(_DISTINCT_NUM.findall(blob)))
            texts = [d.full_text() for d in (self.doc(i) for i in ids) if d is not None]
            missing = [n for n in nums if texts and not any(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", t) for t in texts)]
            if missing:
                self.warnings.append(f"synthesis item dropped: number(s) {missing} not found in the cited paper(s): "
                                     f"{(it.get('headline') or it.get('theme') or it.get('trend') or it.get('gap') or it.get('takeaway') or '')[:80]!r}")
                continue
            kept.append(it)
        return kept


def render_post(cfg: Config, archive: Archive, data: PostData, run_dir: Path | None = None) -> tuple[str, str]:
    """Return (slug, Markdown) for the Hugo post."""
    slug, post, _ = render_post_with_warnings(cfg, archive, data, run_dir)
    return slug, post


def render_post_with_warnings(cfg: Config, archive: Archive, data: PostData, run_dir: Path | None = None,
                              annotated: dict[str, str] | None = None) -> tuple[str, str, list[str]]:
    bcfg = cfg.section("blog")
    corrections = load_corrections(run_dir) if run_dir else []
    synthesis, synth_notes, synth_warnings = correct_synthesis(data.synthesis, corrections)
    data = PostData(**{**data.__dict__, "synthesis": synthesis})
    ctx = _Ctx(cfg, archive, data, {pid: n for n, pid in enumerate(data.paper_ids, start=1)}, corrections)
    ctx.warnings += synth_warnings
    ctx.synthesis_notes = synth_notes
    ctx.annotated = dict(annotated or {})
    titles = {pid: archive.paper(pid)["title"] for pid in data.paper_ids}
    week = _range(*data.window)
    lead = [_short_title(titles[p]) for p in data.paper_ids[:3]]
    description = (f"{len(data.paper_ids)} new arXiv papers on reverse engineering, vulnerability research and "
                   f"AI-assisted security from {week}" + (f", led by {', '.join(lead)}." if lead else "."))
    lines = [
        "---",
        f"title: {yaml_str(bcfg.get('title_prefix', 'Weekly Research Digest') + ': Week of ' + _fmt_day(data.week_of))}",
        f"date: {data.generated.astimezone().isoformat(timespec='seconds')}",
        f"draft: {'true' if bcfg.get('draft', False) else 'false'}",
        f"description: {yaml_str(description)}",
        "tags:", *[f"  - {yaml_str(t)}" for t in bcfg.get("tags", ["Research Digest", "arXiv"])],
    ]
    author = bcfg.get("author")
    if author:
        lines += ["authors:", f"  - name: {yaml_str(author.get('name', ''))}"]
        lines += [f"    {k}: {yaml_str(author[k])}" for k in ("link", "image") if author.get(k)]
    lines += ["digest:", f"  run_id: {data.run_id}", f"  window: {yaml_str(week)}", "---", ""]

    # 1. introduction and disclosure
    lines += [f"This digest covers **{len(data.paper_ids)} papers** submitted to arXiv from {week}, chosen for reverse "
              "engineering, binary analysis, vulnerability research and AI-assisted security tooling.", "", "<!--more-->", "",
              '{{< callout type="info" >}}',
              "**AI-assisted analysis.** A local language model selected, summarised and compared these papers. For papers "
              "marked *PDF analyzed*, the numbers under *Key results* were checked against the paper's PDF and their page "
              "references point to where they appear. This is not independent academic peer review: summaries and "
              "interpretations can be incomplete or wrong, so read the original paper before relying on a finding.",
              "{{< /callout >}}", ""]

    # 2. research overview
    lines += _overview(ctx)

    # 3. highlights (findings)
    exec_items = ctx.cited(data.synthesis.get("executive_summary") or [])
    if exec_items:
        lines += ["## Highlights", "", "*Findings reported by the papers themselves.*", ""]
        lines += [f"- **{text(e.get('headline'))}**: {text(e.get('detail'))}{ctx.ref(e.get('arxiv_ids', []))}" for e in exec_items]
        lines.append("")

    # 4. ranked index
    lines += _index(ctx, titles)

    # 5. paper analyses
    lines += ["## Paper analyses", ""]
    takeaways = {t.get("arxiv_id"): t.get("takeaway") for t in ctx.cited(data.synthesis.get("takeaways") or [])}
    for pid in data.paper_ids:
        lines += _render_paper(ctx, pid, takeaways.get(pid), bool(bcfg.get("include_relevance", True)))

    # 6. synthesis
    lines += _synthesis(ctx)

    # 7. methodology
    lines += _render_method(ctx)
    return slug_for(data.week_of), "\n".join(lines).rstrip() + "\n", ctx.warnings


def _overview(ctx: _Ctx) -> list[str]:
    st, ids = ctx.data.stages, ctx.data.paper_ids
    scores = {pid: ctx.archive.current_score(pid) for pid in ids}
    pdfs = sum(1 for s in scores.values() if s and s.evidence_level in ("full_text_reviewed", "independently_corroborated"))
    cats: dict[str, int] = {}
    for pid in ids:
        c = ctx.archive.paper(pid)["primary_category"]
        cats[c] = cats.get(c, 0) + 1
    scr, tri, disc, inv = (st.get(k, {}) for k in ("screening", "triage", "discovery", "investigation"))
    out = ["## Research overview", ""]
    if disc.get("unique_papers"):
        out.append(f"- **Papers collected:** {disc['unique_papers']:,}")
    if tri.get("attempted") and scr.get("attempted"):
        out.append(f"- **Abstracts screened:** {tri['attempted']:,} checked for relevance; the top {scr['attempted']:,} "
                   "screened in detail")
    elif scr.get("attempted"):
        out.append(f"- **Abstracts screened:** {scr['attempted']:,}")
    out.append(f"- **PDFs analyzed:** {pdfs} of the {len(ids)} selected papers"
               + (f" ({inv['candidates']} in total)" if inv.get("candidates") else ""))
    out.append(f"- **Papers selected:** {len(ids)}")
    if ctx.annotated:
        out.append(f"- **Annotated PDFs:** {len(ctx.annotated)} of {len(ids)} papers")
    out.append("- **arXiv categories represented:** "
               + ", ".join(f"{text(c)} ({n})" for c, n in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))))
    return out + [""]


def _rubric(cfg: Config) -> tuple[str, str]:
    sc = cfg.section("scoring")
    parts = [("topic alignment", "alignment_max"), ("applicability", "applicability_max"),
             ("cross-domain relevance", "cross_domain_max"), ("novelty", "novelty_max"), ("reproducibility", "reproducibility_max")]
    rubric = ", ".join(f"{name} {int(sc[k])}" for name, k in parts if k in sc)
    bands = sorted(sc.get("bands", {}).items(), key=lambda kv: -int(kv[1]))
    return rubric, ", ".join(f"{text(n)} ≥ {int(v)}" for n, v in bands)


def _index(ctx: _Ctx, titles: dict[str, str]) -> list[str]:
    rubric, bands = _rubric(ctx.cfg)
    out = ["## Ranked research index", "",
           "**Rank** is this digest's order. The model compared the shortlisted papers about ten at a time, using "
           "summaries of their abstracts, before any PDF was read. **Score** is the paper's own result out of 100 on a "
           f"fixed rubric ({rubric}), updated after its PDF was analyzed; the label is the score's range ({bands}). "
           "Rank and score come from different steps, so a paper can rank above one with a higher score.", "",
           "| Rank | Paper | Area | Score (of 100) |", "|--:|---|---|---|"]
    for pid in ctx.data.paper_ids:
        s, n = ctx.archive.current_score(pid), ctx.numbering[pid]
        out.append(f"| {n} | [{text(titles[pid], 110)}](#paper-{n}) | {text(ctx.archive.paper(pid)['primary_category'])} "
                   f"| {s.total} · {text(s.band)} |")
    return out + [""]


def _metrics_note(ctx: _Ctx, blob: str) -> str | None:
    new = []
    low = blob.lower()
    for pattern, name, definition in METRICS:
        if name in ctx.defined_metrics or not re.search(pattern, low):
            continue
        if name == "F1" and "Macro F1" in [n for n, _ in new]:
            continue
        ctx.defined_metrics.add(name)
        new.append((name, definition))
    return "*Metrics:* " + "; ".join(f"**{n}** is the {d}" for n, d in new) + "." if new else None


def _render_paper(ctx: _Ctx, pid: str, takeaway: str | None, relevance: bool) -> list[str]:
    cfg, archive = ctx.cfg, ctx.archive
    n = ctx.numbering[pid]
    p = archive.paper(pid)
    s = archive.current_score(pid)
    inv = archive.latest_analysis(pid, "investigation")
    scr = archive.latest_analysis(pid, "screening")
    gh = archive.latest_analysis(pid, "github")
    repos = [r for r in (json.loads(gh["payload"]).get("repos", []) if gh else []) if r.get("status") == "verified"
             and re.fullmatch(r"[\w.-]+", r.get("owner", "")) and re.fullmatch(r"[\w.-]+", r.get("repo", ""))]
    evidence = EVIDENCE.get(s.evidence_level, s.evidence_level)
    out = [f"### {n}. {text(p['title'])} {{#paper-{n}}}", "",
           f"**Rank {n}** · Score **{s.total}/100** ({text(s.band)}) · {text(p['primary_category'])} · {text(evidence)} · "
           f"[arXiv]({abs_url(pid)}) · [PDF]({pdf_url(pid)})"
           + (f" · [Annotated PDF]({ctx.annotated[pid]})" if pid in ctx.annotated else ""), ""]
    details: list[str] = []
    code = "unclear"
    if inv:
        d = json.loads(inv["payload"])
        applied, warnings = apply_corrections(pid, d, ctx.corrections)
        ctx.warnings += warnings
        doc = ctx.doc(pid)
        out += [text(d.get("summary")), ""]
        if takeaway:
            out += [f"**Reader takeaway:** {text(takeaway)}", ""]
        if d.get("technically_interesting"):
            out += [f"**Why it stands out:** {text(d.get('technically_interesting'))}", ""]
        exps = [e for e in d.get("experiments", []) if e.get("reported_result")][:5]
        if exps:
            out += ["**Key results** (numbers checked against the PDF):", ""]
            blob = []
            for e in exps:
                pages = e.get("pages") or []
                if not e.get("pages_fixed"):
                    pages = locate_pages(doc.pages if doc else None, e.get("reported_result"), pages)
                label = text(e.get("dataset_or_benchmark") or e.get("description"), 140)
                metric = f", {text(e.get('metric'), 140)}" if e.get("metric") else ""
                out.append(f"- **{label}**{metric}: {text(e['reported_result'], 450)}{_pages(pages)}")
                blob.append(" ".join(str(e.get(k) or "") for k in ("metric", "reported_result", "dataset_or_benchmark")))
            out.append("")
            note = _metrics_note(ctx, " ".join(blob))
            if note:
                out += [note, ""]
        code = (d.get("reproducibility") or {}).get("code_available", "unclear")
        for c in [c for c in d.get("claims", []) if c.get("status") == "demonstrated"][:4]:
            details.append(f"- **Demonstrated:** {text(c.get('statement'))}{_pages(c.get('pages'))}")
        for c in [c for c in d.get("claims", []) if c.get("status") != "demonstrated"][:3]:
            details.append(f"- **Claimed, not demonstrated:** {text(c.get('statement'))}")
        for lim in d.get("limitations", [])[:3]:
            who = "stated by the authors" if lim.get("source") == "stated_by_authors" else "reviewer's observation"
            details.append(f"- **Limitation** ({who}): {text(lim.get('text'))}")
        if d.get("methodology"):
            details.append(f"- **Method:** {text(d.get('methodology'), 900)}")
        if relevance:
            for r in d.get("project_relevance", [])[:3]:
                details.append(f"- **Relevance** ({text(cfg_project(cfg, r.get('project_id')))}): {text(r.get('explanation'))}")
        details += [f"- **Editorial correction:** {text(note)}" for note in applied]
    elif scr:
        d = json.loads(scr["payload"])
        out += [f"{text(d.get('research_problem'))} {text(d.get('proposed_solution'))}".strip(), ""]
        if takeaway:
            out += [f"**Reader takeaway:** {text(takeaway)}", ""]
        out += ["*Abstract only: no results were checked against the full text.*", ""]
        if relevance:
            for r in d.get("applicable_projects", [])[:3]:
                details.append(f"- **Relevance** ({text(cfg_project(cfg, r.get('project_id')))}): {text(r.get('explanation'))}")
    doc = ctx.doc(pid) if repos else None
    full = doc.full_text() if doc else ""
    own = [r for r in repos if repo_role(full, p["title"], r["owner"], r["repo"], code) == "own"]
    related = [r for r in repos if r not in own]

    def links(rs: list[dict]) -> str:
        return ", ".join(f"[{text(r['owner'])}/{text(r['repo'])}](https://github.com/{r['owner']}/{r['repo']})" for r in rs)

    if own:
        out += [f"**Paper's code:** {links(own)}", ""]
    elif code == "yes":
        out += ["**Paper's code:** the authors say code is available, but no repository link could be verified.", ""]
    if related:
        out += [f"**Related tools and projects** (cited by the paper, not its implementation): {links(related)}", ""]
    if details:
        out += ['{{< details title="Claims, limitations, method and relevance" closed="true" >}}', "", *details, "",
                "{{< /details >}}", ""]
    return out


def _synthesis(ctx: _Ctx) -> list[str]:
    syn = ctx.data.synthesis
    parts = [
        ("How the papers relate", "Interpretation", "comparisons", "theme", "comparison"),
        ("Shared methods", "Interpretation", "shared_methods", "method", "explanation"),
        ("Tensions and disagreements", "Interpretation", "tensions", "tension", "explanation"),
        ("Emerging trends", "Interpretation", "trends", "trend", "explanation"),
        ("Open questions", "Open question", "research_gaps", "gap", "rationale"),
    ]
    body: list[str] = []
    for heading, kind, key, head_field, text_field in parts:
        items = ctx.cited(syn.get(key) or [])
        if items:
            body += [f"### {heading}", "", f"*{kind}: {_KIND_NOTE[kind]}*", ""]
            body += [f"- **{text(it.get(head_field))}**: {text(it.get(text_field))}{ctx.ref(it.get('arxiv_ids', []))}" for it in items]
            body.append("")
    exps = ctx.cited(syn.get("experiments") or [])
    if exps:
        body += ["### Experiments worth trying", "", f"*Proposal: {_KIND_NOTE['Proposal']}*", ""]
        for i, e in enumerate(exps, start=1):
            body.append(f"{i}. **{text(e.get('title'))}** ({text(e.get('complexity'))} effort): {text(e.get('objective'))}"
                        f"{ctx.ref(e.get('arxiv_ids', []))}")
            body += [f"   - {text(step)}" for step in (e.get("first_validation_steps") or [])[:5]]
        body.append("")
    if not body:
        return []
    return ["## Cross-paper synthesis", "",
            "This section compares the papers above. Each part says what kind of statement it makes: a **finding** was "
            "demonstrated in a source paper, an **interpretation** comes from comparing papers, and a **proposal** is an "
            "untested idea. The links point to the paper sections each statement is based on.", "", *body]


_KIND_NOTE = {
    "Interpretation": "drawn by comparing the linked papers, not a result of any single paper.",
    "Open question": "left unanswered by the linked papers.",
    "Proposal": "hypotheses to test, not results.",
}


def _render_method(ctx: _Ctx) -> list[str]:
    cfg, data = ctx.cfg, ctx.data
    st = data.stages
    disc, tri, scr, rk, inv = (st.get(k, {}) for k in ("discovery", "triage", "screening", "rerank", "investigation"))
    model = (st.get("models", {}).get("screener") or {}).get("model") or cfg.profile("screener").model
    rubric, bands = _rubric(cfg)
    budget = int(cfg.section("investigation").get("max_text_chars", 160000))
    cats = ", ".join(text(c) for c in cfg.categories)
    out = ["## Methodology and coverage", "", '{{< details title="How this digest was made" closed="true" >}}', "",
           "**Coverage**", "",
           f"- arXiv submissions from {_range(*data.window)} in {cats}, plus {len(cfg.topic_searches)} targeted topic "
           "searches." + (f" {disc['unique_papers']:,} unique papers were collected." if disc.get("unique_papers") else ""),
           "- Papers in the security-focused categories are always considered; papers from the large machine-learning "
           "categories are considered only if a keyword filter matches. Relevant papers filed elsewhere can be missed.", "",
           "**Selection, ranking and scoring**", ""]
    if tri:
        out.append(f"- **Quick relevance check:** {tri.get('attempted', 0):,} abstracts; the best {tri.get('shortlisted', 0):,} "
                   "went on to detailed screening.")
    if scr:
        out.append(f"- **Detailed screening:** {scr.get('attempted', 0):,} abstracts were summarised and scored on the rubric.")
    if rk:
        out.append(f"- **Ranking:** the model compared papers about ten at a time ({rk.get('calls', 0)} comparisons) using their "
                   "abstract summaries. That order picks the papers in this digest and sets their rank.")
    out += [f"- **Score:** out of 100 ({rubric}). The model rates each part; the total and the label ({bands}) are "
            "computed in code. Papers with no security or reverse-engineering connection are capped below High. A paper "
            "whose PDF was analyzed is re-scored from the full text, which is why scores can disagree with the rank.", "",
            "**Full-text analysis and verification**", ""]
    if inv:
        out.append(f"- The top {inv.get('candidates', 0)} papers' PDFs were analyzed, up to about {budget // 1000}k characters "
                   "of text each; the rest of a longer paper was not read.")
    out += ["- Every number quoted under *Key results* appears in the paper's PDF; results whose numbers could not be found "
            "were dropped. Each result's page reference points to the page where its numbers appear.",
            "- Claims marked *demonstrated* cite pages in the paper; claims without a valid page reference are shown as "
            "*claimed, not demonstrated*.",
            "- GitHub links are checked to exist. A repository counts as the paper's code only if its name matches the paper "
            "or the paper presents it as its own; other links are listed as related tools and projects.",
            "- **Annotated PDFs:** the model reads every page of each selected paper and highlights key aspects (yellow), "
            "win conditions, meaning how success is defined and measured (green), and interesting points (blue), with a comment "
            "on each. A highlight is placed only where the quoted text is found on the page; the "
            "paper text itself is unchanged.",
            "- Corrections made after checking a paper by hand are listed under *Editorial correction* in that paper's "
            "details. Synthesis statements are published only if every result they quote appears in a cited paper.",
            *[f"- **Editorial correction to the synthesis:** {text(n)}" for n in ctx.synthesis_notes], "",
            "**Limitations**", "",
            f"- The analysis is done by a small local model (`{text(model)}` through Ollama), not by domain experts. Summaries "
            "can miss context or overstate results.",
            "- Ranking uses abstract-level summaries and is noisy below the first few places; a strong paper can be missed.",
            "- A number appearing in the PDF confirms it was reported, not that the model interpreted it correctly.",
            "- Nothing here is independent peer review. Check the original paper before relying on any finding.", "",
            "{{< /details >}}", ""]
    return out


# --- publishing ------------------------------------------------------------------------------
def _bundle_annotations(cfg: Config, paper_ids: list[str], post_dir: Path) -> dict[str, str]:
    """Copy the digest papers' annotated PDFs into the post's folder (a Hugo page bundle) and drop stale ones."""
    from arxiv_digest.analysis.annotate import annotated_path

    published: dict[str, str] = {}
    for pid in paper_ids:
        src = annotated_path(cfg, pid)
        if src.exists():
            post_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, post_dir / src.name)
            published[pid] = src.name
    for old in post_dir.glob("*-annotated.pdf") if post_dir.exists() else []:
        if old.name not in published.values():
            old.unlink()
    return published


@dataclass
class BlogResult:
    slug: str | None = None
    path: str | None = None
    mode: str = "write"
    written: bool = False
    hugo_ok: bool | None = None
    in_feed: bool | None = None
    committed: bool = False
    commit: str | None = None
    pushed: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _git(repo: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    # core.autocrlf=false: never let git rewrite line endings in what we commit (it can corrupt PDFs it mistakes for text)
    return subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def hugo_check(repo: Path, slug: str, section: str = "blog", timeout: int = 300) -> tuple[bool, bool, str]:
    """Build the site into a temp dir; return (built, post appears in the section RSS feed, message)."""
    hugo = shutil.which("hugo")
    if hugo is None:
        return False, False, "hugo not found on PATH"
    with tempfile.TemporaryDirectory(prefix="digest-hugo-") as tmp:
        r = subprocess.run([hugo, "--source", str(repo), "--destination", tmp, "--cleanDestinationDir"],
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            out = f"{r.stderr}\n{r.stdout}".strip().splitlines()
            errors = [ln for ln in out if "ERROR" in ln] or out
            return False, False, " | ".join(errors[:3])[-1500:]
        page = Path(tmp) / section / slug / "index.html"
        feed = Path(tmp) / section / "index.xml"
        in_feed = feed.exists() and f"/{section}/{slug}/" in feed.read_text(encoding="utf-8", errors="replace")
        if not page.exists():
            return False, in_feed, f"build succeeded but {section}/{slug}/index.html was not generated (draft or future date?)"
        return True, in_feed, "ok"


def publish(cfg: Config, archive: Archive, run_id: int, mode: str | None = None) -> BlogResult:
    """Write the post into the blog repo, verify it builds, then (mode) commit and/or push only that file.

    mode: "write" (file only), "commit" (write + commit just this post), "push" (commit + push to the remote branch).
    """
    bcfg = cfg.section("blog")
    res = BlogResult(mode=mode or bcfg.get("mode", "write"))
    try:
        repo = Path(bcfg["repo"]).expanduser()
        if not (repo / ".git").exists():
            raise ValueError(f"blog repo not found: {repo}")
        data = load_run(cfg, archive, run_id)
        run_dir = Path(archive.run(run_id)["manifest_path"]).parent
        section = bcfg.get("section", "blog")
        dest = repo / "content" / section / slug_for(data.week_of) / "index.md"
        annotated = _bundle_annotations(cfg, data.paper_ids, dest.parent)
        slug, post, warnings = render_post_with_warnings(cfg, archive, data, run_dir, annotated)
        res.notes += warnings
        rel = dest.relative_to(repo).as_posix()
        rel_dir = dest.parent.relative_to(repo).as_posix()
        res.slug, res.path = slug, str(dest)
        previous = dest.read_text(encoding="utf-8") if dest.exists() else None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(post, encoding="utf-8", newline="\n")
        res.written = True
        if bcfg.get("hugo_check", True):
            ok, in_feed, msg = hugo_check(repo, slug, section)
            res.hugo_ok, res.in_feed = ok, in_feed
            if not ok:
                # leave the site as it was: restore the previous version or remove the new post
                if previous is None:
                    shutil.rmtree(dest.parent, ignore_errors=True)
                else:
                    dest.write_text(previous, encoding="utf-8", newline="\n")
                res.written = False
                raise RuntimeError(f"Hugo build check failed: {msg}")
            if not in_feed and not bcfg.get("draft", False):
                res.notes.append("post built but is not in the RSS feed")
        # Unchanged content only means "nothing to do" if git already has this exact file committed; a post written
        # earlier in write mode is unchanged on disk but has never been committed or pushed.
        in_git = _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode == 0
        dirty = bool(_git(repo, "status", "--porcelain", "--", rel_dir).stdout.strip())
        if previous == post and (res.mode == "write" or (in_git and not dirty)):
            res.notes.append("post unchanged")
            if res.mode != "push" or not in_git:
                return res
        if res.mode in ("commit", "push"):
            branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            want = bcfg.get("branch", "main")
            if branch != want:
                raise RuntimeError(f"blog repo is on branch {branch!r}, expected {want!r}; not committing or pushing")
        if res.mode in ("commit", "push") and (dirty or not in_git):
            r = _git(repo, "add", "-A", "--", rel_dir)  # the post and its annotated PDFs (including removals)
            if r.returncode != 0:
                raise RuntimeError(f"git add failed: {r.stderr.strip()}")
            verb = "Update" if in_git else "Add"
            msg = f"{verb} weekly research digest: week of {_fmt_day(data.week_of)}"
            # pathspec commit: only this post is committed, whatever else is staged in the working tree
            r = _git(repo, "commit", "-m", msg, "--", rel_dir)
            if r.returncode != 0:
                raise RuntimeError(f"git commit failed: {(r.stderr or r.stdout).strip()}")
            res.committed = True
            res.commit = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        if res.mode == "push":
            r = _git(repo, "push", bcfg.get("remote", "origin"), f"HEAD:{bcfg.get('branch', 'main')}", timeout=180)
            if r.returncode != 0:
                raise RuntimeError(f"git push failed (the post is committed locally): {(r.stderr or r.stdout).strip()[-600:]}")
            res.pushed = True
    except Exception as exc:  # noqa: BLE001 - publishing never fails the run; the error is reported
        res.error = f"{type(exc).__name__}: {exc}"
        log.warning("blog publish failed: %s", res.error)
    return res
