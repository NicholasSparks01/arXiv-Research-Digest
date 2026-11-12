"""Annotated PDFs for the digest papers.

For every paper selected for the digest, the local model reads the *whole* PDF in page-aligned chunks and proposes
annotations of three kinds:

- **key aspect**: core contributions, design decisions, threat model or setting;
- **win condition**: what counts as success and how it is measured (criterion, metric or formula, the baseline to
  beat, and the result that shows it won);
- **interesting**: surprising results, clever tricks, caveats and ideas worth reusing.

Each annotation must quote text from the page it refers to. PyMuPDF searches the page for that quote and the
annotation is placed as a coloured highlight with the model's note as its comment. Quotes that cannot be found on the
page (or a neighbouring one) are dropped, so every highlight sits on real text. Page 1 gets a summary note with the
paper's formula for success, a legend and a machine-generated disclosure.

The annotated copy is kept in data/annotated/ and the vault, and is published next to the blog post.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from arxiv_digest.analysis.schemas import _Strict
from arxiv_digest.analysis.untrusted import UNTRUSTED_PREAMBLE, fence, sanitize
from arxiv_digest.archive.db import Archive
from arxiv_digest.config import Config
from arxiv_digest.llm.provider import LLMError, LLMProvider, LLMUnavailable, run_structured
from arxiv_digest.pdf import PDFError, download_pdf, extract

log = logging.getLogger(__name__)
PROMPT_VERSION = "annotate-v1"
AUTHOR = "arXiv Research Digest"

KINDS = {  # kind: (label, highlight colour as RGB 0-1)
    "key_aspect": ("Key aspect", (1.0, 0.85, 0.15)),
    "win_condition": ("Win condition", (0.35, 0.85, 0.35)),
    "interesting": ("Interesting", (0.40, 0.70, 1.0)),
}


class PdfAnnotation(_Strict):
    page: int = Field(ge=1, le=2000)
    quote: str = Field(max_length=400, description="text copied exactly from that page")
    kind: Literal["key_aspect", "win_condition", "interesting"]
    note: str = Field(max_length=600, description="1-3 sentences: why it matters")


class ChunkAnnotations(_Strict):
    annotations: list[PdfAnnotation] = Field(default_factory=list, max_length=25)
    success_formula: str = Field(default="", max_length=600)


def system_prompt() -> str:
    return f"""You annotate a research paper for a security researcher who works on reverse engineering, binary analysis,
vulnerability research and AI-assisted security tooling. You see one part of the paper; pages start with [[PAGE n]].

Mark the passages a careful reader should not miss, of three kinds:
- key_aspect: a core contribution, a key design decision of the method, or the threat model / setting.
- win_condition: what counts as success and how it is measured: the success criterion, the key metric or formula, the
  baseline it has to beat, or the result that shows it won.
- interesting: a surprising result, a clever trick, an important caveat or limitation, or an idea worth reusing.

Rules:
- quote: copy one sentence or phrase CHARACTER FOR CHARACTER from the text of the page given in "page" (6 to 40 words,
  one page only, no ellipses, no paraphrase). Never quote the reference list or acknowledgements.
- note: 1 to 3 sentences in your own words saying why the passage matters. Do not invent numbers.
- About 1 to 3 annotations per page that has substance; skip pages that have none. Prefer win_condition passages that
  define the evaluation and the headline results.
- success_formula: if this part defines how the paper measures success, state it in one or two sentences, for example
  "Success = a proof-of-concept that crashes the vulnerable build but not the patched one, plus flag retrieval;
  headline: 88% recovery, 20% exploitation (Codex)". Otherwise return "".

{UNTRUSTED_PREAMBLE}"""


def user_prompt(title: str, summary: str | None, chunk_text: str, first: int, last: int, total: int) -> str:
    head = f"Paper: {sanitize(title, 300)}\n"
    if summary:
        head += f"Summary from an earlier full-text review (context only, do not quote it): {sanitize(summary, 1200)}\n"
    head += f"This part covers pages {first}-{last} of {total}. Annotate it and return the JSON object.\n\n"
    return head + fence("PAPER TEXT", chunk_text)


def chunk_pages(pages: list[str], chunk_chars: int) -> list[tuple[int, int, str]]:
    """Group whole pages into chunks of about `chunk_chars` characters: [(first_page, last_page, text)]."""
    chunks, cur, start, size = [], [], 1, 0
    for n, text in enumerate(pages, start=1):
        block = f"[[PAGE {n}]]\n{sanitize(text)}\n"
        if cur and size + len(block) > chunk_chars:
            chunks.append((start, n - 1, "".join(cur)))
            cur, start, size = [], n, 0
        cur.append(block)
        size += len(block)
    if cur:
        chunks.append((start, len(pages), "".join(cur)))
    return chunks


# --- anchoring ------------------------------------------------------------------------------
def _candidates(quote: str) -> list[str]:
    q = re.sub(r"\s+", " ", quote).strip().strip('"“”')
    words = q.split(" ")
    out = [q]
    if len(words) > 14:
        out += [" ".join(words[:12]), " ".join(words[-10:])]
    if len(words) > 9:
        out += [" ".join(words[:8]), " ".join(words[-7:])]
    seen, uniq = set(), []
    for c in out:
        if len(c) >= 20 and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def find_quote(doc, page_no: int, quote: str) -> tuple[int, list] | None:
    """(page number, quads) where `quote` (or a long enough part of it) appears; tries the page, then its neighbours."""
    for p in (page_no, page_no - 1, page_no + 1):
        if not 1 <= p <= doc.page_count:
            continue
        page = doc[p - 1]
        for cand in _candidates(quote):
            quads = page.search_for(cand, quads=True)
            if quads:
                return p, quads
    return None


# --- results ---------------------------------------------------------------------------------
@dataclass
class AnnotationResult:
    arxiv_id: str
    path: str | None = None
    pages: int = 0
    chunks: int = 0
    failed_chunks: int = 0
    proposed: int = 0
    placed: int = 0
    dropped: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    success_formula: str = ""
    seconds: float = 0.0
    cached: bool = False
    error: str | None = None
    annotations: list[dict] = field(default_factory=list)


def _summary_note(title: str, arxiv_id: str, model: str, formula: str, placed: list[dict]) -> str:
    lines = [f"{AUTHOR}: machine-generated annotations of arXiv:{arxiv_id}, \"{title}\".",
             f"Written by a local language model ({model}); verify every note against the text.", ""]
    if formula:
        lines += ["Formula for success: " + formula, ""]
    for kind, (label, _) in KINDS.items():
        items = [a for a in placed if a["kind"] == kind]
        if items:
            lines.append(f"{label}s:")
            lines += [f"  p. {a['placed_page']}: {a['note'][:220]}" for a in items[:6]]
            lines.append("")
    lines.append("Highlights: yellow = key aspect, green = win condition, blue = interesting.")
    lines.append(f"Original: https://arxiv.org/abs/{arxiv_id}. Annotations added; the paper text is unchanged.")
    return "\n".join(lines)


def write_annotated(src: Path, dest: Path, arxiv_id: str, title: str, model: str, proposed: list[dict],
                    formula: str) -> tuple[list[dict], int]:
    """Place the proposed annotations on a copy of `src`. Returns (placed annotations, dropped count)."""
    import pymupdf

    doc = pymupdf.open(str(src))
    placed, dropped, used = [], 0, set()
    for a in proposed:
        hit = find_quote(doc, a["page"], a["quote"])
        if hit is None:
            dropped += 1
            continue
        p, quads = hit
        key = (p, tuple(round(v) for v in quads[0].rect))
        if key in used:  # two notes on the same passage: keep the first
            dropped += 1
            continue
        used.add(key)
        label, color = KINDS[a["kind"]]
        page = doc[p - 1]  # keep the page referenced while its annotation is edited (PyMuPDF requirement)
        annot = page.add_highlight_annot(quads)
        annot.set_colors(stroke=color)
        annot.set_info(title=AUTHOR, subject=label, content=f"{label}: {a['note']}")
        annot.set_opacity(0.5)
        annot.update()
        placed.append({**a, "placed_page": p})
    first_page = doc[0]
    note = first_page.add_text_annot(pymupdf.Point(18, 18), _summary_note(title, arxiv_id, model, formula, placed),
                                     icon="Note")
    note.set_info(title=AUTHOR, subject="Digest summary")
    note.set_colors(stroke=KINDS["win_condition"][1])
    note.update()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.pdf")
    # object streams keep the copy about the size of the original (clean=True rewrote page contents: +65% on a test paper)
    doc.save(str(tmp), garbage=3, deflate=True, use_objstms=1)
    doc.close()
    tmp.replace(dest)
    return placed, dropped


# --- driver ----------------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def annotated_path(cfg: Config, arxiv_id: str) -> Path:
    return cfg.data_dir / "annotated" / f"{arxiv_id.replace('/', '_')}-annotated.pdf"


def annotate_paper(cfg: Config, archive: Archive, provider: LLMProvider, run_id: int, arxiv_id: str) -> AnnotationResult:
    acfg = cfg.section("annotation")
    profile = cfg.profile("annotator")
    res = AnnotationResult(arxiv_id)
    t0 = time.monotonic()
    paper = archive.paper(arxiv_id)
    try:
        src = download_pdf(arxiv_id, paper["latest_version"], cfg.pdf_dir, cfg.section("arxiv")["user_agent"],
                           float(cfg.section("investigation").get("download_timeout_s", 120)))
        ext = extract(src)
    except (PDFError, OSError) as exc:
        res.error = f"PDF unavailable: {exc}"
        return res
    res.pages = ext.page_count
    dest = annotated_path(cfg, arxiv_id)
    input_hash = hashlib.sha256(f"{PROMPT_VERSION}|{profile.model}|{acfg.get('chunk_chars')}|{_sha256_file(src)}".encode()).hexdigest()
    cached = archive.find_cached_analysis(arxiv_id, "annotation", profile.model, PROMPT_VERSION, input_hash)
    if cached and dest.exists():
        d = json.loads(cached["payload"])
        known = {k: d[k] for k in AnnotationResult.__dataclass_fields__ if k in d}
        return AnnotationResult(**{**known, "cached": True, "path": str(dest)})

    inv = archive.latest_analysis(arxiv_id, "investigation")
    summary = json.loads(inv["payload"]).get("summary") if inv else None
    proposed, formulas = [], []
    for first, last, chunk in chunk_pages(ext.pages, int(acfg.get("chunk_chars", 60000))):
        res.chunks += 1
        try:
            out = run_structured(provider, profile, system_prompt(),
                                 user_prompt(paper["title"], summary, chunk, first, last, ext.page_count), ChunkAnnotations,
                                 max_retries=int(cfg.section("llm").get("max_validation_retries", 1)))
        except LLMUnavailable:
            raise
        except LLMError as exc:
            res.failed_chunks += 1
            log.warning("annotation of %s pages %d-%d failed: %s", arxiv_id, first, last, exc)
            continue
        value: ChunkAnnotations = out.value
        for a in value.annotations[: int(acfg.get("max_per_chunk", 20))]:
            if first <= a.page <= last and a.quote.strip():
                proposed.append(a.model_dump())
        if value.success_formula.strip():
            formulas.append(value.success_formula.strip())
    res.proposed = len(proposed)
    if res.chunks and res.failed_chunks == res.chunks:
        res.error = "every part of the paper failed to annotate; no annotated PDF written"
        return res
    try:
        placed, res.dropped = write_annotated(src, dest, arxiv_id, paper["title"], profile.model, proposed,
                                              formulas[0] if formulas else "")
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not stop the run
        res.error = f"could not write annotated PDF: {exc}"
        return res
    res.path, res.placed, res.success_formula = str(dest), len(placed), formulas[0] if formulas else ""
    res.by_kind = {k: sum(1 for a in placed if a["kind"] == k) for k in KINDS}
    res.annotations = [{k: a[k] for k in ("placed_page", "kind", "quote", "note")} for a in placed]
    res.seconds = round(time.monotonic() - t0, 1)
    # a partial result (some chunks failed) is used now but not cached, so the next run tries the whole paper again
    archive.add_analysis(arxiv_id, version=paper["latest_version"], run_id=run_id, stage="annotation", model=profile.model,
                         prompt_version=PROMPT_VERSION, input_hash=input_hash if not res.failed_chunks else None,
                         payload=asdict(res))
    log.info("annotated %s: %d of %d proposed annotations placed (%d chunks, %.0f s)", arxiv_id, res.placed, res.proposed,
             res.chunks, res.seconds)
    return res


def annotate_papers(cfg: Config, archive: Archive, provider: LLMProvider, run_id: int, arxiv_ids: list[str]) -> list[AnnotationResult]:
    results = []
    for arxiv_id in arxiv_ids[: int(cfg.section("annotation").get("max_papers", 10))]:
        log.info("annotating %s", arxiv_id)
        results.append(annotate_paper(cfg, archive, provider, run_id, arxiv_id))
    return results


def copy_to(path: str | Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(path).name
    shutil.copyfile(path, out)
    return out
