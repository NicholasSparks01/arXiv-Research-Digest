"""PDF acquisition and text extraction (PyMuPDF). PDFs are untrusted input: parsed for text only, never executed."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from arxiv_digest.analysis.untrusted import sanitize
from arxiv_digest.ids import pdf_url

log = logging.getLogger(__name__)

MAX_PDF_BYTES = 60 * 1024 * 1024
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?"
    r"(Abstract|Introduction|Background|Related Work|Motivation|Overview|Design|Approach|Method(?:ology)?|Implementation|"
    r"Threat Model|System Design|Evaluation|Experiments?(?:al Setup)?|Results|Discussion|Limitations|Case Stud(?:y|ies)|"
    r"Conclusions?|Future Work|References|Bibliography|Appendix|Acknowledg(?:e)?ments?)\b.{0,40}$",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)
ARTIFACT_HOSTS = ("github.com", "gitlab.com", "huggingface.co", "zenodo.org", "figshare.com", "bitbucket.org", "codeberg.org")


@dataclass
class ExtractedDoc:
    path: Path
    pages: list[str]
    sections: list[tuple[str, int]] = field(default_factory=list)  # (heading, 1-based page)
    references_page: int | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def full_text(self) -> str:
        return "\n".join(self.pages)


@dataclass
class PaperContext:
    text: str
    included_pages: list[int]
    truncated_pages: list[int]
    omitted_pages: list[int]
    chars: int


class PDFError(RuntimeError):
    pass


def download_pdf(arxiv_id: str, version: int | None, dest_dir: Path, user_agent: str, timeout_s: float = 120) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = arxiv_id.replace("/", "_")
    path = dest_dir / f"{safe}v{version or 0}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = pdf_url(arxiv_id, version)
    with httpx.Client(headers={"User-Agent": user_agent}, timeout=timeout_s, follow_redirects=True) as http:
        with http.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise PDFError(f"PDF download failed: HTTP {resp.status_code} for {url}")
            data = bytearray()
            for chunk in resp.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise PDFError(f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB: {url}")
    if not bytes(data[:5]).startswith(b"%PDF"):
        raise PDFError(f"response for {url} is not a PDF")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(bytes(data))
    tmp.replace(path)
    return path


def extract(path: Path) -> ExtractedDoc:
    import pymupdf  # imported lazily: only the investigation stage needs it

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001 — any parser failure means unusable input
        raise PDFError(f"cannot open PDF {path.name}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise PDFError(f"{path.name} is encrypted")
        pages = [sanitize(page.get_text("text")) for page in doc]
        sections: list[tuple[str, int]] = []
        try:
            toc = doc.get_toc(simple=True)
        except Exception:  # noqa: BLE001
            toc = []
        if toc:
            sections = [(sanitize(title).strip(), int(pg)) for lvl, title, pg in toc if lvl <= 2 and pg >= 1]
    finally:
        doc.close()

    if not sections:
        for i, text in enumerate(pages, start=1):
            for line in text.splitlines():
                if len(line) < 70 and _HEADING_RE.match(line):
                    sections.append((line.strip(), i))
    ref_page = next((pg for title, pg in sections if re.search(r"\b(references|bibliography)\b", title, re.I)), None)
    if not any(p.strip() for p in pages):
        raise PDFError(f"{path.name}: no extractable text (scanned PDF?)")
    return ExtractedDoc(path=path, pages=pages, sections=sections, references_page=ref_page)


def build_context(doc: ExtractedDoc, max_chars: int, stub_chars: int = 600) -> PaperContext:
    """Page-marked text within a budget. Body pages first; bibliography/appendix only if budget remains."""
    order = list(range(1, doc.page_count + 1))
    body = [p for p in order if doc.references_page is None or p < doc.references_page]
    tail = [p for p in order if p not in body]
    included, truncated, omitted = [], [], []
    chunks: dict[int, str] = {}
    budget = max_chars
    for group in (body, tail):
        for p in group:
            text = doc.pages[p - 1].strip()
            block = f"[[PAGE {p}]]\n{text}\n"
            if len(block) <= budget:
                chunks[p] = block
                included.append(p)
                budget -= len(block)
            elif group is body and budget > 200:
                stub = f"[[PAGE {p}]]\n{text[: min(stub_chars, budget - 100)]}\n[... page truncated ...]\n"
                chunks[p] = stub
                truncated.append(p)
                budget -= len(stub)
            else:
                omitted.append(p)
    text = "".join(chunks[p] for p in sorted(chunks))
    return PaperContext(text=text, included_pages=included, truncated_pages=truncated, omitted_pages=omitted, chars=len(text))


def extract_urls(text: str) -> list[str]:
    urls = []
    # PDF text often splits URLs across lines; only join when the URL clearly continues ("github.com/\nowner/repo").
    joined = re.sub(r"(https?://\S*[/_.\-])\n(?=[\w/])", r"\1", text)
    for m in URL_RE.finditer(joined):
        url = m.group(0).rstrip(".,;:")
        if url not in urls:
            urls.append(url)
    return urls


def artifact_urls(text: str) -> list[str]:
    return [u for u in extract_urls(text) if any(h in u.lower() for h in ARTIFACT_HOSTS)]
