# arXiv Research Digest

A local-first research **collection** system for security / reverse-engineering research. It runs on a Windows 11
workstation:

- **Python** controls every step deterministically.
- **SQLite** is the authoritative archive.
- A **local LLM (Ollama)** does bounded analysis tasks and returns validated JSON.
- **Obsidian Markdown** is the presentation layer.
- **Windows Task Scheduler** runs it weekly.

It uses no cloud LLM APIs and has no Scout quota.

```
Task Scheduler (Mon 08:00, America/New_York)
  └─ scripts/run-weekly.ps1 ──► python -m arxiv_digest run
        1. Discovery      arXiv API · 6 category baselines · 10 topic searches · ≤4 adaptive searches · date windows
        2. Collection     canonical ids · dedup · version tracking (paper_versions) · provenance (run_papers/search_runs)
        3a. Triage        Stage 0 · every selected paper · ids + integer scores only (~1 s/paper) → shortlist (score ≥65, ≤120/week)
        3b. Screening     Stage 1 · shortlisted papers · abstract-only · full structured JSON → Python-validated score
        3c. Ranking       shortlisted papers compared ~10 at a time → order picks full-text candidates and digest papers
        4A. Investigation Stage 2 · top candidates · PDF (PyMuPDF) · page-cited claims · numbers verified verbatim · GitHub check
        4B. Archive       SQLite: scores history, analysis, topics, versions, reports
        5. Synthesis      Stage 3 · top 5–10 · trends, gaps, ≤3 experiments
        6. Report         temp file → validate → commit (never overwrite) → master archive export → SUCCESS
```

## Setup

```powershell
cd C:\Users\nicho\Git\arXiv-Research-Digest
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m arxiv_digest init
```

To write reports into your Obsidian vault, create `config/local.toml` (it's gitignored). The local file is merged over `config/digest.toml`:

```toml
[paths]
vault_dir = "C:/Users/<you>/Obsidian/Research"
```

## Commands

| Command | What it does |
|---|---|
| `digest window` | Shows the next collection window and the arXiv date filter |
| `digest collect [--legacy PATH --expect 67]` | **Phase 1–2**: discovery, archive, version refresh, collection report. No LLM. |
| `digest import-legacy PATH --expect 67 [--dry-run]` | Imports and reconciles the pre-existing Markdown archive (a file or a folder of `.md` files) |
| `digest run [--legacy PATH --expect 67] [--force]` | The full weekly pipeline |
| `digest models` | Checks Ollama, the configured profiles, and VRAM residency |
| `digest benchmark [--models a,b] [--skip-fulltext]` | Benchmarks local models on 5 representative papers |
| `digest triage-eval [--sample 300] [--seed 7]` | Compares compact triage with full screening on a stratified sample of a collected run (read-only) |
| `digest status` | Recent runs, run states, and the last successful window |
| `digest search "decompilation"` | SQLite FTS5 search over the archive |
| `digest export-archive` | Regenerates the Markdown master archive from SQLite |

(`digest` is installed into the venv; `python -m arxiv_digest` is equivalent.)

### First run with the existing 67-paper archive

The archive from the earlier Claude scheduled task (2026-11-11) is saved as `data/legacy/Master Archive.md`:

```powershell
digest import-legacy "data\legacy\Master Archive.md" --expect 67 --dry-run   # parse check only
digest import-legacy "data\legacy\Master Archive.md" --expect 67             # reconcile into SQLite
digest run                                                                  # initial 30-day run
```

You can also fold reconciliation into the first run: `digest run --legacy "...md" --expect 67`. The legacy parser understands
Markdown tables (it uses the `Title` / `Score` / `Date` columns when present) and plain lines that contain arXiv IDs or links. It keeps
every legacy row verbatim in `legacy_records`. It also re-fetches canonical metadata from arXiv and flags four things: duplicates, papers arXiv
no longer resolves, title mismatches, and score conflicts. If the unique count doesn't match `--expect`, the run is recorded as PARTIAL.

## Collection windows

- **Research week:** Monday–Sunday of the last completed week (America/New_York).
- **First run:** the previous 30 days.
- **Later runs:** from *last successful window end − 3 days* to the end of the research week. A gap since the last
  success is back-filled. The overlap catches papers that arXiv announces late (for example, a Friday submission announced the next Monday).
- `last_successful_window_end` advances **only** after a full SUCCESS. A PARTIAL or FAILED run re-covers the same window next time.
- Category baselines (`cat:cs.XX AND submittedDate:[…]`) catch relevant papers that don't use the expected keywords. Topic searches overlap them on
  purpose; dedup happens in SQLite. A topic search that hits `topic_max_results` is recorded as **saturated**.
- arXiv only returns *new submissions* for a date range. **Revisions** of archived papers are detected by re-fetching every
  qualifying paper on each run (an `id_refresh` search) and recorded in `paper_versions`.

## Analysis (local LLM)

| Profile | Default model | Used for |
|---|---|---|
| `triage` | `qwen3.5:9b` | Stage 0: compact triage of every selected abstract |
| `screener` | `qwen3.5:9b` | Stage 1: full screening of the shortlist; proposing adaptive searches |
| `reranker` | `qwen3.5:9b` | Stage 1b: comparative ranking of the screened shortlist |
| `analyst` | `qwen3.5:9b` | Stage 2: full-text investigation of the top 10–20 candidates |
| `synthesizer` | `qwen3.5:9b` | Stage 3: cross-paper synthesis |
| `verifier` | `deepseek-r1:14b` | Optional challenge pass on the top 2–3 papers (`[verification] enabled = true`) |

The defaults come from the 2026-11-11 benchmark on a 16 GB RTX 5080:

- **Screening and analysis:** qwen3.5:9b was more consistent than gpt-oss:20b and 3–4× faster. It also stays fully in GPU memory at a 32k context, while gpt-oss:20b spills 20% to system RAM.
- **Verifier:** off by default. deepseek-r1:14b took about 3 minutes per review, disagreed with every analysis, and mostly raised generic objections.
- **Re-running:** the benchmark is repeatable with `digest benchmark` (stages `screening`, `fulltext` and `verification`).

If a prompt or answer doesn't fit in a profile's `num_ctx`, the call fails with a context-overflow error instead of silently using truncated input. Ollama would otherwise truncate without warning. Full-text investigation then shrinks the paper text to fit and retries once; if that fails too, only that paper is marked failed.

Every request sets its own `num_ctx`, so Ollama's default context length (app setting or `OLLAMA_CONTEXT_LENGTH`) does not
affect the pipeline. Full-text reading uses 64k (`[llm.profiles.analyst]`) with a 160k-character budget: qwen3.5:9b
is then 7.6 GiB and fully in VRAM, and a 43-page paper is read to page 33 instead of page 14, at 41 s instead of 35 s.

The models are set per profile in `config/digest.toml`. The pipeline depends only on `LLMProvider` (`llm/provider.py`), so swapping
models or providers doesn't touch the research logic.

**Which papers get triaged:** papers from topic or adaptive searches, papers in `cs.CR` / `cs.AR` / `cs.SE`, and papers from
`cs.AI` / `cs.CL` / `cs.LG` that trip the taxonomy keyword pre-filter. The pre-filter skips thousands of unrelated ML papers a month;
`digest collect` shows the raw volumes.

**Triage and the shortlist (`[triage]`).** Full screening writes ~650 tokens of prose per paper (~6 s); triage asks the same
model for topic ids, the security connection, the five rubric integers and a recommendation (~100 tokens, ~1 s). Python
scores both the same way. Then:

- papers scoring at least `min_score` (65) go to full screening, best first, up to `shortlist_per_week` (120) per research
  week in the window, capped at `max_shortlist` (300) for a long first run;
- papers up to `recheck_margin` points below the cut-off are triaged once more and the higher score counts, because repeat
  triage scores vary by ~4 points on average;
- papers whose triage failed are fully screened anyway (fail-open);
- adaptive-search papers must clear the same cut-off.

**Comparative ranking (`[rerank]`).** A small model's absolute scores tie heavily: on the first 30-day run, 65 of 300
shortlisted papers scored exactly 81, so "top 10 by score" was decided by arXiv ID. The model is much better at saying
which of ~10 papers is more relevant, so the ranking stage:

- deals the screened papers (score ≥60) into groups of 10, ranks each group, and lets the top 60% advance, until 40 remain;
- slides a 10-paper window up those 40 in steps of 5, three times, to settle the top 15;
- keeps a group's incoming order if its comparison fails (counted in the report).

That order, not the score, picks the 15 full-text candidates and the digest papers; scores are still shown and still gate
qualification. On the 300-paper shortlist it took 87 calls (~1 min). Papers from the legacy archive in the top 10 went
from 3 (by score) to 8.

A triage score only ranks papers. It never qualifies a paper for the master archive or the digest, and it never replaces an
earlier score. Triage results are cached by input, so re-runs and the 3-day overlap cost nothing. Set `enabled = false` to
fully screen everything again. `digest triage-eval` compares triage with full screening on a stratified sample.

Measured on the 2026-11-11 collection (3,557 papers selected over 30 days): triage 1.04 s/paper vs. 6.2 s for full screening.
A weekly run of ~900 papers is about 16 min of triage plus ~13 min of full screening, instead of ~100 min.

Why a score gate and not a fixed top-N: against the 52 legacy-archive papers in that collection, the gate of 65 kept all 13
that the legacy archive rated ≥70 and 24 of 27 rated ≥60. Triage scores bunch at 79–94 for relevant papers, so a top-40
cut falls inside a large tie and would have kept only about 8 of the 13. For a strict top-N, set `min_score = 0` and
`shortlist_per_week = N`.

**What Python enforces on model output.** The model never decides totals or bands. Python:

- validates every response against a pydantic schema and retries once with the validation errors fed back;
- clips each component to its maximum, computes the total, and assigns the band;
- drops unknown topic and project IDs;
- applies the generic-agent guardrail: a paper with no security / RE connection, or one that is only about agent orchestration, is capped at 69 (Medium);
- *full text only:* downgrades a "demonstrated" claim to "claimed" if it has no valid page reference;
- *full text only:* drops any experiment whose reported numbers don't appear **verbatim** in the PDF text, and flags numbers that are in the paper but not on the cited pages;
- *full text only:* keeps artifact URLs only if they appear in the paper, then checks GitHub repositories through the GitHub API (set `GITHUB_TOKEN` for higher rate limits).

**Untrusted input:** abstracts and PDF text are fenced as data, stripped of zero-width/bidi characters, and scanned for injection
phrases. Fence markers can't be forged. The model has no tools, and every output is schema-checked. Suspected injections are flagged in
the report.

The phrase scan distinguishes attacks from papers that *document* prompts, which papers about AI agents do constantly:

- text aimed at a paper-reviewing AI ("as an AI reviewer you must…", "give this paper a high score") is always flagged;
- generic phrases ("ignore previous instructions", "you are an AI assistant") are skipped when they sit inside quotation marks
  or next to a prompt listing (role labels such as `System:`, "user prompt", "Listing 3:", `{placeholders}`, example HTML);
  on reference/appendix pages any such marker on the page counts;
- skipped matches are counted in the stored analysis (`injection_quoted_ignored`), and the model's own
  "suspicious instructions" report still raises the flag.

On the 31 PDFs read so far this cut flagged papers from 9 to 0; every skipped match was a documented prompt or attack example.

**Evidence levels:**

| Level | Meaning |
|---|---|
| Abstract-only | Stage 1 only |
| Full-text reviewed | Stage 2 done |
| Independently corroborated | The verifier agrees with no major challenges **and** a referenced repository was verified |

## Run states and integrity

```
STARTED → DISCOVERED → ANALYZED → REPORT_GENERATED → REPORT_VALIDATED → ARCHIVE_COMMITTED → SUCCESS
```

- Reports are written to a temp file, validated, then moved into place. An existing report is never overwritten; re-runs get a `(run N)` suffix.
- Validation checks the frontmatter, all 7 sections in order and non-empty, that every selected paper appears **exactly once** in
  selection order, and that each paper qualifies in SQLite. A report that fails validation is kept as `rejected-report.md` in the run folder and the run is FAILED.
- The master archive is regenerated from SQLite and checked before and after writing (every qualifying paper exactly once).
  The previous version is kept as `.bak`.
- Recording the reports, the `ARCHIVE_COMMITTED` state, `SUCCESS`, and the advance of `last_successful_window_end` happens in **one SQLite transaction**.
- **FAILED** (window not advanced, exit 1) when:
  - discovery returns nothing;
  - Ollama is unavailable;
  - the report fails validation.
- **PARTIAL** (report saved with a `(PARTIAL)` suffix, window **not** advanced, exit 2) when:
  - a query failed or was incomplete;
  - more than 10% of screenings failed;
  - synthesis failed;
  - there are no qualifying papers;
  - the legacy count doesn't match `--expect`.
- Each run writes `data/runs/run-NNNN/manifest.json`: window, every search with counts and status, prefilter, screening and investigation
  stats, models, timings, state history, problems, and validation results.

## Weekly report

`<vault>/arXiv Research Digest/Weekly/YYYY-MM-DD arXiv Research Digest.md`:

- 01 Executive summary (≤5)
- 02 Research coverage
- 03 Top research papers (5–10; High/Critical first, then topped up to 5)
- 04 Implementation opportunities (≤3)
- 05 Trends and gaps
- 06 Verified open-source resources
- 07 Collection integrity

Papers featured in an earlier successful digest are not repeated. The full index is in `Master Archive.md`.

## Scheduling

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register-task.ps1            # Monday 08:00, StartWhenAvailable
powershell -ExecutionPolicy Bypass -File scripts\run-weekly.ps1               # run once now, exactly as the scheduler would
```

- The task runs as you, only while you're logged on.
- **The workstation must be on.** If it was off at 08:00, `StartWhenAvailable` runs the task at the next opportunity.
- `run-weekly.ps1` starts `ollama serve` if it isn't running.

## Choosing models: the benchmark

`config/benchmark.toml` has five representative papers:

| Paper | Role |
|---|---|
| LLM4Decompile | Decompilation |
| PentestGPT | Agentic pentesting |
| Fuzz4All | LLM fuzzing |
| Spectre | Hardware side channel |
| ReAct | **Negative control**: a generic agent paper that must stay below High |

The gold labels are provisional; adjust them to your own judgement. `digest benchmark` measures:

- first-attempt JSON validity;
- score standard deviation across repeats;
- band agreement and topic recall;
- whether the negative control holds;
- for full text: verbatim-verified vs. dropped experiment numbers, downgraded claims, and invalid page references;
- latency and tokens/s;
- **GPU residency** (`size_vram / size` from `ollama ps`) at the configured context size.

Models that aren't installed are skipped, never pulled.

## Layout

```
config/digest.toml        categories, topic searches, taxonomy, rubric, projects, model profiles
config/benchmark.toml     benchmark papers and labels
src/arxiv_digest/
  discovery/              arXiv client (rate limit, retries, pagination, saturation) and collector
  archive/                SQLite schema/repository, legacy reconciliation
  llm/provider.py         provider abstraction, Ollama, structured output + retry
  analysis/               prefilter, triage + shortlist, screening, investigation, synthesis/adaptive/verification, scoring rules, untrusted-input handling
  report/                 digest render + validation, master archive, collection/reconciliation reports, safe file commits
  pipeline.py             run state machine and manifest
  benchmark.py            model benchmark harness
scripts/                  Task Scheduler entry point and registration
tests/                    offline tests (fake arXiv + fake LLM) covering windows, parsing, guardrails, evidence checks, run states
```
