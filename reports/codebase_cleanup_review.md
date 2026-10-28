# Codebase architecture and cleanup review

Review date: 28 October 2026. Scope: conservative maintenance in the existing repository.
Initial `git status --short` was empty. No existing file was deleted, renamed, or moved.
The README, requirements, configuration, datasets, and historical evidence were preserved.

## Architecture and data flow

```text
source policy PDF
  -> prepare_dataset.py: extraction, heading restoration, manual review inputs
  -> data/processed/cleaned_policy.txt + document metadata
  -> chunking_pipeline.py: fixed-length, sliding-window, lexical TF-IDF chunking
  -> export_reports.py: reviewed provenance, page references, metadata and comparisons
  -> data/chunks/metadata_semantic.json
  -> retriever.py: parent chunks or heading-aligned children, BGE embeddings, cosine ranking
     -> run_queries.py + evaluation.py: Day 06 retrieval grades and logs
     -> qa_pipeline.py: context -> local Ollama -> parsing and citation/answer validation
        -> run_qa_eval.py: Day 07 answer runs and retrieval diagnostics
        -> evaluation/run_day08_evaluation.py: golden cases and controlled prompt experiment
        -> day10_tools.py.read_policy -> web_app.py -> browser answer/evidence display

browser leave request -> day10_tools.py: validate -> stage -> session-owned confirm/cancel
  -> confirmed simulation only -> local tool log (no HR submission)
```

Similarity scores rank evidence; they are not factual-confidence scores. Quote checks,
uncited-sentence filtering, number checks, and the exact abstention answer remain intact.
The embedding model remains `BAAI/bge-small-en-v1.5`. Ollama defaults remain `llama3.2`,
temperature 0, seed 7, context 4096, output limit 400, and timeout 180 seconds.

| File | Responsibility and dependencies |
|---|---|
| `paths.py` | Root-relative locations, strategy names, and required-file checks shared by the stages. |
| `prepare_dataset.py` | Uses pypdf to extract the PDF, restore handbook headings, and write raw/cleaned text, document metadata, and review checklist. |
| `chunking_pipeline.py` | Parses text/PDF, segments content, implements three strategies, builds stable IDs/offset metadata, prints comparisons, and writes chunk JSONL. |
| `export_reports.py` | Requires review confirmation; checks text/offset consistency, adds provenance/pages, exports readable chunks, retrieval metadata, comparisons, and recommendation. Reuses chunker utilities. |
| `retriever.py` | Loads semantic metadata; optionally splits parents at headings in memory; caches the CPU embedding model; exposes ranking/source/inspection helpers. Uses NumPy and sentence-transformers. |
| `evaluation.py` | Day 06 case-field-driven relevance, ranks, margins, section labels, distractors, and aggregate retrieval metrics. Imported by both query runners. |
| `run_queries.py` | Case loading, ad hoc/interactive retrieval, grading, display, run/history logging, parent/child comparisons. |
| `qa_pipeline.py` | Lazy retrieval, context construction, native Ollama HTTP, JSON parsing, grounding checks, sources/debug contract, and health checks. |
| `run_qa_eval.py` | Runs QAPipeline, attaches Day 06 diagnostics, prints answers, and writes QA logs. Reuses query case utilities. |
| `web_app.py` | Flask factory, QA/tool routes, session ownership, errors, web logging, and server CLI. |
| `day10_tools.py` | Request routing/date parsing, policy reads, in-memory pending simulations, expiry/cancellation/supersession/confirmation under an RLock. Logs before state mutation. |
| `run_rag.py` | Stage dispatcher and extraction review gate, including previous-text backup when extraction differs. |
| `evaluation/run_day08_evaluation.py` | Separate 16-case golden evaluator, phrase/citation heuristics, checkpoints, hash controls, and evaluation-process-only prompt experiment. |
| `qa_pipeline - Copy.py` | Copy of the original QA module; no repository consumer found. Preserved. |
| `templates/index.html`, `static/app.js`, `static/style.css` | Existing page, health/ask/action calls, text rendering, answer/citation/evidence display, controls, and styling. Unchanged. |

`evaluation.py` and `evaluation/` serve distinct workflows. No package initializer was
added. The Day 08 script inserts the project root into its import path and reuses QAPipeline.
Its grading deliberately differs from Day 06 retrieval grading.

## Entry points and side effects

Commands run inside `rag`; this terminal requires `.\.venv\Scripts\python.exe` in place of `python`.

| Workflow | Existing command | Side effects |
|---|---|---|
| Day 05 | `python run_rag.py prepare` / `chunk` | Rewrites processed/chunk/comparison artifacts, with existing review gate. Direct `prepare_dataset.py`, `chunking_pipeline.py [path]`, and `export_reports.py --reviewed` remain supported. |
| Day 06 | `python run_rag.py retrieve` | Defaults to both indexes; appends history and writes run files. `--no-log` disables logging. |
| Combined | `python run_rag.py all` | Preparation, chunk/export, retrieval; not executed against existing evidence. |
| Day 07 | `python run_rag.py qa` | Local generation and QA logs; existing file/index/model/tag/no-log options remain. |
| Web/Day 10 | `python run_rag.py web` | Loads retrieval, checks Ollama, starts Flask. User requests write web/tool logs. |
| Day 08 | `python evaluation/run_day08_evaluation.py baseline --output <fresh-directory>` or `experiment`/`after` | Phase artifacts/checkpoints; requires fresh output to preserve completed runs. |
| Launch smoke | `python tests/smoke_launch_day08.py` | Opens historical `outputs/day08/web_launch_verification.txt` in write mode. Not run. |
| Tool smoke | `python tests/smoke_day10_http.py <base-url>` | Calls real QA/actions, appends logs, reads default tool log. Not run. |

Frontend calls remain `/api/health`, `/api/ask`, `/api/action/pending`, and
`/api/action/decision`; `/` and Flask static assets are unchanged. QA responses retain
`{answer, citations, sources, debug}`, actions retain `{action}`, and errors retain their
codes/messages/debug/status behavior. Confirmation still sends action ID and decision;
it cannot replace the reviewed dates. JavaScript fields and DOM IDs were reviewed against
the Flask responses/template; no frontend edits were needed.

## Changes actually made

| File | Change and reason |
|---|---|
| `chunking_pipeline.py` | Corrected layout of the oversized-unit condition; expression unchanged. |
| `export_reports.py` | Removed write-only local `results` dictionary and assignment; added responsibility docstring. |
| `prepare_dataset.py` | Added responsibility docstring; no extraction/cleaning logic changed. |
| `qa_pipeline.py` | Corrected timeout documentation from 120 to existing default 180. No executable/prompt/validation changes. |
| `run_rag.py` | Renamed ambiguous local `l` to `line` in console filtering. |
| `web_app.py` | Renamed logging local `d` to `debug`, sorted two imports, documented existing action endpoints/response variant. |
| `day10_tools.py` | Extracted duplicate expiration logging/state updates into `_expire`; retained caller locks, exact messages, deadlines, and log-before-mutation ordering. Added focused docstrings. |
| `tests/test_day10_tools.py` | Removed unused `ToolError` import, isolated WEB_LOG in existing temporary directory, added two expiration regressions. Existing assertions unchanged. |
| `reports/codebase_cleanup_review.md` | Added this requested report. |

No new runtime module, dependency, framework, configuration layer, or feature was added.

## Unnecessary-code audit

### Category 1: verified safe removal/consolidation, implemented

| Item | Evidence | Risk |
|---|---|---|
| `export_reports.py:main` local `results` | Initialized/populated but never read, returned, or passed elsewhere. Exports/summaries consume `chunks` directly. | Low; removed redundant bookkeeping. |
| `tests/test_day10_tools.py` import `ToolError` | No reference in the test module. Production exception class/handlers remain. | Low; removed import only. |
| `day10_tools.py:decide` and `pending` expiry blocks | Same pending deadline/event/state update; messages differ and remain arguments to helper. | Low but safety-sensitive; existing and added regression coverage. |

### Category 2: suspected unnecessary, preserved

| Item | Evidence and uncertainty | Removal risk |
|---|---|---|
| `qa_pipeline - Copy.py` | Before edits, `git diff --no-index -- qa_pipeline.py "qa_pipeline - Copy.py"` returned no differences. Project source, docs, tests, and frontend searches found no consumer/loader of the copy. Active imports use `qa_pipeline`; external IDE/manual use cannot be excluded. Active module now differs only in timeout documentation. | Medium; deletion needs a separate decision and is prohibited by this task. |
| Overlapping Day 08 report/review prose | Repeated text exists in completed review artifacts, which serve audit purposes rather than runtime execution. | High; preserve historical evidence. |

No dependency was demonstrated unnecessary: `pypdf` supports extraction/direct PDF input,
`numpy` ranking, `sentence-transformers` BGE embeddings, and `flask` the UI/API. No requirement
or installed package was removed. An AST import-name scan found no remaining candidates in
the 19 project/test Python files; this is supporting evidence, not proof against external use.

### Category 3: keep

- `evaluation.py` versus Day 08 evaluator: distinct grading methods/imports; merging risks methodology changes.
- `chunking_pipeline.summarize` versus `evaluation.summarise`: chunk quality versus retrieval aggregates.
- Handbook extraction versus `chunking_pipeline.load_pdf`: reviewed heading restoration versus generic direct PDF input; both are supported entry points.
- `paths.py` locations and aliases (`SOURCE`, `OUT`, `CLEAN`, `CHUNKS_PATH`, `DOC_METADATA_PATH`, `DEFAULT_CASES`, `DEFAULT_LOG_DIR`): used locally/defaults and retained for compatibility.
- Day 07 checks versus Day 06/08 grading: distinct diagnostics; phrase matching alone does not make them interchangeable.
- Web QA/tool log writers: different schemas/audit consumers; no generic logging abstraction introduced.
- CLI helpers, lazy imports, model cache, health/errors, locks, session/expiry/cancellation/idempotency checks, citation validation, and UI evidence rendering.
- Source PDF, processed text, every chunk strategy, datasets, experiment definition, logs, outputs, and historical reports.

## Existing issues and optional future work

- `python` is not on this terminal's PATH. The venv's base executable was inaccessible in the sandbox; approved execution outside that restriction worked. No interpreter/package configuration was changed.
- Day 10 unit tests previously isolated tool logs but appended successful policy reads to the historical web log. Baseline execution redirected WEB_LOG in memory; test setup now isolates it permanently.
- Both live smoke scripts have the write side effects listed above. Preserved; a separate change could add explicit output/log overrides.
- Source inspection: Day 05 summaries/sample selection assume nonempty chunks (`summarize([])` reaches `min(sizes)`). Not an observed policy-PDF failure; changing empty-input behavior was outside scope.
- Source inspection: `Retriever.describe()` reports default CHUNKS_PATH even with a custom constructor path. Preserved because fixing provenance output changes behavior; add a focused regression before a separate fix.
- Day 08 records evaluator/retriever hashes. Editing those sources invalidates old comparison controls, so both were left unchanged and completed evaluations were not rerun.
- Existing quote/phrase heuristics retain their documented limits; this cleanup neither strengthens nor weakens them or treats them as entailment proof.

Separate optional work: isolate smoke outputs, decide whether to retain the copy, and add
empty-input/custom-path coverage before changing those behaviors. No model/performance/prompt
optimization was attempted.

## Verification and actual results

Baseline, before edits: **57 tests passed in 5.104 seconds**, using this wrapper to keep
the original tests from appending web history:

```powershell
@'
import tempfile
import unittest
from pathlib import Path
import web_app
with tempfile.TemporaryDirectory() as directory:
    web_app.WEB_LOG = Path(directory) / 'web_qa_log.jsonl'
    unittest.main(module=None, argv=['unittest', 'discover', '-s', 'tests', '-v'])
'@ | .\.venv\Scripts\python.exe -
```

Initial attempts with unavailable `python`, then the sandbox-restricted venv, failed to
launch. These were environment failures, not baseline test failures. Venv execution with
the required access passed; no baseline unit-test failures were observed.

After cleanup, the requested discovery command using the actual venv executable:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: **59 tests passed in 4.742 seconds**. Existing assertions were not weakened or
skipped. Added checks cover pending-recovery expiration logging exactly once and log failures
preserving pending state through both expiry callers. Existing tests cover context/JSON,
citations/quotes/numbers, abstention, uncited content, errors, Flask contracts, golden rules,
checkpoint overwrite protection, date/session validation, cancellation/supersession, and
concurrent/repeated confirmation. The checkpoint test intentionally prints 16 simulated
generation errors and verifies continuation; those messages are expected test output.

Additional checks were executed as a one-off PowerShell here-string piped into
`.\.venv\Scripts\python.exe -` (no extra script retained):

| Check | Method and result |
|---|---|
| Day 05 behavior equivalence | Loaded original preparation/chunk/export modules from `git show HEAD:<file>` and current modules. Ran both on the actual PDF into separate TemporaryDirectory subtrees, redirecting every output constant. **16 generated files byte-identical; console output identical.** No project data/output artifacts written. |
| Python syntax | In-memory `compile` of all **19** project/test/evaluation Python files: passed. |
| Imports | Imported all **12** primary modules successfully. Asserted `evaluation.__file__` resolves to root `evaluation.py`: passed. No embedding model instantiated by this check. |
| CLI availability | Subprocess help commands below all exited **0**. |
| Unused-import candidates | AST import names compared to loaded names, followed by manual reference review. No remaining candidates reported; no dependencies removed. |
| Diff hygiene | `git diff --check`: passed. Git emitted normal LF/CRLF conversion notices. |
| Preservation | `git diff --exit-code -- data logs outputs requirements.txt .gitignore README.md retriever.py evaluation.py evaluation static templates paths.py test_queries.json heldout_queries.json "qa_pipeline - Copy.py"`: exit **0**. Final source diff reviewed. |

CLI help commands used the venv interpreter:

```powershell
.\.venv\Scripts\python.exe run_rag.py --help
.\.venv\Scripts\python.exe run_rag.py retrieve --help
.\.venv\Scripts\python.exe run_rag.py qa --help
.\.venv\Scripts\python.exe run_rag.py web --help
.\.venv\Scripts\python.exe evaluation/run_day08_evaluation.py --help
```

Live Ollama generation, real embedding inference, full live evaluation, and browser-driven
UI interaction were not performed. Ollama availability was not probed; no claim of complete
live-application verification is made. Frontend contracts were reviewed in source and
exercised through offline Flask tests. Data-producing entry points were verified only in
temporary storage or through source review, not rerun against historical artifacts.

## Safety confirmation

Existing names, paths, function signatures, CLI options, environment defaults, routes,
JSON schemas, chunk IDs, settings, prompts, retrieval/evaluation rules, and UI assets remain.
No existing dataset, source PDF, processed document, chunk, log, output, or historical report
was changed. The sole new report is this file. The intentional test-only behavior change
redirects test web logs into temporary storage; production logging paths/behavior remain.
The private expiry helper is an internal addition; no existing public function was removed
or renamed. No other behavior/interface exception is intended.
