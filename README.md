# RAG Pipeline: Chunking (Day 05) + Retrieval Evaluation (Day 06) + Grounded Answers (Day 07)

This project takes one policy document, the **Acme Corp IT and HR Policy Handbook** (PDF),
and turns it into a searchable knowledge base, then measures how well search works.

1. **Day 05: chunking.** Extract the PDF text, clean it, and split it into small pieces
   ("chunks") using three strategies: fixed-length, sliding-window and semantic (TF-IDF
   topic-shift). Every chunk keeps an ID, page numbers, section names and character offsets.
2. **Day 06: retrieval evaluation.** Embed the semantic chunks and six test questions with
   the same model (`BAAI/bge-small-en-v1.5`), find the top 3 chunks per question by cosine
   similarity, grade them against evidence phrases from the handbook, and log everything.
   It also runs one experiment (heading-aligned "child" chunks) and compares before vs. after.
3. **Day 07: grounded answers + web app.** Send the Day 06 top-3 chunks and the question to a
   **local** `llama3.2` model in [Ollama](https://ollama.com), get a short JSON answer with
   citations, check every citation against the retrieved chunks, and show it all in a small
   Blend-styled web page at `http://127.0.0.1:5000`. See section 7 below.

**Everything runs on your computer.** No cloud LLM, no API key, no LangChain/LlamaIndex, no
agents. The embedding model runs on your CPU; the answer model runs in your local Ollama.

---

## Project layout

```text
rag/
├── run_rag.py                 <- START HERE: one command for each stage
├── paths.py                   all file locations (built from this folder)
│
├── prepare_dataset.py         Day 05 step 1: PDF -> raw + cleaned text, document metadata
├── chunking_pipeline.py       Day 05 step 2: three chunking strategies -> chunks_*.jsonl
├── export_reports.py          Day 05 step 3: add pages/metadata, write metadata_*.json + reports
│
├── retriever.py               Day 06: embeddings + cosine search (parent or child index)
├── evaluation.py              Day 06: grading (evidence rank, margin, label check, summary)
├── run_queries.py             Day 06: runs queries, prints top-k, writes logs
├── test_queries.json          the six evaluation questions + their evidence phrases
│
├── qa_pipeline.py             Day 07: retrieval -> context -> Ollama -> citation checks
├── web_app.py                 Day 07: Flask server (GET /, GET /api/health, POST /api/ask)
├── run_qa_eval.py             Day 07: runs questions through the full pipeline, logs answers
├── heldout_queries.json       Day 07: 8 extra questions (not tuned on) to check over-abstention
├── templates/index.html       Day 07: the single web page
├── static/style.css, app.js   Day 07: page styling + the code that calls /api/ask
├── tests/test_qa_pipeline.py  Day 07: offline unit tests (no Ollama needed)
│
├── data/
│   ├── source/                Acme_Corp_IT_and_HR_Policy.pdf  (the original document)
│   ├── processed/             extracted_raw.txt, cleaned_policy.txt,
│   │                          document_metadata.json, cleaning_report.json
│   └── chunks/                metadata_{fixed_length,sliding_window,semantic}.json
│                              (+ chunks_*.jsonl / chunks_*.txt after re-chunking)
├── outputs/                   Day 05 strategy comparison: comparison_summary.json/.csv,
│                              comparison_console.txt, recommendation.txt
├── logs/                      Day 06 results: retrieval_log.jsonl + runs/<run_id>/
│                              Day 07 results: qa_log.jsonl + qa_runs/<run_id>/, web_qa_log.jsonl
├── reports/                   Day06_Retrieval_Evaluation_Report.md,
│                              Day07_Grounded_QA_Evaluation_Report.md
├── requirements.txt
└── .gitignore
```

## How Day 05 connects to Day 06

```text
data/source/*.pdf
   │  prepare_dataset.py
   ▼
data/processed/cleaned_policy.txt        (text with [[PAGE n]] markers and # headings)
   │  chunking_pipeline.py
   ▼
data/chunks/chunks_*.jsonl               (chunks + metadata, 3 strategies)
   │  export_reports.py                  (adds page numbers + document metadata)
   ▼
data/chunks/metadata_semantic.json       <- the file Day 06 reads
   │  retriever.py + run_queries.py + evaluation.py
   ▼
logs/runs/<run_id>/results.json|csv, comparison.json
```

Each record in `metadata_semantic.json` looks like `{"chunk_id": ..., "text": ..., "metadata": {...}}`.
Chunk IDs look like `cleaned_policy::semantic::012::<hash>`, so do not rename `cleaned_policy.txt`.

The Day 05 pipeline is deterministic: re-running it on the same PDF gives exactly the same chunks.

---

## 1. Open a terminal in the right folder

1. In VS Code, open the `Day 4 Rag` folder (**File > Open Folder...**).
2. Open a terminal: **Terminal > New Terminal** (or press <kbd>Ctrl</kbd>+<kbd>`</kbd>).
   It opens in PowerShell, in the `Day 4 Rag` folder.
3. Move into the project:

   ```powershell
   cd rag
   ```

All commands below are run from inside `rag`. (The scripts find their files from their own
location, so they also work from other folders, but `cd rag` keeps the commands short.)

## 2. Create and activate a virtual environment (first time only)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Each time you open a new terminal, run
`.\.venv\Scripts\Activate.ps1` again (from the `rag` folder).

If `py` is not found, use `python -m venv .venv` instead.

## 3. Install the dependencies (first time only)

```powershell
python -m pip install -r requirements.txt
```

This installs `pypdf`, `numpy`, `sentence-transformers` (which pulls in PyTorch, so it takes
a few minutes) and `flask` (the small Day 07 web server). The first retrieval run downloads the embedding model (~130 MB) from Hugging
Face into your user cache; later runs use the cached copy.

## 4. Run the project

| What you want | Command |
|---|---|
| **Everything**: PDF -> chunks -> retrieval evaluation | `python run_rag.py all` |
| Only regenerate the chunks from the PDF (Day 05) | `python run_rag.py prepare` |
| Re-chunk after you hand-edited `cleaned_policy.txt` | `python run_rag.py chunk` |
| **Only re-run the six retrieval queries** on the existing chunks (Day 06) | `python run_rag.py retrieve` |
| **Ask the six questions with the local LLM** (Day 07) | `python run_rag.py qa` |
| **Start the web app** (Day 07) | `python run_rag.py web` |
| Show the help | `python run_rag.py --help` |

`retrieve` (and the last stage of `all`) runs all six questions in `test_queries.json` on two indexes:

* **parent**: the 47 Day 05 semantic chunks (the baseline), and
* **child**: the same chunks split at their headings (the experiment),

then prints a before-vs-after comparison. For each question you see the top 3 results with
similarity score, chunk ID, page numbers, section name and a text snippet, plus a status
(`YES`, `PARTIAL`, `NO`, or `NO EVIDENCE (expected)` for the unanswerable question q6) and the margin.

Extra options after `retrieve` are passed to `run_queries.py`:

```powershell
python run_rag.py retrieve --index parent          # baseline only
python run_rag.py retrieve --index child           # experiment only
python run_rag.py retrieve --full                  # print full chunk text, not snippets
python run_rag.py retrieve -q "How many sick days do I get?"   # your own question
python run_rag.py retrieve -i                      # interactive prompt (:help inside)
python run_rag.py retrieve --no-log                # do not write log files
```

You can also run the stage scripts directly; they behave as before:

```powershell
python prepare_dataset.py
python chunking_pipeline.py
python export_reports.py --reviewed
python run_queries.py --index both
```

### The review step (why `prepare` may stop)

PDF extraction is never perfect, so Day 05 has a review gate: check `cleaned_policy.txt`
before chunking it. `prepare` / `all` handle this for you:

* If the new extraction is **identical** to the existing (already reviewed)
  `cleaned_policy.txt`, they continue automatically.
* If it is **different** (or there was no previous file), they stop. The old text is saved
  as `data/processed/cleaned_policy.previous.txt` so hand edits are not lost. Review the new
  text, then run `python run_rag.py chunk` (running `chunk` is your confirmation).

## 5. Where to find the results

| Result | Location |
|---|---|
| Original PDF | `data/source/Acme_Corp_IT_and_HR_Policy.pdf` |
| Extracted and cleaned text | `data/processed/extracted_raw.txt`, `data/processed/cleaned_policy.txt` |
| Chunks + metadata used by retrieval | `data/chunks/metadata_semantic.json` |
| Other strategies' chunks | `data/chunks/metadata_fixed_length.json`, `metadata_sliding_window.json` |
| Human-readable chunk listings | `data/chunks/chunks_*.txt` (created by `prepare` / `chunk`) |
| Chunking strategy comparison | `outputs/comparison_summary.csv`, `outputs/recommendation.txt`, `outputs/comparison_console.txt` |
| Every retrieval run ever (append-only) | `logs/retrieval_log.jsonl` |
| One retrieval run | `logs/runs/<date-time>_<mode>_<index>/results.json`, `results.csv`, `comparison.json` |
| Written evaluation report | `reports/Day06_Retrieval_Evaluation_Report.md` |
| Day 07 answers (every run, append-only) | `logs/qa_log.jsonl`; one run: `logs/qa_runs/<run_id>/results.json` |
| Questions asked in the web app | `logs/web_qa_log.jsonl` |
| Day 07 report | `reports/Day07_Grounded_QA_Evaluation_Report.md` |

The report's numbers come from run `logs/runs/20261015-140746_file_both`.
Open `results.csv` in Excel for one row per (question, rank).

## 6. Troubleshooting

**`.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled`**
Allow local scripts for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Or skip activation and call the venv's Python directly: `.\.venv\Scripts\python.exe run_rag.py all`.

**`The term '.\.venv\Scripts\Activate.ps1' is not recognized`**
You are not in the `rag` folder, or the venv does not exist yet. Run `cd rag`
(check with `Get-Location`), then step 2.

**`ModuleNotFoundError: No module named 'pypdf'` / `'sentence_transformers'` / `'numpy'`**
The venv is not active, or dependencies are not installed. Activate it (prompt shows `(.venv)`),
then `python -m pip install -r requirements.txt`. In VS Code, also pick the interpreter
`rag\.venv\Scripts\python.exe` (**Ctrl+Shift+P > Python: Select Interpreter**).

**`Missing required file: ...`**
The scripts never invent replacement data; they name the missing file instead.
* `data\source\Acme_Corp_IT_and_HR_Policy.pdf` missing: put the PDF back in `data/source/`.
* `data\chunks\metadata_semantic.json` missing: run `python run_rag.py prepare`.
* `data\processed\cleaned_policy.txt` missing: run `python run_rag.py prepare`.

**`Cleaned TXT changed after chunking. Run chunking_pipeline.py again.`**
You edited `cleaned_policy.txt` after chunking. Run `python run_rag.py chunk`.

**The model download fails (no internet / proxy)**
The first retrieval run needs to download `BAAI/bge-small-en-v1.5` once. Connect to the internet
and run again. After that it is cached in `%USERPROFILE%\.cache\huggingface`.

**Accented characters look wrong (`cafÃ©`) when you `Get-Content` a file**
The files are UTF-8; use `Get-Content -Encoding UTF8 <file>` or open them in VS Code.

**OneDrive is syncing thousands of files**
`.venv` is inside a OneDrive folder. That works, but you can pause OneDrive sync while installing.
`.venv` is local and can be deleted and recreated at any time with steps 2 and 3.

---

## 7. Day 07: grounded answers and the web app

### 7.1 Start Ollama and check the model (first time only)

1. Install Ollama for Windows from <https://ollama.com/download> if `ollama --version` fails.
2. Make sure Ollama is running: open the **Ollama** app (it sits in the system tray), or run
   `ollama serve` in a separate terminal. Check it with:

   ```powershell
   ollama list
   ```

3. If `llama3.2` is not in the list, download it once (about 2 GB):

   ```powershell
   ollama pull llama3.2
   ```

### 7.2 Start the web app

From the `rag` folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt      # adds flask if you installed before Day 07
python run_rag.py web
```

Wait for `Open http://127.0.0.1:5000 in your browser`, then open **<http://127.0.0.1:5000>**.
Type a question and press **Ask** (or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>). The page shows:

* the **answer**, marked *Grounded answer* or *Insufficient evidence*,
* the **citations and sources**: section, page, similarity score, chunk ID and the quote the
  model used, for each cited chunk,
* **View retrieved evidence**: the actual top-3 chunks and their scores, cited or not.

The first answer after starting Ollama takes longer, because the model is loaded into memory. On a
laptop CPU, expect roughly 10-30 seconds per answer. Stop the server with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

To use another port: `python run_rag.py web --port 8000`. The server listens on `127.0.0.1` only,
so other computers cannot reach it.

### 7.3 Run the questions from the terminal

```powershell
python run_rag.py qa                                  # the six questions in test_queries.json
python run_rag.py qa --file heldout_queries.json      # the eight held-out questions
python run_rag.py qa -q "How many sick days do I get?"
python run_rag.py qa --index parent                   # use the Day 05 chunks instead
```

Each run is logged to `logs/qa_runs/<run_id>/results.json` and appended to `logs/qa_log.jsonl`.

### 7.4 How an answer is produced (`qa_pipeline.py`)

1. **Retrieve**: the Day 06 `Retriever` (unchanged) returns the top 3 chunks. The default is the
   `child` index (heading-aligned chunks). Day 06 showed it labels every evidence chunk with
   its correct section.
2. **Build the context**: each chunk goes into an `<excerpt chunk_id=... section=... pages=...>`
   block, in rank order.
3. **Generate**: the context and question go to Ollama's `/api/chat` with a JSON schema
   (`supported`, `answer`, `citations[{chunk_id, quote}]`) at temperature 0. The system prompt
   says to answer only from the excerpts and to treat them as evidence, not instructions. When
   the excerpts do not answer the question, the model must reply with the exact
   insufficient-evidence sentence.
4. **Validate**: a citation is kept only if its `chunk_id` was retrieved for this question
   **and** its quote is found in that chunk. Answer sentences built from a retrieved chunk the
   model did *not* cite are removed. Numbers in the answer must appear in the cited chunks.
   If nothing verifiable is left, the answer becomes:
   *"Based on the provided documents, there is insufficient evidence to answer this question."*
5. **Return** `{answer, citations, sources, debug}`. `debug` holds the retrieved chunks, the raw
   model output, rejected citations, removed sentences and timings.

### 7.5 Configuration (no code changes needed)

| Environment variable | Default | Example (PowerShell) |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | `$env:OLLAMA_MODEL = "llama3.2:3b"` |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | `$env:OLLAMA_HOST = "http://127.0.0.1:11500"` |
| `OLLAMA_TIMEOUT` | `180` (seconds) | `$env:OLLAMA_TIMEOUT = "300"` |
| `RAG_INDEX` | `child` | `$env:RAG_INDEX = "parent"` |

Set them in the same terminal before `python run_rag.py web`.

### 7.6 Tests

```powershell
python -m unittest discover -s tests -v
```

These run offline, with a fake retriever and a fake Ollama client. They cover context building,
JSON parsing, citation and quote checks (real and fabricated IDs), abstention, error handling and
the HTTP endpoints. `python run_rag.py qa` is the live end-to-end test with the real model.

### 7.7 Troubleshooting the web app

| Message on the page | What to do |
|---|---|
| **Ollama is not running** | Open the Ollama app or run `ollama serve`, then ask again. |
| **Model not installed** | `ollama pull llama3.2` (or set `OLLAMA_MODEL` to a model shown by `ollama list`). |
| **The local model timed out** | Ask again (the first call loads the model), or raise `OLLAMA_TIMEOUT`. |
| **Document chunks are missing** | `python run_rag.py prepare` |
| **The model returned malformed output** | Ask again. The raw output is in the `debug` field of the server's response. |
| `ModuleNotFoundError: No module named 'flask'` | `python -m pip install -r requirements.txt` |
| Port 5000 already in use | `python run_rag.py web --port 8000`, then open `http://127.0.0.1:8000` |

## 8. Day 08: golden evaluation of the existing app

From `rag`, with Ollama running and the existing virtual environment activated:

```powershell
python evaluation/run_day08_evaluation.py baseline --output outputs/day08_new
python evaluation/run_day08_evaluation.py experiment --output outputs/day08_new
```

The first command runs the unchanged dashboard pipeline. The second reuses that baseline,
then runs the single recorded prompt experiment in `evaluation/experiment.json`.
To run both phases with one command, choose a fresh output folder:

```powershell
python evaluation/run_day08_evaluation.py experiment --output outputs/day08_repeat
```

Do not reuse an output folder containing a completed after run: existing phase results are
never overwritten. If a run is interrupted, keep its partial output for audit and use a fresh
folder for a complete rerun. Errors are recorded per case and the runner continues.

The runner uses `QAPipeline.ask()`, exactly as `/api/ask` does. It does not duplicate answer
logic or call a judge model. It preserves `OLLAMA_MODEL`, `RAG_INDEX`, Top-K, temperature,
seed, chunking, citation validation and the UI. The experimental prompt addition is scoped
to the evaluation process; the dashboard retains its existing prompt. The source PDF,
cleaned text, chunk metadata, dataset, retriever and evaluator are hashed for comparison.
Changing a controlled input between phases causes comparison to stop.

The supplied JSON dataset is read unchanged from `evaluation/day08_golden_dataset.json`
(with a project-root fallback). It must contain exactly 16 distinct cases. The existing
Day 06 `evaluation.py` is separate; do not add an `evaluation/__init__.py` that shadows it.

Outputs include `baseline_results.json/.csv`, `baseline_report.md`, `after_results.json/.csv`,
`after_report.md`, `before_after_comparison.csv` and `evaluation_report.md`. JSON includes
all Top-K texts and metadata, raw model output, validated dashboard answer, citations,
latencies, expected behaviour, phrase matches, missing groups and failure explanations.
Each case is checkpointed. A full evidence match for q3 still means **partial-only
answerability**; no general overseas-trip approval rule is implied.

Automatic checks are heuristics. Review paraphrases, conditions, negations, extra claims,
and claim-level citation support in the JSON. A safe standard abstention can fail the
supplied answer-wording group while passing the separate abstention check. A duration
mentioned in a denial goes to manual review rather than automatically being a false claim.

Run all offline tests (no Ollama generation needed):

```powershell
python -m unittest discover -s tests -v
```

Launch the existing dashboard as before:

```powershell
python run_rag.py web
```

Open `http://127.0.0.1:5000`.

The completed 25 October 2026 run is in [outputs/day08/evaluation_report.md](outputs/day08/evaluation_report.md).
It includes manual inspection of all 16 cases. The prompt experiment was unsuccessful and
was not adopted by the dashboard. The recorded review describes this specific run; new
runs generate fresh automatic reports and require their own case-by-case inspection.
The original reviewed comparison additionally includes a `reviewed_outcome` column.

Optional real-server launch smoke check (starts an unused local port, then stops its server):

```powershell
python tests/smoke_launch_day08.py
```
