# RAG Pipeline: Chunking (Day 05) + Retrieval Evaluation (Day 06)

This project takes one policy document, the **Acme Corp IT and HR Policy Handbook** (PDF),
and turns it into a searchable knowledge base, then measures how well search works.

1. **Day 05: chunking.** Extract the PDF text, clean it, and split it into small pieces
   ("chunks") using three strategies: fixed-length, sliding-window and semantic (TF-IDF
   topic-shift). Every chunk keeps an ID, page numbers, section names and character offsets.
2. **Day 06: retrieval evaluation.** Embed the semantic chunks and six test questions with
   the same model (`BAAI/bge-small-en-v1.5`), find the top 3 chunks per question by cosine
   similarity, grade them against evidence phrases from the handbook, and log everything.
   It also runs one experiment (heading-aligned "child" chunks) and compares before vs. after.

**Retrieval only.** There is no LLM answer generation, no LangChain/LlamaIndex, no agents,
and no API key. The embedding model runs locally on your CPU.

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
├── data/
│   ├── source/                Acme_Corp_IT_and_HR_Policy.pdf  (the original document)
│   ├── processed/             extracted_raw.txt, cleaned_policy.txt,
│   │                          document_metadata.json, cleaning_report.json
│   └── chunks/                metadata_{fixed_length,sliding_window,semantic}.json
│                              (+ chunks_*.jsonl / chunks_*.txt after re-chunking)
├── outputs/                   Day 05 strategy comparison: comparison_summary.json/.csv,
│                              comparison_console.txt, recommendation.txt
├── logs/                      Day 06 results: retrieval_log.jsonl + runs/<run_id>/
├── reports/                   Day06_Retrieval_Evaluation_Report.md (the written report)
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

This installs `pypdf`, `numpy` and `sentence-transformers` (which pulls in PyTorch, so it takes
a few minutes). The first retrieval run downloads the embedding model (~130 MB) from Hugging
Face into your user cache; later runs use the cached copy.

## 4. Run the project

| What you want | Command |
|---|---|
| **Everything**: PDF -> chunks -> retrieval evaluation | `python run_rag.py all` |
| Only regenerate the chunks from the PDF (Day 05) | `python run_rag.py prepare` |
| Re-chunk after you hand-edited `cleaned_policy.txt` | `python run_rag.py chunk` |
| **Only re-run the six retrieval queries** on the existing chunks (Day 06) | `python run_rag.py retrieve` |
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
