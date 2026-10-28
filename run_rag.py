"""One entry point for the whole RAG pipeline (Day 05 chunking, Day 06 retrieval,
Day 07 grounded answers with a local Ollama model).

    python run_rag.py all        PDF -> cleaned text -> chunks -> retrieval evaluation
    python run_rag.py prepare    PDF -> cleaned text -> chunks (Day 05 only)
    python run_rag.py chunk      re-chunk the existing cleaned text (after a manual edit)
    python run_rag.py retrieve   retrieval evaluation on the existing chunks (Day 06 only)
    python run_rag.py qa         answer the test questions with Ollama + log them (Day 07)
    python run_rag.py web        start the local web app at http://127.0.0.1:5000 (Day 07)

`retrieve` runs all test queries on both indexes (Day 05 chunks = baseline,
heading-aligned children = experiment) and writes the before/after
comparison. Any extra options are passed to run_queries.py, for example:

    python run_rag.py retrieve --index parent
    python run_rag.py retrieve -q "How many sick days do I get?"
    python run_rag.py retrieve -i

`qa` and `web` need Ollama running with the llama3.2 model (see README).
Options are passed on, for example:

    python run_rag.py qa -q "How many sick days do I get?"
    python run_rag.py qa --file heldout_queries.json
    python run_rag.py web --port 8000
"""
import io
import sys
from contextlib import redirect_stdout

from paths import CHUNKS_DIR, CLEAN_TXT, LOGS_DIR, OUTPUTS_DIR, SOURCE_PDF, require

COMMANDS = ("all", "prepare", "chunk", "retrieve", "qa", "web")


def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def extract():
    """Day 05 step 1: PDF -> cleaned_policy.txt.

    Returns True when the new text matches the previously reviewed text, so
    chunking can continue safely. Otherwise stops so the text can be reviewed
    first (the review gate from the original Day 05 workflow).
    """
    import prepare_dataset

    banner("Day 05 | Step 1: extract and clean the PDF")
    require(SOURCE_PDF, "Put the policy PDF in data/source/.")
    previous = CLEAN_TXT.read_bytes() if CLEAN_TXT.exists() else None
    prepare_dataset.main()
    if previous is None:
        raise SystemExit(
            f"\nNew extraction saved to {CLEAN_TXT}.\n"
            "Review it, then run:  python run_rag.py chunk"
        )
    if CLEAN_TXT.read_bytes() != previous:
        backup = CLEAN_TXT.with_name("cleaned_policy.previous.txt")
        backup.write_bytes(previous)
        raise SystemExit(
            f"\n{CLEAN_TXT.name} differs from the previous version "
            f"(saved as {backup.name}, in case it held manual edits).\n"
            "Review the new text, then run:  python run_rag.py chunk"
        )
    print("Cleaned text is identical to the previously reviewed version.")


def chunk():
    """Day 05 steps 2-3: chunk the cleaned text and export reports/metadata."""
    import chunking_pipeline
    import export_reports

    banner("Day 05 | Step 2: chunk with three strategies")
    require(CLEAN_TXT, "Run:  python run_rag.py prepare")
    console = io.StringIO()
    with redirect_stdout(console):
        chunking_pipeline.main([str(CLEAN_TXT)])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    console_file = OUTPUTS_DIR / "comparison_console.txt"
    console_file.write_text(console.getvalue(), encoding="utf-8")
    # Print the summary table only; the full side-by-side view is in the file.
    lines = console.getvalue().splitlines()
    end = next((i for i, line in enumerate(lines) if "ORPHANED HEADINGS" in line), len(lines))
    print("\n".join(lines[:end]).rstrip())
    print("\n".join(line for line in lines if line.startswith("Saved")))
    print(f"Full comparison output: {console_file}")

    banner("Day 05 | Step 3: export metadata, readable chunks and reports")
    # Running this step is the confirmation that cleaned_policy.txt was reviewed.
    export_reports.main(reviewed=True)
    print(f"Chunks for retrieval: {CHUNKS_DIR / 'metadata_semantic.json'}")


def retrieve(extra_args):
    """Day 06: retrieval evaluation over the Day 05 semantic chunks."""
    import run_queries

    banner("Day 06 | Retrieval evaluation (top-3, cosine similarity)")
    # Default to both indexes (baseline vs experiment); a later --index wins.
    run_queries.main(["--index", "both", *extra_args])
    print(f"\nLogs and results: {LOGS_DIR}")


def qa(extra_args):
    """Day 07: run test questions through retrieval + local generation."""
    import run_qa_eval

    banner("Day 07 | Grounded answers (top-3 retrieval + local Ollama model)")
    return run_qa_eval.main(extra_args)


def web(extra_args):
    """Day 07: the local web app."""
    import web_app

    banner("Day 07 | Blend RAG Assistant (local web app)")
    return web_app.main(extra_args)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    command, extra = argv[0], argv[1:]
    if extra and command in ("prepare", "chunk"):
        raise SystemExit(f"'{command}' takes no options (got {' '.join(extra)})")

    if command in ("all", "prepare"):
        extract()
    if command in ("all", "prepare", "chunk"):
        chunk()
    if command in ("all", "retrieve"):
        retrieve(extra)
    if command == "qa":
        return qa(extra)
    if command == "web":
        return web(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
