import csv
import json
import re
import sys

import chunking_pipeline as pipeline
from paths import (CHUNKS_DIR, CLEAN_TXT, CLEANING_REPORT, DOC_METADATA, OUTPUTS_DIR,
                   STRATEGIES, require)


CLEAN = CLEAN_TXT


def main(reviewed=None):
    if reviewed is None:
        reviewed = "--reviewed" in sys.argv
    if not reviewed:
        raise SystemExit(
            "Review cleaned_policy.txt first, then run with --reviewed."
        )

    hint = "Run chunking_pipeline.py (or: python run_rag.py chunk) first."
    text = require(CLEAN, hint).read_text(encoding="utf-8")
    document_metadata = json.loads(
        require(DOC_METADATA, hint).read_text(encoding="utf-8")
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Page offsets are computed AFTER your manual cleaning.
    markers = list(
        re.finditer(r"(?m)^\[\[PAGE (\d+)\]\]$", text)
    )
    page_spans = [
        (
            marker.start(),
            markers[i + 1].start() if i + 1 < len(markers) else len(text),
            int(marker.group(1)),
        )
        for i, marker in enumerate(markers)
    ]

    if not page_spans:
        raise ValueError("Page markers are missing from the cleaned TXT.")

    results = {}
    summary = {}

    for strategy in STRATEGIES:
        path = require(CHUNKS_DIR / f"chunks_{strategy}.jsonl", hint)
        chunks = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not chunks:
            raise ValueError(f"No chunks found for {strategy}.")

        readable = []
        records = []

        for chunk in chunks:
            body = chunk["text"]
            metadata = chunk["metadata"]
            start = metadata["start_char"]
            end = metadata["end_char"]

            # Detect edits made to the TXT AFTER chunking.
            if text[start:end] != body:
                raise ValueError(
                    "Cleaned TXT changed after chunking. "
                    "Run chunking_pipeline.py again."
                )

            metadata.update(document_metadata)
            metadata["offset_reference"] = CLEAN.name
            metadata["pages"] = pipeline.pages_for(
                (start, end), page_spans
            )

            # Day 06 (retriever.py) loads these {chunk_id, text, metadata} records.
            records.append(
                {"chunk_id": metadata["chunk_id"], "text": body, "metadata": metadata}
            )
            readable.append(
                f"CHUNK {metadata['chunk_index']}\n"
                f"ID: {metadata['chunk_id']}\n"
                f"Section: {metadata['section_path']}\n"
                f"Pages: {metadata['pages']}\n"
                f"Characters: {metadata['char_count']}\n\n"
                f"{body}\n"
                + "-" * 80
            )

        # Rewrite JSONL with the enriched provenance metadata.
        path.write_text(
            "\n".join(
                json.dumps(chunk, ensure_ascii=False)
                for chunk in chunks
            ) + "\n",
            encoding="utf-8",
        )

        (CHUNKS_DIR / f"chunks_{strategy}.txt").write_text(
            "\n\n".join(readable), encoding="utf-8"
        )
        (CHUNKS_DIR / f"metadata_{strategy}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        results[strategy] = chunks
        summary[strategy] = pipeline.summarize(chunks)

    (OUTPUTS_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (OUTPUTS_DIR / "comparison_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["strategy"] + list(summary["fixed_length"]),
        )
        writer.writeheader()
        for strategy, metrics in summary.items():
            writer.writerow({"strategy": strategy, **metrics})

    fixed = summary["fixed_length"]
    sliding = summary["sliding_window"]
    lexical = summary["semantic"]

    recommendation = f"""Day 05 - Chunking Recommendation

Measured results
----------------
Fixed-length:
  {fixed['chunks']} chunks
  {fixed['chunks_cut_mid_word']} chunks flagged for mid-word cuts
  {fixed['broken_table_chunks']} chunks flagged for table problems

Sliding window:
  {sliding['chunks']} chunks
  {sliding['duplicated_chars']} duplicated characters
  {sliding['duplication_pct']}% duplication by the script's definition

TF-IDF lexical topic-shift:
  {lexical['chunks']} chunks
  {lexical['broken_table_chunks']} chunks flagged for table problems
  Maximum chunk size: {lexical['max_chars']} characters

Recommendation
--------------
For this structured policy, provisionally take the TF-IDF lexical
approach forward if visual inspection confirms that tables, headings,
and rule relationships remain understandable.

It can preserve recognized tables and code blocks, and it selects
boundaries using lexical topic changes. However, its size limits,
threshold, and text-segmentation rules affect the result.

Fixed-length is a useful simple baseline, but it can cut words and
separate rules from their context.

Sliding windows avoid mid-word cuts and repeat boundary context, but
they duplicate text and can still split tables or nested lists.

Limitations
-----------
The third method approximates topic changes; it does not establish
semantic understanding. Protected tables/code may exceed 900 chars.
Quality flags are heuristics, not proof of completeness.
No retrieval effectiveness was measured.

Inspect the same hardware, remote-work, and long-paragraph sections
in comparison_console.txt before finalizing this recommendation.
"""

    (OUTPUTS_DIR / "recommendation.txt").write_text(
        recommendation, encoding="utf-8"
    )

    report = json.loads(require(CLEANING_REPORT).read_text(encoding="utf-8"))
    report["manual_review_confirmed"] = True
    report["final_clean_characters"] = len(text)
    report["final_replacement_characters"] = text.count("�")
    CLEANING_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved readable chunks, metadata, CSV/JSON comparison, and recommendation.")


if __name__ == "__main__":
    main()
