"""Run queries against the retriever and log every result to disk.

Query sources (no queries live in this code):
    python run_queries.py                          # test_queries.json
    python run_queries.py --file my_cases.json     # any case file
    python run_queries.py -q "question" -q "another"
    python run_queries.py -i                       # interactive prompt

Options:
    --index parent|child|both   Day 05 chunks, heading-aligned children, or
                                both side by side (writes comparison.json)
    --top-k N                   results per query (default 3)
    --full                      print full chunk text instead of a snippet
    --no-log                    do not write log files

Logs (default ./logs):
    retrieval_log.jsonl         append-only history, one line per query run
    runs/<run_id>/results.json  every record of this run (full chunk text)
    runs/<run_id>/results.csv   one row per (query, rank) for spreadsheets
    runs/<run_id>/comparison.json   only with --index both
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from evaluation import evaluate, summarise
from paths import LOGS_DIR, TEST_QUERIES, require
from retriever import INDEX_MODES, Retriever

DEFAULT_CASES = TEST_QUERIES
DEFAULT_LOG_DIR = LOGS_DIR
CASE_FIELDS = ("id", "category", "question", "expected_evidence", "notes")


# --------------------------------------------------------------------------
# Loading cases
# --------------------------------------------------------------------------
def load_cases(path):
    cases = json.loads(require(Path(path)).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit(f"{path}: expected a JSON list of test cases")
    for n, c in enumerate(cases, start=1):
        if not c.get("question"):
            raise SystemExit(f"{path}: case #{n} has no 'question'")
        c.setdefault("id", f"case{n}")
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"{path}: duplicate ids {sorted(dupes)}")
    return cases


def adhoc_case(question, n):
    return {"id": f"adhoc{n}", "category": "Ad hoc", "question": question}


# --------------------------------------------------------------------------
# Running one query
# --------------------------------------------------------------------------
def run_case(retriever, case, top_k, run_id, mode):
    ranking = retriever.rank_all(case["question"])
    hits = [retriever.format_hit(r, pos, s)
            for r, (pos, s) in enumerate(ranking[:top_k], start=1)]
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        **retriever.describe(),
        "top_k": top_k,
        **{k: case[k] for k in CASE_FIELDS if k in case},
        "unsupported": bool(case.get("unsupported")),
        "evaluation": evaluate(case, hits, ranking, retriever),
        "results": hits,
    }


def print_record(rec, full=False):
    print(f"\n[{rec['index']}] {rec['id']} ({rec.get('category', '')}): {rec['question']}")
    if rec.get("expected_evidence"):
        print(f"  expected: {rec['expected_evidence']}")
    ev = rec["evaluation"]
    if ev:
        print(f"  status: {ev['status']} | evidence rank {ev['evidence_rank']} | "
              f"evidence score {ev['evidence_score']} | best irrelevant "
              f"{ev['best_irrelevant_score']} | margin {ev['margin']} | "
              f"label ok {ev['label_correct']}"
              + (f" | distractor rank {ev['distractor_rank']}"
                 if ev.get("distractor_in_top_k") is not None else ""))
    else:
        print("  status: UNGRADED (inspect manually)")
    for h in rec["results"]:
        src = h["source"]
        tag = f"  [{h['relevance'].upper()}]" if h.get("relevance") else ""
        tag += "  [DISTRACTOR]" if h.get("distractor") else ""
        print(f"  #{h['rank']}  score {h['similarity_score']:.4f} | {h['chunk_id']}"
              f" | p{src['pages']} | {src['section']}{tag}")
        text = h["chunk_text"] if full else " ".join(h["chunk_text"].split())[:220]
        print("      " + text.replace("\n", "\n      "))


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
class RunLogger:
    CSV_COLUMNS = [
        "run_id", "timestamp", "mode", "index", "model", "top_k", "id", "category",
        "question", "status", "rank", "similarity_score", "chunk_id", "document",
        "pages", "section", "section_path", "parent_chunk_id", "relevance", "distractor",
        "chunk_text",
    ]

    def __init__(self, log_dir, run_id, enabled=True):
        self.enabled = enabled
        self.records = []
        if not enabled:
            return
        self.log_dir = Path(log_dir)
        self.run_dir = self.log_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history = self.log_dir / "retrieval_log.jsonl"

    def log(self, rec):
        self.records.append(rec)
        if not self.enabled:
            return
        with self.history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Rewrite per-run files each time so an interrupted session keeps them.
        self._write_run_files()

    def _write_run_files(self):
        (self.run_dir / "results.json").write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")
        with (self.run_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            w.writeheader()
            for rec in self.records:
                status = rec["evaluation"]["status"] if rec["evaluation"] else "UNGRADED"
                for h in rec["results"]:
                    src = h["source"]
                    w.writerow({
                        **{k: rec.get(k) for k in ("run_id", "timestamp", "mode", "index",
                                                   "model", "top_k", "id", "category", "question")},
                        "status": status,
                        "rank": h["rank"],
                        "similarity_score": h["similarity_score"],
                        "chunk_id": h["chunk_id"],
                        "document": src["document"],
                        "pages": " ".join(map(str, src["pages"])),
                        "section": src["section"],
                        "section_path": src["section_path"],
                        "parent_chunk_id": src["parent_chunk_id"] or "",
                        "relevance": h.get("relevance") or "",
                        "distractor": "yes" if h.get("distractor") else "",
                        "chunk_text": h["chunk_text"],
                    })

    def write_json(self, name, data):
        if self.enabled:
            (self.run_dir / name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def location(self):
        return self.run_dir if self.enabled else None


# --------------------------------------------------------------------------
# Comparison (--index both)
# --------------------------------------------------------------------------
def compare(records, indexes):
    by_index = {ix: [r for r in records if r["index"] == ix] for ix in indexes}
    per_query = []
    for recs in zip(*by_index.values()):
        row = {"id": recs[0]["id"], "question": recs[0]["question"]}
        for ix, rec in zip(indexes, recs):
            ev = rec["evaluation"] or {}
            row[ix] = {
                "status": ev.get("status", "UNGRADED"),
                "evidence_rank": ev.get("evidence_rank"),
                "partial_rank": ev.get("partial_rank"),
                "margin": ev.get("margin"),
                "top1_score": rec["results"][0]["similarity_score"],
                "top1_chunk": rec["results"][0]["chunk_id"],
                "top1_section": rec["results"][0]["source"]["section"],
                "label_correct": ev.get("label_correct"),
                "distractor_rank": ev.get("distractor_rank"),
                "distractor_in_top_k": ev.get("distractor_in_top_k"),
            }
        per_query.append(row)
    return {
        "summary": {ix: summarise(recs) for ix, recs in by_index.items()},
        "per_query": per_query,
    }


def print_comparison(cmp, indexes):
    a, b = indexes
    print(f"\n================ {a.upper()} vs {b.upper()} ================")
    for row in cmp["per_query"]:
        x, y = row[a], row[b]
        print(f"{row['id']}: {x['status']} -> {y['status']} | rank {x['evidence_rank']} -> "
              f"{y['evidence_rank']} | partial {x['partial_rank']} -> {y['partial_rank']} | "
              f"margin {x['margin']} -> {y['margin']} | top1 {x['top1_score']} -> "
              f"{y['top1_score']} | '{x['top1_section']}' -> '{y['top1_section']}' | "
              f"label {x['label_correct']} -> {y['label_correct']}"
              + (f" | distractor rank {x['distractor_rank']} -> {y['distractor_rank']}"
                 if x.get("distractor_in_top_k") is not None else ""))


# --------------------------------------------------------------------------
# Interactive mode
# --------------------------------------------------------------------------
HELP = """Commands:
  <text>              run a query
  :k N                set top-k
  :index parent|child switch index
  :full               toggle full chunk text
  :chunk <fragment>   show raw chunk(s) whose id contains <fragment>
  :help               this help
  :quit               exit (Ctrl+Z/Ctrl+D also works)"""


def interactive(get_retriever, index, top_k, full, logger, run_id):
    print(HELP)
    n = 0
    while True:
        try:
            # Piped input on Windows can start with a UTF-8 BOM.
            line = input(f"\n[{index} k={top_k}] query> ").replace("﻿", "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line == ":help":
            print(HELP)
        elif line.startswith(":k "):
            try:
                top_k = max(1, int(line.split()[1]))
            except ValueError:
                print("usage: :k N")
        elif line.startswith(":index "):
            choice = line.split()[1]
            if choice in INDEX_MODES:
                index = choice
            else:
                print(f"index must be one of {INDEX_MODES}")
        elif line == ":full":
            full = not full
            print(f"full text: {full}")
        elif line.startswith(":chunk "):
            matches = get_retriever(index).find_chunk(line.split(maxsplit=1)[1])
            if not matches:
                print("no chunk id contains that fragment")
            for c in matches[:5]:
                print(f"\n--- {c['chunk_id']}")
                print(json.dumps({k: v for k, v in c["metadata"].items()
                                  if k != "original_source_sha256"}, ensure_ascii=False))
                print(c["text"])
            if len(matches) > 5:
                print(f"... {len(matches) - 5} more")
        elif line.startswith(":"):
            print("unknown command; :help for the list")
        else:
            n += 1
            rec = run_case(get_retriever(index), adhoc_case(line, n), top_k, run_id, "interactive")
            print_record(rec, full)
            logger.log(rec)


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", type=Path, help="JSON list of test cases (default test_queries.json)")
    src.add_argument("-q", "--query", action="append", help="ad-hoc query (repeatable)")
    src.add_argument("-i", "--interactive", action="store_true", help="interactive prompt")
    p.add_argument("--index", choices=(*INDEX_MODES, "both"), default="parent")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--full", action="store_true", help="print full chunk text")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)
    if args.top_k < 1:
        p.error("--top-k must be >= 1")

    mode = "interactive" if args.interactive else "adhoc" if args.query else "file"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{mode}_{args.index}"
    logger = RunLogger(args.log_dir, run_id, enabled=not args.no_log)

    retrievers = {}

    def get_retriever(ix):
        if ix not in retrievers:
            retrievers[ix] = Retriever(index=ix)
        return retrievers[ix]

    if args.interactive:
        start = "parent" if args.index == "both" else args.index
        interactive(get_retriever, start, args.top_k, args.full, logger, run_id)
    else:
        if args.query:
            cases = [adhoc_case(q, n) for n, q in enumerate(args.query, start=1)]
        else:
            path = args.file or DEFAULT_CASES
            cases = load_cases(path)
            print(f"Loaded {len(cases)} cases from {path}")
        indexes = list(INDEX_MODES) if args.index == "both" else [args.index]
        for ix in indexes:
            r = get_retriever(ix)
            print(f"\n######## index={ix} | {r.describe()['num_chunks']} chunks | "
                  f"model {r.model_name} | cosine | top_k {args.top_k}")
            for case in cases:
                rec = run_case(r, case, args.top_k, run_id, mode)
                print_record(rec, args.full)
                logger.log(rec)
        for ix in indexes:
            print(f"\nsummary [{ix}]: {summarise([x for x in logger.records if x['index'] == ix])}")
        if len(indexes) == 2:
            cmp = compare(logger.records, indexes)
            print_comparison(cmp, indexes)
            logger.write_json("comparison.json", cmp)

    if logger.location() and logger.records:
        print(f"\nLogged {len(logger.records)} query runs to {logger.location()} "
              f"(+ history in {logger.history})")


if __name__ == "__main__":
    sys.exit(main())
