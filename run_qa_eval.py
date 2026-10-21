"""Day 07: run test questions through the full RAG pipeline and log the answers.

    python run_qa_eval.py                       # the six cases in test_queries.json
    python run_qa_eval.py -q "your question"    # ad-hoc question(s)
    python run_qa_eval.py --index parent        # Day 05 chunks instead of children
    python run_qa_eval.py --tag v2              # label the run (before/after)

For every question this records the answer, citations and sources, plus the
Day 06 retrieval grade of the same Top-3 (evaluation.py), so a weak answer can
be traced to retrieval or to generation. The answers themselves are not
auto-graded: read them against the policy (see the Day 07 report).

Logs (default ./logs):
    qa_log.jsonl                  append-only history, one line per question
    qa_runs/<run_id>/results.json every record of this run
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from evaluation import evaluate
from paths import LOGS_DIR, TEST_QUERIES
from qa_pipeline import INSUFFICIENT, OllamaClient, QAError, QAPipeline, RAG_INDEX
from run_queries import adhoc_case, load_cases


def checks(case, result, hits_eval):
    """Generic, field-driven checks (no question wording is used)."""
    texts = {r["chunk_id"]: r["text"] for r in result["debug"]["retrieved"]}
    cited = [texts[c] for c in result["citations"]]
    out = {"abstained": result["answer"] == INSUFFICIENT,
           "retrieval_status": hits_eval["status"] if hits_eval else None}
    if case.get("unsupported"):
        out["expected_abstain"] = True
    phrases = (case.get("evidence_phrases") or []) + (case.get("partial_phrases") or [])
    if phrases:
        out["cites_evidence_chunk"] = any(p in t for p in phrases for t in cited)
    if case.get("distractor_phrases"):
        out["cites_distractor_chunk"] = any(p in t for p in case["distractor_phrases"] for t in cited)
    return out


def print_result(case, rec):
    print(f"\n[{case['id']}] {case['question']}")
    if "error" in rec:
        print(f"  ERROR {rec['error']['code']}: {rec['error']['message']}")
        return
    r, d = rec["result"], rec["result"]["debug"]
    print(f"  status: {d['status']}" + (f" ({d['reason']})" if d["reason"] else "")
          + f" | retrieval: {rec['checks']['retrieval_status']}"
          + f" | {d['timings_ms']['total']} ms")
    print(f"  answer: {r['answer']}")
    for s in r["sources"]:
        print(f"  cite #{s['rank']} {s['chunk_id']} | p{s['pages']} | {s['section']} "
              f"| quote {s['quote_match']}: \"{s['quote'][:90]}\"")
    for c in d["rejected_citations"]:
        print(f"  REJECTED {c['chunk_id']}: {c['reason']}")
    if d["unsupported_numbers"]:
        print(f"  unsupported numbers: {d['unsupported_numbers']}")
    if d["model_answer"] != r["answer"]:
        print(f"  (model said: {d['model_answer']})")
    for h in d["retrieved"]:
        print(f"    retrieved #{h['rank']} {h['similarity_score']:.4f} {h['chunk_id']} | {h['section']}"
              + ("  [cited]" if h.get("cited") else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", type=Path, help="JSON list of test cases (default test_queries.json)")
    src.add_argument("-q", "--query", action="append", help="ad-hoc question (repeatable)")
    p.add_argument("--index", choices=("parent", "child"), default=RAG_INDEX)
    p.add_argument("--model", help="Ollama model name (default OLLAMA_MODEL or llama3.2)")
    p.add_argument("--tag", default="", help="label stored with the run, e.g. baseline / v2")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)

    cases = ([adhoc_case(q, n) for n, q in enumerate(args.query, start=1)]
             if args.query else load_cases(args.file or TEST_QUERIES))
    client = OllamaClient(model=args.model) if args.model else OllamaClient()
    pipeline = QAPipeline(index=args.index, client=client)
    health = pipeline.health()
    if not health["ollama"]["ok"]:
        raise SystemExit(health["ollama"].get("message", "Ollama is not ready."))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_qa_{args.index}" + (f"_{args.tag}" if args.tag else "")
    print(f"Run {run_id} | model {client.model} | index {args.index} | {len(cases)} questions")
    records = []
    for case in cases:
        rec = {"run_id": run_id, "tag": args.tag,
               "timestamp": datetime.now().isoformat(timespec="seconds"),
               "model": client.model, "index": args.index,
               **{k: case[k] for k in ("id", "category", "question", "expected_evidence") if k in case}}
        try:
            result = pipeline.ask(case["question"])
            r = pipeline.retriever
            ranking = r.rank_all(case["question"])
            hits = [r.format_hit(k, pos, s) for k, (pos, s) in enumerate(ranking[:pipeline.top_k], 1)]
            rec.update(result=result, checks=checks(case, result, evaluate(case, hits, ranking, r)))
        except QAError as e:
            rec["error"] = {"code": e.code, "message": str(e), "debug": e.debug}
        print_result(case, rec)
        records.append(rec)

    if not args.no_log:
        run_dir = LOGS_DIR / "qa_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with (LOGS_DIR / "qa_log.jsonl").open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nLogged {len(records)} answers to {run_dir} (+ history in {LOGS_DIR / 'qa_log.jsonl'})")
    return 1 if any("error" in r for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
