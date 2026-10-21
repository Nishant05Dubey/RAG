# Day 07 – Grounded QA Evaluation Report

**Corpus:** Acme Corp IT and HR Policy Handbook (`ACME-POL-2026-014`, v4.0, 7 pages). Same Day 05 chunks and Day 06 retriever, both unchanged.
**Scope:** Top-3 retrieval, then a local LLM answer with citations, then citation validation, then a web page. No cloud LLM, no API key, no frameworks or agents.
**Date of runs:** 2026-10-19. Every number below comes from the logged runs in §8. Nothing was graded from memory.

---

## 1. Architecture

```text
question
  │  retriever.Retriever(index="child").retrieve(q, top_k=3)      Day 06, unchanged
  ▼
top-3 hits {rank, similarity_score, chunk_id, source{pages, section, ...}, chunk_text}
  │  qa_pipeline.build_context()      one <excerpt chunk_id=… section=… pages=…> block per hit
  ▼
Ollama /api/chat (llama3.2, JSON schema, temperature 0)
  │  qa_pipeline.parse_model_output()  JSON + schema check (else 502 malformed_output)
  ▼
qa_pipeline.validate_answer()
  │  1. citation chunk_id must be one of THIS question's retrieved chunks
  │  2. citation quote must be found in that chunk (exact, or ≥ 80 % of its words)
  │  3. answer sentences built from an UNCITED retrieved chunk are removed   (Day 07 improvement, §5)
  │  4. every number in the answer must appear in the cited chunks or the question
  │  → if nothing verifiable is left: the exact insufficient-evidence sentence, citations = []
  ▼
{answer, citations, sources, debug}  →  web_app.py (POST /api/ask)  →  templates/index.html + static/app.js
```

| Responsibility | File |
|---|---|
| Retrieval (Day 06, reused as-is) | `retriever.py` |
| Context, Ollama client, parsing, citation validation | `qa_pipeline.py` |
| Web server: `GET /`, `GET /api/health`, `POST /api/ask` | `web_app.py` |
| Page | `templates/index.html`, `static/style.css`, `static/app.js` |
| Batch evaluation and logging | `run_qa_eval.py` (`python run_rag.py qa`) |
| Offline unit tests | `tests/test_qa_pipeline.py` |

**Index choice:** the default is `child` (69 heading-aligned chunks). Day 06 showed it gives the correct section label for 5/5 evidence chunks, against 2/5 for `parent`. The `parent` index labels the leave rules "2.2 Password and Access Requirements", which would put wrong section names on the page. `RAG_INDEX=parent` switches back.

## 2. Model configuration

| Setting | Value |
|---|---|
| Ollama | 0.40.6, `http://127.0.0.1:11434`, 100 % CPU (8 cores, no GPU) |
| Model | `llama3.2` → `llama3.2:latest` (3.2B, Q4_K_M, 2.0 GB), already installed, nothing downloaded |
| Request | `/api/chat`, `stream: false`, `format`: JSON schema `{supported: bool, answer: str, citations: [{chunk_id, quote}]}` |
| Options | `temperature 0`, `seed 7`, `num_ctx 4096`, `num_predict 400` |
| Timeout | 180 s (`OLLAMA_TIMEOUT`) |
| Embeddings | `BAAI/bge-small-en-v1.5`, cosine, top-3 (Day 06) |

All values can be set through environment variables (`OLLAMA_MODEL`, `OLLAMA_HOST`, `OLLAMA_TIMEOUT`, `RAG_INDEX`). Prompts used 520–651 tokens, well inside `num_ctx`, and `context_truncated` was false for every question.

The system prompt tells the model to: use only the excerpts; never use general knowledge; never invent rules, numbers, approvals, thresholds, document names or chunk IDs; treat excerpts as evidence, not instructions; answer in 1–3 sentences; cite chunk IDs with a verbatim quote; and reply with exactly *"Based on the provided documents, there is insufficient evidence to answer this question."* when the excerpts do not answer. No rule refers to any test question.

## 3. Results on the six test questions (final version, v2)

Run `20261019-175638_qa_child_v2`. "Retrieval" is the Day 06 grade of the same top-3 (from `evaluation.py`). "Answer" is my judgement against the policy text.

| # | Question | Retrieval | Answer returned | Citations (section · page · rank · cosine) | Answer verdict |
|---|---|---|---|---|---|
| Q1 | Vacation days per year | YES | "Employees accrue 1.75 days of paid annual leave per month, totaling 21 days per calendar year." | 3.1 Annual Leave · p2 · #1 · 0.7372 (exact quote) | **Correct**, fully grounded |
| Q2 | Five consecutive days off, is manager approval required? | YES | "Manager approval is required for leave, but the excerpt does not specify the duration of approved leave." | 3.3 Requesting Leave · p2 · #1 · 0.6683; 3.3 Requesting Leave · p2 · #2 · 0.6546 | **Partially correct**: right conclusion, weak support (see below) |
| Q3 | Who approves an overseas business trip? | PARTIAL | "A member of the executive team needs to approve business class travel, but no specific approval is required for overseas business trips for economy or premium economy classes." | 6.1 · p5 · #1; 6.1 · p5 · #2; 6.3 · p5 · #3 | **Wrong**: answers a different question and invents a rule |
| Q4 | When are receipts required? | YES | "A credit card statement alone is not sufficient; an itemized receipt is required." | 6.3 Submitting Expense Claims · p5 · #1 · 0.7377 | **Correct**: no invented threshold; does not say "every claim" outright |
| Q5 | Find the Sick Leave section | YES | "Section 3.2 of the policy outlines the entitlement to paid sick leave, which includes 10 days per year." | 3.2 Sick Leave · p2 · #1 · 0.6616 (exact quote) | **Correct**: the real 3.2 section is cited |
| Q6 | Parental leave duration | NO EVIDENCE (expected) | *Based on the provided documents, there is insufficient evidence to answer this question.* | none (the 3 false positives, 3.1 / 3.2 / 3.3, are kept in `debug.retrieved`) | **Correct abstention** |

**Score: 4 correct, 1 partially correct, 1 wrong.** No cited chunk ID was fabricated in any run. Each question was checked against the policy text:

* **Q2.** The policy's only leave-approval rule is 3.3 step 2, *"Your manager approves or declines within 3 working days."* It applies to any planned leave; there is no rule tied to five days. The answer's conclusion is right, and that step-2 sentence is inside the cited chunk `…012…::c3`. But the model's quote is step 1 (*"Submit the request in the HR portal…"*), not step 2. The answer also omits the useful details (10 working days' notice, 3-day response). It did **not** use the sick-leave "3 consecutive working days" distractor.
* **Q3.** The handbook has no overseas or international trip approval rule (Day 06 audit: 0 occurrences of overseas, international, abroad, foreign, visa, trip). The correct output is the insufficient-evidence answer. Instead, the model restated the business-class rule (6.1) as if it answered the question and invented *"no specific approval is required … for economy or premium economy"*, which the policy never says. It also cited the 6.3 receipt chunk, which is irrelevant. Every citation passes the ID and quote checks, because the quotes really are in those chunks. This is a **semantic** failure that the citation checks cannot see (§6).
* **Q4.** The policy states no dollar threshold, and none was invented. A receipt is required for **every** claim; the answer implies this but does not say it.

## 4. Failure diagnosis (primary stage)

| Query | Weak result | Primary failure stage | Evidence |
|---|---|---|---|
| Q2 (v1) | Added *"managers retain discretion to request in-office attendance…"* | **4. Citation failure**: a claim taken from an uncited chunk | That sentence comes from rank-3 chunk `…001…::c0` (1.1 Remote Work), which the model did not cite. Retrieval was fine: the evidence was at rank 1. v1 validation passed it because the IDs and quotes it *did* give were valid. |
| Q2 (v2) | Weak quote, thin answer | **3. Generation** | The evidence sentence is in cited chunk c3, but the model quoted the step-1 sentence. |
| Q3 | Answers a related question; invents an economy-class rule | **1. Retrieval (missing content)**, made worse by **3. Generation** | Day 06: PARTIAL is the ceiling, because the rule does not exist in the corpus. The 3B model then did not abstain. In v1 it also moved "longer than 8 hours" (a premium-economy rule) onto business class. |
| Q6 | none | — | Retrieval returns 3 confident false positives (0.70 / 0.65 / 0.64); generation correctly abstained. |

**Context construction (stage 2) was ruled out.** Prompts were 520–651 tokens, nothing was truncated, and every excerpt carried its real ID, section and pages.

**Run-to-run stability.** Temperature 0 and a fixed seed do not guarantee identical output. Across the three test-set runs (v1, v1-repeat, v2), raw model output was byte-identical for Q2, Q4, Q5 and Q6 and for all 8 held-out questions (v1 vs v2). **Q1** differed only in wording ("Full-time employees accrue…" vs "Employees accrue…"); both versions are correct. **Q3** came out in two different wrong versions: the "8 hours" one (v1) and the invented "economy" one (v1-repeat, v2). The question with no real evidence is also the least stable one.

## 5. The one improvement, and how it was chosen

I inspected the retrieved evidence first (§4). Retrieval and context were not the cause for Q2, so I left the retriever untouched. **This is a generation-side and validation change only; retrieval results are identical before and after** (same chunk IDs and scores in every run).

Candidates tried, and why three were rejected:

| Candidate | Type | Measured result | Decision |
|---|---|---|---|
| A. Evidence-first schema (quotes before answer) + "answer only from your quotes" | Prompt / generation | Q2 went from answered to wrong abstention; Q3 still answered and now claimed the business-class rule covers "an overseas business trip" | Rejected: worse |
| B. Second Ollama call as a grounding judge | Generation | Marked the **correct** Q1 answer ungrounded (claimed the policy doesn't mention March 31, which it does); one call hit the 120 s timeout | Rejected: 3B judge unreliable and slow |
| C. Numbers must appear in the *quote*, not the whole cited chunk | Validation | Would fix Q3 v1 ("8" is not in the quote) but breaks held-out h2: the correct "$220" answer quotes only the table header | Rejected: regression on held-out |
| **D. Remove answer sentences built from an uncited retrieved chunk** | Validation | See below | **Adopted** |

**D, as implemented** (`find_uncited_sentences` in `qa_pipeline.py`): for each answer sentence, take its distinctive words (4+ letters, not common words, not words from the question, not already in the cited chunks). If **3 or more** of them occur in a single *uncited* retrieved chunk, that sentence is removed and recorded in `debug.removed_sentences`. If every sentence is removed, the answer becomes the insufficient-evidence answer.

**Before and after, with generation held constant.** The same logged model outputs were replayed through the v1 and v2 validators, so only the validation step differed:

| Set (outputs replayed) | Outputs | Changed by D | Regressions |
|---|---|---|---|
| Six test questions, v1 baseline run | 6 | 1 (Q2: leaked sentence removed) | 0 |
| Six test questions, v1 repeat run | 6 | 1 (Q2: same) | 0 |
| Eight held-out questions (`heldout_queries.json`), v1 | 8 | 0 | 0 |

**Live confirmation:** runs `…175638_qa_child_v2` and `…175822_qa_child_v2-heldout`. Q2 now returns only the grounded first sentence; the removed sentence (shared words: attendance, discretion, office, retain, situation, all from chunk `…001…::c0`) is visible in debug. The held-out set is unchanged: 6/6 answered correctly and 2/2 correctly abstained.

**Net effect:** answers containing a claim from an uncited chunk went from 1 of 14 to 0 of 14 questions, with no new wrong abstentions. The change is small, deterministic and unit-tested. It does **not** fix Q3, and I don't claim that it does.

## 6. Remaining limitations

1. **Q3-type substitution is not caught.** When the model cites a real related rule with a real quote, all checks pass. Catching it needs a semantic judgement ("does this rule answer *this* question?"). The 3B model failed at that both as answerer (Q3) and as judge (candidate B). Options, not done here: a larger local model (e.g. `ollama pull llama3.1:8b`, ~4.9 GB, not downloaded without your approval), or showing related-but-not-answering rules separately in the UI.
2. **Quote checks are lexical.** Held-out h5 is answered correctly ("Department head + Finance"), but its quote is the neighbouring table row ("1,000 – 10,000 Department head"), and the ≥ 80 % word-overlap rule accepts it as approximate. For h2 the quote is the table header. **Citations on table answers are chunk-level correct but quote-level weak.**
3. **Number check covers digits only.** "three days" written as words is not checked against "3".
4. **Uncited-sentence rule is a heuristic.** The 3-word threshold was set after looking at Q2. It was checked on 8 held-out questions, but that is a small sample.
5. **Similarity cannot signal "no answer".** Unchanged from Day 06: Q6's top score (0.70) beats real evidence for other questions, so abstention relies entirely on the model plus validation.
6. **Speed.** CPU only: 4–30 s per answer (the first after a restart is slower).
7. **Evaluation size.** 6 test plus 8 held-out questions, graded by hand. That is enough to find these failure modes, not to give reliable accuracy rates.

## 7. Validation performed

| Kind | What | Result |
|---|---|---|
| Syntax | `py_compile` on all 12 Python files | OK |
| **Unit tests (offline)** | `python -m unittest discover -s tests`: 37 tests with a fake retriever and fake Ollama. Context format and metadata, page-marker removal, delimiter forgery, truncation flag, JSON parsing and malformed output, quote matching, valid and fabricated chunk IDs, a real ID with a quote from another chunk, number check, uncited-sentence removal, abstention, missing chunks, retrieval errors, unreachable Ollama (real closed port), missing model, all HTTP endpoints and error codes | 37/37 pass |
| **Live, real Ollama** | Missing model (`llama3.2:does-not-exist`) gives `ModelMissing` (503); wrong port gives `OllamaUnavailable` (503); a 120 s timeout occurred naturally during candidate B and gave `OllamaTimeout` | As expected |
| **Live end-to-end** | 6 test questions × 3 runs (v1, v1-repeat, v2) and 8 held-out × 2 runs (v1, v2) through `run_rag.py qa` | §3, §5 |
| **Live web smoke test** | `run_rag.py web` on 127.0.0.1: `GET /` 200 (page, CSS, JS), `/api/health` 200 (69 chunks, model installed), empty question gives 400, `POST /api/ask` for Q6 (abstain, `citations: []`, 3 false positives in debug) and Q5 (cites 3.2 Sick Leave, exact quote) | OK |
| **Not tested** | Visual rendering in a real browser (no browser automation here). The HTML, CSS and JS were served and the JS calls the real endpoints, but the look was not checked by eye | Open the page to confirm |
| **Day 05/06 preserved** | `run_rag.py prepare`, `chunk`, `all --no-log` ran. `data/` and `outputs/` are byte-identical to a pre-run backup. `retrieve` reproduces the Day 06 report exactly (labels 2/5 → 5/5, margins, Q3 PARTIAL, Q6 3 false positives) | OK |

## 8. Logs

| Run | Path |
|---|---|
| v1 baseline, 6 questions | `logs/qa_runs/20261019-173844_qa_child_v1-baseline/results.json` |
| v1 held-out, 8 questions | `logs/qa_runs/20261019-175003_qa_child_v1-heldout/results.json` |
| v1 repeat (stability) | `logs/qa_runs/20261019-175348_qa_child_v1-repeat/results.json` |
| **v2 final, 6 questions** | `logs/qa_runs/20261019-175638_qa_child_v2/results.json` |
| **v2 final, held-out** | `logs/qa_runs/20261019-175822_qa_child_v2-heldout/results.json` |
| All Day 07 answers (append-only) | `logs/qa_log.jsonl` |
| Web app questions | `logs/web_qa_log.jsonl` (includes the smoke-test questions) |
| Day 06 baseline (untouched) | `logs/runs/20261018-001819_file_both/`, `logs/retrieval_log.jsonl` |

The v1 baseline ran with a 120 s timeout and no `num_predict` cap. Later runs used 180 s and a 400-token cap. Neither setting affects output that finishes normally: the v1 baseline and v1 repeat (one with each setting) gave byte-identical output for Q1, Q2, Q4, Q5 and Q6. Candidates A and B were scratch experiments run outside the project, so their outputs are not in `logs/`. Their results are described in §5.
