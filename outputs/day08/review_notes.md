# Day 08 experiment review — 25 October 2026

**Do not adopt the experimental prompt.** Both runs completed all 16 cases using local
`llama3.2` without execution errors. The one tested change was the qualified-answer
clarification recorded before the rerun in `evaluation/experiment.json`. It did not fix
q3 or q16, and introduced regressions. The dashboard retains its original prompt.

The deterministic fact count fell from **24/39 to 20/39**. Cases with no automatic flags
fell from four (q1, q5, q10, q15) to three (q1, q10, q15). These are diagnostic counts,
not accuracy or grounding scores. All 16 cases appear in both JSON/CSV outputs and the
comparison CSV. The golden dataset and evaluation rules were unchanged between runs.

## Case-by-case inspection

| Case | Baseline → after | Reviewed finding and likely stage |
|---|---|---|
| q1 | 2/2 → 2/2 | Annual-leave amount remains correct and cited to 3.1/page 2. After answer drops the full-time qualifier: a scope concern that phrase checks miss. |
| q2 | 0/2 → 0/2 | Still omits both leave deadlines despite retrieval. After answer discusses maximum consecutive days, not the lack of a special five-day approval rule. Generation/citation review; limitation check worsens. |
| q3 | 2/3 → 2/3 | Still partial-only. Both answers transfer the eight-hour premium-economy condition to business class. After adds an unsupported distinction about other classes instead of stating the missing general overseas-trip approval rule. Generation/grounding failure despite correct 6.1/page 5 citation. |
| q4 | 2/3 → 1/3 | Regression: after omits the credit-card-statement exclusion as well as explicit every-claim scope. It also conflates receipt contents with separate claim requirements. Generation/grounding. |
| q5 | 2/2 → 1/2 | After correctly locates 3.2 Sick Leave/page 2, but drops the ten-day entitlement required by this dataset. Dataset-completeness regression; the narrower user question is still answered. |
| q6 | 0/1 → 0/1 | Safe in both: standard insufficient-evidence answer, no duration, no citations. Abstention passes. The phrase-group failure is a wording mismatch, not a fabricated parental-leave rule. |
| q7 | 1/2 → 1/2 | Unchanged omission of the 90-day probation requirement. "Eligible" passes the broad condition heuristic but does not explain eligibility. Generation/grounding review. |
| q8 | 2/3 → 2/3 | Unchanged: threshold and advance notice correct, HR portal omitted. Generation completeness. |
| q9 | 1/2 → 1/2 | Tradeoff hidden by equal counts: baseline gives the after-hours hotline; after gives the one-hour IT Service Desk rule but loses the hotline. Neither supplies both necessary facts. |
| q10 | 4/4 → 4/4 | Complete MFA list in both, supported by 2.2/page 2. |
| q11 | 1/3 → 1/3 | Both reject the free-tool exception but omit catalogue/request procedure. Retrieval's 3/4 is a quotation-mark false negative: substantive evidence is present. Quote formatting also needs review. |
| q12 | 2/2 → 2/2 | Substantive regression masked by fact counts: after says USD 20,000 is between $1,000 and $10,000. Approvers are correct, explanation is wrong. Both cite the right 5.3/pages 4-5 chunk but quote the wrong/approximate table row. Citation and generation failure. |
| q13 | 1/2 → 1/2 | Correct 15-minute target in both, but no active-breach/confirmed-exposure description required by the dataset. Direct question is answered; completeness flag remains. |
| q14 | 2/3 → 0/3 | Regression: baseline gives a partial disclosure answer; after's raw answer consists of a chunk-ID reference and the existing number validator replaces it with abstention. Retrieval still misses disclosure-route evidence. Mixed retrieval/generation failure. |
| q15 | 2/2 → 2/2 | Correct budget and no-carry-over rule in both, supported by 8.2/page 6. |
| q16 | 0/3 → 0/3 | No improvement: every-claim receipt evidence is retrieved, but both runs fully abstain. They should reject the implied amount threshold and give the actual rule. The generic limitation pass is insufficient; generation and missing citation remain failures. |

## What works and what remains unresolved

Direct amount/list questions work best: annual leave, MFA coverage and learning budget.
The sick-leave section lookup and SEV-1 response target also answer the literal question,
although the dataset requires extra facts. Multi-part procedures and qualified answers
remain unreliable. All cases except q1, q10 and q15 have automatic after-run flags; q1
also warrants scope review for the omitted full-time qualifier.

Most returned citations identify the expected section/page. That does not establish
claim-level support: q3 cites the right text while combining its conditions incorrectly;
q12 selects the wrong cost-band quotation; q14's correct section label covers only a
cut-off bullet. q2 and q11 have quote-support/formatting concerns. q16 has no citation.
q6 correctly has no citation to the irrelevant leave chunks it retrieves.

No complete case was rescued by the prompt addition. q9 gained one required fact while
losing another. q4, q5 and q14 lost fact coverage; q12 gained a false numerical explanation.
The proposed change is therefore rejected, with no second prompt trial or retrieval change.

**One recommended next step:** build a small held-out set of partial-evidence/false-premise
questions and manually annotate claim-to-evidence support before testing another prompt.
This will distinguish safe qualification from plausible but unsupported answers without
optimizing repeatedly against these same 16 cases.

## Verification and limits

- `python -m unittest discover -s tests -v`: **44 tests passed**, including seven new Day 08 tests, plus the existing pipeline and HTTP tests.
- `python tests/smoke_launch_day08.py`: **passed**. The actual `python run_rag.py web` command started on an unused local port; `/` and `/api/health` returned HTTP 200, with `llama3.2` ready. The smoke-test server was stopped afterward.
- Both model runs completed 16/16 with no exceptions. Dataset SHA-256 is `e3cfe0fd0d0cf09e15cd59bec405d132127cfc4414eed1e54b7db2d804acdec6`.
- JSON retains raw generation and post-validation answers, including removed sentences, rejected citations, retrieved full text and provenance. CSV summarizes checks; it does not replace the JSON audit.
- Phrase matching can miss valid paraphrases or typography (q6/q11), accept incomplete qualifications (q7/q16), and miss added false claims (q12). Relevance is inferred from fact coverage, not independently established. Forbidden-claim matching cannot enumerate every invented claim; ambiguous denials go to review.
- Source absence cannot be proven from Top-K. Correct source labels, quotes and fact overlap do not prove entailment. These are 16 development cases, not a held-out generalization test; one deterministic-seeded run per condition does not establish statistical significance.
- The prompt addition is process-local to the experiment. Production prompt, model, retrieval and citation validation remain as before. The only pipeline edit adds document, section-path and parent-chunk metadata to debug output.

The generated automatic comparison and per-case after report follow. Their heuristic labels
are retained unchanged even where the inspection above identifies false positives/negatives.

---

