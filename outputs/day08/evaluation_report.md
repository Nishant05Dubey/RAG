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

# Controlled experiment

{
  "variable": "one system-prompt clarification appended to the existing prompt",
  "observed_failures": [
    "Baseline q3 retrieves the business-class approval rule but answers without stating that general overseas-trip approval is unspecified, and incorrectly attaches the eight-hour premium-economy condition to business class.",
    "Baseline q16 retrieves every-claim/itemized-receipt evidence but returns blanket insufficient evidence instead of rejecting the implied dollar threshold and giving the actual rule."
  ],
  "hypothesis": "Clarifying partial support will encourage a bounded useful answer rather than an overgeneralized rule or blanket abstention. This is a development-set experiment, not independent evidence of generalization.",
  "expected_improvement": "q3 should distinguish missing general approval from the flight-class rule; q16 should explain that the amount is unspecified and cite the every-claim receipt rule. q6 should continue abstaining without a duration or unrelated citations.",
  "prompt_addition": "Clarification of rules 4 and 5: When a question contains an unsupported premise or asks for a broader rule than the excerpts establish, but the excerpts contain a relevant narrower rule, set supported to true and give a qualified answer: explicitly state what the provided documents do not specify, then give the narrower supported rule with its own conditions and citation. Keep each condition attached to the rule it actually qualifies. Do not replace this qualified answer with blanket abstention; use the insufficient-evidence response only when no relevant rule can be given.",
  "controls": "Identical dataset, source documents, chunks, embedding model, local llama3.2, Top-K, index, temperature, seed, token limits, citation validation and deterministic evaluator. Only this prompt addition changes; dashboard default remains unchanged."
}

Only the system-prompt addition differs. The UI default is preserved pending review.

| Case | Changed checks | Before stage | After stage |
|---|---|---|---|
| q1_annual_leave_days | unchanged | none | none |
| q2_five_days_manager_approval | limitation | mixed | mixed |
| q3_overseas_trip_approval | unchanged | generation_or_grounding | generation_or_grounding |
| q4_expense_receipts | facts | generation_or_grounding | generation_or_grounding |
| q5_find_sick_leave_section | facts, failure_stage | none | generation_or_grounding |
| q6_parental_leave_unsupported | unchanged | generation_or_grounding | generation_or_grounding |
| q7_remote_work_days | unchanged | generation_or_grounding | generation_or_grounding |
| q8_remote_work_from_another_country | unchanged | generation_or_grounding | generation_or_grounding |
| q9_lost_or_stolen_device | unchanged | generation_or_grounding | generation_or_grounding |
| q10_mfa_coverage | unchanged | none | none |
| q11_unapproved_freemium_tool | unchanged | mixed | mixed |
| q12_software_license_approval_20000 | unchanged | citation | citation |
| q13_sev1_response_target | unchanged | generation_or_grounding | generation_or_grounding |
| q14_disclose_gift_150 | facts, citation | mixed | mixed |
| q15_learning_budget | unchanged | none | none |
| q16_receipt_minimum_dollar_threshold | unchanged | mixed | mixed |

# Day 08 after

Cases: 16/16. Model: llama3.2. Full gold evidence does not imply full answerability (especially q3).

| Case | Evidence | Facts | Citations | Abstention | Limitation | Stage | Review |
|---|---|---|---|---|---|---|---|
| q1_annual_leave_days | full | 2/2 | supported_heuristic | not_required | not_required | none | False |
| q2_five_days_manager_approval | full | 0/2 | manual_review | not_required | missing | mixed | True |
| q3_overseas_trip_approval | full | 2/3 | supported_heuristic | not_required | missing | generation_or_grounding | True |
| q4_expense_receipts | full | 1/3 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q5_find_sick_leave_section | full | 1/2 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q6_parental_leave_unsupported | not_expected | 0/1 | not_required | pass | pass_heuristic | generation_or_grounding | True |
| q7_remote_work_days | full | 1/2 | supported_heuristic | not_required | pass_heuristic | generation_or_grounding | True |
| q8_remote_work_from_another_country | full | 2/3 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q9_lost_or_stolen_device | full | 1/2 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q10_mfa_coverage | full | 4/4 | supported_heuristic | not_required | not_required | none | False |
| q11_unapproved_freemium_tool | partial | 1/3 | manual_review | not_required | not_required | mixed | True |
| q12_software_license_approval_20000 | full | 2/2 | manual_review | not_required | not_required | citation | True |
| q13_sev1_response_target | full | 1/2 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q14_disclose_gift_150 | not_found | 0/3 | missing | not_required | not_required | mixed | True |
| q15_learning_budget | full | 2/2 | supported_heuristic | not_required | not_required | none | False |
| q16_receipt_minimum_dollar_threshold | full | 0/3 | missing | not_required | pass_heuristic | mixed | True |

## q1_annual_leave_days

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: Employees accrue 1.75 days of paid annual leave per month, totaling 21 days per calendar year.

## q2_five_days_manager_approval

Required wording groups missing; inspect paraphrases. Citation support needs inspection: manual_review Missing limitation or asserted forbidden claim.

Answer: Manager approval is required for leave requests, but the policy does not specify a maximum number of consecutive days that can be taken.

## q3_overseas_trip_approval

Required wording groups missing; inspect paraphrases. Missing limitation or asserted forbidden claim.

Answer: A member of the executive team needs to approve business class travel for flights longer than 8 hours, but for other classes of travel, no approval is specified.

## q4_expense_receipts

Required wording groups missing; inspect paraphrases.

Answer: Receipts are required for travel expenses when they are submitted as part of an expense claim, specifically an itemized receipt that includes the business purpose and, for meals, the names of attendees.

## q5_find_sick_leave_section

Required wording groups missing; inspect paraphrases.

Answer: The policy section about Sick Leave is 3.2 Sick Leave.

## q6_parental_leave_unsupported

Required wording groups missing; inspect paraphrases.

Answer: Based on the provided documents, there is insufficient evidence to answer this question.

## q7_remote_work_days

Required wording groups missing; inspect paraphrases.

Answer: An eligible employee can work remotely for up to three days per week.

## q8_remote_work_from_another_country

Required wording groups missing; inspect paraphrases.

Answer: Working remotely from another country for more than 14 consecutive calendar days requires an exception request, which must be submitted at least 30 days in advance.

## q9_lost_or_stolen_device

Required wording groups missing; inspect paraphrases.

Answer: Report any lost or stolen device to the IT Service Desk within 1 hour of noticing.

## q10_mfa_coverage

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: The systems that require multi-factor authentication are email, VPN, the HR portal, and all production systems.

## q11_unapproved_freemium_tool

Not all gold evidence groups retrieved. Required wording groups missing; inspect paraphrases. Citation support needs inspection: manual_review

Answer: Based on the provided documents, free trials and 'freemium' tools are not exempt from this process if they will process any company data.

## q12_software_license_approval_20000

Citation support needs inspection: manual_review

Answer: The annual software license must be approved by the Department head + Finance, since the cost is between $1,000 and $10,000.

## q13_sev1_response_target

Required wording groups missing; inspect paraphrases.

Answer: The response target for a SEV-1 security incident is 15 minutes.

## q14_disclose_gift_150

Not all gold evidence groups retrieved. Required wording groups missing; inspect paraphrases. Citation support needs inspection: missing

Answer: Based on the provided documents, there is insufficient evidence to answer this question.

## q15_learning_budget

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: The annual learning budget for a full-time employee is USD 1,200. Unused budget does not carry over.

## q16_receipt_minimum_dollar_threshold

Required wording groups missing; inspect paraphrases. Citation support needs inspection: missing

Answer: Based on the provided documents, there is insufficient evidence to answer this question.

## Evaluator limits

Phrase checks miss valid paraphrases and can accept negated or unrelated facts. Citation labels, exact quotes and gold evidence overlap are heuristics, not entailment. Absence from Top-K does not prove absence from the handbook. Limitations, negation, partial answers and flagged cases require human review. Complete raw model output, validated UI answer and retrieved text are retained in JSON.

Final artifact verification: all 16 retrieved rankings, similarity scores, full texts and provenance fields match exactly across baseline and after; comparison CSV has 16 rows. Git diff whitespace check passed.

