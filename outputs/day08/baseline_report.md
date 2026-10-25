# Day 08 baseline

Cases: 16/16. Model: llama3.2. Full gold evidence does not imply full answerability (especially q3).

| Case | Evidence | Facts | Citations | Abstention | Limitation | Stage | Review |
|---|---|---|---|---|---|---|---|
| q1_annual_leave_days | full | 2/2 | supported_heuristic | not_required | not_required | none | False |
| q2_five_days_manager_approval | full | 0/2 | manual_review | not_required | pass_heuristic | mixed | True |
| q3_overseas_trip_approval | full | 2/3 | supported_heuristic | not_required | missing | generation_or_grounding | True |
| q4_expense_receipts | full | 2/3 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q5_find_sick_leave_section | full | 2/2 | supported_heuristic | not_required | not_required | none | False |
| q6_parental_leave_unsupported | not_expected | 0/1 | not_required | pass | pass_heuristic | generation_or_grounding | True |
| q7_remote_work_days | full | 1/2 | supported_heuristic | not_required | pass_heuristic | generation_or_grounding | True |
| q8_remote_work_from_another_country | full | 2/3 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q9_lost_or_stolen_device | full | 1/2 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q10_mfa_coverage | full | 4/4 | supported_heuristic | not_required | not_required | none | False |
| q11_unapproved_freemium_tool | partial | 1/3 | manual_review | not_required | not_required | mixed | True |
| q12_software_license_approval_20000 | full | 2/2 | manual_review | not_required | not_required | citation | True |
| q13_sev1_response_target | full | 1/2 | supported_heuristic | not_required | not_required | generation_or_grounding | True |
| q14_disclose_gift_150 | not_found | 2/3 | manual_review | not_required | not_required | mixed | True |
| q15_learning_budget | full | 2/2 | supported_heuristic | not_required | not_required | none | False |
| q16_receipt_minimum_dollar_threshold | full | 0/3 | missing | not_required | pass_heuristic | mixed | True |

## q1_annual_leave_days

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: Full-time employees accrue 1.75 days of paid annual leave per month, totaling 21 days per calendar year.

## q2_five_days_manager_approval

Required wording groups missing; inspect paraphrases. Citation support needs inspection: manual_review

Answer: Manager approval is required for leave, but the excerpt does not specify the duration of approved leave.

## q3_overseas_trip_approval

Required wording groups missing; inspect paraphrases. Missing limitation or asserted forbidden claim.

Answer: A member of the executive team needs to approve business class travel for flights longer than 8 hours.

## q4_expense_receipts

Required wording groups missing; inspect paraphrases.

Answer: A credit card statement alone is not sufficient; an itemized receipt is required.

## q5_find_sick_leave_section

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: Section 3.2 of the policy outlines the entitlement to paid sick leave, which includes 10 days per year.

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

Answer: Call the 24/7 security hotline at extension 4357.

## q10_mfa_coverage

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: The systems that require multi-factor authentication are email, VPN, the HR portal, and all production systems.

## q11_unapproved_freemium_tool

Not all gold evidence groups retrieved. Required wording groups missing; inspect paraphrases. Citation support needs inspection: manual_review

Answer: No, free trials and freemium tools are not exempt from this process if they will process any company data.

## q12_software_license_approval_20000

Citation support needs inspection: manual_review

Answer: Department head + Finance

## q13_sev1_response_target

Required wording groups missing; inspect paraphrases.

Answer: The response target for a SEV-1 security incident is 15 minutes.

## q14_disclose_gift_150

Not all gold evidence groups retrieved. Required wording groups missing; inspect paraphrases. Citation support needs inspection: manual_review

Answer: Based on the provided documents, accepting gifts or hospitality worth more than USD 100 from a current or prospective business is a conflict of interest. The gift offered is worth USD 150, so you need to disclose it.

## q15_learning_budget

Automatic checks satisfied; phrase matching is not proof of groundedness.

Answer: The annual learning budget for a full-time employee is USD 1,200. Unused budget does not carry over.

## q16_receipt_minimum_dollar_threshold

Required wording groups missing; inspect paraphrases. Citation support needs inspection: missing

Answer: Based on the provided documents, there is insufficient evidence to answer this question.

## Evaluator limits

Phrase checks miss valid paraphrases and can accept negated or unrelated facts. Citation labels, exact quotes and gold evidence overlap are heuristics, not entailment. Absence from Top-K does not prove absence from the handbook. Limitations, negation, partial answers and flagged cases require human review. Complete raw model output, validated UI answer and retrieved text are retained in JSON.

## Baseline inspection before selecting the experiment

All 16 cases completed through the actual local model, without execution errors.
The dataset was not edited. SHA-256: `e3cfe0fd0d0cf09e15cd59bec405d132127cfc4414eed1e54b7db2d804acdec6`.

- q1, q5, q10 and q15 satisfy the automatic checks: direct leave, MFA-list and budget lookups work well.
- q2 retrieves the requesting-leave deadlines but omits them. Its surviving answer and citation need review; the generic limitation detector passes wording that does not adequately explain the five-day premise.
- q3 is **partial_only**, despite full coverage of its one gold evidence group. It fails to state that general overseas-trip approval is absent and transfers an eight-hour premium-economy condition onto business class. The right section/page and an exact quote do not prove the answer is supported.
- q4 omits the explicit every-claim scope. q7 omits the 90-day probation condition; saying "eligible" passes the broad limitation heuristic but does not supply that condition.
- q6 safely abstains, gives no duration and cites nothing. Its phrase-group failure is a wording mismatch, not an unsafe parental-leave answer.
- q8 omits the HR portal; q9 omits reporting within one hour to the IT Service Desk; q13 gives the correct 15-minute target but omits the incident description required by this dataset.
- q11's 3/4 retrieval score is a quotation-mark mismatch, not missing substantive evidence. The answer correctly rejects the free-tool exception but omits the catalogue and software-request procedure. Its quote formatting also triggers strict citation review.
- q12 gives the correct approvers and cites the correct section/pages (5.3, pages 4-5), but the approximate quote selects the wrong cost band. Inspect the table row, not just the citation label.
- q14 has a real retrieval gap: Top-K contains a cut-off conflicts-of-interest bullet and two hardware-budget chunks, not the disclosure form/HR portal instructions. Its expected section/page is right, but support is incomplete.
- q16 retrieves the every-claim receipt rule and then abstains entirely. It should instead explain the absent dollar threshold and give the actual rule.

The selected single variable is recorded in `evaluation/experiment.json`: one prompt clarification about qualified answers when evidence is partial or a premise is unsupported. No retrieval, model, sampling or citation-validator settings are changed. This is a transparent development-set intervention, not a held-out validation result.
