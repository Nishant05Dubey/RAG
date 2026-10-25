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
