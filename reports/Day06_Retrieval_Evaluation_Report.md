# Day 06 – Retrieval Evaluation Report

**Corpus:** Acme Corp IT and HR Policy Handbook (`ACME-POL-2026-014`, v4.0, 7 pages), using the Day 05 chunks
**Scope:** Retrieval only. No LLM generation, no agents, no MCP tools.
**Status:** Baseline built and evaluated on all 6 test cases. One experiment (Option C, child–parent chunks) was run and compared. Audited against the source PDF on 2026-10-18 (see §8).
**Runs:** all numbers come from `logs/runs/20261018-001819_file_both` (latest run, after the audit). It reproduces `20261018-000702_file_both` and the original `20261015-140746_file_both` exactly, with zero differences in any score, rank, status, chunk ID, margin or label (checked programmatically).

---

## 0. Outcome classification (latest run)

| Class | Definition (from `evaluation.py` status) |
|---|---|
| **Successful retrieval** | Answerable query; a chunk with the full evidence is in the top-3 (`YES`). |
| **Partial retrieval** | Only partial or related evidence is in the top-3 (`PARTIAL`). |
| **Retrieval failure** | Answerable query; no evidence in the top-3 (`NO`). |
| **Negative control** | No evidence exists in the source. Every hit is a false positive, however high its score (`NO EVIDENCE (expected)`). |

Qualifiers such as a wrong citation label, a distractor in the top-3, or a thin margin are recorded alongside the class. They do not change it.

| Query | Baseline (47 Day 05 chunks) | Experiment (69 heading-aligned children) |
|---|---|---|
| Q1 vacation days | **Successful**: rank 1, 0.6811; wrong label ("2.2 Password…") | **Successful**: rank 1, 0.7372; label correct |
| Q2 leave approval | **Successful**: rank 1, 0.6575; wrong label; the sick-leave distractor is *in the same rank-1 chunk* | **Successful**: rank 1, 0.6683; label correct; distractor down to rank 13 |
| Q3 overseas trip approval | **Partial**: related rule at rank 2; wrong-domain software-approval table at rank 1 (0.7147). Not answerable from the corpus. | **Partial**: related rule at rank 2; rank 1 is the 6.1 intro (right section, no approval rule); distractor down to rank 4. Not answerable from the corpus. |
| Q4 receipts | **Successful**: rank 1, 0.7377; margin 0.0603 | **Successful**: rank 1, 0.7377; margin only 0.0168 |
| Q5 sick-leave section | **Successful**: rank 1, 0.6141; wrong label; margin 0.0214 | **Successful**: rank 1, 0.6616; label correct; margin 0.0997 |
| Q6 parental leave | **Negative control**: 3 false positives, top-1 0.6877 | **Negative control**: 3 false positives, top-1 0.7002 |

**Totals per index:** 4 successful, 1 partial, **0 retrieval failures**, 1 negative control returning 3/3 false positives. Earlier versions of this report labelled Q3 and Q6 "FAIL". That conflated two things. Q3 is a *partial* retrieval capped by missing content. Q6 is a negative control where the retriever cannot abstain. Neither is a ranking miss on an answerable question.

---

## 1. Executive Summary & Setup

| Item | Value |
|---|---|
| Chunks | Day 05 **semantic (TF-IDF topic-shift)** chunks, `data/chunks/metadata_semantic.json` (originally `day05-chunking/outputs/`; 47 chunks, 85–881 chars). Chosen because Day 05 recommended it (0 mid-word cuts, 0 broken tables). |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, 512-token limit, so the longest chunk of 881 chars is never truncated) via `sentence-transformers` 6.1.0 on CPU |
| Query encoding | Same model. Queries get BGE's documented retrieval instruction (`"Represent this sentence for searching relevant passages: "`); passages are embedded as-is. |
| Distance metric | Cosine similarity (L2-normalised vectors, dot product) |
| Vector store | In-memory NumPy matrix (47 × 384 baseline, 69 × 384 experiment). No external DB. |
| Top-k | 3 |
| Grading | A chunk is **relevant** if its text contains the evidence phrase taken from the handbook (defined per case in `test_queries.json`, applied by `evaluation.py`). Ranks are measured over the **whole** index, not only the top-3. |

**Headline findings**

1. **Ranking looks good, separation does not.** The baseline puts full evidence at rank 1 for all 4 answerable queries. But the gap between that evidence and the best irrelevant chunk is only **0.021–0.060** cosine.
2. **Similarity is not confidence.** The unsupported Q6 (parental leave) scores **0.688** at top-1. That is *higher* than the true evidence for Q2 (0.658) and Q5 (0.614). The wrong-domain top-1 for Q3 scores **0.715**, higher than the true evidence for Q1, Q2 and Q5. No single cosine threshold separates "answerable" from "not answerable" in this corpus.
3. **Root cause: chunks that span several sections.** Day 05 chunk 12 holds the end of 2.2 plus all of 3.1 Annual Leave, 3.2 Sick Leave and 3.3 Requesting Leave. Its metadata labels it "2.2 Password and Access Requirements". Three different queries (Q1, Q2, Q5) all "hit" that one diluted vector, with the wrong citation.
4. **Experiment (Option C, heading-aligned child chunks with parent links):** citation labels went from **2/5 to 5/5 correct**. The Q5 margin rose **4.7×** (0.021 → 0.100) and Q2's improved (0.028 → 0.039). Q3's top-1 moved from a software-licence table to the travel section. **Regressions:** the Q1 and Q4 margins shrank. Q6 is **not fixed**: it still returns 3 confident false positives, as predicted.

### Ground-truth corrections

The brief assumed some evidence that this handbook does not contain. I graded against the document as written. Each claim below was checked against the **PDF itself**, using two independent `pypdf` extraction modes (plain and layout; both gave the same counts), not only Day 05's cleaned text. The PDF is text-only: 0 images, 0 annotations, 0 attachments and 0 form fields on all 7 pages, so nothing sits outside the extracted text.

| Query | Brief assumed | What the handbook actually says (verbatim) |
|---|---|---|
| Q2 | A special approval rule for leave of "more than three consecutive days" | **No such rule.** No rule ties approval to leave length. **3.3 Requesting Leave** is the only leave-approval text: *"1. Submit the request in the HR portal at least 10 working days in advance for planned leave. 2. Your manager approves or declines within 3 working days. 3. Update your Slack status and calendar once approved."* Step 2 is a single sentence that both makes the manager the decision-maker *and* sets a 3-working-day response time. It is therefore the correct evidence for "is manager approval required?", even though it never uses the word "required" and is stated for *planned* leave. "consecutive" occurs only twice: the sick-leave rule *"A medical certificate is required for absences longer than 3 consecutive working days"* (a numeric distractor about certificates, not approval) and the 1.1 remote-work rule (14 consecutive calendar days abroad). |
| Q3 | International travel approval tiers | **No overseas or international trip approval rule exists.** "overseas", "international", "abroad", "foreign", "visa" and "trip" each occur **0 times**. The only approval rule in travel (§6) is **6.1**: *"business class requires prior written approval from a member of the executive team."* It governs cabin class, not whether a trip is approved. The only cross-border rule is **1.1**: remote work *"from a country other than their country of employment for longer than 14 consecutive calendar days"* needs a Remote Work Exception Request. That is remote work, not business travel. Best achievable grade: **PARTIAL**. |
| Q4 | A receipt dollar threshold | **No threshold.** The PDF has no "$" amounts, and "receipt" occurs once. **6.3**: *"Expense claims must be submitted within 30 days of the expense being incurred. Every claim requires: An itemized receipt (a credit card statement alone is not sufficient). The business purpose and, for meals, the names of attendees. The correct cost center code."* A receipt is required for **every** claim. The only USD thresholds in the document are elsewhere (gifts > USD 100, learning budget USD 1,200, certification > USD 3,000, the 5.3 licence table, the 6.2 allowance table). |
| Q6 | No evidence | Confirmed: "parental", "maternity", "paternity", "family leave", "adoption" and "birth" each occur 0 times. |

Test-case changes from this audit are listed in §8. Only the descriptive `expected_evidence` and `notes` fields and a new reporting-only `distractor_phrases` field changed; **no grading phrase changed**.

---

## 2. Baseline Retrieval Results

Margin = cosine of the best fully relevant chunk − cosine of the best irrelevant chunk (whole index).

| # | Query | Evidence in Top-3? | Evidence rank & cosine | Top-1 source (chunk · page · Day 05 section label) | Margin | Class (§0) and qualifiers |
|---|---|---|---|---|---|---|
| Q1 | How many vacation days do employees get each year? | **YES** | 1 · 0.6811 | 012 · p2 · "2.2 Password and Access Requirements" ⚠ wrong label | 0.0527 | **Successful**; wrong citation |
| Q2 | Five consecutive days off — is manager approval required? | **YES** | 1 · 0.6575 | 012 · p2 · "2.2 Password and Access Requirements" ⚠ | 0.0280 | **Successful**; wrong citation; distractor in the same chunk (rank 1) |
| Q3 | Who needs to approve an overseas business trip? | **PARTIAL** | partial at rank 2 · 0.6449 | 028 · p4–5 · "5.3 Software License Costs" ✗ wrong domain, 0.7147 | n/a | **Partial**; distractor at rank 1; no full evidence exists in the corpus |
| Q4 | When are receipts required for travel expenses? | **YES** | 1 · 0.7377 | 032 · p5 · "6.3 Submitting Expense Claims" | 0.0603 | **Successful** |
| Q5 | Find the policy section about Sick Leave. | **YES** | 1 · 0.6141 | 012 · p2 · "2.2 Password and Access Requirements" ⚠ | 0.0214 | **Successful**; wrong citation; thinnest margin |
| Q6 | What is the company's parental leave duration? | **NO** (expected) | none exists | 012 · p2 · leave chunk, **0.6877** false positive | n/a | **Negative control**: 3/3 false positives |

Evidence labels correct: **2 of 5** (Q3 partial, Q4). The other three cite "2.2 Password and Access Requirements" for leave rules.

---

## 3. Granular Trace (Baseline, Top-3)

Extracts are the first ~250 characters of each chunk with whitespace collapsed. Full text is in `logs/runs/20261015-140746_file_both/results.json`.

### Q1 — Semantic paraphrase: "How many vacation days do employees get each year?"

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | 0.6811 | 012 · p2 · 2.2 Password… | "Do not store any customer data on the device until encryption is confirmed. ## 3. Leave and Time Off ### 3.1 Annual Leave Full-time employees accrue 1.75 days of paid annual leave per month, **totaling 21 days per calendar year**…" | FULL |
| 2 | 0.6284 | 031 · p5 · 6.2 Daily Allowances | "Partial days (departure and return) are paid at 75% of the daily rate. Hotel Cap Region Meals (USD/day)…" | no |
| 3 | 0.6278 | 030 · p5 · 6.2 Daily Allowances | "### 6.2 Daily Allowances Meal and incidental allowances are paid per full travel day." | no |

**Commentary:** The dense model bridged the paraphrase ("vacation" occurs 0 times; the document says "annual leave"), which is a genuine strength. Ranks 2–3 drift towards "per day" allowance tables (day/rate vocabulary). The winning chunk is the 4-section chunk 12, and its label points the user to the password policy.

### Q2 — Approval rule: "I want to take five consecutive days off. Is manager approval required?"

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | 0.6575 | 012 · p2 · 2.2 Password… | Same chunk 12. It contains *"A medical certificate is required for absences longer than 3 consecutive working days"* **and** *"2. Your manager approves or declines within 3 working days."* | FULL |
| 2 | 0.6296 | 001 · p1 · 1.1 Eligibility and Philosophy | "## 1. Remote Work Policy … employees … eligible to work remotely for up to three days per week … managers retain the discretion…" | no |
| 3 | 0.6252 | 026 · p4 · "This line is part of a quote, not a heading." | "Disconnect from the network and wait for instructions … ## 5. Software and SaaS Procurement ### 5.1 Approved Software … without additional approval…" | no |

**Commentary:** Rank 1 is correct only because everything about leave sits in one chunk. That chunk also carries the sick-leave "3 consecutive working days" rule, so the numeric trap from the brief is present *inside the retrieved context*. The run now records this as `distractor_rank 1`, `distractor_in_top_k true`.

*Audit note:* the evidence sentence "Your manager approves or declines within 3 working days" carries both the approval rule (the manager decides) and a response-time rule (within 3 working days). It is not a different rule from the one asked about. It is the only approval statement for leave, and it applies to planned leave of any length. The 220-character console snippet shows only the start of chunk 12 (the end of 2.2 and 3.1). The evidence sentence is the last line of the chunk; see `chunk_text` in `results.json`. Rank 2 drifts on "days per week" and "managers". Rank 3 drifts on "approval", and its section label is a Day 05 heading-detection bug. Step 3 of 3.3 ("…once approved") sits in another chunk and ranks **11th**.

### Q3 — Travel approval: "Who needs to approve an overseas business trip?"

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | **0.7147** | 028 · p4–5 · 5.3 Software License Costs | "The approval thresholds are: Annual Cost (USD) Approver Under 1,000 Team manager … 1,000 – 10,000 Department head … Over 50,000 CFO and CTO (joint approval) ## 6. Travel and Expenses ### 6.1 Booking Business Travel All flights, hotels, and rail travel must be booked through the corporate travel platform…" | no (software approvals) |
| 2 | 0.6449 | 029 · p5 · 6.1 Booking Business Travel | "Economy class is the default for all flights; premium economy is permitted for flights longer than 8 hours, and **business class requires prior written approval from a member of the executive team**." | PARTIAL |
| 3 | 0.6297 | 032 · p5 · 6.3 Submitting Expense Claims | "Every claim requires: - An itemized receipt…" | no |

**Commentary:** This is the highest top-1 score of any query, and it is wrong. A downstream LLM would very likely answer "Department head + Finance / CFO and CTO", which is a fabricated travel-approval hierarchy.

### Q4 — Expense detail: "When are receipts required for travel expenses?"

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | 0.7377 | 032 · p5 · 6.3 Submitting Expense Claims | "Every claim requires: - **An itemized receipt (a credit card statement alone is not sufficient).** - The business purpose and, for meals, the names of attendees." | FULL |
| 2 | 0.7144 | 031 · p5 · 6.2 Daily Allowances | "…Major hubs (NYC, London, SF) 90 20 350 ### 6.3 Submitting Expense Claims Expense claims must be submitted within 30 days…" | PARTIAL (6.3 content labelled 6.2) |
| 3 | 0.6774 | 030 · p5 · 6.2 Daily Allowances | "Meal and incidental allowances are paid per full travel day." | no |

**Commentary:** The cleanest result. Note that the 6.3 rule is split across chunks 031/032/033: the "within 30 days" deadline sits under a 6.2 label, and the "90 days late" rule is in chunk 033. Chunk 032 begins mid-section ("Every claim requires:"), so the receipt rule is retrieved without its section header.

*Audit note:* the "Every claim requires:" list is itself split. Chunk 032 holds the receipt and business-purpose items, and its third item, *"The correct cost center code."*, opens chunk 033. Chunk 033 ranks 14th (baseline) and 7th (child). The grade is still correct, because the receipt rule is complete in 032. The PDF states no dollar threshold: a receipt is required for every claim.

### Q5 — Exact terminology: "Find the policy section about Sick Leave."

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | 0.6141 | 012 · p2 · 2.2 Password… | Chunk 12, containing "### 3.2 Sick Leave Employees are entitled to 10 days of paid sick leave per year…" | FULL |
| 2 | 0.5927 | 046 · p7 · 10. Glossary | "SEV – Severity level … ## 11. Contacts For questions about this policy, contact the People & Technology Office…" | no |
| 3 | 0.5789 | 000 · p1 · (none) | "# Acme Corp IT and HR Policy Handbook Document ID: ACME-POL-2026-014 … This handbook applies to all full-time employees…" | no |

**Commentary:** The brief asked whether dense search drifts into general health/medical text. This corpus has no such text. Instead, the drift goes to **navigational boilerplate** ("this policy", "contact", the handbook preamble), because the query's "find the policy section" framing outweighs the two-word topic. The exact header is found, but with the smallest margin of any query (0.021) and a citation that says "2.2 Password".

### Q6 — Unsupported: "What is the company's parental leave duration?"

| Rank | Cos | Chunk · page · label | Extract | Relevant? |
|---|---|---|---|---|
| 1 | **0.6877** | 012 · p2 · 2.2 Password… | Annual / sick / requesting leave (chunk 12) | **False positive** |
| 2 | 0.6341 | 001 · p1 · 1.1 Eligibility and Philosophy | Remote-work eligibility ("90-day probation", "days per week") | **False positive** |
| 3 | 0.6125 | 043 · p7 · 9.1 New Joiner Checklist | "### 9.2 Leaving Acme Corp Employees who resign must give the notice period…" | **False positive** |

**Commentary:** Every hit is a false positive, and the top one outscores the true evidence for Q2 and Q5. Rank 3 is surface-form drift: "leave" (absence) matches "Leaving Acme Corp" (resignation). Rank 1 offers durations ("21 days", "10 days") that an LLM could easily misquote as a parental-leave entitlement.

---

## 4. Deep-Dive Failure Analysis

### Negative control — Q6: confident false positives on an unanswerable query

Q6 stays an **unsupported negative control**. No chunk is evidence, whatever its similarity. Its high scores measure topical closeness ("leave"), not support for an answer.

| Factor | Diagnosis |
|---|---|
| What happened | Top-3 are all false positives: 0.6877 / 0.6341 / 0.6125 (baseline) and 0.7002 / 0.6499 / 0.6404 (child). |
| Category | **Missing content** plus **no abstention mechanism**. This is not a boundary or dilution problem. |
| Why the embedding failed | Dense retrieval is a nearest-neighbour search, so it *always* returns k results. "Parental leave" sits in the same embedding neighbourhood as "annual/sick leave" (the leave topic), so the nearest real chunk is the leave chunk. BGE also compresses scores into a narrow band: every top-1 across the 6 queries falls between 0.61 and 0.74, so a "high" score carries little information. |
| Why a threshold can't fix it | Unsupported separation = min(true-evidence cosine over Q1/Q2/Q4/Q5) − Q6 top-1 = 0.6141 − 0.6877 = **−0.074**. Any cutoff that rejects Q6 also rejects real answers for Q2 and Q5. |
| Secondary drift | "Leave" (time off) vs "Leaving Acme Corp" (resignation): a word-sense collision. |

### Partial retrieval — Q3: no sufficient evidence in either index

**Why neither index retrieved sufficient evidence: the evidence does not exist.** The PDF has no international or overseas trip approval requirement (§1, ground truth). No chunking, metadata or ranking change can retrieve it, so PARTIAL is the ceiling for both indexes. The other three factors explain only *what ranks above* the closest related rule:

| Candidate cause | Verdict | Evidence (latest run plus whole-index ranks) |
|---|---|---|
| **Missing evidence in the corpus** | **Root cause** | 0 occurrences of overseas, international, abroad, foreign, visa and trip in the PDF (text-only). The closest related rule (6.1 business-class approval) is about cabin class, not trip approval. |
| **Chunking** | Contributing (baseline only) | Day 05 chunk 028 joins the end of 5.3 (the licence approval table) to the start of 6.1 (the headers and the first sentence). One vector carries both "approval/approver" and "business travel", which are the query's two halves. The 6.1 business-class sentence was cut into a separate 195-character chunk (029) with no header. Children split 028 at the heading (028·c0 = table, 028·c1 = 6.1 intro), but 029 stays headerless, because children only split *at* headings and the Day 05 split falls *inside* 6.1. |
| **Retrieval ranking** | Contributing | Baseline: the distractor (licence table) ranks **1** at 0.7147, and the related rule ranks 2 at 0.6449. "Approve" matches "approval thresholds / Approver / joint approval" more strongly than the single "requires prior written approval". Child: the distractor drops to **rank 4**. Rank 1 is now 6.1·c1 (0.6816), which is the right section but holds the booking-platform rule, not an approval. The related rule stays at rank 2 (0.6449) in both indexes. The 1.1 work-from-another-country rule ranks 8 (baseline) and 7 (child). |
| **Missing or wrong metadata** | Contributing (baseline only) | 028 is labelled "5.3 Software License Costs" although its second half is 6.1. Both 028 and 029 carry the false H1 "This line is part of a quote, not a heading." in `section_path`. In the child index the related rule's label is correct ("6.1 Booking Business Travel", `label_correct` true in both runs). |

**Risk if passed to generation:** the baseline rank-1 table offers a ready-made approval hierarchy ("Department head + Finance", "CFO and CTO"). A generator could present it as a travel-approval rule. The correct behaviour is to say the policy does not define overseas-trip approval, and to cite 6.1 for business class only.

### Related weakness — Q5 / Q2: vector dilution in chunk 12 (motivates the experiment)

Chunk 12 (708 chars) spans four sections: the end of 2.2, then 3.1, 3.2 and 3.3. The sick-leave text is about a quarter of the chunk, so its vector is an average of four topics. Q1, Q2, Q5 *and* the false-positive Q6 all land on this one vector, with margins of only 0.021–0.053. Because Day 05 records `section_path` at a chunk's **start**, all of this content is cited as "2.2 Password and Access Requirements". A second Day 05 bug compounds this: 21 of 47 chunks have the false H1 "This line is part of a quote, not a heading." in their `section_path`.

---

## 5. Experiment Description & Hypothesis

**Option chosen: C (child–parent / boundary repair), implemented as heading-aligned child chunks.**

**Hypothesis.** The Q3 failure and the thin Q2/Q5 margins come from multi-section chunks: their vectors average several topics, and their labels cite the wrong section. If every indexed vector covers exactly one section, then:
- (a) the right section's vector will separate further from its neighbours (larger margin);
- (b) Q3's mixed "approval table + travel header" vector disappears, so the wrong-domain top-1 should drop;
- (c) section labels become correct.

**Prediction for Q6:** no fix. Its failure is missing content, and no ranking change can create evidence.

**The single change.** `split_at_headings()` in `retriever.py` splits each Day 05 chunk before every markdown heading it contains. A heading stays attached to the content beneath it, so `## 3. Leave` + `### 3.1 Annual Leave` stay together. Each child:
- keeps all parent metadata plus a `parent_chunk_id` (so Day 07 can expand back to the parent);
- gets a `section_path` rebuilt from a document-order heading stack, so its label is the section it actually contains.

47 parents → 69 children (10–843 chars). For example, chunk 12 becomes `c0` "2.2 tail" · `c1` "3.1 Annual Leave" · `c2` "3.2 Sick Leave" · `c3` "3.3 Requesting Leave".

**Held constant:** embedding model, query instruction, cosine metric, top-k = 3, chunk text (children are exact substrings; nothing is prepended), test cases and grading.

**Why not A or B:**
- **Option B** (prepend the Day 05 `[Section: …]`) would stamp "2.2 Password and Access Requirements" onto the leave content and "5.3 Software License Costs" onto the travel text. That reinforces exactly the wrong associations diagnosed above.
- **Option A** (BM25 + RRF) cannot help Q3, because "overseas" matches nothing and "approve" with stemming would *boost* the software approval table. It cannot help Q6 either, because "leave" matches the same leave chunks. It would mainly help Q5's exact-term match, which is not the core failure.

---

## 6. Before vs. After

Same 6 queries, same model, top-3. Baseline = 47 Day 05 chunks; Experiment = 69 heading-aligned children.

| Query | Status (before → after) | Evidence rank | Margin (evidence − best irrelevant) | Top-1 cosine | Top-1 section label | Label correct? |
|---|---|---|---|---|---|---|
| Q1 | YES → YES | 1 → 1 | 0.0527 → **0.0306** ▼ | 0.6811 → 0.7372 | 2.2 Password… → **3.1 Annual Leave** | ✗ → ✓ |
| Q2 | YES → YES | 1 → 1 (partial step: 11 → **2**) | 0.0280 → **0.0387** ▲ | 0.6575 → 0.6683 | 2.2 Password… → **3.3 Requesting Leave** | ✗ → ✓ |
| Q3 | PARTIAL → PARTIAL | partial 2 → 2 | n/a | 0.7147 → 0.6816 | **5.3 Software License Costs → 6.1 Booking Business Travel** | ✓ → ✓ |
| Q4 | YES → YES | 1 → 1 (partial: 2 → 3) | 0.0603 → **0.0168** ▼ | 0.7377 → 0.7377 | 6.3 → 6.3 | ✓ → ✓ |
| Q5 | YES → YES | 1 → 1 | 0.0214 → **0.0997** ▲ (4.7×) | 0.6141 → 0.6616 | 2.2 Password… → **3.2 Sick Leave** | ✗ → ✓ |
| Q6 | 3 FP → 3 FP | none | n/a | 0.6877 → 0.7002 | leave chunk → 3.1 / 3.2 / 3.3 children | n/a |

| Aggregate metric | Baseline | Experiment |
|---|---|---|
| Answerable queries with evidence at rank 1 (Q1, Q2, Q4, Q5) | 4 / 4 | 4 / 4 |
| Evidence citation labels correct (Q1–Q5) | **2 / 5** | **5 / 5** |
| Mean margin (Q1, Q2, Q4, Q5) | 0.0406 | 0.0464 |
| Unsupported separation (min true-evidence cos − Q6 top-1 cos) | −0.0736 | −0.0386 (still negative) |
| Q6 false positives in top-3 | 3 | 3 |
| Distractor rule in top-3 (Q2 sick-leave certificate, Q3 licence table) | **2 / 2** (both at rank 1) | **0 / 2** (ranks 13 and 4) |
| Successful / partial / retrieval failure / negative control | 4 / 1 / 0 / 1 | 4 / 1 / 0 / 1 |
| Index size | 47 vectors | 69 vectors |

### Experiment trace (top-3, abbreviated)

| Query | #1 | #2 | #3 |
|---|---|---|---|
| Q1 | 0.737 · 3.1 Annual Leave ✓ | 0.707 · 3.2 Sick Leave | 0.664 · 2.1 Hardware Allowance |
| Q2 | 0.668 · 3.3 Requesting Leave ("Your manager approves or declines…") ✓ | 0.655 · 3.3 ("…once approved") partial | 0.630 · 1.1 Remote Work Eligibility |
| Q3 | 0.682 · 6.1 Booking Business Travel ("All flights, hotels… corporate travel platform") | 0.645 · 6.1 (business class → executive approval) partial | 0.630 · 6.3 Expense Claims |
| Q4 | 0.738 · 6.3 itemized receipt ✓ | 0.721 · 6.1 Booking Business Travel | 0.710 · 6.3 ("within 30 days") partial |
| Q5 | 0.662 · 3.2 Sick Leave ✓ | 0.562 · 1.1 Remote Work Eligibility | 0.561 · 11. Contacts |
| Q6 | 0.700 · 3.1 Annual Leave ✗ | 0.650 · 3.2 Sick Leave ✗ | 0.640 · 3.3 Requesting Leave ✗ |

### Interpretation

- **Hypothesis (a), margins: partly confirmed.** Q5 improved strongly and Q2 moderately. With one section per vector, Q2's top-1 no longer carries the sick-leave "3 consecutive working days" distractor.
- **Regressions:**
  - **Q1** narrowed because its nearest competitor is now the separate *sibling* 3.2 Sick Leave child (0.707). That is a benign, same-topic confusion, but it is real.
  - **Q4** narrowed (0.060 → 0.017) because the 6.1 travel intro is now its own clean "travel" vector (0.721) competing with the receipt rule.
  - Both stay correct at rank 1, but with less headroom.
  - More, smaller vectors (69 vs 47; the smallest is 10 chars) means more near-duplicate candidates.
- **Hypothesis (b), Q3: partly confirmed.** The wrong-domain software table left the top-1 (now in the right domain, at 0.682 instead of 0.715), but the grade stays PARTIAL because the corpus has no overseas-travel rule. The business-class sentence is still headerless (a Day 05 split inside 6.1), so it ranks 2nd behind 6.1's intro.
- **Hypothesis (c), labels: confirmed (2/5 → 5/5).** This is the most valuable result for Day 07 citations.
- **Q6: as predicted, not fixed.** Separation improved from −0.074 to −0.039 but stays negative, and Q6's top-1 rose to 0.700. Now all three false positives are leave sections that state concrete day counts, which is arguably *more* dangerous for generation.

**Verdict:** Adopt heading-aligned children as the **ranking unit** for Day 07, because of the correct citations and sharper Q2/Q5. Do **not** treat this as a fix for unanswerable queries. Watch the Q1/Q4 margin regressions.

---

## 7. Recommendations for Day 07 (prompt augmentation & generation)

1. **Rank on children, feed parents (or whole sections) to the LLM.** Children give precise ranking and correct labels. But the complete rule can span several Day 05 chunks: 3.3's steps 1–3 cross chunks 12 → 13, and 6.3's rules cross chunks 031 → 032 → 033. Expand each hit through `parent_chunk_id` and merge adjacent children that share a `section_path` before building the prompt.
2. **Never use cosine as an answerability signal.** In this corpus, unsupported and wrong-domain hits (0.688, 0.715) outscore true evidence (0.614–0.658). Day 07 needs explicit abstention:
   - a prompt rule: *"Answer only from the provided sections; if the policy does not state it, say so"*;
   - plus a cheap lexical-coverage check, e.g. flag queries whose key content terms ("parental", "overseas") occur 0 times in the corpus.
3. **Guard against numeric and rule conflation.** Q2's context places "3 consecutive working days (sick-leave certificate)" next to "manager approves within 3 working days". Require the model to quote the rule it relies on, with its section, so the user can catch a conflation.
4. **Pass citations through.** Include `section_path` + `pages` for each context block. These are now correct for children, but were wrong for 3 of 5 evidence chunks at baseline.
5. **Fix Day 05 metadata at the source:**
   - set `section_path` per contained heading, not per chunk start;
   - stop the false H1 "This line is part of a quote, not a heading." from entering 21 chunks' paths;
   - avoid splits inside short sections like 6.1, which left the business-class rule headerless.
6. **Diversify top-k.** With children, the top-3 can be three siblings of one section (Q6: 3.1 / 3.2 / 3.3). Consider de-duplicating by parent, or MMR, so that k = 3 covers more than one section when the question is ambiguous.
7. **Next experiment candidate: hybrid BM25 + RRF on top of children.** It targets exact-term queries like Q5 and supplies the zero-match signal for recommendation 2. Run it as a separate controlled change, so its effect stays measurable.
8. **Grow the evaluation set.** Six queries cannot distinguish a 0.017 margin from noise. Ground truth must come from the document: three of the brief's six expectations (Q2, Q3, Q4) did not match this handbook.

---

## 8. Audit (2026-10-18)

**Scope.** Q2, Q3, Q4 and Q6 were re-checked against the source PDF, the Day 05 chunks, `test_queries.json`, `evaluation.py`, `retriever.py` and run `20261018-000702_file_both`. Retrieval only; no LLM was used.

**Findings**

| Query | Question checked | Result | Criteria change? |
|---|---|---|---|
| Q2 | Does the policy require special approval for leave over 3 consecutive days? | **No.** 3.3 step 2 ("Your manager approves or declines within 3 working days") is the only leave-approval rule. It covers planned leave of any length and carries both the approval and the response time in one sentence. | No grading change: the evidence phrase is supported. Descriptive text corrected ("every leave request" → verbatim "planned leave" wording). |
| Q3 | Exact international travel approval requirement? | **None exists.** Root cause of the PARTIAL is missing evidence. Chunking, metadata and ranking only affect what outranks the related 6.1 rule (§4). | No grading change. Descriptive text now says the question is not answerable from the corpus. |
| Q4 | Dollar threshold for itemized receipts? | **None.** "Every claim requires: An itemized receipt (a credit card statement alone is not sufficient)." | No grading change. Verbatim wording recorded. |
| Q6 | Still a negative control? | **Yes.** 0 occurrences of parental, maternity, paternity, family leave, adoption or birth. | None. Note added that scores are not evidence. |

**Changes made**

1. `test_queries.json`: rewrote `expected_evidence` and `notes` for q2, q3 and q4 using verbatim PDF wording, and added a note to q6. Added `distractor_phrases` to q2 (`"longer than 3 consecutive working days"`) and q3 (`"CFO and CTO (joint approval)"`). Questions, `evidence_phrases`, `partial_phrases`, `expected_section` and `unsupported` are **unchanged**.
2. `evaluation.py`: optional `distractor_phrases`. Each hit gets `distractor: true/false`, each case gets `distractor_rank` (whole index) and `distractor_in_top_k`, and summaries get `distractor_in_top_k`. Distractors are **reported only**: they never count as evidence and never change status, rank, margin or label. This makes the "trap inside the retrieved context" claim in §3 a logged measurement instead of a manual observation.
3. `run_queries.py`: shows `[DISTRACTOR]` on hits and the distractor rank in the status and comparison lines, and adds a `distractor` column to `results.csv` and fields to `comparison.json`.
4. Retrieval architecture (model, query instruction, cosine, top-k, chunks, child splitting): **unchanged**.

**Before vs after the audit change** (all 6 queries, both indexes, rerun as `20261018-001819_file_both`):

| | Before (`20261018-000702`) | After (`20261018-001819`) |
|---|---|---|
| Status, evidence rank, partial rank, scores, margins, labels, top-3 chunk IDs | as in §2 and §6 | **identical** (0 differences across 12 records) |
| Summary metrics (rank-1 4/4; labels 2/5 → 5/5; mean margin 0.0406 → 0.0464; separation −0.0736 → −0.0386) | as listed | **identical** |
| Q2 distractor (sick-leave certificate rule) | not measured | baseline rank **1**, in top-3 (inside the evidence chunk) → child rank **13** |
| Q3 distractor (licence approval table) | not measured | baseline rank **1**, in top-3 → child rank **4** |

**Remaining failures and weaknesses (latest run)**

- **Q6:** 3/3 confident false positives in both indexes. Unsupported separation is still negative (−0.0736 baseline, −0.0386 child), so no cosine threshold can abstain. This needs an abstention mechanism (§7.2), not better ranking.
- **Q3:** capped at PARTIAL by missing content. In the baseline, a wrong-domain approval table is the top hit.
- **Baseline citations:** 3 of 5 evidence labels are wrong (Q1, Q2, Q5 cite "2.2 Password…"). They are fixed in the child index.
- **Thin margins:** Q5 baseline 0.0214, Q4 child 0.0168. Six queries cannot separate these from noise.
- **Split rules:** 6.3's "Every claim requires" list (chunks 032 → 033) and 3.3's steps (chunks 012 → 013) are still split across Day 05 chunks. The child index does not repair splits that fall inside a section.

---

## Reproduce

No queries are hardcoded in Python. Test cases live in `test_queries.json`. Ad-hoc queries come from the command line or an interactive prompt.

Paths below are relative to the unified `rag/` project folder (formerly `day06-retrieval/`). `python run_rag.py retrieve` is equivalent to `run_queries.py --index both`.

```powershell
cd rag
.\.venv\Scripts\python.exe run_queries.py --index both     # this report: baseline vs experiment on test_queries.json
.\.venv\Scripts\python.exe run_queries.py                  # baseline (parent) only
.\.venv\Scripts\python.exe run_queries.py --index child    # experiment only
.\.venv\Scripts\python.exe run_queries.py --file my.json   # any other case file
.\.venv\Scripts\python.exe run_queries.py -q "your query"  # ad-hoc (repeat -q for more)
.\.venv\Scripts\python.exe run_queries.py -i               # interactive: :k N, :index child, :chunk <id>, :full, :quit
```

| File | Purpose |
|---|---|
| `retriever.py` | `Retriever(index="parent"/"child")`: loads Day 05 chunks + document metadata, embeds, NumPy cosine index, `retrieve(query, top_k=3)` → `rank`, `similarity_score`, `chunk_id`, `source`, `chunk_text`; `split_at_headings()` |
| `test_queries.json` | The 6 test cases. Optional grading fields: `evidence_phrases`, `partial_phrases`, `expected_section`, `unsupported`. Optional reporting-only field: `distractor_phrases`. Cases without grading fields are logged as UNGRADED. |
| `evaluation.py` | Generic grading (status, evidence rank over the whole index, margin, label check) and run summaries |
| `run_queries.py` | Runs file / ad-hoc / interactive queries and logs everything |
| `logs/retrieval_log.jsonl` | Append-only history: one line per query run, with full top-k chunks, scores and metadata |
| `logs/runs/<run_id>/` | `results.json`, `results.csv` (one row per query × rank), and `comparison.json` for `--index both` |

The numbers in this report come from run `20261018-001819_file_both`, the latest run, which includes the distractor fields. The original run `20261015-140746_file_both` and `20261018-000702_file_both` produce identical scores, ranks and statuses.
