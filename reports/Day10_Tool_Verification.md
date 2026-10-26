# Day 10 local tool verification

Verified on 25 October 2026 using the existing local `llama3.2`, child retriever and web app.

- `python -m unittest discover -s tests -v`: 57 tests passed (13 Day 10 tests plus 44 existing tests).
- JavaScript syntax check using Node's `vm.Script`: passed.
- Live server: `python run_rag.py web --port 5010`.
- `python tests/smoke_day10_http.py http://127.0.0.1:5010`: passed policy read, cancellation, invalid date, confirmation and duplicate-confirmation checks.
- Policy read returned a brief policy-philosophy answer with a citation to 1.1 Eligibility and Philosophy, page 1, under Remote Work Policy. This verifies routing and source display data, not exhaustive answer completeness.
- Cancelled action: zero execution records. Invalid calendar date: HTTP 400 and no execution. Confirmed action retried twice: exactly one execution record, marked simulated.
- Additional offline checks cover reversed dates, missing dates, malformed JSON, expiry, another browser session, changing confirmation payload dates, replacing pending requests, concurrent confirmations and log-write failure.

Actual browser clicking could not be tested: this session exposed no browser connection and the in-app browser was unavailable. Live HTTP endpoints, page controls in served HTML, and JavaScript syntax were checked instead. Browser test instructions are in README section 9.

The demo has no HR client or integration. The simulated write only appends a local execution event. Pending requests are held in memory and invalidated on server restart. Dates are validated for format, calendar validity and ordering; there is no leave-balance or eligibility decision. The RAG prompt/model/retrieval remain unchanged.
