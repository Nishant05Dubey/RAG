"""Generic grading driven entirely by fields in the test-case JSON.

Optional per-case fields (all may be omitted; the case is then UNGRADED and
only logged for manual inspection):
    evidence_phrases  list[str]  chunk text containing any of these = "full"
    partial_phrases   list[str]  chunk text containing any of these = "partial"
    expected_section  str        section title the evidence should be cited as
    unsupported       bool       negative control: every hit is a false positive
    distractor_phrases list[str] a source rule that looks relevant but answers a
                                 different question; reported (rank, in top-k),
                                 never counted as evidence and never changes status
"""


def is_distractor(case, text):
    return any(p in text for p in case.get("distractor_phrases") or [])


def relevance(case, text):
    """'full', 'partial' or None for one chunk text."""
    if any(p in text for p in case.get("evidence_phrases") or []):
        return "full"
    if any(p in text for p in case.get("partial_phrases") or []):
        return "partial"
    return None


def is_graded(case):
    return bool(
        case.get("unsupported")
        or case.get("evidence_phrases")
        or case.get("partial_phrases")
    )


def evaluate(case, hits, ranking, retriever):
    """Grade one query.

    hits:    retriever.retrieve() output (annotated in place with 'relevance')
    ranking: retriever.rank_all() output, used to locate evidence anywhere in
             the index, not only inside the top-k.
    Returns None for ungraded cases.
    """
    if not is_graded(case):
        return None
    has_distractors = bool(case.get("distractor_phrases"))
    for h in hits:
        h["relevance"] = relevance(case, h["chunk_text"])
        if has_distractors:
            h["distractor"] = is_distractor(case, h["chunk_text"])

    full_rank = full_score = partial_rank = best_irrelevant = distractor_rank = None
    for pos, (idx, score) in enumerate(ranking, start=1):
        text = retriever.chunks[idx]["text"]
        if has_distractors and distractor_rank is None and is_distractor(case, text):
            distractor_rank = pos
        rel = relevance(case, text)
        if rel == "full" and full_rank is None:
            full_rank, full_score = pos, score
        elif rel == "partial" and partial_rank is None:
            partial_rank = pos
        elif rel is None and best_irrelevant is None:
            best_irrelevant = score

    rels = [h["relevance"] for h in hits]
    if case.get("unsupported"):
        status = f"NO EVIDENCE (expected) - {len(hits)} false positives"
    elif "full" in rels:
        status = "YES"
    elif "partial" in rels:
        status = "PARTIAL"
    else:
        status = "NO"

    evidence_hit = next((h for h in hits if h["relevance"]), None)
    expected_section = case.get("expected_section")
    label_correct = None
    if evidence_hit and expected_section:
        label_correct = evidence_hit["source"]["section"] == expected_section

    return {
        "status": status,
        "evidence_rank": full_rank,
        "partial_rank": partial_rank,
        "evidence_score": round(full_score, 4) if full_score is not None else None,
        "best_irrelevant_score": round(best_irrelevant, 4) if best_irrelevant is not None else None,
        "margin": round(full_score - best_irrelevant, 4)
        if full_score is not None and best_irrelevant is not None else None,
        "expected_section": expected_section,
        "evidence_hit_section": evidence_hit["source"]["section"] if evidence_hit else None,
        "label_correct": label_correct,
        "distractor_rank": distractor_rank,
        "distractor_in_top_k": any(h["distractor"] for h in hits) if has_distractors else None,
    }


def summarise(records):
    """Aggregate metrics over one index's graded records."""
    graded = [r for r in records if r["evaluation"]]
    answerable = [r for r in graded if r["evaluation"]["evidence_score"] is not None]
    unsupported = [r for r in graded if r.get("unsupported")]
    labels = [r["evaluation"]["label_correct"] for r in graded
              if r["evaluation"]["label_correct"] is not None]
    margins = [r["evaluation"]["margin"] for r in answerable
               if r["evaluation"]["margin"] is not None]

    separation = None
    if answerable and unsupported:
        min_evidence = min(r["evaluation"]["evidence_score"] for r in answerable)
        max_unsupported = max(r["results"][0]["similarity_score"] for r in unsupported)
        # Negative = an unanswerable query outscores some real evidence,
        # so no single similarity threshold can separate them.
        separation = round(min_evidence - max_unsupported, 4)

    with_distractors = [r["evaluation"].get("distractor_in_top_k") for r in graded
                        if r["evaluation"].get("distractor_in_top_k") is not None]

    return {
        "queries": len(records),
        "graded": len(graded),
        "evidence_at_rank_1": f"{sum(r['evaluation']['evidence_rank'] == 1 for r in answerable)}/{len(answerable)}",
        "labels_correct": f"{sum(labels)}/{len(labels)}",
        "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
        "unsupported_separation": separation,
        "distractor_in_top_k": f"{sum(with_distractors)}/{len(with_distractors)}",
    }
