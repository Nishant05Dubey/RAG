"""Deterministic Day 08 evaluation. Run this file from any working directory."""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qa_pipeline as qa


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = text.translate(str.maketrans({'’': "'", '–': '-', '—': '-'}))
    return ' '.join(text.split())


def groups(text, expected):
    hay = normalize(text)
    matches = [next((v for v in group if normalize(v) in hay), None) for group in expected]
    return {'passed': sum(m is not None for m in matches), 'total': len(expected),
            'matches': matches, 'missing_groups': [g for g, m in zip(expected, matches) if m is None]}


def load_dataset(path=None):
    candidates = [Path(path)] if path else [ROOT / 'evaluation/day08_golden_dataset.json', ROOT / 'day08_golden_dataset.json']
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise ValueError(f'Copy the supplied dataset to {candidates[0]}; no substitute will be generated.')
    data = json.loads(found.read_text(encoding='utf-8-sig'))
    cases = data['cases']
    if len(cases) != 16 or len({c['id'] for c in cases}) != 16:
        raise ValueError('Expected exactly 16 uniquely identified golden cases')
    required = ('question', 'category', 'answerability', 'expected_response_behavior',
                'expected_evidence_groups', 'required_answer_fact_groups', 'expected_citation_sections',
                'expected_source_pages', 'citation_required', 'must_abstain', 'must_state_limitation', 'forbidden_claims')
    for c in cases:
        for key in required:
            if key not in c:
                raise ValueError(f"{c['id']}: missing {key}")
        for key in ('expected_evidence_groups', 'required_answer_fact_groups'):
            if not isinstance(c[key], list) or any(not isinstance(g, list) or not g or any(not isinstance(v, str) or not v.strip() for v in g) for g in c[key]):
                raise ValueError(f"{c['id']}: invalid phrase groups")
    return found, data


MISSING = re.compile(r"(?:does not|doesn't|do not|cannot|can't|not)\s+(?:explicitly\s+)?(?:state|specify|say|give|provide|confirm|contain|stated|specified)|insufficient evidence|not in (?:the|this)|no (?:general|minimum|dollar)|could(?: not|n't) find")


def forbidden_checks(answer, phrases):
    text = normalize(answer)
    findings = []
    for phrase in phrases:
        for match in re.finditer(re.escape(normalize(phrase)), text):
            left = max(text.rfind('.', 0, match.start()), text.rfind(';', 0, match.start())) + 1
            prefix = text[left:match.start()]
            denied = bool(re.search(r"(?:does not|doesn't) (?:say|state|claim)|not true that|incorrect (?:to say|that)|not the case that", prefix))
            ambiguous = not denied and bool(re.search(r"\b(?:not|never|no|deny|denies)\b", prefix))
            findings.append({'phrase': phrase, 'status': 'denied' if denied else 'manual_review' if ambiguous else 'asserted', 'context': text[left:match.end()+60]})
    return findings


def citation_checks(case, result):
    hits = {h['chunk_id']: h for h in result.get('debug', {}).get('retrieved', [])}
    details = []
    for src in result.get('sources', []):
        hit = hits.get(src['chunk_id'], {})
        text = hit.get('text', '')
        evidence = groups(text, case['expected_evidence_groups'])
        label = src.get('section') in case['expected_citation_sections'] and bool(set(src.get('pages', [])) & set(case['expected_source_pages']))
        quote = normalize(src.get('quote', ''))
        exact = bool(quote) and quote in normalize(text)
        details.append({'chunk_id': src['chunk_id'], 'section_page_match': label,
                        'quote_in_text': exact, 'evidence': evidence,
                        'supports_expected_evidence_heuristic': label and exact and evidence['passed'] > 0})
    cited_text = ' '.join(hits.get(s['chunk_id'], {}).get('text', '') for s in result.get('sources', []))
    coverage = groups(cited_text, case['expected_evidence_groups'])
    if case['must_abstain']:
        status = 'suspicious' if result.get('citations') else 'not_required'
    elif not details:
        status = 'missing' if case['citation_required'] else 'not_required'
    elif not all(d['supports_expected_evidence_heuristic'] for d in details):
        status = 'manual_review'
    elif coverage['passed'] < coverage['total']:
        status = 'partial'
    else:
        status = 'supported_heuristic'
    return {'status': status, 'details': details, 'cited_evidence': coverage,
            'caveat': 'Correct labels and phrase overlap do not prove claim-level entailment.'}


def evaluate(case, result):
    answer = result.get('answer', '')
    text = ' '.join(h['text'] for h in result.get('debug', {}).get('retrieved', []))
    evidence, facts = groups(text, case['expected_evidence_groups']), groups(answer, case['required_answer_fact_groups'])
    evidence['status'] = ('not_expected' if not evidence['total'] else 'full' if evidence['passed'] == evidence['total'] else 'partial' if evidence['passed'] else 'not_found')
    citations = citation_checks(case, result)
    forbidden = forbidden_checks(answer, case['forbidden_claims'])
    missing = bool(MISSING.search(normalize(answer)))
    source_named = bool(re.search(r'handbook|document|excerpt|source|provided', normalize(answer)))
    # Detect novel durations too, not only the supplied examples.
    duration = bool(re.search(r'\b(?:\d+|one|two|three|four|six|eight|ten|twelve|sixteen|twenty-six)\s+(?:weeks?|months?|days?)\b', normalize(answer)))
    abstention = 'not_required'
    if case['must_abstain']:
        abstention = 'pass' if missing and source_named and not duration else 'manual_review' if missing and duration else 'fail'
    limitation = 'not_required'
    if case['must_state_limitation']:
        if case['answerability'] == 'answerable_with_conditions':
            limitation = 'pass_heuristic' if re.search(r'probation|eligible|eligibility|provided that|only if', normalize(answer)) else 'missing'
        else:
            limitation = 'pass_heuristic' if missing and source_named else 'missing'
    relevance = 'covered_heuristic' if facts['passed'] == facts['total'] else 'partial_heuristic' if facts['passed'] else 'not_demonstrated'
    failures, notes = [], []
    if evidence['status'] in ('partial', 'not_found'):
        failures.append('retrieval'); notes.append('Not all gold evidence groups retrieved.')
    if facts['passed'] < facts['total']:
        failures.append('generation_or_grounding'); notes.append('Required wording groups missing; inspect paraphrases.')
    if citations['status'] in ('missing', 'partial', 'manual_review', 'suspicious'):
        failures.append('citation'); notes.append('Citation support needs inspection: ' + citations['status'])
    if abstention in ('fail', 'manual_review'):
        failures.append('abstention'); notes.append('Unsupported response not verified safe.')
    if limitation == 'missing' or any(f['status'] == 'asserted' for f in forbidden):
        failures.append('generation_or_grounding'); notes.append('Missing limitation or asserted forbidden claim.')
    if result.get('error'):
        failures = ['retrieval' if result['error']['code'] in ('retrieval_failed', 'chunks_missing') else 'generation_or_grounding']
        notes.append('Execution error: ' + result['error']['message'])
    stages = sorted(set(failures))
    manual = bool(stages) or case['must_state_limitation'] or any(f['status'] == 'manual_review' for f in forbidden)
    return {'evidence': evidence, 'facts': facts, 'relevance': relevance, 'citation_support': citations,
            'abstention': abstention, 'limitation': limitation, 'forbidden_claims': forbidden,
            'failure_stage': 'mixed' if len(stages) > 1 else stages[0] if stages else 'manual_review' if manual else 'none',
            'failure_components': stages, 'manual_review': manual,
            'note': ' '.join(notes) or 'Automatic checks satisfied; phrase matching is not proof of groundedness.'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(dataset, pipeline):
    from paths import RETRIEVAL_CHUNKS, SOURCE_PDF, CLEAN_TXT, DOC_METADATA
    return {'dataset_sha256': digest(dataset), 'files': {p.name: digest(p) for p in (RETRIEVAL_CHUNKS, SOURCE_PDF, CLEAN_TXT, DOC_METADATA, ROOT/'retriever.py')},
            'model': pipeline.client.model, 'index': pipeline.index, 'top_k': pipeline.top_k,
            'options': {'temperature': 0, 'seed': 7, 'num_ctx': qa.NUM_CTX, 'num_predict': qa.NUM_PREDICT},
            'evaluator_sha256': digest(__file__), 'system_prompt': qa.SYSTEM_PROMPT}


def flat(row):
    e = row['evaluation']
    return {'id': row['id'], 'answerability': row['answerability'], 'evidence': e['evidence']['status'],
            'facts': f"{e['facts']['passed']}/{e['facts']['total']}", 'citation': e['citation_support']['status'],
            'abstention': e['abstention'], 'limitation': e['limitation'], 'failure_stage': e['failure_stage'],
            'manual_review': e['manual_review'], 'note': e['note'], 'answer': row['result'].get('answer', '')}


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(run):
    rows = [flat(r) for r in run['cases']]
    lines = [f"# Day 08 {run['phase']}", '', f"Cases: {len(rows)}/16. Model: {run['manifest']['model']}. Full gold evidence does not imply full answerability (especially q3).", '',
             '| Case | Evidence | Facts | Citations | Abstention | Limitation | Stage | Review |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(r[k]) for k in ('id','evidence','facts','citation','abstention','limitation','failure_stage','manual_review')) + ' |')
    for r in rows:
        lines += ['', '## ' + r['id'], '', r['note'], '', 'Answer: ' + r['answer']]
    lines += ['', '## Evaluator limits', '', 'Phrase checks miss valid paraphrases and can accept negated or unrelated facts. Citation labels, exact quotes and gold evidence overlap are heuristics, not entailment. Absence from Top-K does not prove absence from the handbook. Limitations, negation, partial answers and flagged cases require human review. Complete raw model output, validated UI answer and retrieved text are retained in JSON.']
    return '\n'.join(lines) + '\n'


def save(run, out):
    (out / f"{run['phase']}_results.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding='utf-8')
    if run['cases']:
        csv_write(out / f"{run['phase']}_results.csv", [flat(r) for r in run['cases']])
    (out / f"{run['phase']}_report.md").write_text(report(run), encoding='utf-8')


def run_cases(phase, path, data, out, pipeline=None):
    pipeline = pipeline or qa.QAPipeline()
    target = out / f'{phase}_results.json'
    if target.exists():
        raise ValueError(f'Refusing to overwrite {target}; choose a new --output directory.')
    run = {'phase': phase, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'manifest': manifest(path, pipeline), 'cases': []}
    for case in data['cases']:
        start = time.perf_counter()
        try:
            result = pipeline.ask(case['question'])
        except Exception as exc:
            result = {'answer': '', 'citations': [], 'sources': [], 'debug': getattr(exc, 'debug', {}),
                      'error': {'code': getattr(exc, 'code', type(exc).__name__), 'message': str(exc)}}
        row = {**case, 'result': result, 'latency_seconds': round(time.perf_counter()-start, 3), 'evaluation': evaluate(case, result)}
        run['cases'].append(row)
        save(run, out)  # checkpoint after every case, including exceptions
        print(phase, case['id'], row['evaluation']['failure_stage'], flush=True)
    return run


def compare(before, after):
    for key, value in before['manifest'].items():
        if key != 'system_prompt' and value != after['manifest'].get(key):
            raise ValueError('Uncontrolled change: ' + key)
    if [r['id'] for r in before['cases']] != [r['id'] for r in after['cases']] or len(before['cases']) != 16:
        raise ValueError('Comparison requires the same 16 cases in order')
    rows = []
    for b, a in zip(before['cases'], after['cases']):
        bf, af = flat(b), flat(a)
        changes = [k for k in ('evidence','facts','citation','abstention','limitation','failure_stage') if bf[k] != af[k]]
        rows.append({'id': b['id'], 'changed_checks': ', '.join(changes) or 'unchanged',
                     **{'before_'+k: bf[k] for k in ('evidence','facts','citation','abstention','limitation','failure_stage')},
                     **{'after_'+k: af[k] for k in ('evidence','facts','citation','abstention','limitation','failure_stage')}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'after', 'experiment'])
    parser.add_argument('--dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/day08')
    args = parser.parse_args()
    path, data = load_dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'baseline':
        run_cases('baseline', path, data, args.output)
    else:
        baseline = args.output/'baseline_results.json'
        before = json.loads(baseline.read_text(encoding='utf-8')) if baseline.exists() else run_cases('baseline', path, data, args.output)
        decision = ROOT/'evaluation/experiment.json'
        if not decision.exists():
            raise ValueError('Inspect baseline and record one justified prompt change in evaluation/experiment.json before running the experiment.')
        selected = json.loads(decision.read_text(encoding='utf-8'))
        if not selected.get('hypothesis') or not selected.get('observed_failures'):
            raise ValueError('Experiment needs recorded evidence and hypothesis')
        qa.SYSTEM_PROMPT = before['manifest']['system_prompt'] + '\n' + selected['prompt_addition']
        # Verify controls before spending a second run.
        current = manifest(path, qa.QAPipeline())
        for key, value in before['manifest'].items():
            if key != 'system_prompt' and current[key] != value:
                raise ValueError('Uncontrolled change before rerun: ' + key)
        after = run_cases('after', path, data, args.output)
        rows = compare(before, after)
        csv_write(args.output/'before_after_comparison.csv', rows)
        text = '# Controlled experiment\n\n' + json.dumps(selected, indent=2) + '\n\n'
        text += 'Only the system-prompt addition differs. The UI default is preserved pending review.\n\n'
        text += '| Case | Changed checks | Before stage | After stage |\n|---|---|---|---|\n'
        for r in rows:
            text += f"| {r['id']} | {r['changed_checks']} | {r['before_failure_stage']} | {r['after_failure_stage']} |\n"
        text += '\n' + report(after)
        (args.output/'evaluation_report.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
