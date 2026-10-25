"""Offline tests for Day 08 rules and experiment controls."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import run_day08_evaluation as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.path, self.data = ev.load_dataset()

    def test_dataset_exactly_16_and_partial_case_preserved(self):
        self.assertEqual(len(self.data['cases']), 16)
        self.assertEqual(self.data['cases'][2]['answerability'], 'partial_only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'missing.json'
            with self.assertRaisesRegex(ValueError, 'Copy the supplied'):
                ev.load_dataset(path)
            path.write_text(json.dumps({'cases': self.data['cases'][:-1]}))
            with self.assertRaisesRegex(ValueError, '16'):
                ev.load_dataset(path)

    def test_phrase_groups_variants_and_whitespace(self):
        result = ev.groups('A  CREDIT\ncard costs 10,001–50,000.', [['credit card'], ['10,001-50,000', 'alternative'], ['missing']])
        self.assertEqual((result['passed'], result['total']), (2, 3))
        self.assertEqual(ev.groups('unrelated', [])['total'], 0)

    def test_forbidden_claim_denial_and_assertion(self):
        phrase = 'parental leave lasts 12 weeks'
        self.assertEqual(ev.forbidden_checks('The handbook does not say ' + phrase, [phrase])[0]['status'], 'denied')
        self.assertEqual(ev.forbidden_checks(phrase.upper(), [phrase])[0]['status'], 'asserted')
        self.assertEqual(ev.forbidden_checks('I do not know whether ' + phrase, [phrase])[0]['status'], 'manual_review')

    def test_unsupported_novel_duration_and_safe_abstention(self):
        case = self.data['cases'][5]
        self.assertEqual(ev.evaluate(case, {'answer': ev.qa.INSUFFICIENT})['abstention'], 'pass')
        self.assertEqual(ev.evaluate(case, {'answer': 'Parental leave is 19 weeks.'})['abstention'], 'fail')
        denied = {'answer': 'The handbook does not say parental leave lasts 12 weeks.'}
        self.assertEqual(ev.evaluate(case, denied)['abstention'], 'manual_review')
        self.assertEqual(ev.evaluate(case, denied)['forbidden_claims'][0]['status'], 'denied')

    def test_correct_label_without_support_is_not_enough(self):
        case = self.data['cases'][0]
        result = {'citations': ['a'], 'sources': [{'chunk_id': 'a', 'section': '3.1 Annual Leave', 'pages': [2], 'quote': 'unrelated facts'}],
                  'debug': {'retrieved': [{'chunk_id': 'a', 'text': 'unrelated facts'}]}}
        self.assertEqual(ev.citation_checks(case, result)['status'], 'manual_review')
        result['debug']['retrieved'][0]['text'] = 'totaling 21 days per calendar year'
        result['sources'][0]['quote'] = '21 days per calendar year'
        self.assertEqual(ev.citation_checks(case, result)['status'], 'supported_heuristic')
        result['sources'][0]['pages'] = [5]
        self.assertEqual(ev.citation_checks(case, result)['status'], 'manual_review')

    def test_compare_checks_identity_and_controls(self):
        rows = [{**c, 'result': {}, 'evaluation': ev.evaluate(c, {})} for c in self.data['cases']]
        before = {'manifest': {'system_prompt': 'old', 'model': 'local'}, 'cases': rows}
        after = copy.deepcopy(before)
        after['manifest']['system_prompt'] = 'new'
        self.assertEqual(len(ev.compare(before, after)), 16)
        self.assertTrue(all(r['changed_checks'] == 'unchanged' for r in ev.compare(before, after)))
        after['manifest']['model'] = 'other'
        with self.assertRaisesRegex(ValueError, 'Uncontrolled'):
            ev.compare(before, after)
        after = copy.deepcopy(before)
        after['cases'].pop()
        with self.assertRaises(ValueError):
            ev.compare(before, after)

    def test_checkpoint_continues_after_exception(self):
        class Broken:
            client = type('Client', (), {'model': 'fake'})()
            index, top_k = 'child', 3
            def ask(self, question):
                raise ev.qa.OllamaUnavailable('test failure')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run = ev.run_cases('baseline', self.path, self.data, out, Broken())
            self.assertEqual(len(run['cases']), 16)
            self.assertTrue(all(r['result']['error'] for r in run['cases']))
            self.assertEqual(len(json.loads((out/'baseline_results.json').read_text())['cases']), 16)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                ev.run_cases('baseline', self.path, self.data, out, Broken())


if __name__ == '__main__':
    unittest.main()
