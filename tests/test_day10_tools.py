"""Day 10 browser API tests; no real HR service or LLM required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day10_tools import LocalTools, ToolError, is_leave_request
from web_app import create_app
from test_qa_pipeline import pipeline, good_reply

QUESTION = 'Request annual leave from 2026-11-02 to 2026-11-04 reason: holiday'


class ToolWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'tools.jsonl'
        self.pipeline = pipeline(good_reply())
        self.app = create_app(self.pipeline, tool_log_path=self.path)
        self.client = self.app.test_client()

    def events(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []

    def stage(self):
        response = self.client.post('/api/ask', json={'question': QUESTION})
        self.assertEqual(response.status_code, 200)
        return response.json['action']

    def decide(self, action, decision):
        return self.client.post('/api/action/decision', json={'action_id': action['id'], 'decision': decision})

    def test_policy_uses_read_tool_and_returns_sources(self):
        result = self.client.post('/api/ask', json={'question': 'What is the remote work policy?'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['debug']['tool'], 'read_policy')
        self.assertTrue(result.json['sources'])
        self.assertEqual(self.events()[-1]['tool'], 'read_policy')

    def test_cancel_never_executes_and_cannot_confirm_later(self):
        action = self.stage()
        self.assertEqual(action['status'], 'pending')
        self.assertEqual(self.decide(action, 'cancel').json['action']['status'], 'cancelled')
        self.assertEqual(self.decide(action, 'confirm').status_code, 409)
        self.assertFalse(any(row['executed'] for row in self.events()))
        self.assertFalse(self.pipeline.client.calls)

    def test_confirmation_executes_once(self):
        action = self.stage()
        self.assertFalse(any(row['executed'] for row in self.events()))
        for _ in range(2):
            result = self.decide(action, 'confirm')
            self.assertEqual(result.json['action']['status'], 'executed')
            self.assertTrue(result.json['action']['simulated'])
        self.assertEqual(sum(row['executed'] for row in self.events()), 1)
        self.assertEqual(self.decide(action, 'cancel').status_code, 409)

    def test_invalid_dates_no_pending_or_execution(self):
        for dates in ('2026-02-30 to 2026-03-02', '2026-11-05 to 2026-11-02', '2026-13-01 to 2026-13-02', 'tomorrow', '2026-1-02 to 2026-01-03'):
            with self.subTest(dates=dates):
                response = self.client.post('/api/ask', json={'question': 'Request leave from '+dates})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json['error']['code'], 'invalid_input')
        self.assertIsNone(self.client.get('/api/action/pending').json['action'])
        self.assertFalse(any(row['executed'] for row in self.events()))
        self.assertEqual(len(self.events()), 5)

    def test_refresh_recovers_pending(self):
        action = self.stage()
        self.assertEqual(self.client.get('/api/action/pending').json['action']['id'], action['id'])

    def test_other_session_cannot_confirm(self):
        action = self.stage()
        response = self.app.test_client().post('/api/action/decision', json={'action_id': action['id'], 'decision': 'confirm'})
        self.assertEqual(response.status_code, 404)
        self.assertFalse(any(row['executed'] for row in self.events()))

    def test_expiry_prevents_execution(self):
        action = self.stage()
        self.app.extensions['day10_tools'].records[action['id']]['expires'] = 0
        self.assertEqual(self.decide(action, 'confirm').status_code, 409)
        self.assertFalse(any(row['executed'] for row in self.events()))

    def test_confirmation_cannot_change_reviewed_dates(self):
        action = self.stage()
        response = self.client.post('/api/action/decision', json={'action_id': action['id'], 'decision': 'confirm', 'start_date': '2027-01-01'})
        self.assertEqual(response.json['action']['request']['start_date'], '2026-11-02')

    def test_invalid_json_and_confirmation(self):
        for body in ([], 'wrong', {'action_id': 'x', 'decision': True}):
            self.assertEqual(self.client.post('/api/action/decision', json=body).status_code, 400)
        self.assertEqual(self.client.post('/api/ask', json=['x']).status_code, 400)

    def test_new_request_supersedes_old(self):
        old, new = self.stage(), self.stage()
        self.assertEqual(self.decide(old, 'confirm').status_code, 409)
        self.assertEqual(self.decide(new, 'cancel').status_code, 200)
        self.assertFalse(any(row['executed'] for row in self.events()))

    def test_parallel_confirmation_executes_once(self):
        tools = LocalTools(self.pipeline, self.path)
        action = tools.stage_leave(QUESTION, 'owner')
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: tools.decide(action['id'], 'confirm', 'owner'), range(2)))
        self.assertTrue(all(r['status'] == 'executed' for r in results))
        self.assertEqual(sum(row['executed'] for row in self.events()), 1)

    def test_policy_questions_are_not_write_requests(self):
        for question in ('How do I request leave?', 'What is the leave policy?', 'Who approves leave requests?'):
            self.assertFalse(is_leave_request(question))
        self.assertTrue(is_leave_request('Please request a simulated leave action from 2026-11-02 to 2026-11-03'))

    def test_log_failure_does_not_mark_execution(self):
        action = self.stage()
        tools = self.app.extensions['day10_tools']
        tools.log_path = Path(self.temp.name)  # directory, not writable as a log file
        result = self.decide(action, 'confirm')
        self.assertEqual(result.status_code, 503)
        self.assertEqual(tools.records[action['id']]['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
