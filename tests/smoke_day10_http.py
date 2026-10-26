"""Live Day 10 endpoint smoke check against a running local web app."""
import http.cookiejar
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:5010'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base+path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with opener.open(req, timeout=200) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)

with opener.open(base, timeout=10) as response:
    html = response.read().decode()
    assert 'confirm-action' in html and 'cancel-action' in html
status, result = call('/api/ask', {'question': 'What is the remote work policy?'})
assert status == 200 and result['debug']['tool'] == 'read_policy', result
assert result['debug']['status'] == 'answered' and result['answer'], result
assert any('Remote Work Policy' in source['section_path'] and 1 in source['pages'] for source in result['sources']), result
print('PASS policy read: ' + result['answer'], flush=True)
print('Sources: ' + ', '.join(s['section'] for s in result['sources']), flush=True)
question = 'Request annual leave from 2026-11-02 to 2026-11-04 reason: Day 10 smoke test'
status, staged = call('/api/ask', {'question': question})
assert status == 200 and staged['action']['status'] == 'pending'
action_id = staged['action']['id']
status, result = call('/api/action/decision', {'action_id': action_id, 'decision': 'cancel'})
assert status == 200 and result['action']['status'] == 'cancelled'
print('PASS cancellation', flush=True)
status, result = call('/api/ask', {'question': 'Request leave from 2026-02-30 to 2026-03-02'})
assert status == 400 and result['error']['code'] == 'invalid_input'
print('PASS invalid date: '+result['error']['message'], flush=True)
status, staged = call('/api/ask', {'question': question})
confirmed_id = staged['action']['id']
for _ in range(2):
    status, result = call('/api/action/decision', {'action_id': confirmed_id, 'decision': 'confirm'})
    assert status == 200 and result['action']['status'] == 'executed'
rows = [json.loads(line) for line in (Path(__file__).resolve().parents[1]/'logs/tool_execution_log.jsonl').read_text(encoding='utf-8').splitlines()]
assert not any(r['executed'] and r.get('action_id') == action_id for r in rows)
assert sum(r['executed'] and r.get('action_id') == confirmed_id for r in rows) == 1
print('PASS confirmation exactly once; cancelled action has zero execution records', flush=True)
