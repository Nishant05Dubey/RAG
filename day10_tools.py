"""Local Day 10 tools. The write tool only records a simulation in a JSONL log."""
import json
import re
import secrets
import threading
import time
from datetime import date, datetime, timezone

from paths import LOGS_DIR

TOOL_LOG = LOGS_DIR / 'tool_execution_log.jsonl'


class ToolError(ValueError):
    def __init__(self, message, code='invalid_input', status=400):
        super().__init__(message)
        self.code, self.http_status = code, status


def is_leave_request(question):
    text = question.strip().lower()
    # Informational questions stay on the read path, even when they mention requests.
    if re.match(r'^(how|what|when|who|where|why|is|are|does|do)\b', text):
        return False
    return bool(re.search(
        r'\b(?:request|submit|book|schedule|apply for|take)\b.{0,65}\b(?:leave|time off)\b|'
        r'\b(?:i want|i need|i would like)\b.{0,40}\b(?:leave|time off)\b', text))


def parse_leave(question):
    dates = re.findall(r'\b\d{4}-\d{1,2}-\d{1,2}\b', question)
    if len(dates) != 2:
        raise ToolError('Include a start and end date as YYYY-MM-DD. Example: Request annual leave from 2026-11-02 to 2026-11-04. For one day, repeat the date.')
    try:
        if any(not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) for value in dates):
            raise ValueError()
        start, end = [date.fromisoformat(value) for value in dates]
    except ValueError:
        raise ToolError('Invalid date. Use real calendar dates in YYYY-MM-DD format (for example, 2026-11-02).') from None
    if end < start:
        raise ToolError('The end date must be on or after the start date.')
    # These are simulation fields, not a claim about handbook eligibility or balances.
    kind = re.search(r'\b(annual|sick|unpaid)\s+leave\b', question, re.I)
    reason = re.search(r'\breason\s*:\s*(.+)$', question, re.I)
    return {'leave_type': kind.group(1).lower() if kind else 'unspecified',
            'start_date': start.isoformat(), 'end_date': end.isoformat(),
            'calendar_days': (end-start).days + 1,
            'reason': reason.group(1).strip() if reason else ''}


class LocalTools:
    def __init__(self, pipeline, log_path=TOOL_LOG, ttl=900, clock=time.monotonic):
        self.pipeline, self.log_path, self.ttl, self.clock = pipeline, log_path, ttl, clock
        self.records = {}
        self.lock = threading.RLock()

    def log(self, tool, status, **fields):
        if self.log_path is None:
            return
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), 'tool': tool,
               'status': status, 'executed': status == 'executed', **fields}
        with self.lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def read_policy(self, question):
        try:
            result = self.pipeline.ask(question)
        except Exception as exc:
            self.log('read_policy', 'error', question=question, error=str(exc))
            raise
        self.log('read_policy', 'completed', question=question, result=result)
        result['debug']['tool'] = 'read_policy'
        return result

    @staticmethod
    def public(record):
        return {key: record[key] for key in ('id', 'status', 'simulated', 'request', 'message')}

    def stage_leave(self, question, owner):
        try:
            details = parse_leave(question)
        except ToolError as exc:
            self.log('simulate_leave', 'validation_error', simulated=True, question=question, error=str(exc))
            raise
        record = {'id': secrets.token_urlsafe(24), 'owner': owner, 'status': 'pending',
                  'simulated': True, 'request': details, 'expires': self.clock()+self.ttl,
                  'message': 'Review this simulated leave request. Nothing has been executed. Confirm or cancel below.'}
        with self.lock:
            for old in self.records.values():
                if old['owner'] == owner and old['status'] == 'pending':
                    self.log('simulate_leave', 'cancelled', simulated=True, action_id=old['id'], reason='superseded')
                    old.update(status='cancelled', message='Replaced by a new request; not executed.')
            self.log('simulate_leave', 'pending', simulated=True, action_id=record['id'], request=details)
            self.records[record['id']] = record
        return self.public(record)

    def decide(self, action_id, decision, owner):
        if not isinstance(action_id, str) or decision not in ('confirm', 'cancel'):
            raise ToolError('Choose Confirm simulation or Cancel for the displayed request.')
        with self.lock:
            record = self.records.get(action_id)
            if not record or record['owner'] != owner:
                raise ToolError('Request not found in this browser session. Enter a new request.', 'unknown_action', 404)
            if record['status'] == 'pending' and self.clock() >= record['expires']:
                self.log('simulate_leave', 'expired', simulated=True, action_id=action_id)
                record.update(status='expired', message='Confirmation expired; nothing was executed. Enter a new request.')
            if record['status'] != 'pending':
                if (decision == 'confirm' and record['status'] == 'executed') or (decision == 'cancel' and record['status'] == 'cancelled'):
                    return self.public(record)  # retry: do not execute/log a second action
                raise ToolError(record['message'], 'action_closed', 409)
            status = 'executed' if decision == 'confirm' else 'cancelled'
            message = ('Simulation executed and logged. No request was sent to HR and no leave was approved.'
                       if decision == 'confirm' else 'Cancelled. No simulated leave action was executed.')
            # The local log is the only write-tool effect; never call a real HR service.
            self.log('simulate_leave', status, simulated=True, action_id=action_id,
                     request=record['request'], result=message)
            record.update(status=status, message=message)
            return self.public(record)

    def pending(self, owner):
        with self.lock:
            for record in reversed(list(self.records.values())):
                if record['owner'] == owner and record['status'] == 'pending':
                    if self.clock() >= record['expires']:
                        self.log('simulate_leave', 'expired', simulated=True, action_id=record['id'])
                        record.update(status='expired', message='Confirmation expired; nothing was executed.')
                    return self.public(record)
        return None
