"""Offline Day 11 examples and execution-boundary checks; no Ollama required."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day10_tools import LocalTools
from day11_agent import MAX_ITERATIONS, ReActAgent


def decision(action, question=None, answer=''):
    return {'thought': 'Use the next step needed to answer the question.', 'action': action,
            'action_input': {} if question is None else {'question': question}, 'final_answer': answer}


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.messages = []

    def chat(self, messages, schema):
        self.messages.append(copy.deepcopy(messages))
        return {'message': {'content': json.dumps(next(self.decisions))}}


class ReActTests(unittest.TestCase):
    def test_two_policy_calls_then_final(self):
        question = 'What are the remote work and annual leave policies?'
        remote = {'answer': 'Remote work requires approval.', 'citations': ['remote-1'],
                  'sources': [{'chunk_id': 'remote-1', 'quote': 'requires approval'}], 'debug': {}}
        leave = {'answer': 'Annual leave is 21 days.', 'citations': ['leave-1'],
                 'sources': [{'chunk_id': 'leave-1', 'quote': '21 days'}], 'debug': {}}
        model = ScriptedModel([decision('read_policy', 'What is the remote work policy?'),
                               decision('read_policy', 'What is the annual leave policy?'),
                               decision('final', answer='Remote work requires approval [remote-1]; annual leave is 21 days [leave-1].')])
        tools, emitted = Mock(spec=LocalTools), []
        tools.read_policy.side_effect = [remote, leave]
        result = ReActAgent(client=model, tools=tools, emit=emitted.append).run(question)
        self.assertEqual(tools.read_policy.call_count, 2)
        self.assertEqual([call.args[0] for call in tools.read_policy.call_args_list],
                         ['What is the remote work policy?', 'What is the annual leave policy?'])
        self.assertEqual(json.loads(model.messages[1][-1]['content'])['observation'], remote)
        self.assertEqual(json.loads(model.messages[2][-1]['content'])['observation'], leave)
        self.assertEqual([event['action'] for event in result['trace'] if 'action' in event],
                         ['read_policy', 'read_policy', 'final'])
        self.assertEqual([event['observation'] for event in result['trace'] if 'observation' in event], [remote, leave])
        self.assertEqual(result['stop_reason'], 'final')
        self.assertEqual(result['answer'], result['trace'][-1]['final_answer'])
        self.assertEqual(result['trace'], emitted)
        tools.stage_leave.assert_not_called()
        tools.decide.assert_not_called()
        print('\nDay 11 TWO-TOOL trace (mocked facts): ' + json.dumps(result['trace']), flush=True)

    def test_direct_answer_without_tools(self):
        model = ScriptedModel([decision('final', answer='Hello! How can I help?')])
        tools = Mock(spec=LocalTools)
        result = ReActAgent(client=model, tools=tools).run('Hello!')
        self.assertEqual(tools.mock_calls, [])
        self.assertEqual(len(model.messages), 1)
        self.assertEqual(result['answer'], 'Hello! How can I help?')
        self.assertEqual(result['stop_reason'], 'final')
        self.assertFalse(any('observation' in event for event in result['trace']))
        print('\nDay 11 ZERO-TOOL trace: ' + json.dumps(result['trace']), flush=True)

    def test_invalid_tools_and_arguments_never_execute(self):
        invalid = [decision('invented_tool', 'anything'), decision('decide', 'confirm'),
                   decision('read_policy'), decision('read_policy', 123),
                   {**decision('read_policy', 'policy?'), 'action_input': {'question': 'policy?', 'code': 'exec()'}},
                   decision('simulate_leave', 'Request leave from 2026-11-02 to 2026-11-04')]
        for bad in invalid:
            with self.subTest(decision=bad):
                tools = Mock(spec=LocalTools)
                model = ScriptedModel([bad, decision('final', answer='I cannot execute that request.')])
                result = ReActAgent(client=model, tools=tools).run('What is the leave policy?')
                self.assertEqual(tools.mock_calls, [])
                self.assertIn('error', result['trace'][0].get('observation', result['trace'][1].get('observation', {})))
                self.assertIn('error', json.loads(model.messages[1][-1]['content'])['observation'])

    def test_five_turn_limit_returns_trace_without_sixth_model_call(self):
        tools = Mock(spec=LocalTools)
        tools.read_policy.return_value = {'answer': 'Insufficient evidence.'}
        model = ScriptedModel([decision('read_policy', 'policy?')] * MAX_ITERATIONS)
        result = ReActAgent(client=model, tools=tools).run('What is the policy?')
        self.assertEqual(len(model.messages), 5)
        self.assertEqual(tools.read_policy.call_count, 5)
        self.assertEqual(result['stop_reason'], 'iteration_limit')
        self.assertEqual(result['answer'], result['trace'][-1]['final_answer'])

    def test_simulation_without_human_callback_stays_pending(self):
        question = 'Request annual leave from 2026-11-02 to 2026-11-04'
        tools = LocalTools(Mock(), log_path=None)
        model = ScriptedModel([decision('simulate_leave', question), decision('final', answer='Awaiting confirmation.')])
        result = ReActAgent(client=model, tools=tools).run(question)
        self.assertEqual(result['trace'][1]['observation']['status'], 'pending')
        self.assertEqual(next(iter(tools.records.values()))['status'], 'pending')

    def test_only_human_callback_can_confirm_or_cancel(self):
        question = 'Request annual leave from 2026-11-02 to 2026-11-04'
        for human_decision, status in [('confirm', 'executed'), ('cancel', 'cancelled'), ('yes', 'pending')]:
            with self.subTest(human_decision=human_decision):
                tools = LocalTools(Mock(), log_path=None)
                model = ScriptedModel([decision('simulate_leave', question), decision('final', answer='See action result.')])
                human = Mock(return_value=human_decision)
                result = ReActAgent(client=model, tools=tools, confirm=human).run(question)
                self.assertEqual(human.call_args.args[0]['status'], 'pending')
                self.assertEqual(next(iter(tools.records.values()))['status'], status)
                observation = result['trace'][1]['observation']
                if status == 'pending':
                    self.assertIn('error', observation)
                else:
                    self.assertEqual(observation['status'], status)
                    self.assertTrue(observation['simulated'])


if __name__ == '__main__':
    unittest.main()
