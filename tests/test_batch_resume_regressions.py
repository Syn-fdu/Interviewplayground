"""Offline regression tests for interrupted batch resume and native bridge rehydration."""
import threading
import time
import unittest
from types import SimpleNamespace
from dataclasses import dataclass
from unittest.mock import patch

from playground.interview_agent.outline.models import Outline, Topic, Question
from playground.interview_agent.state import InterviewState, Decision
from playground.research import group_interviewer_bridge as bridge
from playground.research.batch_experiments import BatchError, BatchManager


class FakeInterviewer:
    def __init__(self, outline, llms, max_turns):
        self.state = InterviewState(outline)

    def step(self, reply):
        self.state.start_turn(Decision('questioning', question_id='q')).utterance = 'Original question'
        return 'Original question'


class BatchResumeRegressionTests(unittest.TestCase):
    def test_restores_missing_in_memory_session_from_native_checkpoint(self):
        outline = Outline('test', topics=[Topic('t', 'topic', [Question('q', 'Original question', 't')])])
        provider, sid, fp, rev = 'local', 'scene_batch', 'fingerprint', 3
        original_session = 'originalsession123'
        resumed_session = 'resumedsession123'
        provenance = {'dataset': 'example', 'candidate_id': 'one', 'source_fingerprint_sha256': fp,
                      'guide_level': 'level1'}
        registration = {'provenance': provenance}
        cache_key = (fp, 'example', 'one', rev, 'level1')
        with bridge._LOCK:
            bridge._OUTLINES[cache_key] = outline.to_dict()
        @dataclass
        class DummyConfig:
            timeout: float = 105
            model: str = 'offline-model'
        clients = SimpleNamespace(cfg=DummyConfig())
        def request(session, history, checkpoint=None):
            value = dict(protocol='native_interviewer_v1', provider=provider, scenario_id=sid,
                         session_id=session, history=history, language='zh',
                         source_fingerprint_sha256=fp, research_revision=rev)
            if checkpoint is not None:
                value['checkpoint'] = checkpoint
            return value
        try:
            with patch.object(bridge, 'Interviewer', FakeInterviewer), patch.object(
                    bridge, '_clients_for', return_value={'planner': clients}):
                first = bridge.run_next(request(original_session, []), registration, b'', rev, provider)
                self.assertEqual(first['question'], 'Original question')
                checkpoint = bridge.export_checkpoint(provider, original_session, sid, fp, rev)
                history = [{'question': first['question'], 'answer': 'Test answer'}]
                with bridge._LOCK:
                    bridge._SESSIONS.pop((provider, original_session, sid, fp, rev))
                next_question = bridge.run_next(
                    request(resumed_session, history, checkpoint), registration, b'', rev, provider)
                self.assertEqual(next_question['question'], 'Original question')
                self.assertEqual(bridge._SESSIONS[(provider, resumed_session, sid, fp, rev)]['history_length'], 1)
                with self.assertRaisesRegex(bridge.GroupBridgeError, 'no native checkpoint'):
                    bridge.run_next(request('missingsession123', history), registration, b'', rev, provider)
        finally:
            with bridge._LOCK:
                bridge._OUTLINES.pop(cache_key, None)
                for key in list(bridge._SESSIONS):
                    if key[2] == sid:
                        bridge._SESSIONS.pop(key, None)

    def test_resume_while_detached_provider_call_is_still_blocked(self):
        import tempfile
        released = threading.Event()
        entered = threading.Event()
        resumed = threading.Event()
        attempts = []
        def runner(batch, candidate, turns, stop):
            attempts.append(candidate)
            if candidate == 'a' and not turns:
                turns.append({'question': 'Q1', 'answer': 'A1'})
                batch['_report_progress']('turn_saved', turn_count=1, checkpoint={'state': 1})
                entered.set()
                released.wait(5)
                # Simulate a delayed provider return racing with the resumed job.
                turns.append({'question': 'STALE', 'answer': 'STALE'})
                batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'state': 999})
                return {'status': 'completed'}
            if candidate == 'a':
                self.assertEqual(turns, [{'question': 'Q1', 'answer': 'A1'}])
                turns.append({'question': 'Q2', 'answer': 'A2'})
                batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'state': 2})
                resumed.set()
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as directory:
            manager = BatchManager(directory, runner)
            job = manager.start('dataset', 'fp', [{'candidate_id': 'a', 'scenario_eligible': True},
                                                   {'candidate_id': 'b', 'scenario_eligible': True}], {})['experiment_id']
            self.assertTrue(entered.wait(2))
            manager.stop(job)
            self.assertEqual(manager.status(job)['state'], 'interrupted')
            self.assertTrue(any(t.is_alive() for t in manager.inflight.get(job, ())))
            # Continue is not blocked by an old uncooperative LLM request.
            self.assertEqual(manager.resume(job, 'fp')['progress']['resume_from_turn'], 1)
            self.assertTrue(resumed.wait(2))
            for _ in range(200):
                if manager.status(job)['state'] == 'completed':
                    break
                time.sleep(.01)
            self.assertEqual(manager.status(job)['state'], 'completed')
            self.assertEqual(attempts, ['a', 'a', 'b'])
            released.set()
            for thread in list(manager.inflight.get(job, ())):
                thread.join(2)
            saved = manager._load_partial(job, 1)
            self.assertEqual(saved['turns'], [{'question': 'Q1', 'answer': 'A1'},
                                              {'question': 'Q2', 'answer': 'A2'}])
            self.assertEqual(saved['checkpoint'], {'state': 2})
            self.assertEqual(manager.status(job)['interviews'][0]['turn_count'], 2)


