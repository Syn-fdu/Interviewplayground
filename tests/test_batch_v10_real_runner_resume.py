import os
"""No-API integration: execute actual server batch runner, stop and restore checkpoint."""
import copy
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from playground.research.batch_experiments import BatchManager


@unittest.skipIf(os.name == "nt", "Requires POSIX fork/process-group termination; Windows runtime uses non-isolated logical cancellation")
class RealRunnerResume(unittest.TestCase):
    def test_resume_uses_saved_history_in_actual_server_runner(self):
        import server
        from playground.research import group_interviewer_bridge as bridge
        import playground.common.model_services as model_services
        from playground.interview_agent.outline.models import Outline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop_on_second = root / 'block_on_second'
            original_store = server.STORE
            fake_store = server.LocalStore()
            fake_store.catalog = {
                'source_fingerprint': 'fingerprint',
                'candidates': [{'candidate_id': 'one', 'dataset': 'dataset', 'scenario_eligible': True}],
                'source_index': {'dataset': 0},
                'datasets': [{'id': 'dataset', 'label': 'Dataset'}]
            }
            fake_store.source_zip = b'fake source'
            fake_store.public_pool = None
            fake_store.public_pool_source = None
            fake_scenario = {
                'scenario_id': 'scenario',
                'provenance': {'dataset': 'dataset', 'candidate_id': 'one',
                               'source_fingerprint_sha256': 'fingerprint',
                               'guide_level': 'level1'},
                'construction_mode': 'full'
            }
            def setup_scenario(*args, **kwargs):
                return copy.deepcopy(fake_scenario)
            def fake_bundle(*args, **kwargs):
                return {'audit': {'revision': 1}}
            def fake_bind(scenario, *args):
                return None
            def fake_prepare(*args, **kwargs):
                return {'ready': True}
            def fake_resp(self, scenario, request, model):
                key = (scenario['scenario_id'], request['session_id'])
                state = fake_store.research_sessions.get(key, {'turn': 0, 'history': []})
                assert request['turn_index'] == state['turn'], (request['turn_index'], state['turn'])
                answer = 'A' + str(state['turn'] + 1)
                state['history'].append({'question': request['question'], 'answer': answer})
                state['turn'] += 1
                fake_store.research_sessions[key] = state
                return {'question_echo': request['question'], 'answer': answer}
            def fake_next(request, scenario, source, revision, provider=None):
                history = request['history']
                if len(history) == 1 and not request.get('checkpoint'):
                    # First generation: freeze AFTER turn one committed
                    stop_on_second.write_text('blocked')
                    while True:
                        time.sleep(.05)
                if len(history) == 1:
                    self.assertTrue(request.get('checkpoint'), 'Native checkpoint was not restored')
                if len(history) == 2:
                    return {'question': 'Finished', 'action': 'conclude', 'finished': True}
                return {'question': 'Q' + str(len(history) + 1), 'action': 'ask', 'finished': False}
            def fake_export(*args, **kwargs):
                return {'format': 'native_interviewer_checkpoint_v1', 'provider': 'local',
                        'fingerprint': 'fingerprint', 'native': {'outline': {'fake': True}}}
            def fake_research_prep(*args, **kwargs):
                return None
            fake_outline = types.SimpleNamespace(questions=lambda: [1], to_dict=lambda: {'fake': True})
            original_check = self.assertEqual
            self._assert_turn = original_check
            with patch.object(server, 'STORE', fake_store), \
                 patch.object(server, 'build_from_v2', side_effect=setup_scenario), \
                 patch.object(server, 'make_research_bundle', side_effect=fake_bundle), \
                 patch.object(server, 'prepare_evidence_index'), \
                 patch.object(server, '_run_preparation_request', side_effect=fake_research_prep), \
                 patch('playground.research.guide_level_projection.bind_batch_level', side_effect=fake_bind), \
                 patch('playground.research.interviewer_preparation.prepare_public_outline', side_effect=fake_prepare), \
                 patch('playground.common.model_services.get', return_value=types.SimpleNamespace(model='fake-model')), \
                 patch.object(bridge, 'run_next', side_effect=fake_next), \
                 patch.object(bridge, 'export_checkpoint', side_effect=fake_export), \
                 patch.object(Outline, 'from_dict', return_value=fake_outline), \
                 patch.object(server.PlaygroundHandler, 'respondent_v08', side_effect=fake_resp):
                options = {'interviewer_provider': 'local', 'respondent_provider': 'local',
                           'preparation_mode': 'fast_hybrid', 'max_turns': 3,
                           'language': 'zh', 'interviewer_timeout_seconds': 60,
                           'respondent_timeout_seconds': 60, 'retrieval_timeout_seconds': 60}
                manager = BatchManager(root / 'batch', server._run_batch_interview_scoped, isolate_calls=True)
                job = manager.start('dataset', 'fingerprint',
                    [{'candidate_id': 'one', 'scenario_eligible': True}], options)['experiment_id']
                deadline = time.monotonic() + 5
                while not stop_on_second.exists() and time.monotonic() < deadline:
                    time.sleep(.015)
                self.assertTrue(stop_on_second.exists(), f'Real batch runner did not reach turn two: {manager.status(job)}')
                stopped = manager.stop(job)
                self.assertEqual(stopped['interviews'][0]['turn_count'], 1)
                for _ in range(100):
                    if manager.status(job)['active_provider_calls'] == 0:
                        break
                    time.sleep(.01)
                self.assertEqual(manager.status(job)['active_provider_calls'], 0)
                saved = manager._load_partial(job, 1)
                self.assertEqual([t['question'] for t in saved['turns']], ['Q1'])
                self.assertTrue(saved.get('checkpoint'))
                # Another process/manager reads solely from disk.
                manager2 = BatchManager(root / 'batch', server._run_batch_interview_scoped,
                                        isolate_calls=True)
                self.assertEqual(manager2.resume(job, 'fingerprint')['progress']['resume_from_turn'], 1)
                deadline = time.monotonic() + 5
                while manager2.status(job)['state'] not in ('completed', 'finished_with_errors') and time.monotonic() < deadline:
                    time.sleep(.015)
                status = manager2.status(job)
                self.assertEqual(status['state'], 'completed', status)
                saved = manager2._load_partial(job, 1)
                self.assertEqual([t['question'] for t in saved['turns']], ['Q1', 'Q2'])
                self.assertEqual(saved['turn_count'], 2)

if __name__ == '__main__': unittest.main()
