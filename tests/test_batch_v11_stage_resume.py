import os
"""End-to-end no-API test of two candidates and a scenario-stage checkpoint.

Candidate one completes and yields a valid interview result. Candidate two
constructs its scenario and blocks inside respondent preparation. Resume must
not call build_from_v2 for candidate two a second time.
"""
import copy
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from playground.research.batch_experiments import BatchManager


@unittest.skipIf(os.name == "nt", "Requires POSIX fork/process-group termination; Windows runtime uses non-isolated logical cancellation")
class StageResumeIntegration(unittest.TestCase):
    def test_complete_one_interrupt_second_preparation_resume_without_scene_rebuild(self):
        import server
        from playground.research import group_interviewer_bridge as bridge
        from playground.interview_agent.outline.models import Outline

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entered = root / 'respondent_generation_blocked'
            constructed = root / 'scenario_build_events.jsonl'
            original_store = server.LocalStore()
            original_store.catalog = {
                'source_fingerprint': 'fp-stage',
                'candidates': [{'candidate_id': cid, 'dataset': 'ds', 'scenario_eligible': True}
                               for cid in ('one', 'two')],
                'source_index': {'ds': 0},
                'datasets': [{'id': 'ds', 'label': 'Dataset'}],
            }
            original_store.source_zip = b'deterministic simulated source'
            original_store.public_pool = None
            original_store.public_pool_source = None

            def scenario_builder(source, item, fingerprint, **kw):
                with constructed.open('a') as output:
                    output.write(json.dumps({'candidate_id': item['candidate_id']}) + '\n')
                return {'scenario_id': 'scene_' + item['candidate_id'],
                        'provenance': {'dataset': 'ds', 'candidate_id': item['candidate_id'],
                                       'source_fingerprint_sha256': fingerprint, 'guide_level': 'level1'},
                        'construction_mode': 'full'}

            def fake_preparation(payload):
                if payload['scenario_id'].startswith('scene_two') and not entered.exists():
                    entered.write_text('entered')
                    while True:
                        time.sleep(.05)
                return {'ok': True, 'prepared': True}

            def fake_next(request, scenario, source, revision, provider=None):
                if request['history']:
                    return {'question': '谢谢', 'action': 'conclude', 'finished': True}
                return {'question': 'Q for ' + scenario['provenance']['candidate_id'],
                        'action': 'ask', 'finished': False}

            def fake_resp(self, scenario, request, model):
                cid = scenario['provenance']['candidate_id']
                question = request['question']
                key = (scenario['scenario_id'], request['session_id'])
                state = original_store.research_sessions.get(key, {'turn': 0, 'history': []})
                state['history'].append({'question': question, 'answer': '模拟受访者回答 ' + cid})
                state['turn'] += 1
                original_store.research_sessions[key] = state
                return {'question_echo': question, 'answer': '模拟受访者回答 ' + cid}

            options = {'interviewer_provider': 'local', 'respondent_provider': 'local',
                       'preparation_mode': 'fast_hybrid', 'max_turns': 3,
                       'language': 'zh', 'interviewer_timeout_seconds': 60,
                       'respondent_timeout_seconds': 60, 'retrieval_timeout_seconds': 60}
            with patch.object(server, 'STORE', original_store), \
                 patch.object(server, 'build_from_v2', side_effect=scenario_builder), \
                 patch.object(server, 'make_research_bundle', return_value={'audit': {'revision': 1}}), \
                 patch.object(server, 'prepare_evidence_index'), \
                 patch.object(server, '_run_preparation_request', side_effect=fake_preparation), \
                 patch('playground.research.guide_level_projection.bind_batch_level'), \
                 patch('playground.research.interviewer_preparation.prepare_public_outline', return_value={'ready': True}), \
                 patch('playground.common.model_services.get', return_value=types.SimpleNamespace(model='fake-model')), \
                 patch.object(bridge, 'run_next', side_effect=fake_next), \
                 patch.object(bridge, 'export_checkpoint', return_value={
                     'format': 'native_interviewer_checkpoint_v1', 'provider': 'local',
                     'fingerprint': 'fp-stage', 'native': {'outline': {'fake': True}}}), \
                 patch.object(server.PlaygroundHandler, 'respondent_v08', side_effect=fake_resp):
                first = BatchManager(root / 'batch', server._run_batch_interview_scoped, isolate_calls=True)
                job = first.start('ds', 'fp-stage', [{'candidate_id': x, 'scenario_eligible': True}
                          for x in ('one', 'two')], options)['experiment_id']
                deadline = time.monotonic() + 8
                while not entered.exists() and time.monotonic() < deadline:
                    time.sleep(.015)
                self.assertTrue(entered.exists(), first.status(job))
                document_one = first._load_partial(job, 1)
                self.assertEqual(document_one['status'], 'completed')
                self.assertEqual(len(document_one['turns']), 1)
                self.assertTrue(document_one['checkpoint'])
                self.assertTrue(first._scenario_path(job, 2).exists())
                first.stop(job)
                deadline = time.monotonic() + 4
                while first.status(job)['active_provider_calls'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(first.status(job)['active_provider_calls'], 0)
                # A fresh manager restores from disk only, as after service restart.
                second = BatchManager(root / 'batch', server._run_batch_interview_scoped, isolate_calls=True)
                second.resume(job, 'fp-stage')
                deadline = time.monotonic() + 8
                while second.status(job)['state'] not in ('completed', 'finished_with_errors') and time.monotonic() < deadline:
                    time.sleep(.015)
                status = second.status(job)
                self.assertEqual(status['state'], 'completed', status)
                result_two = second._load_partial(job, 2)
                self.assertEqual(result_two['status'], 'completed')
                self.assertEqual(len(result_two['turns']), 1)
                events = [json.loads(x)['candidate_id'] for x in constructed.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events, ['one', 'two'], 'Resume rebuilt the interrupted scenario')
                self.assertEqual(second._load_partial(job, 1)['turns'], document_one['turns'])


if __name__ == '__main__':
    unittest.main()
