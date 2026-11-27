import os
"""Real BatchManager -> isolated server interview -> durable pending question -> resume.

No LLM API: interviewer emits a valid native-shape checkpoint; respondent blocks
until force-stopped; a new manager reloads the staged question from disk.
"""
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from playground.research.batch_experiments import BatchManager


@unittest.skipIf(os.name == "nt", "Requires POSIX fork/process-group termination; Windows runtime uses non-isolated logical cancellation")
class RespondentStageResume(unittest.TestCase):
    def test_interviewer_generated_respondent_blocked_then_resume_exact_stage(self):
        import server
        from playground.research import group_interviewer_bridge as bridge
        from playground.interview_agent.outline.models import Outline
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generated = root / 'generated.jsonl'
            constructed = root / 'constructed.jsonl'
            answering = root / 'respondent_entered'
            store = server.LocalStore()
            store.catalog = {'source_fingerprint': 'fp-stage',
                             'candidates': [{'candidate_id': 'one', 'dataset': 'ds', 'scenario_eligible': True}],
                             'source_index': {'ds': 0}, 'datasets': [{'id': 'ds', 'label': 'Dataset'}]}
            store.source_zip = b'synthetic dataset for isolated no-api test'
            store.public_pool = None
            store.public_pool_source = None

            def build(source, item, fp, **kw):
                with constructed.open('a') as f:
                    f.write('scene\n')
                return {'scenario_id': 'scene_one', 'provenance': {'dataset': 'ds',
                        'candidate_id': 'one', 'source_fingerprint_sha256': fp,
                        'guide_level': 'level1'}, 'construction_mode': 'full'}

            def next_question(req, scenario, source, revision, provider=None):
                with generated.open('a') as f:
                    f.write(json.dumps({'history_length': len(req['history'])})+'\n')
                if req['history']:
                    return {'question': '谢谢参与', 'finished': True, 'action': 'conclude'}
                return {'question': '请谈谈您的经历？', 'finished': False, 'action': 'ask'}

            export_log = root / 'checkpoint_exports.log'
            def export_once(*args, **kwargs):
                with export_log.open('a') as file:
                    file.write('export\n')
                if len(export_log.read_text(encoding="utf-8").splitlines()) > 1:
                    raise bridge.GroupBridgeError('Cannot checkpoint a missing interviewer session')
                return native

            native = {'format': 'native_interviewer_checkpoint_v1', 'provider': 'local',
                      'fingerprint': 'fp-stage', 'revision': 1, 'language': 'zh',
                      'history_length': 0, 'history': [], 'last_result': {'question':'请谈谈您的经历？'},
                      'native': {'outline': {'fake': True}, 'turns': [{'utterance': '请谈谈您的经历？'}]}}

            def respond(self, scenario, req, model):
                if not answering.exists():
                    answering.write_text('blocked')
                    while True:
                        time.sleep(.03)
                key = (scenario['scenario_id'], req['session_id'])
                turn = {'question': req['question'], 'answer': '模拟回复'}
                store.research_sessions[key] = {'turn': 1, 'history': [turn]}
                return {'question_echo': req['question'], 'answer': '模拟回复'}

            options = {'interviewer_provider': 'local', 'respondent_provider': 'local',
                       'preparation_mode': 'fast_hybrid', 'max_turns': 3,
                       'language': 'zh', 'interviewer_timeout_seconds': 60,
                       'respondent_timeout_seconds': 60, 'retrieval_timeout_seconds': 60}
            stub_outline = types.SimpleNamespace(questions=lambda: ['q1'], to_dict=lambda: {'fake': True})
            with patch.object(server, 'STORE', store), \
                 patch.object(server, 'build_from_v2', side_effect=build), \
                 patch.object(server, 'make_research_bundle', return_value={'audit': {'revision': 1}}), \
                 patch.object(server, 'prepare_evidence_index'), \
                 patch.object(server, '_run_preparation_request', return_value={'prepared': True}), \
                 patch('playground.research.guide_level_projection.bind_batch_level'), \
                 patch('playground.research.interviewer_preparation.prepare_public_outline', return_value={'ready': True}), \
                 patch('playground.common.model_services.get', return_value=types.SimpleNamespace(model='mock-model')), \
                 patch.object(Outline, 'from_dict', return_value=stub_outline), \
                 patch.object(bridge, 'run_next', side_effect=next_question), \
                 patch.object(bridge, 'export_checkpoint', side_effect=export_once), \
                 patch.object(server.PlaygroundHandler, 'respondent_v08', side_effect=respond):
                manager = BatchManager(root / 'batch', server._run_batch_interview_scoped, isolate_calls=True)
                job = manager.start('ds', 'fp-stage', [{'candidate_id':'one', 'scenario_eligible': True}], options)['experiment_id']
                deadline = time.monotonic() + 10
                while not answering.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(answering.exists(), manager.status(job))
                stage = manager._question_path(job, 1)
                self.assertTrue(stage.exists(), 'Interviewer question must be durably saved before respondent starts')
                question_doc = json.loads(stage.read_text(encoding="utf-8"))
                self.assertEqual(question_doc['question'], '请谈谈您的经历？')
                self.assertEqual(question_doc['turn_index'], 0)
                manager.stop(job)
                deadline = time.monotonic() + 5
                while manager.status(job)['active_provider_calls'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(manager.status(job)['active_provider_calls'], 0)
                self.assertTrue(stage.exists())
                again = BatchManager(root / 'batch', server._run_batch_interview_scoped, isolate_calls=True)
                again.resume(job, 'fp-stage')
                deadline = time.monotonic() + 10
                while again.status(job)['state'] not in ('completed', 'finished_with_errors') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(again.status(job)['state'], 'completed', again.status(job))
                result = again._load_partial(job, 1)
                self.assertEqual(result['turns'], [{'question':'请谈谈您的经历？', 'answer':'模拟回复'}])
                self.assertFalse(stage.exists(), 'Committed respondent answer must clear pending question')
                self.assertEqual(export_log.read_text(encoding="utf-8").splitlines(), ['export'], 'Resume must reuse the already saved native checkpoint')
                self.assertEqual(constructed.read_text(encoding="utf-8").splitlines(), ['scene'])
                self.assertEqual([json.loads(x)['history_length'] for x in generated.read_text(encoding="utf-8").splitlines()], [0, 1],
                                 'Question at history length 0 was regenerated after resuming')

if __name__ == '__main__':
    unittest.main()
