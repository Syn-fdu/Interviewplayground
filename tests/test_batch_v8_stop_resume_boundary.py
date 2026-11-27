"""End-to-end batch manager stop/resume with no external LLM or API credentials."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from playground.research.batch_experiments import BatchManager
from playground.common.interruptible_http import model_call_scope, cancellable_sleep


class StopResumeBoundaryV8(unittest.TestCase):
    def test_fast_stop_durable_turn_resume_and_no_old_write(self):
        first_saved = threading.Event()
        calls = []
        def fake_llm(batch, candidate, turns, cancelled):
            calls.append((candidate, len(turns)))
            if candidate == 'first' and len(turns) == 0:
                turns.append({'question': 'first question', 'answer': 'first answer'})
                batch['_report_progress']('turn_saved', turn_count=1, checkpoint={'turn': 1})
                first_saved.set()
                cancellable_sleep(60)
            elif candidate == 'first':
                assert len(turns) == 1 and batch['_checkpoint'] == {'turn': 1}
                turns.append({'question': 'second question', 'answer': 'second answer'})
                batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'turn': 2})
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as tmp:
            manager = BatchManager(tmp, fake_llm)
            job = manager.start('dataset', 'fingerprint', [
                {'candidate_id': 'first', 'scenario_eligible': True},
                {'candidate_id': 'second', 'scenario_eligible': True}], {})['experiment_id']
            self.assertTrue(first_saved.wait(3))
            ts = time.monotonic()
            stopped = manager.stop(job)
            self.assertLess(time.monotonic() - ts, 0.5)
            self.assertEqual(stopped['interviews'][0]['turn_count'], 1)
            self.assertEqual(manager.status(job)['active_provider_calls'], 0)
            self.assertEqual(manager.resume(job, 'fingerprint')['progress']['resume_from_turn'], 1)
            for _ in range(200):
                if manager.status(job)['state'] == 'completed':
                    break
                time.sleep(0.01)
            self.assertEqual(manager.status(job)['state'], 'completed')
            self.assertEqual(calls, [('first', 0), ('first', 1), ('second', 0)])
            disk = json.loads((Path(tmp)/job/'results'/'interview_0001.json').read_text(encoding="utf-8"))
            self.assertEqual(disk['turn_count'], 2)
            self.assertEqual(disk['checkpoint'], {'turn': 2})
            self.assertEqual([r['question'] for r in disk['turns']], ['first question', 'second question'])

if __name__ == '__main__':
    unittest.main()
