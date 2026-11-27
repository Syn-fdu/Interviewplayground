import os
"""Kill a genuinely uncooperative provider process; no external API."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from playground.research.batch_experiments import BatchManager


@unittest.skipIf(os.name == "nt", "Requires POSIX fork/process-group termination; Windows runtime uses non-isolated logical cancellation")
class IsolatedBatchTermination(unittest.TestCase):
    def test_uncooperative_provider_stops_at_zero_and_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / 'provider_entered'
            def fake_provider(batch, candidate, turns, stop):
                if not turns:
                    turns.append({'question':'Q1','answer':'A1'})
                    batch['_report_progress']('turn_saved', 1, checkpoint={'turn':1})
                    state.write_text('entered')
                    # Deliberately IGNORE stop: no HTTP/API; cannot be cancelled cooperatively.
                    while True:
                        time.sleep(0.1)
                self.assertEqual(len(turns), 1)
                self.assertEqual(batch['_checkpoint'], {'turn':1})
                turns.append({'question':'Q2','answer':'A2'})
                batch['_report_progress']('turn_saved', 2, checkpoint={'turn':2})
                return {'status':'completed'}
            manager = BatchManager(Path(tmp)/'batch', fake_provider, isolate_calls=True)
            job = manager.start('dataset','fingerprint',[{'candidate_id':'a','scenario_eligible':True}],{})['experiment_id']
            for _ in range(200):
                if state.exists(): break
                time.sleep(0.02)
            self.assertTrue(state.exists(), 'provider did not start')
            started = time.monotonic()
            manager.stop(job)
            self.assertLess(time.monotonic()-started, 2)
            for _ in range(100):
                if manager.status(job)['active_provider_calls'] == 0 and not any(
                    t.is_alive() and t.name.startswith('batch-call-') for t in threading.enumerate()
                ):
                    break
                time.sleep(0.02)
            self.assertEqual(manager.status(job)['active_provider_calls'], 0)
            self.assertFalse(any(t.is_alive() and t.name.startswith('batch-call-') for t in threading.enumerate()))
            self.assertEqual(manager.resume(job,'fingerprint')['progress']['resume_from_turn'],1)
            for _ in range(200):
                if manager.status(job)['state'] == 'completed': break
                time.sleep(0.02)
            self.assertEqual(manager.status(job)['state'], 'completed')
            saved = json.loads((Path(tmp)/'batch'/job/'results'/'interview_0001.json').read_text(encoding="utf-8"))
            self.assertEqual(saved['turn_count'],2)
            self.assertEqual(saved['checkpoint'], {'turn':2})
            self.assertEqual([x['question'] for x in saved['turns']],['Q1','Q2'])
            self.assertEqual(manager.status(job)['active_provider_calls'],0)

if __name__ == '__main__': unittest.main()
