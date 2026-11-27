"""Persisted progress cannot silently roll back to turn zero."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from playground.research.batch_experiments import BatchManager, BatchError


class NoSilentReset(unittest.TestCase):
    def test_missing_result_file_does_not_reset_acknowledged_round(self):
        with tempfile.TemporaryDirectory() as folder:
            def provider(batch, candidate, turns, stop):
                turns.append({'question': 'Q1', 'answer': 'A1'})
                batch['_report_progress']('turn_saved', 1, checkpoint={'round': 1})
                stop.wait(4)
                return {'status': 'incomplete'}
            manager = BatchManager(folder, provider)
            job = manager.start('ds', 'fp', [{'candidate_id': 'one', 'scenario_eligible': True}], {})['experiment_id']
            deadline = time.monotonic() + 4
            while manager.status(job)['interviews'][0]['turn_count'] < 1 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(manager.status(job)['interviews'][0]['turn_count'], 1)
            manager.stop(job)
            partial = Path(folder) / job / 'results' / 'interview_0001.json'
            self.assertTrue(partial.exists())
            partial.unlink()
            restarted = BatchManager(folder, provider)
            with self.assertRaisesRegex(BatchError, 'refusing to restart at turn 0'):
                restarted.resume(job, 'fp')
            self.assertEqual(restarted.status(job)['interviews'][0]['turn_count'], 1)

if __name__ == '__main__': unittest.main()
