"""Adversarial batch lifecycle tests added during V15 audit."""
import json
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from playground.research.batch_experiments import BatchManager, atomic_json, utc_now


class TestWorkerFailureIsDurable(unittest.TestCase):
    def test_bad_stage_does_not_leave_ghost_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            manager = BatchManager(Path(directory), lambda *args: calls.append(args))
            job_id = uuid.uuid4().hex
            root = Path(directory) / job_id
            record = {
                'experiment_id': job_id, 'dataset': 'dataset-x',
                'source_fingerprint_sha256': 'fingerprint-x', 'options': {'guide_level': 'level1'},
                'created_at': utc_now(), 'updated_at': utc_now(), 'finished_at': None,
                'state': 'running', 'progress': {'phase': 'queued'},
                'interviews': [{'candidate_id': 'participant-x', 'status': 'pending',
                                'turn_count': 0, 'error': None}]
            }
            atomic_json(root / 'experiment.json', record)
            # Metadata mismatch should fail before any model call, and MUST NOT
            # strand experiment.json in running state after worker death.
            atomic_json(root / 'stages' / 'scenario_0001.json', {
                'experiment_id': 'wrong-experiment', 'candidate_id': 'participant-x',
                'scenario': {'scenario_id': 'old'}})
            manager._launch(job_id)
            deadline = time.monotonic() + 3
            while job_id in manager.running and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertNotIn(job_id, manager.running)
            saved = json.loads((root / 'experiment.json').read_text(encoding="utf-8"))
            self.assertEqual(saved['state'], 'finished_with_errors')
            self.assertEqual(saved['interviews'][0]['status'], 'failed')
            self.assertIn('Saved scenario stage does not match', saved['interviews'][0]['error'])
            self.assertEqual(saved['interviews'][0]['turn_count'], 0)
            self.assertEqual(calls, [])
            # The user can still request resume; it should fail again with a
            # diagnostic state, not falsely report successful completion.
            manager.resume(job_id, 'fingerprint-x')
            deadline = time.monotonic() + 3
            while job_id in manager.running and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(manager.status(job_id)['state'], 'finished_with_errors')
