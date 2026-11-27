"""V5 cancellation lifecycle and durable resume boundary regression tests."""
import tempfile
import threading
import time
import unittest

from playground.research.batch_experiments import BatchManager, BatchError
from playground.common.interruptible_http import cancellable_sleep, model_call_scope, InterruptedModelCall


class StopAndResumeV5Tests(unittest.TestCase):
    def wait_state(self, manager, job, target, seconds=5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            state = manager.status(job)
            if state['state'] == target:
                return state
            time.sleep(.01)
        self.fail(f'{target} not reached: {manager.status(job)}')

    def test_stop_during_retry_sleep_exits_and_resumes_saved_round(self):
        entered = threading.Event()
        attempts = []
        def run_one(batch, cid, turns, stop):
            attempts.append((cid, len(turns)))
            if not turns:
                turns.append({'question': 'Question one', 'answer': 'Answer one'})
                batch['_report_progress']('turn_saved', turn_count=1, checkpoint={'round': 1})
                entered.set()
                cancellable_sleep(60)
            else:
                self.assertEqual(turns, [{'question': 'Question one', 'answer': 'Answer one'}])
                self.assertEqual(batch['_checkpoint'], {'round': 1})
                turns.append({'question': 'Question two', 'answer': 'Answer two'})
                batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'round': 2})
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as root:
            manager = BatchManager(root, run_one)
            job = manager.start('dataset', 'hash', [{'candidate_id': 'same', 'scenario_eligible': True}], {})['experiment_id']
            self.assertTrue(entered.wait(2))
            started = time.monotonic()
            manager.stop(job)
            stopped = self.wait_state(manager, job, 'interrupted', 2)
            self.assertLess(time.monotonic() - started, 2)
            self.assertEqual(stopped['active_provider_calls'], 0)
            self.assertEqual(stopped['interviews'][0]['turn_count'], 1)
            self.assertEqual(manager._load_partial(job, 1)['checkpoint'], {'round': 1})
            self.assertEqual(manager.resume(job, 'hash')['progress']['resume_from_turn'], 1)
            done = self.wait_state(manager, job, 'completed')
            self.assertEqual(done['interviews'][0]['turn_count'], 2)
            self.assertEqual(attempts, [('same', 0), ('same', 1)])

    def test_uncooperative_provider_does_not_block_continue_or_write_stale_turn(self):
        entered, release, resumed = threading.Event(), threading.Event(), threading.Event()
        attempts = []
        def run_one(batch, cid, turns, stop):
            attempts.append(len(turns))
            if not turns and not entered.is_set():
                turns.append({'question': 'Q1', 'answer': 'A1'})
                batch['_report_progress']('turn_saved', turn_count=1, checkpoint={'round': 1})
                entered.set()
                release.wait(5)
                turns.append({'question': 'OLD', 'answer': 'STALE'})
                batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'round': 999})
                return {'status': 'completed'}
            self.assertEqual(len(turns), 1)
            self.assertEqual(batch['_checkpoint'], {'round': 1})
            turns.append({'question': 'Q2', 'answer': 'A2'})
            batch['_report_progress']('turn_saved', turn_count=2, checkpoint={'round': 2})
            resumed.set()
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as root:
            manager = BatchManager(root, run_one)
            job = manager.start('dataset', 'hash', [{'candidate_id': 'same', 'scenario_eligible': True}], {})['experiment_id']
            self.assertTrue(entered.wait(2))
            started = time.monotonic()
            manager.stop(job)
            self.assertLess(time.monotonic() - started, .5)
            self.assertEqual(manager.status(job)['state'], 'interrupted')
            self.assertEqual(manager.status(job)['active_provider_calls'], 1)
            self.assertEqual(manager.resume(job, 'hash')['progress']['resume_from_turn'], 1)
            self.assertTrue(resumed.wait(2))
            self.wait_state(manager, job, 'completed')
            release.set()
            time.sleep(.1)
            saved = manager._load_partial(job, 1)
            self.assertEqual(saved['turns'], [{'question':'Q1','answer':'A1'}, {'question':'Q2','answer':'A2'}])
            self.assertEqual(saved['checkpoint'], {'round':2})
            self.assertEqual(attempts, [0,1])


if __name__ == '__main__':
    unittest.main()
