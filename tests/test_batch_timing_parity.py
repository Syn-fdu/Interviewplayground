"""Batch timing sidecar survives failure/resume/export without changing history."""
import io
import json
import tempfile
import time
import unittest
import zipfile
from playground.research.batch_experiments import BatchManager


class BatchTimingParity(unittest.TestCase):
    def test_timings_survive_resume_and_keep_turns_plain(self):
        with tempfile.TemporaryDirectory() as directory:
            def runner(batch, cid, turns, stop):
                idx = len(turns)
                turns.append({'question': f'Q{idx}', 'answer': f'A{idx}'})
                batch['_report_progress'](
                    'turn_saved', turn_count=len(turns),
                    last_turn_seconds=6.0 + idx,
                    checkpoint={'turn': len(turns)},
                    artifact={'timing': {
                        'interviewer_seconds': 1.5 + idx,
                        'respondent_seconds': 4.2,
                        'interviewer_restored': False,
                        'respondent_retrieval_ms': 1200,
                        'answer_rewrite_count': 2}})
                if not idx:
                    raise RuntimeError('test resume after one committed round')
                return {'status': 'completed'}
            manager = BatchManager(directory, runner)
            job = manager.start('ds', 'fp', [{'candidate_id':'one',
                'scenario_eligible':True}], {'max_turns':3})['experiment_id']
            def wait():
                for _ in range(300):
                    status = manager.status(job)
                    if status['state'] not in ('running','stopping'):
                        return status
                    time.sleep(.01)
                self.fail('worker timed out')
            failed = wait()
            self.assertEqual(failed['state'],'finished_with_errors')
            self.assertEqual(failed['interviews'][0]['last_interviewer_seconds'], 1.5)
            manager.resume(job, 'fp')
            completed = wait()
            self.assertEqual(completed['state'], 'completed')
            self.assertEqual(completed['progress']['last_respondent_seconds'], 4.2)
            self.assertEqual(completed['interviews'][0]['last_interviewer_seconds'],2.5)
            with zipfile.ZipFile(io.BytesIO(manager.export(job))) as zipped:
                doc = json.loads(zipped.read('results/interview_0001.json'))
            self.assertEqual(doc['turns'], [
                {'question':'Q0','answer':'A0'},
                {'question':'Q1','answer':'A1'}])
            self.assertEqual(doc['turn_timings'][0]['interviewer_seconds'],1.5)
            self.assertEqual(doc['turn_timings'][1]['respondent_retrieval_ms'],1200)
            self.assertEqual(len(doc['turn_timings']),len(doc['turns']))


if __name__ == '__main__':
    unittest.main()
