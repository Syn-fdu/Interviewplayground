"""Offline batch contract tests: no API keys, no model traffic, no source fixtures."""
import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from playground.research.batch_experiments import BatchManager, BatchError, transcript_text


class BatchExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _wait(self, manager, job):
        for _ in range(200):
            record = manager.status(job)
            if record['state'] not in ('running', 'stopping'):
                return record
            time.sleep(.01)
        self.fail('Batch worker timed out')

    def test_export_and_resumption_does_not_rerun_success(self):
        attempts = {}
        def runner(batch, cid, turns, stop):
            attempts[cid] = attempts.get(cid, 0) + 1
            if cid == 'a':
                turns.append({'question':'Hello?', 'answer':'Yes'})
                return {'status':'completed', 'closing_message':'Thank you.'}
            if attempts[cid] == 1:
                turns.append({'question':'Question?', 'answer':'Partial answer'})
                batch['_report_progress']('turn_saved',turn_count=len(turns),checkpoint={'test':'state'})
                raise RuntimeError('Synthetic failure')
            self.assertEqual(len(turns), 1)
            turns.append({'question':'Continued question?', 'answer':'Recovered'})
            batch['_report_progress']('turn_saved',turn_count=len(turns),checkpoint={'test':'state 2'})
            return {'status':'completed', 'closing_message':'Finished.'}
        manager = BatchManager(self.temp.name, runner)
        batch = manager.start('dataset','checksum',[{'candidate_id':'a','scenario_eligible':True},
                                                    {'candidate_id':'b','scenario_eligible':True},
                                                    {'candidate_id':'c','scenario_eligible':False}],{'max_turns':4})
        job = batch['experiment_id']
        state = self._wait(manager, job)
        self.assertEqual(state['state'],'finished_with_errors')
        self.assertEqual(state['counts']['completed'],1)
        self.assertEqual(state['counts']['failed'],1)
        self.assertEqual(state['counts']['skipped'],1)
        with zipfile.ZipFile(io.BytesIO(manager.export(job))) as archive:
            self.assertIn('transcripts/interview_0001.txt',archive.namelist())
            self.assertIn('results/interview_0002.json',archive.namelist())
            self.assertEqual(json.loads(archive.read('results/interview_0002.json'))['status'],'failed')
            self.assertIn('Partial answer',archive.read('transcripts/interview_0002.txt').decode())
        with self.assertRaises(BatchError): manager.resume(job,'wrong checksum')
        manager.resume(job,'checksum')
        state = self._wait(manager,job)
        self.assertEqual(state['state'],'completed')
        self.assertEqual(attempts,{'a':1,'b':2})
        self.assertEqual(state['interviews'][1]['turn_count'],2)
        self.assertEqual(state['counts']['completed'],2)
        with zipfile.ZipFile(io.BytesIO(manager.export(job))) as archive:
            self.assertIn('Recovered',archive.read('transcripts/interview_0002.txt').decode())

    def test_extended_interview_configuration_survives_export(self):
        settings = {'max_turns': 3, 'interviewer_timeout_seconds': 300,
                    'respondent_timeout_seconds': 450, 'retrieval_timeout_seconds': 360,
                    'preparation_mode': 'python_binary',
                    'behavioral_traits': {'knowledge': .9, 'understanding': .8,
                                          'reflexivity': .7, 'memory': .6,
                                          'verbosity': .5, 'disclosure': .4}}
        manager = BatchManager(self.temp.name, lambda *args: {'status':'completed'})
        job = manager.start('dataset','checksum',
                            [{'candidate_id':'a','scenario_eligible':True}], settings)['experiment_id']
        self._wait(manager, job)
        with zipfile.ZipFile(io.BytesIO(manager.export(job))) as archive:
            exported = json.loads(archive.read('experiment.json'))
            self.assertEqual(exported['options'], settings)

    def test_batch_interface_sends_editable_timeouts_and_traits(self):
        js = (Path(__file__).resolve().parents[1] / 'assets' / 'batch_experiments.js').read_text(encoding="utf-8")
        for name in ('batchInterviewerTimeout', 'batchRespondentTimeout',
                     'batchRetrievalTimeout', 'behavioral_traits:batchTraits()'):
            self.assertIn(name, js)
        self.assertNotIn('interviewer_timeout_seconds:105,', js)

    def test_stop_is_immediate_and_discards_late_model_output(self):
        import threading
        entered = threading.Event()
        release = threading.Event()
        attempts = []
        def blocked_runner(batch, cid, turns, stop):
            attempts.append(cid)
            turns.append({'question':'saved question','answer':'saved response'})
            batch['_report_progress']('turn_saved', turn_count=1, last_turn_seconds=.25, checkpoint={'saved':'state'})
            entered.set()
            release.wait(3)
            turns.append({'question':'late question','answer':'late response'})
            return {'status':'completed'}
        manager = BatchManager(self.temp.name, blocked_runner)
        records = [{'candidate_id':c,'scenario_eligible':True} for c in ('a','b')]
        job = manager.start('dataset','checksum',records,{'max_turns':4})['experiment_id']
        self.assertTrue(entered.wait(2))
        started = time.monotonic()
        stopping = manager.stop(job)
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(stopping['state'], 'interrupted')
        # The interview stops immediately even when the provider request is stuck.
        self.assertEqual(manager.status(job)['state'], 'interrupted')
        self.assertEqual(manager.status(job)['active_provider_calls'], 1)
        release.set()
        result = self._wait(manager, job)
        self.assertEqual(result['state'],'interrupted')
        self.assertEqual(result['counts']['interrupted'],1)
        self.assertEqual(result['counts']['pending'],1)
        self.assertEqual(result['interviews'][0]['turn_count'],1)
        release.set()
        time.sleep(.05)
        with zipfile.ZipFile(io.BytesIO(manager.export(job))) as archive:
            doc=json.loads(archive.read('results/interview_0001.json'))
            self.assertEqual(doc['status'],'interrupted')
            self.assertEqual(len(doc['turns']),1)
            self.assertNotIn('late response',archive.read('transcripts/interview_0001.txt').decode())
        self.assertEqual(attempts,['a'])

    def test_progress_fields_are_visible(self):
        def runner(batch, cid, turns, stop):
            batch['_report_progress']('interviewer_preparation')
            turns.append({'question':'Q','answer':'A'})
            batch['_report_progress']('turn_saved',turn_count=1,last_turn_seconds=1.25,checkpoint={'saved':'state'})
            return {'status':'completed'}
        manager = BatchManager(self.temp.name,runner)
        job=manager.start('ds','fp',[{'candidate_id':'a','scenario_eligible':True}],{})['experiment_id']
        result=self._wait(manager,job)
        self.assertEqual(result['interviews'][0]['last_turn_seconds'],1.25)
        self.assertEqual(result['progress']['last_turn_seconds'],1.25)
        self.assertIsNotNone(result['progress']['last_activity_at'])


    def test_continue_keeps_turn_count_after_interruption_and_delete_clears_only_selected(self):
        import threading
        begun=threading.Event()
        release=threading.Event()
        def runner(batch,cid,turns,stop):
            if not turns:
                turns.append({'question':'First','answer':'Answer one'})
                batch['_report_progress']('turn_saved',turn_count=1,checkpoint={'phase':1})
                begun.set()
                release.wait(2)
                return {'status':'incomplete'}
            self.assertEqual(len(turns),1)
            self.assertEqual(batch['_checkpoint'], {'phase':1})
            turns.append({'question':'Second','answer':'Answer two'})
            batch['_report_progress']('turn_saved',turn_count=2,checkpoint={'phase':2})
            return {'status':'completed'}
        m=BatchManager(self.temp.name,runner)
        job=m.start('ds','fp',[{'candidate_id':'x','scenario_eligible':True}],{'max_turns':3})['experiment_id']
        self.assertTrue(begun.wait(2))
        m.stop(job)
        old=self._wait(m,job)
        self.assertEqual(old['interviews'][0]['turn_count'],1)
        self.assertEqual(old['counts']['interrupted'],1)
        with zipfile.ZipFile(io.BytesIO(m.export(job))) as z:
            self.assertIn('Answer one',z.read('transcripts/interview_0001.txt').decode())
        release.set()
        m.resume(job,'fp')
        done=self._wait(m,job)
        self.assertEqual(done['interviews'][0]['turn_count'],2)
        with zipfile.ZipFile(io.BytesIO(m.export(job))) as z:
            transcript=z.read('transcripts/interview_0001.txt').decode()
            self.assertEqual(transcript.count('[Turn '),2)
            self.assertIn('Answer one',transcript)
            self.assertIn('Answer two',transcript)
        with self.assertRaises(BatchError):m.prune([job,'../nope'])
        self.assertTrue(Path(self.temp.name,job,'experiment.json').exists())
        m.prune([job])
        self.assertFalse(Path(self.temp.name,job).exists())

    def test_legacy_partial_requires_native_checkpoint(self):
        def runner(batch,cid,turns,stop):
            turns.append({'question':'Old','answer':'Response'})
            return {'status':'incomplete'}
        m=BatchManager(self.temp.name,runner)
        job=m.start('ds','fp',[{'candidate_id':'x','scenario_eligible':True}],{'max_turns':3})['experiment_id']
        self._wait(m,job)
        with self.assertRaisesRegex(BatchError,'no native checkpoint'):
            m.resume(job,'fp')

    def test_live_export_includes_saved_turn_before_interrupt(self):
        import threading
        ready=threading.Event()
        finish=threading.Event()
        def runner(batch,cid,turns,stop):
            turns.append({'question':'Question','answer':'Already answered'})
            batch['_report_progress']('turn_saved',turn_count=1,checkpoint={'saved':True})
            ready.set()
            finish.wait(2)
            return {'status':'incomplete'}
        m=BatchManager(self.temp.name,runner)
        job=m.start('ds','fp',[{'candidate_id':'x','scenario_eligible':True}],{})['experiment_id']
        self.assertTrue(ready.wait(2))
        with zipfile.ZipFile(io.BytesIO(m.export(job))) as z:
            self.assertIn('Already answered',z.read('transcripts/interview_0001.txt').decode())
        m.stop(job)
        self._wait(m,job)
        finish.set()

    def test_native_interviewer_state_can_be_serialized_and_rehydrated(self):
        import threading
        from playground.interview_agent.outline.models import Outline, Topic, Question, Slot
        from playground.interview_agent.state import InterviewState, Decision
        from playground.research.group_interviewer_bridge import (
            _LOCK, _SESSIONS, export_checkpoint, _restore_checkpoint)
        outline=Outline('test', topics=[Topic('t','topic', [
            Question('q','Original question','t',required_slots=[Slot('s','detail')])])])
        state=InterviewState(outline)
        state.start_turn(Decision('questioning',question_id='q')).utterance='Original question'
        session=('shubi','session123456','scenario','fingerprint',1)
        entry={'agent':type('Dummy',(),{'state':state})(), 'history_length':0,
               'history':[], 'last_result':{'question':'Original question'},
               'language':'zh', 'lock':threading.RLock()}
        try:
            with _LOCK: _SESSIONS[session]=entry
            checkpoint=json.loads(json.dumps(export_checkpoint(*session)))
            recovered=InterviewState(outline)
            replacement={'agent':type('Dummy',(),{'state':recovered})(), 'history_length':-1,
                         'history':[], 'language':'zh'}
            _restore_checkpoint(replacement, checkpoint,
                [{'question':'Original question','answer':'Real answer'}],
                'shubi','fingerprint',1,'zh')
            self.assertEqual(len(recovered.turns),1)
            self.assertEqual(recovered.turns[0].utterance,'Original question')
            self.assertEqual(replacement['history_length'],0)
            with self.assertRaises(Exception):
                _restore_checkpoint(replacement, checkpoint,
                    [{'question':'Changed question','answer':'Real answer'}],
                    'shubi','fingerprint',1,'zh')
        finally:
            with _LOCK: _SESSIONS.pop(session,None)

    def test_cannot_delete_running_experiment(self):
        import threading
        ready=threading.Event()
        release=threading.Event()
        def runner(batch,cid,turns,stop):
            ready.set()
            release.wait(2)
            return {'status':'completed'}
        manager=BatchManager(self.temp.name,runner)
        job=manager.start('ds','fp',[{'candidate_id':'one','scenario_eligible':True}],{})['experiment_id']
        self.assertTrue(ready.wait(1))
        with self.assertRaisesRegex(BatchError,'running'):
            manager.delete(job)
        manager.stop(job)
        self._wait(manager,job)
        release.set()
        manager.delete(job)
        self.assertFalse(Path(self.temp.name,job).exists())

    def test_transcript_contains_real_turns_and_closing(self):
        self.assertEqual(transcript_text([],None),'')
        output=transcript_text([{'question':'问？','answer':'答。'}],'结束。')
        self.assertIn('Interviewer: 问？',output)
        self.assertIn('Respondent: 答。',output)
        self.assertIn('[Closing]',output)

    def test_bad_identifiers_cannot_traverse_filesystem(self):
        manager=BatchManager(self.temp.name,lambda *_:None)
        for value in ('../escape','x'*32,'','/tmp/a'):
            with self.assertRaises(BatchError): manager.export(value)


if __name__=='__main__': unittest.main()
