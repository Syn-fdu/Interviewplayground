"""Active runtime contracts: backend startup, Fast scoring, core imports, hard interrupt."""
import json,os,pathlib,socket,subprocess,sys,time,unittest,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from playground.evaluation.strict_benchmark_integration import _fast_report_complete,_stop_process_tree
from standalone_evaluation.fast import fast_benchmark as fast

class TestRuntimeContract(unittest.TestCase):
    def test_frontend_backend_version_and_skills(self):
        import server
        self.assertEqual(server.verify_frontend_backend_version(),'1.9.4')
        self.assertTrue(server.verify_interviewer_skills())

    def test_runtime_modules_import_without_versioned_files(self):
        import server
        from playground.memory import integrated_memory, semantic_memory_compat, semantic_selector
        from playground.research import interviewer_preparation
        from playground.evaluation import spoken_truth_core
        from playground.memory import legacy_memory_access,python_binary_factbank
        self.assertTrue(callable(python_binary_factbank.classify_source))

    def test_new_fast_protocol_preserves_seven_independent_metrics(self):
        outline={'topics':[{'questions':[{'id':'Q1','text':'How do you use AI?',
           'required_slots':[{'id':'timeline','description':'When adoption started'}],
           'optional_deepening_slots':[{'id':'motivation','description':'Why adoption began'}]}]}]}
        slots=fast.classified_outline_slots(outline)
        turns=[{'round':1,'question':'When did you start using AI and why?',
                'answer':'I started coding with AI in 2022 because my team needed faster reviews.'}]
        def model(stage,prompt,task,schema):
            self.assertEqual(stage,'fast_classified_slot_all')
            self.assertIn('quality.relevance',prompt)
            return {'slot_assessments':[{
                'slot_id':slot['slot_id'],'status':'covered',
                'round':1,'quote':turns[0]['answer'],
                'reason':'The answer supplies a specific timeframe and a concrete motivation.'
                } for slot in task['allowed_slots']],
                'quality':{'relevance':2,'followup':1,'clarity':2,'leading':0,'rapport':1}}
        class Event:
            def event(self,*args,**kwargs):pass
        checked=fast.score_classified_v9(turns,slots,model,Event())
        report=fast.assemble_classified_v9({},turns,slots,checked)
        self.assertTrue(_fast_report_complete(report))
        self.assertEqual(report['metrics']['required_slot_coverage']['denominator'],1)
        self.assertEqual(report['metrics']['optional_deepening_slot_coverage']['denominator'],1)
        self.assertEqual(len(report['quality_metrics']),5)
        self.assertEqual(report['quality_metrics']['non_leading']['value'],100)

    @unittest.skipIf(os.name=='nt','POSIX process groups use a different signal contract than Windows')
    def test_ctrl_c_equivalent_terminates_current_scoring_process(self):
        proc=subprocess.Popen([sys.executable,'-S','-c',
                   'import signal,time; signal.signal(signal.SIGINT,lambda *a:None); time.sleep(30)'],
                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            time.sleep(0.15)
            self.assertIsNone(proc.poll())
            started=time.monotonic()
            _stop_process_tree(proc)
            self.assertIsNotNone(proc.poll())
            self.assertLess(time.monotonic()-started,3)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

    def test_actual_http_server_starts_from_git_checkout(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        proc=subprocess.Popen([sys.executable,'-S','server.py','--port',str(port)],
              cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
              env={**os.environ,'PYTHONNOUSERSITE':'1'})
        try:
            ok=False
            for _ in range(75):
                try:
                    with urllib.request.urlopen(
                         f'http://127.0.0.1:{port}/',timeout=1.5) as resp:
                        html=resp.read().decode('utf8')
                        ok=resp.status==200 and 'content="1.9.4"' in html
                    break
                except Exception:
                    if proc.poll() is not None:break
                    time.sleep(.07)
            self.assertTrue(ok,'Server failed to start or frontend version was inconsistent')
        finally:
            proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__': unittest.main()
