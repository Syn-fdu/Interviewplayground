"""Offline contracts for one-build/three-level scenes and independent batch model routing."""
import copy
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from playground.common import model_services
from playground.research.guide_schema import with_validation_snapshot
from playground.research.guide_level_projection import attach_all_levels,select_attached_level
from playground.research.batch_model_profiles import validate_profiles,register,activated,clear


def sample_scene():
    guide={
        "goal":"Collect interview experience","source_kind":"reconstructed_from_transcript",
        "review_status":"needs_review",
        "research_questions":[{"id":"RQ1","text":"Experience"}],
        "topics":[{"topic_id":"T01","title":"Experience",
          "linked_research_question_ids":["RQ1"],
          "subtopics":[{"subtopic_id":"T01-S01","label":"Explore experience",
            "suggested_probes":["Tell me about your experience"],
            "probe_intents":[{"intent":"explore","description":"Background"}],
            "completion_signals":[{"signal":"Answer given"}],
            "linked_research_question_ids":["RQ1"]}]}]}
    return {"scenario_id":"test_scene","source_type":"v2_restricted_local",
      "provenance":{"dataset":"ai_developer_interviews","candidate_id":"CAND_LOCAL_TEST",
         "source_fingerprint_sha256":"f"*64},
      "public":{"interview_guide":with_validation_snapshot(guide),
         "plan_questions":["Tell me"],"topics":["Experience"]},
      "private":{"memory_bank":[{"fact_id":"E01","answer":"PRIVATE"}],
          "respondent_persona":{"name":"never show"},"facts":[{"fact_id":"E01","answer":"Private"}] }}


class AttachedLevelsTest(unittest.TestCase):
    def test_build_binds_three_levels_once_without_changing_private_memory(self):
        scenario=sample_scene()
        evidence=copy.deepcopy(scenario["private"])
        bound=attach_all_levels(scenario)
        self.assertEqual(set(bound),{"level0","level1","level2"})
        self.assertEqual(scenario["provenance"]["available_guide_levels"],
                         ["level0","level1","level2"])
        identities={level:json.dumps(info,sort_keys=True) for level,info in bound.items()}
        self.assertEqual(len(set(identities.values())),3)
        for level in ("level2","level0","level1"):
            select_attached_level(scenario,level)
            self.assertEqual(scenario["public"]["plan_questions"],
                             bound[level]["plan_questions"])
            self.assertEqual(scenario["private"],evidence)
        with self.assertRaises(ValueError):
            select_attached_level(scenario,"level3")


class IndependentBatchProfilesTest(unittest.TestCase):
    def tearDown(self):
        model_services.reset_for_tests()
        clear(["b"*32])

    def test_thread_scoped_keys_cannot_leak_into_single(self):
        model_services.set_selection("interviewer","shubi","single-model",
            "single-key-12345","https://api.shubiaobiao.cn/v1")
        public,private=validate_profiles({
            "interviewer":{"provider":"shubi","model":"batch-model",
                "base_url":"https://example.org/v1","api_key":"batch-key-12345"},
            "respondent":{"provider":"maas","model":"batch-answerer",
                "base_url":"https://example.net/v1","api_key":"respondent-key-12345"}})
        self.assertNotIn("batch-key-12345",json.dumps(public))
        self.assertNotIn("respondent-key-12345",json.dumps(public))
        register("b"*32,public,private)
        results=[]
        def run_batch():
            with activated("b"*32):
                results.append(("batch",model_services.get("shubi","interviewer").model,
                         model_services.get("shubi","interviewer").api_key))
                results.append(("batch-respondent",model_services.get("maas","respondent").model,
                         model_services.get("maas","respondent").api_key))
        thread=threading.Thread(target=run_batch)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0],("batch","batch-model","batch-key-12345"))
        self.assertEqual(results[1],("batch-respondent","batch-answerer","respondent-key-12345"))
        self.assertEqual(model_services.get("shubi","interviewer").model,"single-model")
        self.assertEqual(model_services.get("shubi","interviewer").api_key,"single-key-12345")

    def test_remote_keys_are_role_isolated(self):
        model_services.set_selection("interviewer","maas","model-a","interview-key-12345",
                                     "https://maas.qianwenaiapi.com/compatible-mode/v1")
        self.assertNotEqual(model_services.get("maas","respondent",False).api_key,
                            "interview-key-12345")

    def test_batch_does_not_allow_blank_remote_key_for_new_url(self):
        with self.assertRaisesRegex(ValueError,"API Key"):
            validate_profiles({
                "interviewer":{"provider":"shubi","model":"batch-model","base_url":"https://example.org/v1"},
                "respondent":{"provider":"local","model":"qwen3:8b"}})


class LocalSelectGuideAPITest(unittest.TestCase):
    def test_select_only_changes_server_bound_guide(self):
        from server import PlaygroundHandler,STORE
        scenario=sample_scene()
        attach_all_levels(scenario)
        sid=scenario["scenario_id"]
        original_private=copy.deepcopy(scenario["private"])
        bundle={"audit":{"revision":8},
                "interview_guide":copy.deepcopy(scenario["public"]["interview_guide"]),
                "source_scenario":{"scenario_id":sid,
                    "provenance":copy.deepcopy(scenario["provenance"])},
                "respondent":{"behavior_parameters":{
                    k:1.0 for k in ("knowledge","understanding","reflexivity",
                                     "memory","verbosity","disclosure")}}}
        with STORE.lock:
            old_scene=STORE.scenarios.get(sid)
            old_bundle=STORE.research_bundles.get(sid)
            STORE.scenarios[sid]=scenario
            STORE.research_bundles[sid]=bundle
        httpd=ThreadingHTTPServer(("127.0.0.1",0),PlaygroundHandler)
        thread=threading.Thread(target=httpd.serve_forever,daemon=True)
        thread.start()
        try:
            conn=http.client.HTTPConnection("127.0.0.1",httpd.server_port,timeout=10)
            payload=json.dumps({"scenario_id":sid,"guide_level":"level2",
                                "local_consent":True})
            conn.request("POST","/api/v3/scenario/select-guide",payload,
                {"Content-Type":"application/json","X-Playground-Local":"intake-v2"})
            reply=conn.getresponse()
            body=json.loads(reply.read())
            conn.close()
            self.assertEqual(reply.status,200,body)
            self.assertEqual(body["scenario"]["provenance"]["guide_level"],"level2")
            self.assertEqual(body["scenario"]["research_revision"],9)
            self.assertNotIn("_attached_levels",body["scenario"])
            self.assertNotIn("memory_bank",body["scenario"]["private"])
            with STORE.lock:
                self.assertEqual(STORE.scenarios[sid]["private"],original_private)
                self.assertEqual(STORE.research_bundles[sid]["audit"]["revision"],9)
            self.assertEqual(set(body["available_levels"]),{"level0","level1","level2"})
        finally:
            httpd.shutdown();httpd.server_close();thread.join(timeout=4)
            with STORE.lock:
                if old_scene is None:STORE.scenarios.pop(sid,None)
                else:STORE.scenarios[sid]=old_scene
                if old_bundle is None:STORE.research_bundles.pop(sid,None)
                else:STORE.research_bundles[sid]=old_bundle


class BatchStartHTTPTest(unittest.TestCase):
    def test_batch_start_keeps_role_keys_out_of_experiment_json(self):
        import tempfile
        import server
        from pathlib import Path
        from playground.research.batch_experiments import BatchManager
        old_catalog=server.STORE.catalog
        old_batch=server.BATCH
        with tempfile.TemporaryDirectory() as tmp:
            manager=BatchManager(Path(tmp)/"batches",lambda *args:None)
            manager._launch=lambda job_id:None  # no remote model call
            server.BATCH=manager
            with server.STORE.lock:
                server.STORE.catalog={"source_fingerprint":"a"*64,
                    "datasets":[{"id":"safe_fixture","label":"Synthetic fixture"}],
                    "candidates":[{"dataset":"safe_fixture","candidate_id":"CAND_TEST",
                                   "scenario_eligible":True}]}
            httpd=ThreadingHTTPServer(("127.0.0.1",0),server.PlaygroundHandler)
            thread=threading.Thread(target=httpd.serve_forever,daemon=True)
            thread.start()
            try:
                profiles={
                    "interviewer":{"provider":"shubi","model":"batch-only-i",
                        "base_url":"https://batch-a.example/v1","api_key":"private-batch-i-123"},
                    "respondent":{"provider":"maas","model":"batch-only-r",
                        "base_url":"https://batch-b.example/v1","api_key":"private-batch-r-456"}}
                payload={"dataset":"safe_fixture","candidate_id":"CAND_TEST",
                    "local_consent":True,"external_processing_consent":True,
                    "guide_level":"level1","guide_category":None,
                    "interviewer_provider":"shubi","respondent_provider":"maas",
                    "language":"zh","preparation_mode":"python_binary",
                    "max_turns":0,"model_profiles":profiles,
                    "interviewer_timeout_seconds":105,"respondent_timeout_seconds":300,
                    "retrieval_timeout_seconds":240}
                conn=http.client.HTTPConnection("127.0.0.1",httpd.server_port,timeout=5)
                conn.request("POST","/api/batch/start",json.dumps(payload),
                    {"Content-Type":"application/json","X-Playground-Local":"intake-v2"})
                response=conn.getresponse()
                result=json.loads(response.read())
                conn.close()
                self.assertEqual(response.status,202,result)
                batch_id=result["experiment_id"]
                saved=(Path(tmp)/"batches"/batch_id/"experiment.json").read_text("utf-8")
                self.assertNotIn("private-batch-i-123",saved)
                self.assertNotIn("private-batch-r-456",saved)
                self.assertIn("batch-only-i",saved)
                with activated(batch_id):
                    self.assertEqual(model_services.get("shubi","interviewer").api_key,
                                     "private-batch-i-123")
            finally:
                httpd.shutdown();httpd.server_close();thread.join(timeout=3)
                server.BATCH=old_batch
                with server.STORE.lock:
                    server.STORE.catalog=old_catalog
                if "batch_id" in locals():clear([batch_id])


if __name__=="__main__":unittest.main()
