"""Regression: production batch dispatch actually reaches the configured HTTP model."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import server
from playground.research.batch_experiments import BatchManager
from playground.research.batch_model_profiles import register, validate_profiles, clear
from playground.common.model_services import get
from playground.interview_agent.llm.openai_compat import OpenAICompatClient


class CaptureLLM(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        content = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        self.server.requests.append((self.path, content, self.headers.get('Authorization')))
        reply = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': '模型返回的实际内容'}}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


class APITransportRegression(unittest.TestCase):
    def test_production_runner_profile_reaches_http_and_continues(self):
        http = ThreadingHTTPServer(('127.0.0.1', 0), CaptureLLM)
        http.requests = []
        background = threading.Thread(target=http.serve_forever, daemon=True)
        background.start()
        endpoint = f'http://127.0.0.1:{http.server_port}/v1'
        profiles = {
            role: {'provider': 'shubi', 'model': 'test-model', 'base_url': endpoint, 'api_key': 'test-only-key'}
            for role in ('interviewer', 'respondent')
        }
        public, private = validate_profiles(profiles)
        invocation_boundaries = []

        def real_transport(batch, candidate, turns, stop):
            invocation_boundaries.append(len(turns))
            cfg = get('shubi', 'interviewer')
            message = OpenAICompatClient(cfg, retries=1).chat([
                {'role': 'user', 'content': 'Generate interview question'}])
            turns.append({'question': message, 'answer': '受访者回答'})
            batch['_report_progress']('turn_saved', turn_count=len(turns),
                                      checkpoint={'turn': len(turns)})
            if len(turns) == 1:
                stop.wait(5)
                return {'status': 'incomplete'}
            return {'status': 'completed'}

        try:
            with tempfile.TemporaryDirectory() as folder:
                with patch.object(server, '_run_batch_interview_scoped', side_effect=real_transport):
                    mgr = BatchManager(folder, server._run_batch_interview, isolate_calls=False)
                    result = mgr.start('dataset', 'sha', [{'candidate_id': 'person', 'scenario_eligible': True}],
                                       {'model_profiles': public},
                                       before_launch=lambda job: register(job, public, private))
                    job = result['experiment_id']
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and len(http.requests) < 1:
                        time.sleep(.02)
                    self.assertEqual(len(http.requests), 1, 'Real model HTTP request never reached endpoint')
                    while time.monotonic() < deadline and mgr.status(job)['interviews'][0]['turn_count'] != 1:
                        time.sleep(.02)
                    self.assertEqual(mgr.status(job)['interviews'][0]['turn_count'], 1)
                    mgr.stop(job)
                    while time.monotonic() < deadline and mgr.status(job)['state'] != 'interrupted':
                        time.sleep(.02)
                    mgr.resume(job, 'sha')
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and mgr.status(job)['state'] != 'completed':
                        time.sleep(.02)
                    self.assertEqual(mgr.status(job)['state'], 'completed')
                    self.assertEqual(invocation_boundaries, [0, 1])
                    self.assertEqual(len(http.requests), 2)
                    self.assertTrue(all(p == '/v1/chat/completions' for p, _, _ in http.requests))
                    self.assertTrue(all(a == 'Bearer test-only-key' for _, _, a in http.requests))
                    self.assertEqual(mgr._load_partial(job, 1)['turn_count'], 2)
                    clear([job])
        finally:
            http.shutdown()
            http.server_close()


if __name__ == '__main__':
    unittest.main()
