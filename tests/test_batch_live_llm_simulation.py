"""End-to-end local LLM wire tests: actual requests, cancellation and batch resume.

The local provider is deterministic, NOT a real remote LLM. It speaks the
OpenAI-compatible protocol and produces interviewer/respondent text.
"""
import http.server
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from playground.common.interruptible_http import InterruptedModelCall
from playground.interview_agent.config import ModelConfig
from playground.interview_agent.llm.openai_compat import OpenAICompatClient
from playground.research.batch_experiments import BatchManager


class SimulatedLLMHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        payload = json.loads(raw)
        self.server.requests.append(payload)
        content = payload['messages'][-1]['content']
        if 'BLOCK_RESPONSE' in content:
            self.server.block_started.set()
            self.server.release_block.wait(7)
        if 'interviewer' in content:
            answer = '你能详细介绍你最近一次使用这项服务的经历吗？'
        elif 'respondent' in content:
            answer = '上周我使用过一次，流程顺畅，但等待时间有点长。'
        else:
            answer = '收到请求。'
        body = json.dumps({'id': 'local-simulated-llm', 'choices': [
            {'message': {'role': 'assistant', 'content': answer}}]}, ensure_ascii=False).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (OSError, ConnectionError):
            pass

    def log_message(self, *args):
        pass


class LiveSimulationTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), SimulatedLLMHandler)
        self.server.requests = []
        self.server.block_started = threading.Event()
        self.server.release_block = threading.Event()
        self.background = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.background.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.release_block.set)

    def _client(self, scheme='http'):
        return OpenAICompatClient(ModelConfig(model='local-simulated-llm',
            base_url=f'{scheme}://127.0.0.1:{self.server.server_port}/v1',
            api_key='local-only', timeout=10), retries=1)

    def test_real_http_round_trip_for_interviewer_and_respondent(self):
        client = self._client()
        q = client.chat([{'role': 'user', 'content': 'interviewer: generate question'}])
        a = client.chat([{'role': 'user', 'content': 'respondent: answer question'}])
        self.assertIn('介绍', q)
        self.assertIn('上周', a)
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.server.requests[0]['model'], 'local-simulated-llm')

    def test_https_connection_really_works_with_batch_scope(self):
        if not shutil.which('openssl'):
            self.skipTest('openssl is needed for the local TLS fixture')
        with tempfile.TemporaryDirectory() as directory:
            cert = str(Path(directory, 'cert.pem'))
            key = str(Path(directory, 'key.pem'))
            created = False
            # Git for Windows/MSYS OpenSSL may convert /CN=localhost to a
            # filesystem path.  Try the standard DN first, then the MSYS form.
            subjects = ['/CN=localhost']
            if os.name == 'nt':
                subjects.append('//CN=localhost')
            for subject in subjects:
                completed = subprocess.run(
                    ['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                     '-keyout', key, '-out', cert, '-days', '1',
                     '-subj', subject],
                    check=False, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True)
                if completed.returncode == 0:
                    created = True
                    break
            if not created:
                self.skipTest('Local OpenSSL could not generate a test certificate on this system')
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
            # Local self-signed certificate is deliberately insecure, never production.
            with patch('http.client._create_https_context', return_value=ssl._create_unverified_context()):
                from playground.common.interruptible_http import model_call_scope
                with model_call_scope(threading.Event()):
                    answer = self._client('https').chat([
                        {'role': 'user', 'content': 'interviewer: one question'}])
            self.assertIn('介绍', answer)

    def test_live_batch_stop_and_resume_current_candidate(self):
        client = self._client()
        with tempfile.TemporaryDirectory() as directory:
            attempts = []
            def run_one(batch, candidate, turns, stop):
                attempts.append((candidate, len(turns)))
                if candidate == 'first' and not turns:
                    q = client.chat([{'role': 'user', 'content': 'interviewer: Q1'}])
                    a = client.chat([{'role': 'user', 'content': 'respondent: A1'}])
                    turns.append({'question': q, 'answer': a})
                    batch['_report_progress']('turn_saved', turn_count=1,
                                              checkpoint={'round': 1})
                    # Simulate an LLM currently producing the second question.
                    client.chat([{'role': 'user', 'content': 'interviewer BLOCK_RESPONSE'}])
                    return {'status': 'completed'}
                if candidate == 'first':
                    self.assertEqual(len(turns), 1)
                    self.assertEqual(batch['_checkpoint'], {'round': 1})
                    q = client.chat([{'role': 'user', 'content': 'interviewer: Q2'}])
                    a = client.chat([{'role': 'user', 'content': 'respondent: A2'}])
                    turns.append({'question': q, 'answer': a})
                    batch['_report_progress']('turn_saved', turn_count=2,
                                              checkpoint={'round': 2})
                else:
                    q = client.chat([{'role': 'user', 'content': 'interviewer: other candidate'}])
                    a = client.chat([{'role': 'user', 'content': 'respondent: other candidate'}])
                    turns.append({'question': q, 'answer': a})
                    batch['_report_progress']('turn_saved', turn_count=1,
                                              checkpoint={'round': 1})
                return {'status': 'completed'}
            manager = BatchManager(directory, run_one)
            job = manager.start('local-model-test', 'fingerprint', [
                {'candidate_id': 'first', 'scenario_eligible': True},
                {'candidate_id': 'second', 'scenario_eligible': True}], {})['experiment_id']
            self.assertTrue(self.server.block_started.wait(3), 'Local LLM was not called')
            stopping = manager.stop(job)
            self.assertEqual(stopping['state'], 'interrupted')
            self.assertGreaterEqual(stopping['progress']['cancelled_http_sockets'], 1)
            self._wait_for(manager, job, 'interrupted')
            self.assertEqual(manager._load_partial(job, 1)['checkpoint'], {'round': 1})
            manager.resume(job, 'fingerprint')
            final = self._wait_for(manager, job, 'completed')
            self.assertEqual(attempts, [('first', 0), ('first', 1), ('second', 0)])
            self.assertEqual(final['interviews'][0]['turn_count'], 2)
            self.assertEqual(manager._load_partial(job, 1)['checkpoint'], {'round': 2})
            self.assertEqual(len(manager._load_partial(job, 1)['turns']), 2)
            self.server.release_block.set()

    def _wait_for(self, manager, job, expected):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            result = manager.status(job)
            if result['state'] == expected:
                return result
            if result['state'] in ('finished_with_errors', 'completed', 'interrupted'):
                self.fail(f'Expected {expected}; got {result}')
            time.sleep(.02)
        self.fail(f'Timed out waiting for {expected}: {manager.status(job)}')
