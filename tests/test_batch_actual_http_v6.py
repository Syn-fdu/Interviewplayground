import os
"""End-to-end batch manager -> actual local HTTP mock LLM -> cancel -> checkpoint -> resume."""
import contextvars
import json
import tempfile
import threading
import time
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playground.common.interruptible_http import (interruptible_urlopen, model_call_scope,
                                                    InterruptedModelCall)
from playground.research.batch_experiments import BatchManager


class LLM(BaseHTTPRequestHandler):
    calls = []
    lock = threading.Lock()
    second_started = threading.Event()

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        request = json.loads(self.rfile.read(length))
        with self.lock:
            self.calls.append(request)
        if request.get('block'):
            self.second_started.set()
            # Send response headers, then mimic a provider that stalls mid-body.
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            time.sleep(1.8)
            try:
                self.wfile.write(b'{"message":"stale"}')
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        encoded = json.dumps({'message': 'LLM answer for turn ' + str(request['turn'])}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class BatchActualHTTPV6(unittest.TestCase):
    def test_stop_resume_reuses_checkpoint_over_real_http(self):
        LLM.calls, LLM.second_started = [], threading.Event()
        server = ThreadingHTTPServer(('127.0.0.1', 0), LLM)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = 'http://127.0.0.1:%d/llm' % server.server_port
        attempts = []

        def call_model(turn, block=False):
            body = json.dumps({'turn': turn, 'block': block}).encode()
            req = urllib.request.Request(endpoint, data=body, headers={'Content-Type':'application/json'})
            with interruptible_urlopen(req, timeout=5) as response:
                return json.loads(response.read())['message']

        def run_one(batch, cid, turns, stop):
            attempts.append(len(turns))
            for turn in range(len(turns), 3):
                if stop.is_set():
                    return {'status': 'incomplete'}
                block = turn == 1 and attempts == [0]
                answer = call_model(turn, block=block)
                if stop.is_set():
                    return {'status': 'incomplete'}
                turns.append({'question': 'question %d' % turn, 'answer': answer})
                batch['_report_progress']('turn_saved', turn_count=len(turns),
                                          checkpoint={'last_completed': turn})
            return {'status': 'completed'}

        try:
            with tempfile.TemporaryDirectory() as folder:
                manager = BatchManager(folder, run_one)
                job = manager.start('dataset', 'fingerprint',
                                    [{'candidate_id':'person-1', 'scenario_eligible':True}], {})['experiment_id']
                self.assertTrue(LLM.second_started.wait(4), 'second HTTP request did not start')
                self.assertEqual(manager._load_partial(job, 1)['turns'][0]['answer'], 'LLM answer for turn 0')
                manager.stop(job)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and manager.status(job)['state'] != 'interrupted':
                    time.sleep(.02)
                self.assertEqual(manager.status(job)['state'], 'interrupted')
                if os.name != 'nt':
                    self.assertEqual(manager.status(job)['active_provider_calls'], 0)
                # On Windows an underlying socket read may remain live until
                # the provider returns.  It is still reported honestly and
                # cannot commit after this interrupted generation.
                self.assertEqual(manager._load_partial(job, 1)['checkpoint'], {'last_completed': 0})
                self.assertEqual(manager.resume(job, 'fingerprint')['progress']['resume_from_turn'], 1)
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline and manager.status(job)['state'] != 'completed':
                    time.sleep(.02)
                result = manager._load_partial(job, 1)
                self.assertEqual(manager.status(job)['state'], 'completed')
                self.assertEqual(len(result['turns']), 3)
                self.assertEqual([item['answer'] for item in result['turns']],
                                 ['LLM answer for turn 0', 'LLM answer for turn 1', 'LLM answer for turn 2'])
                self.assertEqual(attempts, [0, 1])
        finally:
            server.shutdown()
            server.server_close()

    def test_resume_after_manager_recreation_uses_durable_rounds(self):
        calls = []
        def run_one(batch, cid, turns, stop):
            calls.append(len(turns))
            if not turns:
                turns.append({'question': 'one', 'answer': 'answer one'})
                batch['_report_progress']('turn_saved', turn_count=1,
                                          checkpoint={'last_completed': 0})
                stop.wait(5)
                return {'status': 'incomplete'}
            self.assertEqual(batch['_checkpoint'], {'last_completed': 0})
            turns.append({'question': 'two', 'answer': 'answer two'})
            batch['_report_progress']('turn_saved', turn_count=2,
                                      checkpoint={'last_completed': 1})
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as root:
            first = BatchManager(root, run_one)
            job = first.start('dataset', 'fingerprint',
                              [{'candidate_id': 'same', 'scenario_eligible': True}], {})['experiment_id']
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and first.status(job)['interviews'][0]['turn_count'] < 1:
                time.sleep(.01)
            self.assertEqual(first.status(job)['interviews'][0]['turn_count'], 1)
            first.stop(job)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and first.status(job)['state'] != 'interrupted':
                time.sleep(.01)
            self.assertEqual(first.status(job)['state'], 'interrupted')
            second = BatchManager(root, run_one)
            self.assertEqual(second.resume(job, 'fingerprint')['progress']['resume_from_turn'], 1)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and second.status(job)['state'] != 'completed':
                time.sleep(.01)
            self.assertEqual(second.status(job)['state'], 'completed')
            self.assertEqual(second._load_partial(job, 1)['turn_count'], 2)
            self.assertEqual(calls, [0, 1])

    def test_worker_pool_inherits_stop_scope(self):
        stop = threading.Event()
        seen = threading.Event()
        def inner():
            from playground.common.interruptible_http import cancellable_sleep
            seen.set()
            cancellable_sleep(30)
        with model_call_scope(stop):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(contextvars.copy_context().run, inner)
                self.assertTrue(seen.wait(2))
                stop.set()
                with self.assertRaises(InterruptedModelCall):
                    future.result(timeout=2)


if __name__ == '__main__':
    unittest.main()
