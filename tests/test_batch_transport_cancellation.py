import os
"""Tests that batch Stop can break an in-flight urllib model socket."""
import http.server
import tempfile
import threading
import time
import unittest
import urllib.request

from playground.common.interruptible_http import (
    InterruptedModelCall, abort_event, interruptible_urlopen, model_call_scope,
)


class SlowHandler(http.server.BaseHTTPRequestHandler):
    entered = None
    release = None

    def do_GET(self):
        type(self).entered.set()
        type(self).release.wait(5)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        except (ConnectionError, OSError):
            pass

    def log_message(self, *args):
        pass


class BatchTransportCancellationTests(unittest.TestCase):
    def test_socket_shutdown_interrupts_blocked_model_read(self):
        SlowHandler.entered = threading.Event()
        SlowHandler.release = threading.Event()
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), SlowHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cancel = threading.Event()
        finished = threading.Event()
        errors = []

        def model_request():
            try:
                with model_call_scope(cancel):
                    with interruptible_urlopen('http://127.0.0.1:%d/' % server.server_port, timeout=10) as r:
                        r.read()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=model_request, daemon=True)
        worker.start()
        try:
            self.assertTrue(SlowHandler.entered.wait(2))
            cancel.set()
            self.assertGreaterEqual(abort_event(cancel), 1)
            if os.name == 'nt' and not finished.wait(2):
                # Windows may not interrupt a socket recv from another thread.
                # Let the fixture respond; _CancelAwareResponse must still
                # reject the late reply as cancelled, never deliver stale data.
                SlowHandler.release.set()
            self.assertTrue(finished.wait(4), 'Cancelled request did not finish')
            self.assertTrue(errors, 'A cancelled network operation must not succeed')
        finally:
            SlowHandler.release.set()
            server.shutdown()
            server.server_close()
            worker.join(2)

    def test_delayed_model_reply_is_rejected_even_if_socket_did_not_abort(self):
        # Windows socket cancellation is best-effort. Correctness additionally
        # requires that a late answer cannot escape to the interviewer state.
        from playground.common.interruptible_http import _CancelAwareResponse
        class LateReply:
            def __init__(self):
                self.closed = False
            def read(self):
                cancel.set()
                return b'late answer'
            def close(self):
                self.closed = True
        cancel = threading.Event()
        fake = LateReply()
        with self.assertRaises(InterruptedModelCall):
            with _CancelAwareResponse(fake, cancel) as reply:
                reply.read()
        self.assertTrue(fake.closed)

    def test_no_new_requests_after_stop(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedModelCall):
            with model_call_scope(cancel):
                interruptible_urlopen('http://127.0.0.1:11434/')


if __name__ == '__main__':
    unittest.main()
