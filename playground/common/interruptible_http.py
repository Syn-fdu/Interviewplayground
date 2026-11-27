"""Best-effort transport interruption for batch-only urllib model requests.

Unlike cancelling a browser fetch, abort_event attempts to shutdown the actual
HTTP socket, unblocking a thread waiting for a provider response. A DNS lookup
or a third-party library that does not use interruptible_urlopen cannot be
forcibly stopped; the batch manager reports those calls as still active.
"""
from __future__ import annotations

import contextlib
import contextvars
import http.client
import socket
import threading
import urllib.request
import weakref

_STOP = contextvars.ContextVar("batch_stop_event", default=None)
_LOCK = threading.RLock()
_CONNECTIONS = weakref.WeakKeyDictionary()


class InterruptedModelCall(RuntimeError):
    pass


def _register(event, sock):
    if sock is None:
        return
    with _LOCK:
        if event.is_set():
            _shutdown(sock)
            raise InterruptedModelCall('Batch model request interrupted')
        _CONNECTIONS.setdefault(event, set()).add(sock)


def _shutdown(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def abort_event(event):
    """Close active model HTTP sockets; never pretend this kills other code."""
    with _LOCK:
        sockets = list(_CONNECTIONS.pop(event, ()))
    for sock in sockets:
        _shutdown(sock)
    return len(sockets)


@contextlib.contextmanager
def model_call_scope(event):
    previous = _STOP.get()
    token = _STOP.set(event)
    try:
        if event.is_set():
            raise InterruptedModelCall('Batch model request interrupted')
        yield
    finally:
        _STOP.reset(token)
        # Releasing references after a normal completion avoids retained sockets.
        # A later abort is harmless for a completed worker.
        if previous is None:
            with _LOCK:
                _CONNECTIONS.pop(event, None)



class _CancelAwareResponse:
    """Preserve urllib response behavior and reject a response after Stop.

    On Windows a thread blocked inside socket/SSL recv may not wake immediately
    when another thread calls shutdown().  The returned data must nevertheless
    never be treated as a valid model answer after the stop event was set.
    """
    def __init__(self, response, event):
        self._response = response
        self._event = event

    def _check(self):
        if self._event.is_set():
            self._response.close()
            raise InterruptedModelCall('Batch model response arrived after interruption')

    def read(self, *args, **kwargs):
        self._check()
        try:
            data = self._response.read(*args, **kwargs)
        except (OSError, ValueError) as exc:
            if self._event.is_set():
                raise InterruptedModelCall('Batch model request interrupted') from exc
            raise
        self._check()
        return data

    def readline(self, *args, **kwargs):
        self._check()
        try:
            data = self._response.readline(*args, **kwargs)
        except (OSError, ValueError) as exc:
            if self._event.is_set():
                raise InterruptedModelCall('Batch model request interrupted') from exc
            raise
        self._check()
        return data

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        return self._response.close()

    def __getattr__(self, name):
        return getattr(self._response, name)

def interruptible_urlopen(request, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
    """Use a socket-tracked urllib opener only within a batch model scope."""
    event = _STOP.get()
    if event is None:
        return urllib.request.urlopen(request, timeout=timeout, **kwargs)
    if event.is_set():
        raise InterruptedModelCall('Batch model request interrupted')

    class TrackedHTTPConnection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            _register(event, self.sock)

    class TrackedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            _register(event, self.sock)

    class HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(TrackedHTTPConnection, req)

    class HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(TrackedHTTPSConnection, req, context=self._context)

    opener = urllib.request.build_opener(HTTPHandler(), HTTPSHandler())
    try:
        response = opener.open(request, timeout=timeout)
        if event.is_set():
            response.close()
            raise InterruptedModelCall('Batch model request interrupted')
        return _CancelAwareResponse(response, event)
    except (OSError, ValueError) as exc:
        if event.is_set():
            raise InterruptedModelCall('Batch model request interrupted') from exc
        raise


def cancellable_sleep(seconds):
    """Wait for retry backoff, waking immediately when a batch is cancelled."""
    event = _STOP.get()
    if event is None:
        import time
        time.sleep(seconds)
    elif event.wait(max(0, seconds)):
        raise InterruptedModelCall('Batch model request interrupted during retry delay')


def check_cancelled():
    event = _STOP.get()
    if event is not None and event.is_set():
        raise InterruptedModelCall('Batch model request interrupted')
