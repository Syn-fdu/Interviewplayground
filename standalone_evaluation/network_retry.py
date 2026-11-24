"""A single retry for transport instability only; never retry model/schema/content failures."""
import errno, socket, time
from urllib.error import HTTPError, URLError

# No retry for 400/401/403/404/422/429, JSON, schema or policy failures.
RETRY_HTTP = {408, 502, 503, 504}
RETRY_ERRNOS = {10060, 10054, 10053, 10061, 110, 104, 103, 101, 111, 32}

def is_network_instability(exc):
    if isinstance(exc, HTTPError):
        return exc.code in RETRY_HTTP
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionResetError,
                        ConnectionAbortedError, ConnectionRefusedError, BrokenPipeError)):
        return True
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            return is_network_instability(reason)
        reason = str(reason).lower()
        return any(term in reason for term in (
            'timed out', 'timeout', 'connection reset', 'connection aborted',
            'connection refused', 'temporarily unavailable', 'winerror 10060',
            'winerror 10054', 'network is unreachable'))
    if isinstance(exc, OSError):
        return getattr(exc, 'winerror', None) in RETRY_ERRNOS or exc.errno in RETRY_ERRNOS
    return False

class NetworkInstabilityError(OSError):
    """Network failed after exactly one automatic retry."""

def perform_with_one_network_retry(action, notify=None, backoff=2):
    for attempt in range(2):
        try:
            return action()
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            if not is_network_instability(exc):
                raise
            if notify:
                notify('network_instability_detected', attempt=attempt + 1,
                       max_attempts=2, reason=str(exc)[:180])
            if attempt == 1:
                raise NetworkInstabilityError(
                    '网络不稳定：同一请求已自动重试1次，仍无法连接 API；'
                    '已保存成功的历史评分，请点击「继续机评」恢复。'
                    ' 最后错误：' + str(exc)) from exc
            if notify:
                notify('network_retry_scheduled', attempt=2, max_attempts=2)
            time.sleep(backoff)


def request_with_one_network_retry(opener, request, timeout, notify=None, backoff=2):
    return perform_with_one_network_retry(lambda: opener(request, timeout=timeout), notify, backoff)
