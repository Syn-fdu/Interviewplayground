from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import List, Optional

from playground.common.interruptible_http import interruptible_urlopen, InterruptedModelCall, cancellable_sleep

from ..config import ModelConfig
from .base import LLMClient, LLMError, Message


class RemoteAPIError(LLMError):
    """Safe, structured provider failure; never embeds upstream response bodies or keys."""
    def __init__(self, kind: str, status: int | None = None, retryable: bool = False):
        self.kind = kind
        self.status = status
        self.retryable = retryable
        super().__init__(f"Remote API {kind}" + (f" (upstream HTTP {status})" if status else ""))


class OpenAICompatClient(LLMClient):
    """OpenAI-compatible chat API with bounded retries for transient failures."""
    RETRYABLE_HTTP = frozenset((408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524))

    def __init__(self, cfg: ModelConfig, retries: int = 3):
        self.cfg = cfg
        self.retries = max(1, min(int(retries), 3))

    def chat(self, messages: List[Message], temperature: Optional[float] = None,
             json_mode: bool = False) -> str:
        cfg = self.cfg
        body = {"model": cfg.model, "messages": messages,
                "temperature": cfg.temperature if temperature is None else temperature}
        if json_mode and cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(cfg.extra_body)
        request = urllib.request.Request(
            cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        )
        deadline=time.monotonic()+float(cfg.timeout)
        for attempt in range(self.retries):
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise RemoteAPIError("connection_or_timeout",retryable=True)
                with interruptible_urlopen(request, timeout=remaining) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RemoteAPIError("invalid_or_empty_response", retryable=False)
                return content
            except InterruptedModelCall:
                raise
            except urllib.error.HTTPError as exc:
                # Never include raw upstream error bodies: some providers echo sensitive data.
                status = int(exc.code)
                error = RemoteAPIError("upstream_unavailable" if status in self.RETRYABLE_HTTP
                                       else "authentication_or_permission" if status in (401, 403)
                                       else "model_or_request_configuration" if status in (400, 404, 422)
                                       else "upstream_http_error", status,
                                       retryable=status in self.RETRYABLE_HTTP)
                exc.close()
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                error = RemoteAPIError("connection_or_timeout", retryable=True)
            except (KeyError, ValueError, TypeError, IndexError, UnicodeError):
                error = RemoteAPIError("invalid_response_structure", retryable=False)
            except RemoteAPIError:
                raise
            if not error.retryable or attempt == self.retries - 1:
                raise error
            delay=min(2.0 * (2 ** attempt), 4.0)
            if deadline-time.monotonic()<=delay:raise error
            cancellable_sleep(delay)
        raise AssertionError("unreachable")
