"""Model configuration. Everything is overridable via environment variables so the
backing model can be swapped without touching code.

Global:   LLM_PROVIDER (openai_compat | mock), LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
          LLM_TEMPERATURE, LLM_TIMEOUT, LLM_EXTRA_BODY (JSON), LLM_JSON_MODE (0/1)
Per role: <ROLE>_LLM_MODEL / _BASE_URL / _API_KEY / _TEMPERATURE ... where ROLE is one of
          PLANNER, REFLECTION, QUESTIONING, FOLLOWUP, RESPONSE, DISRUPTIVE, PREPROCESS
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://api.shubiaobiao.cn/v1"


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openai_compat"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    temperature: float = 0.7
    timeout: float = 120.0
    json_mode: bool = False  # send response_format=json_object (not all backends support it)
    extra_body: Dict[str, Any] = field(default_factory=dict)  # provider-specific params


def _env(name: str, role: Optional[str], default: Optional[str]) -> Optional[str]:
    if role:
        v = os.getenv(f"{role.upper()}_{name}")
        if v not in (None, ""):
            return v
    v = os.getenv(name)
    return v if v not in (None, "") else default


def load_model_config(role: Optional[str] = None) -> ModelConfig:
    base = ModelConfig()
    extra = _env("LLM_EXTRA_BODY", role, None)
    base_url = _env("LLM_BASE_URL", role, base.base_url)
    api_key = _env("LLM_API_KEY", role, "") or ""
    if not api_key and urlsplit(base_url or "").hostname == "open.bigmodel.cn":
        api_key = os.getenv("ZHIPUAI_API_KEY", "")
    return replace(
        base,
        provider=_env("LLM_PROVIDER", role, base.provider),
        model=_env("LLM_MODEL", role, base.model),
        base_url=base_url,
        api_key=api_key.strip(),
        temperature=float(_env("LLM_TEMPERATURE", role, base.temperature)),
        timeout=float(_env("LLM_TIMEOUT", role, base.timeout)),
        json_mode=_env("LLM_JSON_MODE", role, "0") in ("1", "true", "True"),
        extra_body=json.loads(extra) if extra else {},
    )
