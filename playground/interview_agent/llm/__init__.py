from __future__ import annotations

from typing import Optional

from ..config import load_model_config
from .base import LLMClient, LLMError, parse_json
from .mock import MockLLM
from .openai_compat import OpenAICompatClient

_PROVIDERS = {"openai_compat": OpenAICompatClient, "openai": OpenAICompatClient, "zhipu": OpenAICompatClient}


def create_llm(role: Optional[str] = None) -> LLMClient:
    """Build the client for a role (planner/questioning/...) from env configuration.
    Register new backends in _PROVIDERS."""
    cfg = load_model_config(role)
    try:
        cls = _PROVIDERS[cfg.provider]
    except KeyError:
        raise LLMError(f"unknown LLM_PROVIDER {cfg.provider!r}; available: {sorted(_PROVIDERS)}")
    if not cfg.api_key:
        raise LLMError("LLM_API_KEY is not set (ZHIPUAI_API_KEY is supported for open.bigmodel.cn)")
    return cls(cfg)


__all__ = ["create_llm", "LLMClient", "LLMError", "MockLLM", "OpenAICompatClient", "parse_json"]
