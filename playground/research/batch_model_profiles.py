"""In-memory, per-experiment provider credentials; never persisted to batch JSON.

Each batch participant executes under its own context-variable model routing.
Interactive model settings and existing batch snapshots cannot mutate each other.
"""
from __future__ import annotations
import threading
from playground.common.model_services import (
    PROVIDERS, _MODELS, _env_value, normalize_base_url,
    batch_config_context,
)
from playground.interview_agent.config import ModelConfig

_LOCK=threading.RLock()
_PROFILES={}


def validate_profiles(data):
    if not isinstance(data,dict) or set(data)!={"interviewer","respondent"}:
        raise ValueError("Batch requires interviewer and respondent model configurations")
    public={}
    private={}
    for role in ("interviewer","respondent"):
        entry=data[role]
        if not isinstance(entry,dict) or entry.get("provider") not in PROVIDERS:
            raise ValueError("Invalid batch "+role+" provider")
        provider=entry["provider"]
        model=entry.get("model")
        if not isinstance(model,str) or not _MODELS.fullmatch(model):
            raise ValueError("Invalid batch "+role+" model")
        if provider=="local":
            base_url="http://127.0.0.1:11434/v1"
            key="ollama"
        else:
            base_url=normalize_base_url(entry.get("base_url"))
            raw_key=entry.get("api_key") or ""
            if not isinstance(raw_key,str) or len(raw_key)>1024 or any(c in raw_key for c in "\r\n"):
                raise ValueError("Invalid batch API Key")
            _model,env_key,env_url,_source=_env_value(provider,role)
            key=raw_key or (env_key if base_url==env_url else "")
            if not key:
                raise ValueError("Batch "+role+" API Key missing. Configure it in Batch settings.")
        public[role]={"provider":provider,"model":model,"base_url":base_url}
        private[(role,provider)]=ModelConfig(provider="openai_compat",base_url=base_url,model=model,
                                             api_key=key,temperature=0.4,timeout=180)
    return public,private


def register(job_id,public,private):
    with _LOCK:
        _PROFILES[job_id]={"public":public,"private":private}


def rebind(job_id, saved_public, fresh):
    public,private=validate_profiles(fresh)
    if public!=saved_public:
        raise ValueError("Resume requires the exact original batch model / URL / providers")
    register(job_id,public,private)


def activated(job_id):
    with _LOCK:
        record=_PROFILES.get(job_id)
    if record is None:
        raise ValueError("Batch API keys were not restored; re-enter the original batch config to resume")
    return batch_config_context(record["private"])


def clear(job_ids):
    with _LOCK:
        for job_id in job_ids:
            _PROFILES.pop(job_id,None)
