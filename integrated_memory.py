"""v0.12.3: deterministic, source-faithful preparation + persistent integrated card.

No LLM during preparation: guarantees no expensive 19K-token preparation request.
The local source bank remains intact; only permitted source facts enter the card.
The cache is per-source, per-policy, versioned, and stored only locally.
"""
import hashlib
from contextvars import ContextVar
from contextlib import contextmanager
import json
import os
import re
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from respondent_core import StudioError
from respondent_autonomy import SYSTEM_POLICY_ZH
from question_first_retrieval import select_memory
from memory_access import tier_integrated_memory
from knowledge_bank import prepare as prepare_knowledge, signature as knowledge_signature, retrieve as retrieve_knowledge
from trait_postprocessing import normalize, memory_budget, verbosity_guide, behavioral_guide, question_sensitivity, memory_access_level

_CACHE = {}
VERSION = 'v1.6.2-restored-dual-gate'
CACHE_DIR = Path(__file__).resolve().parent / 'prepared_memory'
SAFE = {'none','low','medium'}
# V2 transcript-derived facts default to `unknown`; that is a review state, not a high-sensitivity label.
# Admit these only inside the explicit restricted-local V2 simulation, and report them separately.
# Explicit high/very_high facts remain excluded in every mode.


def _parse_json_object(content):
    """Parse a complete JSON object, tolerating a single Markdown fence only."""
    if not isinstance(content,str):
        return None
    value=content.strip()
    if value.startswith("```") and value.endswith("```"):
        value=re.sub(r"^```(?:json)?\s*", "", value, count=1, flags=re.I)
        value=re.sub(r"\s*```$", "", value, count=1)
    try:
        parsed=json.loads(value)
    except (TypeError,ValueError):
        return None
    return parsed if isinstance(parsed,dict) else None


def _call(model, messages, timeout, num_predict, schema, temperature=0.0, max_attempts=3, strict_schema=False):
    # The unlimited pure-Agent preparation can use the same provider selected
    # for the respondent. It has no total preparation deadline; each call is
    # still subject to the backend's network timeout.
    provider=_REMOTE_RESPONDENT.get()
    if provider in ("maas","shubi"):
        from model_services import get
        from interview_agent.llm.openai_compat import OpenAICompatClient
        from dataclasses import replace
        from uuid import uuid4
        try:
            cfg=get(provider,"respondent")
        except (ValueError,KeyError):
            raise StudioError("Agent 记忆准备 API 配置无效；请核对受访者的服务商、Key 和 URL。",
                              "preparation_api_configuration",409,
                              detail_code="provider_configuration_invalid",retryable=False) from None
        if cfg.model!=model:
            raise StudioError("记忆准备所选模型与网页受访者模型不同。",
                              "preparation_model_mismatch",409,
                              detail_code="model_configuration_mismatch",retryable=False)
        # Keep each upstream request inside the same maximum as the respondent
        # API; unlimited means unlimited batches, not an unlimited HTTP request.
        remote_timeout=min(900.0,max(30.0,float(timeout)))
        # This function is exclusively for structured preparation. Previously
        # it explicitly disabled JSON mode, then parsed the free-form answer as
        # JSON. That mismatch is the direct cause of integrated_invalid_json.
        # Normal respondent dialogue remains plain text in _call_plain_answer.
        extra={**cfg.extra_body, "max_tokens": max(512, min(65536, int(num_predict)))}
        if strict_schema:
            # The provider must enforce the types and enumerations at generation
            # time. Never silently downgrade this request to json_object mode.
            extra["response_format"]={"type":"json_schema","json_schema":{
                "name":"factbank_one_shot_v4","strict":True,"schema":schema}}
        cfg=replace(cfg,timeout=remote_timeout,temperature=temperature,
                    json_mode=True,extra_body=extra)
        diagnosis=_remote_request_diagnostics(messages,provider,model,num_predict,
                                               remote_timeout)
        started=time.monotonic()
        request_messages=list(messages)
        parse_retries=0
        for attempt in range(1,max_attempts+1):
            try:
                content=OpenAICompatClient(cfg,retries=1).chat(
                    request_messages,temperature=temperature,json_mode=True)
            except Exception as exc:
                detail,status,transient=_safe_remote_failure(exc)
                if strict_schema and status in (400, 422):
                    raise StudioError(
                        "当前评分/准备模型接口拒绝严格 JSON Schema 请求（HTTP %s）。"
                        "该接口可能不支持 response_format=json_schema，或不接受当前模型的"
                        "结构化输出参数；已停止而不是退回无约束格式。" % status,
                        "factbank_provider_schema_unsupported", 409,
                        detail_code="provider_rejected_json_schema", retryable=False) from None
                if transient and attempt<max_attempts:
                    time.sleep(float(attempt))
                    continue
                error=StudioError(
                    "Agent 记忆准备 API 调用失败（"+detail+"）。"
                    +("已达到三次调用上限；" if transient and attempt>=max_attempts else
                      "已重试；" if attempt>1 else "")
                    +"本批次未提交，可继续准备。",
                    "preparation_api_failure",status,detail_code=detail,
                    retryable=transient)
                error.diagnostic_id="PREP-"+uuid4().hex[:12]
                error.error_detail=detail
                error.response_structure={
                    **diagnosis,
                    "elapsed_seconds":round(time.monotonic()-started,2),
                    "attempts":attempt,"retry_count":attempt-1,
                    "json_mode_requested":True,
                    "json_format_retry_count":parse_retries,
                    "last_upstream_status":status if detail.startswith("upstream_http_") else None,
                    "transient_failure":transient,
                    "request_text_recorded":False,"api_key_recorded":False,
                }
                raise error from None
            # The provider may return HTTP 200 with prose/Markdown despite a
            # structured request. Retry a malformed completion with one
            # explicit format correction; never checkpoint invalid output.
            parsed=_parse_json_object(content)
            if isinstance(parsed,dict):
                return parsed
            if attempt<max_attempts:
                parse_retries+=1
                request_messages=request_messages+[{
                    "role":"user",
                    "content":("格式校正：上一条输出不是有效 JSON 对象。请重新完成原任务，"
                               "只返回一个完整 JSON 对象；不要添加 Markdown 代码围栏、"
                               "解释文字或对象以外的内容。字段必须符合原任务要求。")
                }]
                time.sleep(0.25)
                continue
            error=StudioError(
                "Agent 记忆准备 API 返回无效 JSON；本次没有自动重发请求，未提交缓存。",
                "integrated_invalid_json",502,
                detail_code="invalid_json_response",retryable=False)
            error.diagnostic_id="PREP-"+uuid4().hex[:12]
            error.error_detail="invalid_json_response"
            error.response_structure={
                **diagnosis,
                "elapsed_seconds":round(time.monotonic()-started,2),
                "attempts":attempt,"retry_count":attempt-1,
                "json_mode_requested":True,
                "json_format_retry_count":parse_retries,
                "request_text_recorded":False,"api_key_recorded":False,
            }
            raise error from None
    else:
        request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
            data=json.dumps({'model':model,'messages':messages,'stream':False,'format':schema,
                             'keep_alive':'30m','options':{'temperature':temperature,
                             'num_predict':num_predict,'num_ctx':max(4096, min(131072, int(num_predict) * 2))}},ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=timeout) as resp:
                out=json.load(resp)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            raise StudioError('整合记忆问答模型调用失败；没有自动重试。',
                              'integrated_model_failure',503) from exc
        content=(out.get('message') or {}).get('content')
    if not isinstance(content,str) or not content.strip():
        reason=out.get('done_reason','unknown')
        raise StudioError(f'受访者没有返回可解析内容（done_reason={reason}）；当前轮停止，不自动重试。',
                          'integrated_empty_model_output',502)
    try:
        parsed=json.loads(content)
    except (TypeError,ValueError) as exc:
        reason=str(out.get('done_reason','unknown'))
        # Do not expose a model completion preview: it may contain private
        # source material or provider-echoed credentials.
        raise StudioError(f'受访者 JSON 解析失败（done_reason={reason}, chars={len(content)}）；本轮停止，没有自动重试。',
                          'integrated_invalid_json',502) from exc
    if not isinstance(parsed,dict):
        raise StudioError('受访者返回的 JSON 不是对象；本轮停止。','integrated_invalid_json',502)
    return parsed



_REMOTE_RESPONDENT=ContextVar("remote_respondent",default=None)
@contextmanager
def respondent_api(provider):
    token=_REMOTE_RESPONDENT.set(provider)
    try:yield
    finally:_REMOTE_RESPONDENT.reset(token)

def _safe_remote_failure(exc):
    """Return a sanitized upstream failure classification.

    The upstream client's LLMError wraps arbitrary urllib errors in strings.
    Never export its raw message: it can contain upstream response bodies or
    private prompts. Preserve only recognized status and exception *types*.
    """
    import urllib.error
    import socket
    import ssl
    import http.client
    # The current OpenAI-compatible client exposes safe structured metadata.
    # Read it before parsing legacy exception strings, otherwise a real 401
    # wrapped as "Remote API ... (upstream HTTP 401)" is misclassified as an
    # unknown transient error and retried.
    structured_status=getattr(exc,"status",None)
    if isinstance(structured_status,int) and not isinstance(structured_status,bool):
        return ("upstream_http_"+str(structured_status),structured_status,
                structured_status in (408,429,500,502,503,504,520,521,522,523,524))
    if getattr(exc,"kind",None)=="connection_or_timeout":
        return "upstream_connection_or_timeout",504,True
    if getattr(exc,"kind",None) in ("invalid_response_structure","invalid_or_empty_response"):
        return "upstream_response_format_invalid",502,False
    message=str(exc)
    status_match=re.match(r"^HTTP\s+(\d{3})\s*:",message,re.I)
    if status_match:
        status=int(status_match.group(1))
        return "upstream_http_"+str(status),status,status in (408,429,500,502,503,504,520,521,522,523,524)
    name=type(exc).__name__
    # These type labels are emitted by OpenAICompatClient after wrapping
    # urllib/socket exceptions as LLMError.
    if (isinstance(exc,(TimeoutError,socket.timeout))
        or re.match(r"^(?:TimeoutError|socket\.timeout):",message,re.I)
        or "timed out" in message.lower()):
        return "upstream_timeout",504,True
    if (isinstance(exc,(urllib.error.URLError,ConnectionError,
                        http.client.RemoteDisconnected,http.client.IncompleteRead))
        or re.match(r"^(?:URLError|RemoteDisconnected|ConnectionResetError|"
                    r"ConnectionAbortedError|BrokenPipeError|ConnectionRefusedError|"
                    r"IncompleteRead|ConnectionError|OSError):",message,re.I)):
        return "upstream_connection_failure",503,True
    if (isinstance(exc,(ssl.SSLError,))
        or re.match(r"^(?:SSLError|SSLEOFError|SSLCertVerificationError):",message,re.I)):
        # A bad certificate may be a configuration issue, not transient.
        return "upstream_tls_failure",502,False
    if (isinstance(exc,(KeyError,ValueError,TypeError))
        or re.match(r"^(?:KeyError|ValueError|TypeError|JSONDecodeError):",message)):
        return "upstream_response_format_invalid",502,False
    # The *wrapper* gives only "LLMError" even for exceptions not handled by
    # the upstream client's built-in clauses (e.g. RemoteDisconnected). Do not
    # claim this proves a real upstream HTTP 502. One bounded retry is useful.
    if name=="LLMError" or message=="None" or not message.strip():
        return "upstream_unclassified_llm_error",502,True
    return "upstream_unclassified_client_error",502,True


def _remote_request_diagnostics(messages, provider, model, max_tokens, timeout):
    """Aggregate only structural sizes. Do not persist questions or API keys."""
    return {
        "provider":provider,
        "model":model,
        "message_count":len(messages),
        "prompt_character_count":sum(
            len(str(m.get("content",""))) for m in messages if isinstance(m,dict)),
        "requested_max_output_tokens":int(max_tokens),
        "request_timeout_seconds":float(timeout),
    }


_TURN_BUDGET=ContextVar("respondent_turn_budget",default=None)

def _call_plain_answer(model,messages,timeout,num_predict,temperature=0.2):
    budget=_TURN_BUDGET.get()
    if budget is None:
        return _call_plain_answer_impl(model,messages,timeout,num_predict,temperature)
    phase=budget.phase
    started=time.monotonic();ok=False
    try:
        result=_call_plain_answer_impl(model,messages,budget.limit(timeout),num_predict,temperature)
        budget.check()
        ok=True
        return result
    finally:
        budget.record(phase,time.monotonic()-started,ok)

def _call_plain_answer_impl(model, messages, timeout, num_predict, temperature=0.2):
    """Single, non-retrying plain-text chat call for conversational answers.

    Preparation continues to use strict JSON.  Respondent dialogue must not
    depend on JSON syntax being completed within a small token budget.
    """
    api_provider=_REMOTE_RESPONDENT.get()
    if api_provider in ("maas","shubi"):
        from model_services import get
        from interview_agent.llm.openai_compat import OpenAICompatClient
        from dataclasses import replace
        from uuid import uuid4
        try:
            cfg=get(api_provider,"respondent")
        except (ValueError,KeyError) as exc:
            raise StudioError("远程受访者 API 配置不完整或格式无效。",
                              "respondent_api_configuration",409,
                              detail_code="provider_configuration_invalid",
                              retryable=False) from None
        if cfg.model!=model:
            raise StudioError("受访者模型配置与会话不一致；请新建会话。",
                              "respondent_model_changed",409)
        remote_timeout=float(timeout)
        if not 0<remote_timeout<=900:
            raise StudioError("受访者单轮时限应为 30–900 秒",
                              "respondent_api_timeout_invalid",400)
        budget=_TURN_BUDGET.get()
        is_answer=budget is not None and budget.phase=="answer"
        extra={**cfg.extra_body,"max_tokens":int(num_predict)}
        thinking=os.getenv("INTERVIEW_RESPONDENT_THINKING","auto").strip().lower()
        if is_answer and thinking in ("off","on"):
            extra["enable_thinking"]=(thinking=="on")
        # The semantic selector asks only for structured source IDs. Qwen's
        # default reasoning mode may spend most of the 120-second HTTP window
        # on hidden thinking before returning those IDs. Disable reasoning for
        # this *retrieval* request only; leave the answer model and all
        # retrieval/ranking/fallback logic untouched. For non-Qwen models do
        # not send the Qwen-specific request option.
        is_semantic=budget is not None and budget.phase=="retrieval"
        semantic_thinking=os.getenv("INTERVIEW_SEMANTIC_THINKING","off").strip().lower()
        if is_semantic and model.lower().startswith("qwen") and semantic_thinking in ("off","on"):
            extra["enable_thinking"]=(semantic_thinking=="on")
        cfg=replace(cfg,temperature=temperature,timeout=remote_timeout,extra_body=extra)
        info=_remote_request_diagnostics(messages,api_provider,model,num_predict,remote_timeout)
        attempt_cap=remote_timeout
        if is_answer:
            try:attempt_cap=max(30,min(900,float(os.getenv("INTERVIEW_RESPONDENT_ATTEMPT_TIMEOUT","120"))))
            except ValueError:attempt_cap=120
            info["answer_attempt_cap_seconds"]=attempt_cap
            info["answer_thinking_mode"]=thinking if thinking in ("off","on") else "auto"
        started=time.monotonic()
        # A transient error can be retried once, while the turn is still
        # uncommitted. Reject 400/401/403/404 and invalid responses immediately.
        # The native group Interviewer request is unaffected by these retries.
        for attempt in range(1,4):
            try:
                remaining=remote_timeout-(time.monotonic()-started)
                if remaining<=0:
                    raise TimeoutError("request budget exhausted")
                # Reserve time for another attempt when the whole answer budget allows it.
                attempt_limit=min(remaining,attempt_cap) if is_answer else remaining
                attempt_cfg=replace(cfg,timeout=max(.001,attempt_limit))
                attempt_started=time.monotonic()
                response=OpenAICompatClient(attempt_cfg,retries=1).chat(
                    messages,temperature=temperature)
                if not isinstance(response,str) or not response.strip():
                    raise StudioError("远程受访者返回空文本。","remote_respondent_empty",
                                      502,detail_code="upstream_empty_output",
                                      retryable=False)
                budget=_TURN_BUDGET.get()
                if budget is not None:
                    budget.record(budget.phase+"_remote_attempts",time.monotonic()-(attempt_started if is_answer else started),True,attempt)
                return response.strip()
            except StudioError:
                raise
            except Exception as exc:
                detail,status,transient=_safe_remote_failure(exc)
                budget=_TURN_BUDGET.get()
                if budget is not None:
                    budget.record(budget.phase+"_remote_failure",time.monotonic()-(attempt_started if is_answer and "attempt_started" in locals() else started),False,attempt)
                delay=float(attempt*2)
                # HTTP 524 is a gateway timeout after an already long-running
                # request. Replaying the identical semantic prompt does not
                # restore the lost response and may exhaust the entire turn.
                # Keep bounded retries for other transient errors unchanged.
                if (transient and detail != "upstream_http_524" and attempt<3
                        and remote_timeout-(time.monotonic()-started)>delay+1):
                    time.sleep(delay)
                    continue
                error=StudioError(
                    "远程受访者 API 失败（"+detail+"）；已保留之前成功的轮次。"
                    +(" 已尝试自动重试。" if attempt>1 else " 请检查 API 模型与请求配置。"),
                    "remote_respondent_api_failure",
                    status,detail_code=detail,retryable=transient)
                error.diagnostic_id="RAPI-"+uuid4().hex[:12]
                error.error_detail=detail
                error.response_structure={
                    **info,
                    "elapsed_seconds":round(time.monotonic()-started,2),
                    "attempts":attempt,
                    "retry_count":attempt-1,
                    "last_upstream_status":status if detail.startswith("upstream_http_") else None,
                    "transient_failure":transient,
                    "request_text_recorded":False,
                    "api_key_recorded":False,
                }
                raise error from None
    payload = {'model': model, 'messages': messages, 'stream': False,
               'keep_alive': '30m', 'options': {'temperature': temperature,
               'num_predict': num_predict, 'num_ctx': 4096}}
    request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            out = json.load(resp)
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        raise StudioError('受访者 Ollama 调用失败；未自动重试。',
                          'integrated_model_failure', 503) from exc
    content = (out.get('message') or {}).get('content')
    if not isinstance(content, str) or not content.strip():
        raise StudioError('受访者没有返回文本；本轮未提交。',
                          'integrated_empty_model_output', 502)
    # A length stop means the reply may end mid-sentence. Never save it as a
    # completed turn, and never fabricate the missing ending.
    if out.get('done_reason') == 'length':
        raise StudioError('受访者回答达到生成长度限制，未提交不完整回答；没有自动重试。',
                          'integrated_answer_truncated', 502)
    text = content.strip()
    # Compatibility with models that voluntarily wrap their reply in JSON.
    # Only decode a COMPLETE object; never attempt to repair broken JSON.
    if text.startswith('{') and text.endswith('}'):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get('answer'), str):
                text = obj['answer'].strip()
        except (ValueError, TypeError):
            pass
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:text|markdown)?\s*\n?', '', text, count=1, flags=re.I)
        text = re.sub(r'\n?```$', '', text).strip()
    if not text:
        raise StudioError('受访者回答为空；本轮未提交。', 'integrated_empty_answer', 502)
    return text

def key_for(bundle,model):
    source={'memory_bank':bundle['respondent']['memory_bank'],
            'persona':bundle['respondent'].get('persona'),
            'model':model,'source_type':(bundle.get('source_scenario') or {}).get('source_type'),
            'usage':(bundle.get('source_scenario') or {}).get('usage'),
            'revision':bundle['audit']['revision'],
            'public_pool':(bundle['respondent'].get('public_memory_pool') or {}).get('memories',[]),
            'seed':(bundle.get('generation') or {}).get('seed'),
            'algorithm':VERSION, 'domain_knowledge':knowledge_signature(bundle)}
    return hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def _load(key):
    path=CACHE_DIR / (key + '.json')
    if not path.is_file():return None
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('key')==key and data.get('algorithm')==VERSION:
            return data['result']
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return None


def _save(key,result):
    CACHE_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
    # Never write full private source facts into this cache.
    fd,tmp=tempfile.mkstemp(prefix='preparing_',suffix='.tmp',dir=str(CACHE_DIR))
    try:
        try:os.fchmod(fd,0o600)
        except (AttributeError,OSError):pass
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            json.dump({'key':key,'algorithm':VERSION,'result':result},handle,ensure_ascii=False)
        os.replace(tmp,CACHE_DIR/(key+'.json'))
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _compact(text,limit=240):
    """Keep faithful original excerpts, never invent a summary or causal link."""
    value=re.sub(r'\s+',' ',str(text or '')).strip()
    if len(value)<=limit:return value
    # Prefer a complete opening sentence when available.
    prefix=value[:limit]
    breaks=[m.end() for m in re.finditer(r'[。！？.!?；;]',prefix)]
    cut=max(breaks) if breaks and max(breaks)>=max(60,limit//2) else limit
    return prefix[:cut].rstrip() + '…'





















def _load_agent(key):
    path=CACHE_DIR/(key+'.agent.json')
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        return data['result'] if data.get('key')==key else None
    except (OSError,ValueError,KeyError,TypeError):return None


def _save_agent(key,result):
    CACHE_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='preparing_agent_',suffix='.tmp',dir=str(CACHE_DIR))
    try:
        try:os.fchmod(fd,0o600)
        except (AttributeError,OSError):pass
        with os.fdopen(fd,'w',encoding='utf-8') as h:
            json.dump({'key':key,'result':result},h,ensure_ascii=False)
        os.replace(tmp,CACHE_DIR/(key+'.agent.json'))
    finally:
        if os.path.exists(tmp):os.unlink(tmp)















def _llm_factbank_key(bundle, model, hybrid=False):
    from llm_factbank import PROMPT_VERSION
    # Preserve original FactBank identity for local runs. Remote full-cache
    # identities additionally bind the selected provider: a validated result
    # from one service must not certify a different service's preparation.
    provider = _REMOTE_RESPONDENT.get()
    provider_suffix = (":provider:" + provider) if provider in ("maas", "shubi") else ""
    # A bulk response has different completeness semantics from legacy batch checkpoints.
    bulk_suffix = ":strict-structured-wire-v4" if hybrid else ""
    return hashlib.sha256((("hybrid-v155:" if hybrid else "llm-factbank:")
        + PROMPT_VERSION + ":" + key_for(bundle, model) + provider_suffix + bulk_suffix).encode()).hexdigest()


def _llm_factbank_progress_path(key):
    return CACHE_DIR/(key+".llm_factbank_progress.json")


def _llm_factbank_progress_save(key, signature, completed):
    CACHE_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
    path=_llm_factbank_progress_path(key)
    fd,tmp=tempfile.mkstemp(prefix="llm_factbank_progress_",suffix=".tmp",dir=str(CACHE_DIR))
    try:
        try:os.fchmod(fd,0o600)
        except (OSError,AttributeError):pass
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump({"key":key,"source_signature":signature,
                       "completed_batches":completed},stream,ensure_ascii=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _llm_factbank_progress_load(key, signature, batches):
    try:
        data=json.loads(_llm_factbank_progress_path(key).read_text(encoding="utf-8"))
        if data.get("key")!=key or data.get("source_signature")!=signature:
            return []
        completed=data.get("completed_batches")
        if not isinstance(completed,list):return []
        valid=[]
        for idx,row in enumerate(completed):
            if idx>=len(batches) or not isinstance(row,dict):break
            expected=[x["id"] for x in batches[idx]]
            if row.get("source_ids")!=expected or not isinstance(row.get("units"),list):break
            if [x.get("source_id") for x in row["units"] if isinstance(x,dict)]!=expected:break
            if any(x.get("llm_status")!="accepted" for x in row["units"]):break
            valid.append(row)
        return valid
    except (OSError,ValueError,TypeError,KeyError):
        return []


def _native_factbank_prepare(bundle, model, hybrid=False):
    """Default FactBank preparation: LLM semantic classification + source IDs.

    The LLM creates a retrieval index only. Original quotes remain in the
    source bundle and are materialized later by authorized source ID.
    """
    from agent_memory_export import eligible_source_facts
    from llm_factbank import (SCHEMA_VERSION, build_prompt, response_schema,
                              validate_batch, summary, signature)
    from source_evidence import _event_id, _time_expressions
    started=time.monotonic()
    key=_llm_factbank_key(bundle,model,hybrid=hybrid)
    method=("python_agent_hybrid_bilingual_v155" if hybrid else "llm_factbank_semantic_index_v154")
    cached=_CACHE.get(key) or _load_agent(key)
    if (cached is not None and cached.get("preparation_method")==method
            and cached.get("cache_version")==SCHEMA_VERSION
            and cached.get("llm_preparation_complete") is True
            and (cached.get("factbank_llm_summary") or {}).get("unresolved_count")==0
            and (cached.get("factbank_llm_summary") or {}).get("total",0)>0
            and len(cached.get("factbank_llm_index") or [])==(cached.get("factbank_llm_summary") or {}).get("total")
            and all(x.get("llm_status")=="accepted" for x in cached.get("factbank_llm_index",[]))):
        _CACHE[key]=cached
        cache_copy=dict(cached)
        cache_copy["preparation_cache_hit"]=True
        cache_copy["llm_api_call_count_this_run"]=0
        return key,cache_copy
    records=eligible_source_facts(bundle)
    if not records:
        raise StudioError("No permitted source facts are available for LLM FactBank preparation.",
                          "llm_factbank_no_permitted_facts",409)
    try:
        batch_size=max(1,min(16,int(os.getenv("INTERVIEW_FACTBANK_BATCH_SIZE","6"))))
    except (TypeError,ValueError):
        batch_size=6
    try:
        char_budget=max(500,min(6000,int(os.getenv("INTERVIEW_FACTBANK_BATCH_CHARS","2400"))))
    except (TypeError,ValueError):
        char_budget=2400
    batches=[];current=[];chars=0
    for item in records:
        size=len(item["text"])+len(item.get("source_question") or "")
        if size>6500:
            raise StudioError("A source fact exceeds the safe FactBank preparation limit; split it manually.",
                              "llm_factbank_fact_too_long",409)
        if current and (chars+size>char_budget or len(current)>=batch_size):
            batches.append(current);current=[];chars=0
        current.append(item);chars+=size
    if current:batches.append(current)
    # LLM-sort mode: one complete source-grounded request, one response.
    # No per-fact model requests and no incomplete checkpoint pretending success.
    single_request = bool(hybrid)
    if single_request:
        batches = [records]
        batch_size = len(records)
        char_budget = sum(len(x["text"]) + len(x.get("source_question") or "") for x in records)
    source_signature=hashlib.sha256(json.dumps(
        [[x["id"],x["text"],x.get("source_question","")] for x in records],
        ensure_ascii=False,separators=(",",":" )).encode()).hexdigest()
    try:
        budget=max(600,min(14400,int(os.getenv("INTERVIEW_FACTBANK_PREPARE_SECONDS","3600"))))
    except (ValueError,TypeError):
        budget=3600
    completed=_llm_factbank_progress_load(key,source_signature,batches)
    # v1.5.5: reusable validated fact-level cache, independent of batching.
    # Previous complete batches are verified as always. Cache never replaces
    # model-produced bilingual semantics with deterministic guesses.
    from hybrid_preparation import load_unit, save_unit, deterministic_metadata, local_features, validate_semantic_unit
    provider=_REMOTE_RESPONDENT.get()
    from llm_factbank import PROMPT_VERSION
    prepared_local_features=([local_features(x) for x in records] if hybrid else [])
    nonlocal_cache_hits=[0]
    per_fact_lock=__import__('threading').Lock()
    # Two independent batches may run concurrently. Checkpoints are committed
    # in source order, and each worker inherits the selected API provider.
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    try:
        concurrency=max(1,min(3,int(os.getenv("INTERVIEW_FACTBANK_PARALLEL_BATCHES","2"))))
    except (ValueError,TypeError):
        concurrency=2
    def process_batch(batch_idx):
        batch=batches[batch_idx]
        accepted={}
        pending=[]
        calls=0
        if hybrid:
            for source in batch:
                existing=load_unit(CACHE_DIR,source,model,provider,PROMPT_VERSION)
                if existing is not None:
                    accepted[source["id"]]=existing
                    with per_fact_lock:
                        nonlocal_cache_hits[0]+=1
                else:
                    pending.append(source)
        else:
            pending=list(batch)
        if single_request:
            # One compact response avoids repeating eleven JSON property names
            # for each fact. The model still sees EVERY original source fact.
            # Only the server maps ordinal indexes back onto immutable source IDs.
            from llm_factbank import compact_prompt, compact_schema, expand_compact_rows
            elapsed=time.monotonic()-started
            if elapsed>=budget-10:
                raise StudioError("全量 FactBank 准备超时，未写入缓存。",
                                  "llm_factbank_prepare_timeout",503)
            output_tokens=max(4096,min(65536,155*len(records)+800))
            result=_call(model,compact_prompt(records),
                timeout=max(30,min(900,budget-elapsed)),
                num_predict=output_tokens,schema=compact_schema(len(records)),
                temperature=0.0,max_attempts=1,strict_schema=True)
            calls+=1
            expanded, diagnostic=expand_compact_rows(result,records)
            if diagnostic:
                raise StudioError(
                    "一次性 FactBank 结果不完整："+diagnostic+
                    "。没有自动重发请求，也没有写入成功缓存。",
                    "llm_factbank_bulk_incomplete",502)
            units=validate_batch({"units":expanded},records)
            missing=[]
            for source,unit in zip(records,units):
                validated=validate_semantic_unit(source,unit)
                if validated is None:
                    missing.append(source["id"])
                else:
                    accepted[source["id"]]=validated
            if missing:
                from collections import Counter
                reasons=Counter(row.get("classification_reason","unknown")
                    for row in units if row.get("llm_status")!="accepted")
                raise StudioError(
                    "一次性 FactBank 返回 %d 条，但 %d 条语义字段不合格（%s）；例如 %s。"
                    "没有自动重发请求，未保存成功缓存。" % (
                    len(expanded),len(missing),str(dict(reasons)),", ".join(missing[:5])),
                    "llm_factbank_bulk_incomplete",502)
            for source in records:
                save_unit(CACHE_DIR,source,model,provider,PROMPT_VERSION,
                          accepted[source["id"]])
            return {"batch_index":batch_idx,"source_ids":[x["id"] for x in batch],
                    "units":[accepted[x["id"]] for x in batch]},calls
        for attempt in range(4):
            if not pending:
                break
            chunk_size=len(pending) if attempt==0 else (3 if attempt==1 else 1)
            next_pending=[]
            for offset in range(0,len(pending),chunk_size):
                chunk=pending[offset:offset+chunk_size]
                elapsed=time.monotonic()-started
                if elapsed>=budget-10:
                    raise StudioError("FactBank preparation budget exhausted; resume to continue.",
                                      "llm_factbank_prepare_timeout",503)
                result=_call(model,build_prompt(chunk),
                    timeout=max(10,min(180,budget-elapsed)),
                    num_predict=max(768,520*len(chunk)),schema=response_schema(),temperature=0.0)
                calls+=1
                units=validate_batch(result,chunk)
                for source,unit in zip(chunk,units):
                    if unit.get("llm_status")=="accepted":
                        if hybrid:
                            unit=validate_semantic_unit(source,unit)
                        if unit is not None:
                            accepted[source["id"]]=unit
                            if hybrid:
                                save_unit(CACHE_DIR,source,model,provider,PROMPT_VERSION,unit)
                        else:
                            next_pending.append(source)
                    else:
                        next_pending.append(source)
            pending=next_pending
        if pending:
            raise StudioError(
                "FactBank semantic classification failed for %d facts in batch %d/%d "
                "after targeted retries; valid checkpoints retained." %
                (len(pending),batch_idx+1,len(batches)),
                "llm_factbank_invalid_batch",502)
        return {"batch_index":batch_idx,"source_ids":[x["id"] for x in batch],
                "units":[accepted[x["id"]] for x in batch]},calls

    llm_api_call_count=0
    start_batch=len(completed)
    if start_batch<len(batches):
        # Executor.map yields results in input order, even when later batches
        # finish earlier, so resumes remain deterministic and source-grounded.
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures=[executor.submit(copy_context().run,process_batch,idx)
                     for idx in range(start_batch,len(batches))]
            try:
                for future in futures:
                    row,calls=future.result()
                    llm_api_call_count+=calls
                    completed.append(row)
                    _llm_factbank_progress_save(key,source_signature,completed)
            except BaseException:
                _llm_factbank_progress_save(key,source_signature,completed)
                for future in futures:
                    future.cancel()
                raise
    index_by_id={}
    for row in completed:
        for unit in row["units"]:
            index_by_id[unit["source_id"]]=unit
    index=[]
    for record in records:
        unit=dict(index_by_id.get(record["id"]) or {})
        unit.setdefault("source_id",record["id"])
        unit["source_text_sha256"]=hashlib.sha256(record["text"].encode()).hexdigest()
        unit["source_question_sha256"]=(hashlib.sha256(
            str(record.get("source_question") or "").encode()).hexdigest()
            if record.get("source_question") else None)
        unit["event_id"]=_event_id(record["id"])
        unit["time_expressions"]=_time_expressions(
            record["text"]+" "+str(record.get("source_question") or ""))
        index.append(unit)
    by_event={}
    for unit in index:
        cue=" ".join(x for x in (unit.get("normalized_text_zh"),
                                  unit.get("normalized_text_en")) if x).strip()
        if not cue:
            source=next(x for x in records if x["id"]==unit["source_id"])
            cue=str(source["text"])[:500]
        by_event.setdefault(unit.get("event_id") or unit["source_id"],[]).append(cue)
    card=[{"theme":"SourceEvent "+str(event),"facts":facts}
          for event,facts in by_event.items()]
    public_sample=[]
    pool=bundle["respondent"].get("public_memory_pool")
    if isinstance(pool,dict) and pool.get("memories"):
        from memory_system import sample_public_cues, compact_public_cues
        selected=sample_public_cues(pool,seed=(bundle.get("generation") or {}).get("seed",0),
            scenario_id=(bundle.get("source_scenario") or {}).get("scenario_id","unbound"),
            turn=0,question="fixed_preparation_sample_v152",language="zh")
        public_sample=compact_public_cues(selected,"zh")
    from llm_factbank import summary as factbank_summary
    factbank_summary_data=factbank_summary(index)
    total=sum(len(m.get("facts") or []) for m in bundle["respondent"]["memory_bank"])
    unresolved=factbank_summary_data["unresolved_count"]
    if unresolved:
        raise StudioError("FactBank contains unresolved semantic entries; success cache not written.",
                          "llm_factbank_incomplete",502)
    result={"card":card,"groups":[],"fact_count":total,
        "personal_memory_count":len(bundle["respondent"]["memory_bank"]),
        "public_distractor_sample_count":len(public_sample),
        "public_distractor_sample":public_sample,
        "included_fact_count":len(records),
        "excluded_private_fact_count":max(0,total-len(records)),
        "provisional_unreviewed_fact_count":sum(x.get("review_pending",False) for x in records),
        "truncated_excerpt_count":0,
        "preparation_seconds":round(time.monotonic()-started,3),
        "preparation_method":method,
        "preparation_prompt_version":"llm_factbank_prepare.strict_wire_v4",
        "preparation_batch_count":len(batches),
        "python_preprocessed_fact_count":len(prepared_local_features),
        "per_fact_cache_hit_count":nonlocal_cache_hits[0],
        "preparation_strategy":"python_orchestration_agent_bilingual_semantics" if hybrid else "llm_led",
        "llm_api_call_count_this_run":llm_api_call_count,
        "preparation_cache_hit":False,
        "preparation_completed_batch_count":len(completed),
        "llm_processed_fact_count":len(index)-unresolved,
        "source_only_unresolved_count":unresolved,
        "preparation_parallel_batches":1 if single_request else concurrency,
        "preparation_request_strategy":"single_strict_json_schema_request" if single_request else "bounded_batches",
        "preparation_batch_size":batch_size,
        "preparation_batch_char_budget":char_budget,
        "factbank_llm_index":index,
        "factbank_llm_index_signature":signature(index),
        "factbank_llm_summary":factbank_summary_data,
        "factbank_schema":SCHEMA_VERSION,
        "model":model,"cache_version":SCHEMA_VERSION,
        "llm_preparation_complete":True,
        "domain_knowledge":prepare_knowledge(bundle)}
    _save_agent(key,result)
    _CACHE[key]=result
    _llm_factbank_progress_path(key).unlink(missing_ok=True)
    return key,result




# Two explicit, versioned choices. LLM fact sorting is never inferred from
# prebuilt source IDs. The no-sort path retains the original permitted facts.
PREPARATION_METHODS = {
    'fast_hybrid': 'native_llm_factbank_v154',
    'no_fact_sort': 'source_only_unified_v2',
    'python_binary': 'python_binary_v15_unified_before_judgement_v1',
}

def _fact_preparation_key(bundle, model, mode):
    # The native v1.5.5 FactBank is the authority for the sorting cache.
    # This derived ID is only for the Playground-facing result/lookup adapter.
    # Do not include the optional vector-model path: it is not an input to
    # the FactBank classification and should not invalidate its semantic cache.
    if mode == 'fast_hybrid':
        native_key = _llm_factbank_key(bundle, model, hybrid=True)
        return hashlib.sha256(('playground-factbank-adapter-v5:' + native_key).encode()).hexdigest()
    provider = _REMOTE_RESPONDENT.get()
    return hashlib.sha256(json.dumps([
        'fact-preparation-v3-native-python-binary-v15.1', mode, provider, model, key_for(bundle, model),
        os.getenv('INTERVIEW_MULTILINGUAL_MODEL_DIR', '')
    ], ensure_ascii=False).encode('utf-8')).hexdigest()

def fast_hybrid_prepare(bundle, model, preparation_mode='fast_hybrid'):
    """Sort every eligible source fact with validated LLM results, or don't sort.

    LLM mode requires 100% valid facts; failure is explicit, never concealed by
    a Python-only fallback. Per-fact validated semantic caches may be reused.
    The memory text and source IDs are immutable in both modes.
    """
    from agent_memory_export import eligible_source_facts
    from factbank import classify
    from llm_factbank import build_prompt, response_schema, validate_batch, signature, PROMPT_VERSION
    from source_evidence import _event_id, _time_expressions
    from hybrid_preparation import load_unit, save_unit, deterministic_metadata, validate_semantic_unit
    from knowledge_bank import prepare as prepare_domain
    if preparation_mode not in PREPARATION_METHODS:
        raise StudioError('未知记忆准备方式', 'unsupported_preparation_mode', 400)
    started = time.monotonic()
    mode = preparation_mode
    method = PREPARATION_METHODS[mode]
    key = _fact_preparation_key(bundle, model, mode)
    if mode == 'fast_hybrid':
        # One complete LLM classification of all eligible source facts.
        # The native FactBank validator, original source IDs, output layout,
        # full cache and downstream retrieval contract remain unchanged.
        _, native = _native_factbank_prepare(bundle, model, hybrid=True)
        result = dict(native)
        accepted = sum(item.get('llm_status') == 'accepted'
                       for item in result.get('factbank_llm_index', []))
        count = len(result.get('factbank_llm_index', []))
        if count == 0 or accepted != count:
            raise StudioError('原版 FactBank 仍有未完成事实，拒绝标记准备成功',
                              'native_factbank_incomplete', 422)
        result.update({
            'preparation_mode': mode,
            'preparation_method': method,
            'preparation_complete': True,
            'llm_accepted_fact_count': accepted,
            'agent_processed_fact_count': accepted,
            'agent_batch_count': result.get('preparation_batch_count', 0),
            'full_bilingual_semantic_complete': True,
            'fast_preparation_quality_status': 'native_llm_factbank_validated',
            'python_only_fact_count': 0,
            'semantic_coverage': 'native_llm_factbank_full_semantic_index',
            'source_id_mapping_complete': True,
            'metadata_origin_counts': {'native_llm_factbank': accepted},
        })
        # Persist the adapter for the existing respondent lookup contract.
        # A new preparation call must always consult the native complete cache
        # first, rather than treating this derived adapter as the source of truth.
        _save_agent(key, result)
        _CACHE[key] = result
        return key, result
    existing = _CACHE.get(key) or _load_agent(key)
    if existing and existing.get('preparation_complete') and existing.get('preparation_method') == method:
        return key, dict(existing, preparation_cache_hit=True, llm_api_call_count_this_run=0)
    records = eligible_source_facts(bundle)
    if not records:
        raise StudioError('没有可使用的来源事实', 'fast_prepare_no_facts', 409)
    provider = _REMOTE_RESPONDENT.get()
    index = []
    uncached = []
    cache_hits = 0
    for source in records:
        # Deterministic metadata is only a locator, never an LLM verdict.
        if mode == 'python_binary':
            from python_binary_factbank import classify_source
            binary=classify_source(source)
            category,reason,confidence=(binary['fact_category'],binary['classification_reason'],binary['confidence'])
        else:
            category, reason, confidence = classify(source) if mode == 'fast_hybrid' else ('memory', 'unified_unsorted_pool', 0.0)
        unit = {'source_id': source['id'], 'fact_category':category,
                'fact_type':'other', 'temporal_scope':'unknown',
                'normalized_text_zh':'', 'normalized_text_en':'',
                'semantic_tags':[], 'importance':'background',
                'confidence':confidence, 'review_required':(binary['review_required'] if mode == 'python_binary' else True),
                'classification_reason':reason,
                'llm_status':'not_requested' if mode != 'fast_hybrid' else 'pending',
                'event_id':_event_id(source['id']),
                'time_expressions':_time_expressions(source['text']+' '+str(source.get('source_question') or ''))}
        if mode == 'python_binary':
            # Exactly one classification dimension is exported: Memory / Reality.
            # Do not attach extra fact_type/temporal subtype/tag labels that can
            # perturb search. Literal source words and source questions remain.
            unit['metadata_origin']='python_binary_v15'
            unit['semantic_review_status']='python_rules_unverified'
        else:
            unit.update(deterministic_metadata(source, category, confidence))
            if source.get('dataset_memory_id'):
                unit['semantic_tags'] = list(dict.fromkeys((unit.get('semantic_tags') or []) + (source.get('prebuilt_semantic_tags') or [])))
        if mode == 'fast_hybrid':
            stored = load_unit(CACHE_DIR, source, model, provider, PROMPT_VERSION)
            if stored is not None:
                unit.update(stored)
                unit['metadata_origin'] = 'validated_llm_cache'
                unit['semantic_review_status'] = 'llm_validated'
                cache_hits += 1
            else:
                uncached.append(source)
        index.append(unit)

    calls = 0
    if mode == 'fast_hybrid':
        # No limit of four facts and no special exemption for MemoryBank IDs.
        # Keep each request bounded, but cover every eligible fact.
        try:
            batch_size = max(1, min(8, int(os.getenv('INTERVIEW_FACT_SORT_BATCH_SIZE', '4'))))
        except (TypeError, ValueError):
            batch_size = 4
        units_by_id = {item['source_id']: item for item in index}
        for offset in range(0, len(uncached), batch_size):
            batch = uncached[offset:offset + batch_size]
            try:
                calls += 1
                raw = _call(model, build_prompt(batch), timeout=120,
                            num_predict=max(1024, 700 * len(batch)),
                            schema=response_schema(), temperature=0.0)
                checked = validate_batch(raw, batch)
                validated = {}
                for source, unit in zip(batch, checked):
                    clean = validate_semantic_unit(source, unit)
                    if clean is None:
                        raise StudioError('LLM事实分拣返回缺失或不合规字段：'+source['id'],
                                          'fact_sort_invalid_result', 422)
                    validated[source['id']] = clean
                # Save only fully validated batches; original source records remain unchanged.
                for source in batch:
                    clean = validated[source['id']]
                    save_unit(CACHE_DIR, source, model, provider, PROMPT_VERSION, clean)
                    units_by_id[source['id']].update(clean)
                    units_by_id[source['id']]['metadata_origin'] = 'validated_llm'
                    units_by_id[source['id']]['semantic_review_status'] = 'llm_validated'
            except StudioError:
                raise
            except Exception as exc:
                raise StudioError('LLM事实分拣失败；未生成成功缓存（批次 '+str(offset//batch_size+1)+'）',
                                  'fact_sort_model_error', 502) from exc
        if any(item.get('llm_status') != 'accepted' for item in index):
            raise StudioError('LLM事实分拣没有覆盖全部事实', 'fact_sort_incomplete', 422)

    # Both paths keep the full immutable source text. A classification has no
    # authority to remove a source fact from the respondent or Benchmark.
    grouped = {}
    for source in records:
        unit = next(item for item in index if item['source_id'] == source['id'])
        grouped.setdefault(unit['event_id'], []).append(source['text'][:500])
    card = [{'theme':'SourceEvent '+str(k), 'facts':v} for k,v in grouped.items()]
    accepted = sum(item.get('llm_status') == 'accepted' for item in index)
    # Optional local vector indexing is independent of fact classification.
    vector_index = None
    vector_error = 'not_requested'
    result = {'card':card, 'groups':[], 'fact_count':len(records),
        'included_fact_count':len(records), 'excluded_private_fact_count':0,
        'personal_memory_count':len(bundle['respondent']['memory_bank']),
        'preparation_seconds':round(time.monotonic()-started, 3),
        'preparation_method':method, 'preparation_mode':mode,
        'preparation_complete':True, 'preparation_cache_hit':False,
        'llm_api_call_count_this_run':calls,
        'llm_accepted_fact_count':accepted,
        'llm_processed_fact_count':accepted,
        'agent_processed_fact_count':accepted,
        'agent_batch_count':calls,
        'full_bilingual_semantic_complete':accepted == len(records),
        'fast_preparation_quality_status': ('full_llm_fact_sort' if mode=='fast_hybrid' else 'python_binary_v15' if mode=='python_binary' else 'original_unsorted_facts'),
        'python_only_fact_count':len(records)-accepted,
        'python_preprocessed_fact_count':len(records)-accepted,
        'per_fact_cache_hit_count':cache_hits,
        'llm_enrichment_error_code':None,
        'semantic_coverage': ('llm_full_semantic_index' if mode=='fast_hybrid' else 'source_text_only'),
        'multilingual_vector_index':vector_index,
        'multilingual_index_status':vector_error,
        'multilingual_vector_ready':False,
        'multilingual_source_coverage':0,
        'source_id_mapping_complete':all(item.get('event_id') for item in index),
        'metadata_origin_counts':{origin:sum(x.get('metadata_origin')==origin for x in index)
                                  for origin in ('python_source_cues','validated_llm','validated_llm_cache')},
        'factbank_llm_index':index,
        'factbank_llm_index_signature':signature(index),
        'factbank_llm_summary':{'schema':'llm_factbank_index.v154' if mode=='fast_hybrid' else 'python_binary_factbank.v15' if mode=='python_binary' else 'unsorted_source_pool.v1',
            'total':len(records),
            'memory_count':sum(x['fact_category']=='memory' for x in index),
            'reality_count':sum(x['fact_category']=='reality' for x in index),
            'llm_accepted_count':accepted,
            'python_only_count':len(records)-accepted,
            'review_count':sum(bool(x['review_required']) for x in index)},
        'model':model, 'cache_version':'fact-preparation-v2',
        'domain_knowledge':prepare_domain(bundle)}
    _save_agent(key, result)
    _CACHE[key] = result
    return key,result


def lookup(bundle, model, preparation_mode='fast_hybrid'):
    if preparation_mode not in PREPARATION_METHODS:
        raise StudioError('只支持 LLM事实分拣 / 不分拣 两种准备方式',
                          'unsupported_preparation_mode', 400)
    key = _fact_preparation_key(bundle, model, preparation_mode)
    cached = _CACHE.get(key) or _load_agent(key)
    if not cached or cached.get('preparation_method') != PREPARATION_METHODS[preparation_mode] or not cached.get('preparation_complete'):
        raise StudioError('所选分拣方式尚未完成准备，请重新执行受访者准备', 'fact_prepare_not_ready', 409)
    if preparation_mode == 'fast_hybrid':
        from llm_factbank import SCHEMA_VERSION
        native_key = _llm_factbank_key(bundle, model, hybrid=True)
        native = _CACHE.get(native_key) or _load_agent(native_key)
        summary = (native or {}).get('factbank_llm_summary') or {}
        units = (native or {}).get('factbank_llm_index') or []
        if (not native or native.get('preparation_method') != 'python_agent_hybrid_bilingual_v155'
                or native.get('cache_version') != SCHEMA_VERSION
                or native.get('llm_preparation_complete') is not True
                or summary.get('unresolved_count') != 0
                or summary.get('total', 0) < 1
                or len(units) != summary.get('total')
                or any(u.get('llm_status') != 'accepted' for u in units)):
            raise StudioError('原生 FactBank 完整缓存缺失或失效，请重新准备',
                              'native_factbank_not_ready', 409)
    _CACHE[key] = cached
    return cached


def _source_summary_fallback(selected_memory, question, language="zh", coverage=None, attribution=None):
    """Brief, independently worded answer; never concatenate raw transcripts.

    This conservative emergency path deliberately does *not* reproduce source
    text, quote a third-person speaker, or claim an unrecorded date/episode.
    Ordinary grounded answers should come from the respondent LLM itself.
    """
    from source_recall import _question_dimension
    dim=_question_dimension(question)
    source=" ".join(str(f) for item in selected_memory[:3]
                    for f in item.get("facts",[])[:1])
    def cite(items):
        if attribution is not None:
            attribution.extend(x["memory_id"] for x in items if x.get("memory_id"))
    en=language=="en"
    if not source:
        if dim=="frequency":
            return ("I can't place a reliable count on each activity right now."
                    if en else "每种任务具体用多频繁，我一下说不准。")
        if dim=="onset_time":
            return ("I don't have a reliable first-use date to give." if en else
                    "关于最早开始使用的具体时间，我目前没有能够确认的信息。")
        if dim=="specific_episode":
            return ("I can't give a reliable recent example without more detail." if en else
                    "要说最近一次具体任务，我目前没有足够的细节能准确描述。")
        return ("I can't give a reliable personal example for that question." if en else
                "这个我一下子想不起来，您可以再提醒我一点。")

    # Frequency is a separate semantic dimension. Only the RESPONDENT'S
    # frequency terms may support "often/sometimes", with the original
    # question used to interpret elliptical replies. Never turn frequency
    # into invented per-day counts for tasks that were not measured.
    if dim=="frequency":
        from source_recall import _FREQUENCY_EVIDENCE, _FREQUENCY_SOURCE_QUESTION
        grounded=[]
        for item in selected_memory:
            origin_q=str(item.get("source_question_context") or "")
            for fact in item.get("facts") or []:
                fact=str(fact)
                if _FREQUENCY_EVIDENCE.search(fact) and (
                    _FREQUENCY_SOURCE_QUESTION.search(origin_q) or
                    re.search(r"AI|Copilot|ChatGPT|code|coding|test|debug|编程",fact,re.I)
                ):
                    grounded.append((origin_q,fact,item))
        if grounded:
            all_text=" ".join(fact for _,fact,_ in grounded).lower()
            often=bool(re.search(r"very often|\boften\b|frequently|经常|频繁",all_text))
            sometimes=bool(re.search(r"\bsometimes\b|occasionally|偶尔|有时",all_text))
            # Record the actual frequency terms used in the selected branch.
            if often:
                cite([item for _,fact,item in grounded if re.search(r"very often|\boften\b|frequently|经常|频繁",fact,re.I)])
                if not en and sometimes and not re.search(r"高还是低|频率高|频率低|偏高|偏低",question):
                    cite([item for _,fact,item in grounded if re.search(r"\bsometimes\b|occasionally|偶尔|有时",fact,re.I)])
            elif sometimes:
                cite([item for _,fact,item in grounded if re.search(r"\bsometimes\b|occasionally|偶尔|有时",fact,re.I)])
            if en:
                if often:
                    return ("I use AI fairly often for coding, but I haven't "
                            "counted exactly how many times I do each of "
                            "those specific tasks.")
                if sometimes:
                    return ("For some activities it's occasional. I couldn't "
                            "give an exact number of times for each one.")
                return ("I have a rough sense of how often, but not an exact "
                        "per-task count.")
            if often:
                if re.search(r"高还是低|频率高|频率低|偏高|偏低",question):
                    return ("整体用 AI 辅助编程算比较频繁。不过你说的这两种"
                            "具体任务，我没有单独算过频率。")
                if sometimes:
                    return ("写代码、测试这些事我用 AI 还挺经常的，"
                            "但文档、学习之类的就没那么频繁。"
                            "你问到每种任务一天几次，我确实没专门数过。")
                return ("我平时用 AI 辅助编程挺经常的，不过要说"
                        "每类任务一天具体用几次，我没单独统计过。")
            if sometimes:
                return ("有些时候会用，但没有频繁到每次都用。"
                        "具体到一天几次，我不太好估。")
        if en:
            return "I can't give a dependable frequency for that particular task."
        return "具体到这类任务的使用频率，我一下说不准。"

    # Elliptic first-use responses need their original interviewer question.
    # Example: original question "When did you start using AI for programming?"
    # followed by the respondent "6-12 months". The *answer* is the evidence
    # for the interval; the interviewer's question provides only its meaning.
    # Keep the estimate relative to the ORIGINAL interview, not today's date.
    if dim=="onset_time":
        from source_recall import _question_dimension, _AI_USE
        for item in selected_memory:
            origin_q=str(item.get("source_question_context") or "")
            if not (_question_dimension(origin_q)=="onset_time" and
                    _AI_USE.search(origin_q)):
                continue
            facts=" ".join(str(x) for x in (item.get("facts") or []))
            period=re.search(
                r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*(months?|years?)\b",
                facts,re.I)
            if period:
                cite([item])
                first,second,unit=period.groups()
                if en:
                    label=f"{first}–{second} {unit.lower()}"
                    return (f"At the time of that interview, I estimated I had started "
                            f"using AI for programming about {label} earlier; "
                            "I didn't give an exact date.")
                unit_zh="个月" if unit.lower().startswith("month") else "年"
                return (f"当时我估计自己大约是在此前{first}到{second}{unit_zh}开始用AI辅助编程的，"
                        "没有说出确切日期。")
    # The following features have to be explicitly present in this source.
    # Naming a product or an institution solely because the interview asked
    # about it would incorrectly fabricate autobiographical experience.
    def present(pattern):
        match=re.search(pattern,source,re.I)
        return match.group(0) if match else None

    tool=present(r"GitHub\s*Copilot|ChatGPT|Swagger\s*UI|VS\s*Code|"
                 r"PyCharm|Jupyter\s*Notebook|TensorFlow|scikit[- ]learn|"
                 r"DeepSeek|Cursor|Claude|通义灵码")
    university=bool(re.search(r"大学|在校|课程|college|university",source,re.I))
    first_use=bool(re.search(r"最早|第一次|初次|开始.{0,15}(?:使用|尝试|接触)|"
                             r"first (?:used|tried|started)|started using|"
                             r"began using|first time",source,re.I))
    ide=bool(re.search(r"\bIDE\b|集成|插件|扩展|code editor|编辑器",source,re.I))
    actual_recent=bool(re.search(r"最近|上次|有一次|具体项目|那次|"
                                 r"recent project|last project|recently",source,re.I))
    def matching(pattern):
        return [item for item in selected_memory[:3] if re.search(pattern," ".join(str(f) for f in item.get("facts",[])[:1]),re.I)]
    if dim=="onset_time":
        if university and first_use:
            cite(matching(r"大学|在校|课程|college|university"))
            cite(matching(r"最早|第一次|初次|开始.{0,15}(?:使用|尝试|接触)|first (?:used|tried|started)|started using|began using|first time"))
            if tool:cite(matching(re.escape(tool)))
            sentence=("I first tried "+(tool or "AI coding tools")+" while I was at university."
                      if en else
                      "我最早是在大学期间开始尝试"+(tool or "这类编程辅助工具")+"的。")
        elif first_use:
            cite(matching(r"最早|第一次|初次|开始.{0,15}(?:使用|尝试|接触)|first (?:used|tried|started)|started using|began using|first time"))
            sentence=("I did start using it, but the record doesn't specify a date." if en else
                      "我确实开始用过这类工具，但具体是哪天，我记不太准了。")
        else:
            sentence=("I can't reliably identify when I first began using it." if en else
                      "最早到底是哪天，我现在一下说不准。")
        return sentence
    if dim=="specific_episode" and not actual_recent:
        if tool:
            cite(matching(re.escape(tool)))
            return (f"I've used {tool}, but I can't give an accurate recent case from these details."
                    if en else
                    f"我平时确实会用{tool}，不过最近一次具体是哪项任务，我一下想不起来。")
        return ("I know the general way I worked, but I can't reliably describe a specific recent case."
                if en else "平时是怎么做的我能聊聊，但最近具体哪一次，我现在确实想不起来。")
    if tool:
        cite(matching(re.escape(tool)))
        if ide:
            cite(matching(r"\bIDE\b|集成|插件|扩展|code editor|编辑器"))
            return (f"I've mainly used {tool} through my coding environment."
                    if en else f"我主要是在开发环境里使用{tool}来辅助编程。")
        return (f"I've used {tool} as part of my work." if en else
                f"我平时会使用{tool}辅助相关任务。")
    # Generic non-copy fallback when the source does not contain structured
    # details we can safely rephrase without inventing a fact.
    return ("There is some relevant experience, but I can't give the specifics confidently."
            if en else "这方面我有过一些接触，不过具体过程我一时记不全。")


def _anchored_onset_period(selected_memory):
    """Find a first-use duration actually stated by the respondent.

    Source question provides the disambiguating topic but is never used to
    fabricate a duration. A precise date is not inferred from the interval.
    """
    from source_recall import _question_dimension, _AI_USE
    for item in selected_memory:
        origin_q=str(item.get("source_question_context") or "")
        if not (_question_dimension(origin_q)=="onset_time"
                and _AI_USE.search(origin_q)):
            continue
        for fact in item.get("facts") or []:
            match=re.search(
                r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*"
                r"(months?|years?)\b",str(fact),re.I)
            if match:
                return match.groups()
    return None


def _uses_anchored_onset_period(text,period):
    """Avoid silently discarding a known 6–12 month onset interval."""
    if period is None:return True
    from difflib import SequenceMatcher
    lo,hi,_=period
    answer=str(text or "").lower()
    normalized=re.sub(r"[^0-9]+"," ",answer).split()
    return lo in normalized and hi in normalized


def _looks_like_source_dump(answer, selected_memory, language="zh"):
    """Reject verbatim quotations, third-person corpus narration and clipping."""
    from difflib import SequenceMatcher
    response=str(answer or "").strip()
    if not response:return True
    if re.search(r"\b(?:the participant|participant reported|interviewee said|"
                 r"speaker said|respondent said|according to the transcript)\b",
                 response,re.I):
        return True
    if re.search(r"(?i)\b(?:participant|interviewee|speaker)\s+(?:can|uses|think|reported)\b",response):
        return True
    if response.endswith(("…","...")) or "我记得的大致情况是：" in response:
        return True
    if language!="en":
        chinese=len(re.findall(r"[\u4e00-\u9fff]",response))
        if chinese<max(8,len(response)*.18):
            return True
    for card in selected_memory:
        for fact in card.get("facts",[]):
            src=str(fact or "").strip()
            if not src:continue
            # A full English source pasted into a Chinese answer, or a long
            # uninterrupted source quotation, is not respondent synthesis.
            if len(response)>=45 and (response in src or src in response):
                return True
            if len(response)>=65 and len(src)>=65:
                if SequenceMatcher(None,response,src).ratio()>.83:
                    return True
    return False


def _non_memory_reply(question, intent, model, language, timeout_seconds,
                      token_limit, behavior, history, knowledge=None):
    """Never apply personal-memory access rolls to consent, social, or concept questions."""
    from question_intent import procedural_reply
    if intent["kind"]=="procedural":
        return procedural_reply(question,language),{"grounding_gate":"non_memory_procedural",
                 "blocked":False,"model_called":False}
    rules=("你只回答当前提出的通用观点或概念。禁止声称自己亲历某个项目、"
           "使用过某个工具、参与过某次事件，不能编造个人年龄、身份或经历。"
           "普通观点可以自然表达，不需要依靠个人记忆；不得用‘想不起来’"
           "替代对一般观点/概念的回答。语言保持简洁自然，不列清单。"
           if language!="en" else
           "Answer the general opinion or concept directly and naturally. Do not "
           "invent personal experience, biographical events or tool usage.")
    from respondent_behavior import natural_style
    recent=[{"question":str(t.get("question",""))[:180],
             "answer":str(t.get("answer",""))[:240]}
            for t in (history or [])[-3:] if isinstance(t,dict)]
    factual_knowledge=[{"topic":str(e.get("topic",""))[:120],
                         "text":str(e.get("text",""))[:750]}
                       for e in (knowledge or []) if isinstance(e,dict)]
    messages=[{"role":"system","content":rules+"\\n"+
               "【知识隔离】以下领域知识不是我亲历的事实，只可用于通用观点和概念。" +
               natural_style("reasoning",history,language)},
              {"role":"user","content":"以下是背景资料，不是新的采访问题："+json.dumps({
                  "recent_history_reference_only":recent,
                  "domain_knowledge_background_not_autobiography":factual_knowledge
              },ensure_ascii=False)},
              {"role":"user","content":"本轮采访问题（请只回答这一问）：\n"+question}]
    text=_call_plain_answer(model,messages,timeout_seconds,token_limit,temperature=.20)
    # A general opinion is not proof of any personal history. Surface that in telemetry.
    return text.strip(),{"grounding_gate":"non_memory_"+intent["kind"],
                         "blocked":False,"model_called":True}

def _roleplay_without_source(question,model,language,timeout_seconds,
                             token_limit,history,temporal_target,behavior,
                             selected_memory=None,memory_coverage=None):
    """A role-play completion, explicitly segregated from sourced FactBank.

    Simulated quotidian plausible details are allowed by design, but they do
    not become factual provenance, bank entries or future ground-truth.
    """
    from respondent_behavior import topical_response,natural_style,uncertainty_guidance
    from source_recall import _question_dimension
    dimension=_question_dimension(question)
    if dimension in ("specific_episode","onset_time","frequency"):
        if selected_memory:
            summary=_source_summary_fallback(
                selected_memory,question,language,memory_coverage or [])
            suffix={
                "specific_episode":("不过，我现在想不起最近一次的具体案例。" if language!="en" else " I cannot recall a specific recent episode."),
                "onset_time":("不过，最早开始使用的准确时间我记不清了。" if language!="en" else " I cannot recall the exact starting time."),
                "frequency":("不过，具体使用频率我说不准。" if language!="en" else " I cannot state the exact frequency."),
            }[dimension]
            answer=(summary.rstrip("。.! ")+suffix) if summary else uncertainty_guidance(question,history,language)
        else:
            answer=uncertainty_guidance(question,history,language)
            answer=re.sub(r"^\s*让我想想[…….]*\s*", "", answer).strip()
        return answer,{"grounding_gate":"no_direct_"+dimension+"_evidence",
            "blocked":True,"model_called":False,
            "fallback":"source_summary_plus_explicit_uncertainty" if selected_memory else "natural_uncertainty",
            "source_status":"partial_background_only" if selected_memory else "no_direct_source",
            "attempts":0}
    # Do not improvisationally validate impossible claims, identifiable
    # biographies or high-stakes personal histories. This is a simulation,
    # not a license to imply an unsupported real person lived these events.
    if re.search(
        r"火星|月球基地|月球城市|外星|航天员|犯罪|被捕|判刑|"
        r"确诊|病史|诊断|欠债|财务账户|"
        r"真实姓名|具体学校|哪个公司|公司全称|身份证|"
        r"Mars|moon base|criminal record|medical diagnosis",
        str(question),re.I):
        return uncertainty_guidance(question,history,language),{
            "grounding_gate":"roleplay_sensitive_or_impossible_detail",
            "blocked":True,"model_called":False,
            "fallback":"natural_unsourced_boundary",
            "source_status":"simulated_unverified",
            "attempts":0}
    recent=[{"question":str(t.get("question",""))[:200],
             "answer":str(t.get("answer",""))[:230]}
            for t in (history or [])[-3:] if isinstance(t,dict)]
    if language=="en":
        rules=("You are role-playing a respondent in an interview, NOT a fact "
               "lookup system. Answer only the current question. No verified "
               "fact was retrieved. You may improvise ordinary situational "
               "thoughts, a plausible viewpoint or a minor illustrative "
               "detail, without claiming verification. Never invent sensitive "
               "biographical events, named employers, precise dates, medical "
               "history, criminal acts or other traceable identity details. "
               "Keep it natural, first-person, concise, with no disclaimers "
               "about datasets or memory banks. Don't contradict earlier "
               "conversation or pretend you know specifics you do not.")
    else:
        rules=(
            "你正在扮演一位参加研究访谈的受访者，而不是事实检索助手。"
            "没有检索到能直接回答这次问题的个人事实。"
            "请真实地代入角色和目前的对话现场：简短、口语化、第一人称，"
            "优先回答采访者真正问的角度，允许自然迟疑、即时观点、"
            "常见工作习惯的合理模拟，以及不具可核查性的生活化小细节。"
            "这些均为模拟即兴补全，不属于真实来源证据。"
            "不允许编造具体姓名、学校、公司、确切日期、可核查的项目、"
            "医疗隐私、犯罪经历、财务或其他敏感个人事实；"
            "不要为了回答而制造具体历史事件。"
            "如果询问的是明确的过去事件而没有可信依据，可以自然说想不起准确那次，"
            "再补充一般的可能做法或感受，不要只说‘信息不足’。"
            "避免‘数据集没有记录’、‘这里没有细节’、‘根据记忆’等系统措辞。"
            "最多两三句话，不列清单；也不要无关地表示‘我们继续聊吧’。")
    messages=[
        {"role":"system","content":rules+"\n"+natural_style("reasoning",history,language)},
        {"role":"user","content":"上一段现场对话，只用于承接语境，不是新问题："+
              json.dumps(recent,ensure_ascii=False)},
        {"role":"user","content":"本轮采访问题（必须正面回应）："+question.strip()},
    ]
    # One on-topic repair only; never write generated details back to bank.
    for attempt in (1,2):
        try:
            candidate=str(_call_plain_answer(
                model,messages,timeout_seconds,min(int(token_limit),360),
                temperature=.34)).strip()
            candidate=re.sub(r"(?i)^\s*(?:回答|answer)\s*[:：]\s*","",candidate)
            if topical_response(question,candidate):
                return candidate,{"grounding_gate":"unsourced_roleplay",
                    "blocked":False,"model_called":True,
                    "fallback":"respondent_improvisation",
                    "source_status":"simulated_unverified",
                    "attempts":attempt}
            messages[0]["content"]+=(
                "\n上一次模拟偏题了：必须直接回应当前问题中的对象和所问角度，"
                "不要只是续聊或复述先前内容。")
        except (StudioError,TimeoutError,OSError):
            # Preserve transport failure classification if the model service
            # itself is unavailable; do not claim the model successfully spoke.
            if attempt==1:raise
            break
    return uncertainty_guidance(question,history,language),{
        "grounding_gate":"improvisation_answer_focus_rejected",
        "blocked":True,"model_called":True,
        "fallback":"topic_specific_natural_uncertainty",
        "source_status":"simulated_unverified",
        "attempts":2}


def _normalize_respondent_output(raw):
    """Normalize provider/agent framing without fabricating or rewriting facts.

    Like the reference Interview Agent, respondent output is an utterance, not
    a JSON report.  Retain content verbatim apart from presentation wrappers.
    """
    text=str(raw or "").strip()
    text=re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I|re.S).strip()
    # Some compatible APIs include an incomplete reasoning prefix.
    text=re.sub(r"^```(?:text|markdown)?\s*\n([\s\S]*?)\n```\s*$",r"\1",text,flags=re.I).strip()
    for _ in range(2):
        newer=re.sub(r"^(?:受访者(?:回答)?|采访对象|Respondent|Answer|回答)\s*[:：]\s*", "", text, flags=re.I).strip()
        if newer==text: break
        text=newer
    return text.strip(' \n\t\"“”')


def _answer_has_clear_evasion(text):
    from respondent_behavior import _plain_evasion
    return _plain_evasion(text)


def _answer_with_fact_citations(model,messages,timeout,num_predict,selected,temperature):
    allowed={str(x.get("memory_id")) for x in selected if x.get("memory_id")}
    if not allowed:
        return _normalize_respondent_output(_call_plain_answer(model,messages,timeout,num_predict,temperature=temperature)),[],True
    copied=[dict(m) for m in messages]
    copied[0]["content"] += ("\nFor internal attribution only, after the natural answer append exactly one final line "
        "MEMORY_IDS: followed by comma-separated Memory ID values actually expressed in your answer. "
        "Do not list merely retrieved facts. If none, write MEMORY_IDS: NONE. "
        "Allowed IDs: "+", ".join(sorted(allowed))+". Never include this line in the spoken answer itself.")
    raw=_normalize_respondent_output(_call_plain_answer(model,copied,timeout,num_predict+64,temperature=temperature))
    import re
    match=re.search(r'(?:^|\n)\s*MEMORY_IDS\s*:\s*([^\n]*)\s*$',raw,re.I)
    if not match:
        # Missing attribution metadata must not invalidate a safe spoken answer.
        # Missing citations remain explicitly incomplete in the export.
        return raw,[],False
    payload=match.group(1).strip()
    ids=[x.strip() for x in re.split(r'[,，;；]',payload) if x.strip()] if payload.upper()!='NONE' else []
    valid=list(dict.fromkeys(i for i in ids if i in allowed))
    complete=(payload.upper()=="NONE" or (bool(ids) and all(i in allowed for i in ids)
        and len(set(ids))==len(ids)))
    return _normalize_respondent_output(raw[:match.start()]),valid,complete


def answer(bundle,question,model,history,traits,language,preparation_mode="python",
           session_id="",turn_index=0,timeout_seconds=300,retrieval_strategy="new_retrieval",
           retrieval_timeout_seconds=240):
    from respondent_timeout import TurnBudget
    budget=TurnBudget(retrieval_timeout_seconds,timeout_seconds)
    token=_TURN_BUDGET.set(budget)
    try:
        result=_answer_impl(bundle,question,model,history,traits,language,preparation_mode,
            session_id,turn_index,timeout_seconds,retrieval_strategy)
        budget.check()
        result[1]["_turn_memory_selection"]["timeout_management"]=budget.snapshot()
        return result
    finally:
        _TURN_BUDGET.reset(token)

def _answer_impl(bundle,question,model,history,traits,language,preparation_mode="python",
           session_id="",turn_index=0,timeout_seconds=300,retrieval_strategy="new_retrieval"):
    turn_started=time.monotonic()
    answer_rewrite_count=0
    answer_rewrite_success=False
    cited_units=[]; citation_complete=False
    template_citations=[]; template_used=False
    from respondent_behavior import route, conversational_reply, natural_style
    behavior=normalize(traits or bundle['respondent']['behavior_parameters'])
    length_guide,token_limit=verbosity_guide(float(behavior['verbosity']))
    # v1.4.41: The behavioral router MUST NOT suppress retrieval. Every
    # incoming turn first probes permitted Memory, then decides whether
    # retrieved source evidence is actually relevant. In particular, a
    # lexical "刚才提到" must never hijack an autobiographical question
    # into a canned clarification reply.
    # The route is a *fallback hint*, not a gate for source inspection.
    prepared=lookup(bundle,model,preparation_mode)
    if not prepared['card'] and prepared.get("fact_count",0):
        raise StudioError('FactBank 尚无许可事实，请先检查来源授权与分类准备。',
                          'factbank_no_permitted_facts',409)
    # Prepared memory is a locator, not an authority to invent personal facts.
    # The respondent only receives eligible source facts after per-turn retrieval.
    max_groups,max_chars=memory_budget(behavior['memory'])
    # Historical answers are conversational context, NOT a new factual source.
    # Repetition never constitutes a transport or validation error.
    previous_answers=[str(t.get('answer') or '').strip() for t in (history or []) if isinstance(t,dict)]
    recent=[]
    for turn in (history or [])[-4:]:
        if isinstance(turn,dict):
            recent.append({k:str(turn[k])[:140] for k in ('question','answer') if k in turn})
    # Cache-only retrieval is never mistaken for proof. The separate source
    # index is prepared from permission-filtered V2 facts, then a reproducible
    # access roll determines if the matching original text may be recovered.
    from source_recall import recall
    semantic_selector=None
    # v1.8 route consolidation: intent is annotation only, never a retrieval gate.
    semantic_skip_reason=None
    if retrieval_strategy in ("python_retrieval", "legacy_cache"):
        # This opt-in baseline cannot guarantee a live LLM selector; fail closed.
        raise StudioError("正式访谈要求实时 LLM 语义检索；Python 基线检索不可用于正式回答。",
                          "semantic_llm_required",409)
    turn_budget=_TURN_BUDGET.get()
    semantic_cache_stats={"hits":0,"misses":0,"actual_api_calls":0}
    # Counters are shared by up to two concurrent memory selection batches.
    from threading import Lock
    semantic_stats_lock=Lock()
    if retrieval_strategy in ("legacy_cache","fused_retrieval","new_retrieval","unified_retrieval","python_binary_retrieval") and not semantic_skip_reason:
        # v0.9.1: semantic selection is independent of the respondent answer.
        # The selector sees question/answer excerpts, but can return only IDs.
        # v1.5: The new main retriever and the original legacy path both use LLM semantics.
        from semantic_selector import select_with_respondent_model, PROMPT as SEMANTIC_PROMPT
        def semantic_selector(q,previous,candidates,limit):
            def live_select(question, context, choices, top_n):
                with semantic_stats_lock:
                    semantic_cache_stats["actual_api_calls"]+=1
                # The default limit remains the tested v1.5 value to avoid
                # truncating JSON or reasoning output. Smaller limits opt-in.
                token_budget=int(os.getenv("INTERVIEW_SEMANTIC_MAX_TOKENS", "512"))
                token_budget=max(256,min(1024,token_budget))
                return select_with_respondent_model(
                    question,context,choices,top_n,
                    lambda messages: _timed_semantic_call(messages,token_budget))
            def _timed_semantic_call(messages,token_budget):
                token=_TURN_BUDGET.set(turn_budget)
                try:
                    return _call_plain_answer(model,messages,120,token_budget,temperature=0)
                finally:
                    _TURN_BUDGET.reset(token)
            # Each formal turn must make a live request. Do not use the exact
            # semantic cache: a hit would otherwise bypass the selector LLM.
            selected=live_select(q,previous,candidates,limit)
            with semantic_stats_lock:
                semantic_cache_stats["misses"]+=1
            return selected
    retrieval_started=time.monotonic()
    selected_memory,selected_knowledge,recall_trace=recall(
        bundle,prepared,question,history or [],behavior,session_id,turn_index,
        retrieval_strategy=retrieval_strategy,semantic_selector=semantic_selector,
        semantic_skip_reason=semantic_skip_reason)
    retrieval_elapsed_ms=round((time.monotonic()-retrieval_started)*1000)
    semantic_stats_snapshot=dict(semantic_cache_stats)
    if not semantic_stats_snapshot["actual_api_calls"]:
        # Empty source bank can still be an interview question: run the model
        # once with an empty candidate set, rather than silently skipping it.
        if semantic_selector is None:
            raise StudioError("语义 LLM 未配置，不能生成正式回答。",
                              "semantic_llm_required",409)
        semantic_selector(question, str(recent[-1].get("answer", "")) if recent else "", [], 0)
        semantic_stats_snapshot=dict(semantic_cache_stats)
    if not semantic_stats_snapshot["actual_api_calls"]:
        raise StudioError("本轮没有执行实时语义检索 LLM。", "semantic_llm_required",502)
    # Explicit stage routing: 1 retrieval -> 2 judgement; retrieval miss
    # chooses 3 cache answer; judgement miss chooses 4 ordinary response.
    # Unlike HR2/HR4, a failed judgement never uses cached source facts.
    four_stage_branch=recall_trace.get("response_branch")
    # Targeted slow-round relief: only genuine retrieval misses receive a larger
    # answering/rewrite request timeout. Successful retrieval keeps the tested
    # v1.6.0 timing and all access gates unchanged.
    if four_stage_branch in ("cache_answer", "no_cache_evidence"):
        timeout_seconds=min(900, float(timeout_seconds)+45)
        recall_trace["answer_timeout_relief_seconds"]=45
    turn_budget.start_answer(relief=45 if four_stage_branch in ("cache_answer","no_cache_evidence") else 0)
    if four_stage_branch=="memory_judgement_failed":
        selected_memory=[]
    # Cache answer is source grounded but must be identified independently of
    # a successful primary retrieval in exported diagnostic fields.
    # Only AFTER retrieval is finished do we consider a conversational mode.
    # The router is an advisory for what to do with absent/partial evidence,
    # never permission to skip the first stage.
    action=route(question,history)
    intent=action["intent"]
    question_analysis={
        "selection_status":recall_trace["fact_match_status"],
        "selected_groups":recall_trace["fact_visible_count"],
        "total_groups":recall_trace["fact_prepared_count"],
        "aspects":["source_grounded"],
        "history_relation":"previous_turns_context_only",
    }
    memory_access_trace={
        "access_policy":"memory_only_probabilistic_fact_retrieval",
        "temporal_target":recall_trace["question_temporal_target"],
        "reality_source_recovered":recall_trace["reality_source_recovered"],
        "source_recovered":(recall_trace["memory_source_recovered"] or
                            recall_trace["reality_source_recovered"]),
        "cache_fallback":not (recall_trace["memory_source_recovered"]
                               or recall_trace["reality_source_recovered"]),
    }
    knowledge_cache=prepared.get("domain_knowledge") or {}
    prompt={'integrated_memory':selected_memory,
            'memory_selection_status':question_analysis['selection_status'],
            'current_question_analysis':{
                'requested_aspects':question_analysis['aspects'],
                'relation_to_history':question_analysis['history_relation'],
                'answer_focus':'先直接回应当前问题提出的角度；没有证据时直接承认不确定，不许编造。',
                'requested_question_dimension':recall_trace.get("question_dimension"),
        'retrieval_evidence_levels':recall_trace.get("memory_evidence_levels",[]),

            },
            'current_question':question,'recent_history':recent,
            'agent_voice_guide_unverified':prepared.get('agent_voice_guide',''),
            'interview_turn':len(previous_answers)+1,
            'follow_up_policy':{'when_prior_topic_revisited':'如果当前问题追问刚才讲过的经历，优先补充此前没有说过、且整合记忆确实支持的具体细节（例如动作、时间、后果、个人感受），避免复述完整经历。',
                                'when_no_new_supported_detail':'如果已经说完，直接用自然口语说没有更多细节、记不清或不方便继续谈；不要为了回答而编造。',
                                'when_question_changes':'问题换了角度就只回答新角度，不得套用上一轮答案。',
                                'never':'不要把公共干扰记忆写成自身经历，不要输出 AI 式总结或建议。'},
            'response_focus':'不要复述、翻译或改写问题本身；直接以第一人称回答问题的核心。没有信息时坦言不知道。',
            'behavior_parameters':behavior,'language':language,
            'answer_style':{'length':length_guide,
                            'answer_first':'首先直接回应当前问题，不主动回答未问到的其他问题',
                            'avoid_repetition':'追问时只新增有事实支持的细节；没有新细节时直接承认无法补充或礼貌回绝，不复述已有整段回答',
                            'human_voice':'像真实被访者在现场聊天一样回应：用我/当时/后来等口语，自然但不刻意演戏，不讲课、给用户建议、分点讲解或写总结',
                            'do_not_replay':'不要套用前几轮的完整说法。若问题涉及同一事件，只补充当前问题询问的那个侧面；没有新增内容就直接说明。',
                            'uncertainty':'没有依据时坦言不确定，不编造'},
            'source_policy':'Only the SELECTED integrated_memory contains personal facts relevant to this turn. '
                'Public distractors are NON-FACTUAL and must never become autobiographical claims. '
                'No memory IDs or citations in answers.'}
    # v0.13.2: the current question is the LAST independent chat message.
    # Earlier versions buried it inside a large JSON object alongside old answers
    # and memory, making small-context models disproportionately follow old text.
    # History and memory are reference DATA, never the new instruction.
    # Keep the reference payload compact enough for Ollama's 4096-token context.
    context_data = {
        'factbank_memory': [m for m in selected_memory
                            if m.get("fact_category")=="memory"],
        'factbank_reality': [m for m in selected_memory
                             if m.get("fact_category")=="reality"],
        'source_time_reference': "as_of_original_interview_not_verified_current",
        'source_evidence_policy': {
            'unit_identity': 'stable source Memory IDs; verbatim source evidence stays in the authorized source bank',
            'event_boundary': 'different event_id values are distinct events and must not be merged',
            'relative_time': 'preserve source-relative time phrases; never convert them to the current date',
            'access_boundary': 'only selected observable-layer sources are usable; latent source records are not answer evidence',
        },
        'domain_knowledge_background_not_autobiography': selected_knowledge,
        'factbank_selection_status': recall_trace.get("fact_match_status"),
        'memory_selection_status': question_analysis['selection_status'],
        'requested_aspects': question_analysis['aspects'],
        'history_relation': question_analysis['history_relation'],
        'recent_history_reference_only': recent,
        'behavior_parameters': behavior,
        'behavioral_instruction': behavioral_guide(behavior, sensitivity=question_sensitivity(question)),
        'memory_access_protocol': memory_access_trace,
        'requested_question_dimension':recall_trace.get("question_dimension"),
        'answer_evidence_format': 'Read source snippets internally, then answer naturally in first person. '
             'Never output transcript excerpts, third-person narrations, or unfinished quotations. '
             'If requested a first-use date, source must actually establish onset. '
             'If asked a recent example, general tool familiarity is not a recent episode.',
        'source_recall_limitations': {
            'memory_source_recovered':recall_trace['memory_source_recovered'],
            'knowledge_source_recovered':recall_trace['knowledge_source_recovered'],
            'cache_only_when_failed':'低等级或失败状态仅允许解释已给的简短线索。不得凭概念常识补写原始经历。',
        },
        'answer_length_guide': length_guide,
        'language': language,
    }
    # Agent-produced voice instructions are not a verified personal-memory
    # source. Do not inject potentially invented autobiographical summaries.
    system_text = (SYSTEM_POLICY_ZH + """
【FactBank：来源事实与角色化表达】
你是一位参与访谈的受访者。先代入人物、理解当前问题，再用自然口语回答。
事实索引包含两类：factbank_memory 是过去经历，受 Memory 参数影响；
factbank_reality 是原始采访时的日常状态、习惯、观点，不受 Memory 参数影响。
两者均来自许可源。现实事实的“现在”特指原始访谈时间，并不自动等同于今天。
对于能检索到的个人事实，必须以本轮所给来源为依据，不编造新的可核查经历。
缓存中的简短线索不是恢复细节的许可，禁止凭常识、模型已有知识或历史答案补全人物、项目、
日期、人数、动机、心理、步骤、结果、引用和因果关系；不能以“可能”“大概”包装虚构事实。
如果缓存仅提供大意，就根据已给大意回答主要事实；仅在询问的特定时间、数量或细节没有证据时说明那一处不确定。
如果没有对应经历的可靠信息，可以说目前没有足够依据；不要把整个问题机械回答为“我忘记了”；
不要强行回答所有问题，也不要为了看起来专业而新增技术经历。
不得逐字输出或翻译贴出记忆原文，特别是 “the participant …” 等第三人称访谈记录。
必须先识别这轮询问的是时间、最近案例、使用方式还是态度，并用自然的一人称中文重新组织。
问题询问“最早何时”时，不能拿一般的 AI 发展观点充当首次使用经历。
问题询问“最近一次具体案例”时，不能拿平时使用某工具的笼统描述冒充具体项目。
如果只有相关背景但缺少所问细节，应先回答有依据的那部分，再说明该细节无可靠信息。
通用领域知识只能用于解释概念，绝不能成为你亲身做过某事的证据。
提供完整原文时也只复述有依据的内容，不得推断原文没有说的细节。
""" + "\n【知识来源隔离】domain_knowledge_background_not_autobiography 仅是通用领域知识，不是我亲历的事实。"
        "允许自然使用通用知识解释术语，但不得据此虚构经历、身份、行动、动机、个人观点或事实。"
        "有证据的个人经历只能依据本轮 factbank_memory / factbank_reality 的来源内容；"
        "没有来源时进入标记为模拟的角色化应对，不得把即兴说法当真实来源。" + "\n【问题隔离规则】最后一条 user 消息中的‘本轮采访问题’是唯一要回答的问题。"
        "前一条 user 消息只是历史及可用事实，不是采访者的新问题。"
        "只回答最后一条问题询问的角度；不复述历史问题，不为了使用记忆而讲旧故事。"
        "如果问的是频率，回答粗略的频繁程度或具体次数；仅仅说明使用 Copilot 或 IDE 不是频率回答。"
        "如果来源没有给出精确使用次数，可以自然说没数过，不要猜一天几次。"
        "没有相关证据就自然说不知道、记不清或不想谈。重复回答也属有效回答，不中断。"
        "使用第一人称自然口语，不输出 JSON、标题、列表、记忆编号或分析步骤。")
    system_text += "\n【现场交流行为】" + natural_style(action["mode"],history,language)
    system_text += (
        "\n【记忆检索质量】检索到的条目不代表一定能回答问题。"
        "如果只找到同一主题的背景信息，而没有问题所要求的事实维度，"
        "只能说明已知部分，并自然表达无法确认的那部分。"
        "不要将使用工具的记录推论为使用频率、动机、情绪或具体事件。"
        "原采访问题只是帮助理解短回答的上下文，不能独立当作个人事实。"
    )

    # Intent is descriptive only. Recording response guidance must not enter
    # the normal grounded or cached-source prompt before retrieval/judgement.
    # Only the context-only coping path may receive this guidance.
    from question_intent import classify as classify_turn_intent
    from respondent_behavior import recording_answer_guidance

    # Only remote answer generation; retrieval already finished. Preserve all
    # selected sources, history, behavioral instructions and source boundaries.
    if _REMOTE_RESPONDENT.get() in ("maas","shubi") and os.getenv("INTERVIEW_RESPONDENT_COMPACT_PROMPT","1") != "0":
        system_text = '你是正在接受研究访谈的人，以自然第一人称口语回答最后一条用户消息。前一条仅是资料。\n只依据本轮许可的 memory/reality 或缓存线索陈述个人事实；现实状态以原访谈时点为准，不换算到今天。不同 event_id 不合并成同一经历。通用知识仅解释概念，不能当本人经历；公共干扰记忆不是事实。\n直接回答当前角度；追问只补充有依据的新细节。同一事实可支持不同角度；没有新增信息就自然说明，不机械复读，不自行结束访谈。遵守行为参数、长度和披露边界，可礼貌拒答；未知与不愿披露不同。\n不编造时间、数量、人物、项目、动机、情绪、步骤、结果或因果。原采访问题仅解释来源短答，不是本人陈述。问首次时间必须有起始证据；问近期案例不能以工具习惯冒充；问频率不能以工具名称代替，也不猜次数。只有部分证据就回答有依据部分，明确其余不确定；缓存大意不授权补全细节。无依据的角色化应对不能当真实来源。\n勿逐字贴出或翻译第三人称源记录，勿复述提问、讲课、建议、列表、标题或分析。按指定语言回答。' + "\n【现场交流行为】" + natural_style(action["mode"],history,language)
        for k in ("source_evidence_policy","answer_evidence_format","memory_selection_status","source_time_reference"):
            context_data.pop(k,None)
        # Keep source text and identity, original question, event, category,
        # and every remaining metadata field; only remove redundant instructions.
        context_data["source_time_reference"]="as_of_original_interview"
    last_message = '本轮采访问题（请只回答这一问）：\n' + question.strip()
    messages = [
        {'role':'system','content':system_text},
        {'role':'user','content':'以下是上下文资料，不是本轮新问题：\n'+json.dumps(context_data, ensure_ascii=False)},
        {'role':'user','content':last_message},
    ]
    # Non-sensitive hashes are exported so mis-routing is distinguishable from
    # the LLM reusing a prior answer. Never export the source excerpts here.
    question_digest = hashlib.sha256(question.strip().encode('utf-8')).hexdigest()
    model_last_question = messages[-1]['content'].split('\n',1)[1]
    if model_last_question != question.strip():
        raise StudioError('模型请求的最终问题与当前轮不一致；未调用模型。', 'integrated_question_routing_mismatch', 409)
    selected_digests = [hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]
                        for item in selected_memory]
    from grounding_guard import verify
    from respondent_behavior import topical_response
    # The intent classifier remains available as a descriptive hint, but
    # neither intent nor the question dimension selects a response path.
    purely_live=False
    asks_personal=None
    usable_memory=bool(selected_memory)
    if four_stage_branch=="cache_answer":
        consolidated_route="cache_after_retrieval_miss"
    elif four_stage_branch=="no_cache_evidence":
        consolidated_route="context_only_after_retrieval_miss"
    elif four_stage_branch in ("grounded_answer","memory_judgement_failed"):
        consolidated_route="judged_primary_retrieval"
    else:
        raise StudioError("未知的检索判定状态，已终止本轮。","unknown_retrieval_branch",502)

    # Nothing blocked by the memory probability gate may enter the answer
    # prompt, even if it appeared among the originally retrieved candidates.
    if consolidated_route=="context_only_after_retrieval_miss" or four_stage_branch=="memory_judgement_failed":
        selected_memory=[]
    if consolidated_route=="cache_after_retrieval_miss" and not selected_memory:
        raise StudioError("缓存判定成功却没有获得许可的缓存来源。","cache_source_missing",502)

    if selected_memory:
        if consolidated_route=="cache_after_retrieval_miss":
            # Preserve the original cached-source answer prompt for ALL modes,
            # while the prerequisite primary retrieval remains live semantic.
            from legacy_memory_compat import legacy_messages
            answer_messages=legacy_messages(question,selected_memory,recent,
                behavior,language,length_guide,
                selection_trace=recall_trace.get("fusion_memory_access") if retrieval_strategy=="fused_retrieval" else None)
            answer_temperature=0.25
        else:
            answer_messages=messages
            answer_temperature=0.12 if consolidated_route=="cache_after_retrieval_miss" else 0.20
    else:
        # Context-only path: no refused memory, source cache, or knowledge
        # bank facts are passed. Ordinary social replies still use the LLM.
        policy=(
            "你是一位接受研究访谈的受访者。只回答本轮问题；参考上一段对话的语言风格与语境。"
            "本轮没有获准使用的个人事实，不得编造具体人物、时间、工具使用经历、项目细节、因果或可核查的身份。"
            "如果问到没有证据的具体经历，请自然说明不确定或记不清；可以回答普通社交或通用观点。"
            "必须直接回应问题，不要只说‘可以，我们继续聊吧’。用第一人称简洁口语回答。"
            if language!='en' else
            "You are the interview respondent. Answer the current question, using "
            "prior conversation only as conversational context. No verified personal "
            "facts have been authorized. Do not invent specific experiences, tools, "
            "dates, identity or project details. Express uncertainty when needed. "
            "Answer on topic, naturally and concisely."
        )
        answer_messages=[
            {"role":"system","content":policy+"\n"+natural_style(action["mode"],history,language)},
            {"role":"user","content":"对话上下文（仅供理解，不作为个人事实证据）："+json.dumps(recent,ensure_ascii=False)},
            {"role":"user","content":"本轮采访问题（只回答这一问）：\n"+question.strip()},
        ]
        if classify_turn_intent(question).get("reason")=="recording_consent":
            answer_messages[0]["content"] += "\n" + recording_answer_guidance(language)
        answer_temperature=0.20

    # Presentation-only guidance shared by grounded, cache, and context-only
    # answer routes. Inject after routing; cannot change retrieval or access.
    from respondent_behavior import answer_focus_guidance
    answer_messages[0]["content"] += "\n" + answer_focus_guidance(question,language)

    candidate,cited_units,citation_complete=_answer_with_fact_citations(
        model,answer_messages,timeout_seconds,token_limit,selected_memory,
        temperature=answer_temperature)
    if not candidate.strip():
        raise StudioError("LLM 返回了空回答。","respondent_empty_answer",502)

    def evaluate_answer(text):
        # Facet/keyword overlap is a retrieval heuristic, not a hard validity
        # predicate for natural paraphrases or cross-language interview speech.
        # Reject empty text and unmistakable transitions; factual support is
        # still enforced separately by the grounding guard.
        if not text.strip() or len(text)>2000 or _answer_has_clear_evasion(text):
            return False,text,{"grounding_gate":"answer_off_topic","blocked":True}
        if selected_memory:
            # Preserve the existing question-dimension quality rules in the
            # consolidated route. This is safety behavior, not a fast path.
            dimension=recall_trace.get("question_dimension")
            if dimension=="onset_time":
                anchored=_anchored_onset_period(selected_memory)
                if anchored and not _uses_anchored_onset_period(text,anchored):
                    return False,text,{"grounding_gate":"model_omitted_source_grounded_onset_interval","blocked":True}
            if dimension=="frequency":
                from source_recall import _FREQUENCY_EVIDENCE, _FREQUENCY_SOURCE_QUESTION
                has_frequency=any(
                    _FREQUENCY_EVIDENCE.search(str(f)) and (
                        _FREQUENCY_SOURCE_QUESTION.search(str(item.get("source_question_context") or ""))
                        or re.search(r"AI|Copilot|ChatGPT|coding|test|debug|编程",str(f),re.I))
                    for item in selected_memory for f in item.get("facts",[]))
                if has_frequency and not re.search(
                    r"经常|频繁|偶尔|有时|每天|每周|几次|次数|高|低|说不准|记不清|"
                    r"不能确定|统计|没数|often|sometimes|rarely|weekly|daily",text,re.I):
                    return False,text,{"grounding_gate":"model_answer_does_not_address_frequency","blocked":True}
            safe,audit=verify(text,selected_memory,language)
            if audit.get("blocked") or _looks_like_source_dump(text,selected_memory,language):
                return False,text,{"grounding_gate":audit.get("grounding_gate","source_dump"),"blocked":True}
            return True,safe,audit
        # With zero personal facts, guard specific unverifiable statements.
        # A Python template is never returned as the respondent's answer.
        named_claim=re.search(r"(?:\d{4}年|\d+\s*(?:个月|年|次|岁)|我在[^。\n]{0,35}(?:公司|学校)工作)",text)
        if named_claim and re.search(r"以前|当时|那次|曾经|过去|上次|首次|具体",question):
            return False,text,{"grounding_gate":"unsupported_unsourced_specific","blocked":True}
        return True,text,{"grounding_gate":"llm_context_only","blocked":False}

    # Always preserve the LLM's actual utterance for downstream research.
    # Content-quality checks may suggest a rewrite but must never discard a
    # non-empty model response or silently replace it with a Python template.
    initial_model_answer = candidate
    answer_attempts = [{"attempt": 1, "text": candidate}]
    good,clean_answer,validation=evaluate_answer(candidate)
    answer_attempts[0]["validation"] = dict(validation)
    if not good:
        answer_rewrite_count+=1
        repair_messages=[
            {"role":"system","content":(
                "你是研究访谈受访者。上一版回答未通过安全或相关性检查，重新回答当前问题。"
                "必须正面回答所问角度，只使用以下获准的个人事实。没有事实时自然承认不确定，"
                "不得编造历史事件、细节、原因或人物；不复述来源原文。不输出 JSON 或来源编号。"
                if language!='en' else
                "You are the interview respondent. Rewrite the answer to address "
                "the question directly. Only authorized personal facts are evidence; "
                "if none, express uncertainty without inventing experiences."
            )},
            {"role":"user","content":json.dumps({
                "current_question":question,
                "recent_conversation_context_only":recent,
                "authorized_personal_facts":selected_memory,
                "requested_dimension":recall_trace.get("question_dimension"),
                "language":language,
                "validation_failure_reason":validation.get("grounding_gate"),
                "repair_focus":(
                    "前一次回答偏题或空泛。请用不同措辞直接回应当前问题；若无证据，"
                    "只说不确定的那一部分，不要重复套话。录音同意属于现场安排，"
                    "允许对用途、保密和额外使用前征求同意作条件性回答。"
                    if language!="en" else
                    "Address the current question directly, with a concrete on-topic "
                    "response or natural uncertainty; recording consent is a present "
                    "conversational choice, not autobiographical evidence."
                ),
            },ensure_ascii=False)}
        ]
        if not selected_memory and classify_turn_intent(question).get("reason")=="recording_consent":
            repair_messages[0]["content"] += "\n" + recording_answer_guidance(language)
        revised,cited_units,citation_complete=_answer_with_fact_citations(
            model,repair_messages,timeout_seconds,min(token_limit,450),selected_memory,
            temperature=0.05)
        if revised and revised.strip():
            answer_attempts.append({"attempt": 2, "text": revised})
            good,clean_answer,validation=evaluate_answer(revised)
            answer_attempts[-1]["validation"] = dict(validation)
        else:
            # An empty repair must not erase the initial usable model answer.
            good = False
            validation = {"grounding_gate": "empty_rewrite", "blocked": True}
        if not good:
            # One LAST tightly scoped LLM repair, never a Python-generated
            # utterance. Do not expose blocked memories or add evidence.
            answer_rewrite_count+=1
            last_prompt=(
                "只输出受访者真正要说的一段自然口语，不输出角色名、JSON、解释或来源。"
                "严格依据以下获准事实，未授权的历史事实不可使用。"
                "如果缺少问题所要的具体信息，直接自然说明不确定。"
                "不要添加任何未在获准事实中出现的工具名称、数字、日期或个人经历。"
                "不要直接照抄原文。针对本轮问题回答，避免空泛的续聊语。"
                if language!='en' else
                "Output only the respondent's spoken answer. Use only the authorized "
                "personal evidence; if the needed detail is absent, say so naturally. "
                "Do not invent named entities, dates, numbers or episodes. Do not "
                "quote source records or output JSON, labels or IDs."
            )
            last_messages=[
                {"role":"system","content":last_prompt},
                {"role":"user","content":json.dumps({
                    "question":question,
                    "authorized_personal_facts":selected_memory,
                    "recent_conversation_context_only":recent,
                    "failure_to_correct":validation.get("grounding_gate"),
                },ensure_ascii=False)},
            ]
            if not selected_memory and classify_turn_intent(question).get("reason")=="recording_consent":
                last_messages[0]["content"] += "\n" + recording_answer_guidance(language)
            revised,cited_units,citation_complete=_answer_with_fact_citations(
                model,last_messages,timeout_seconds,min(token_limit,450),selected_memory,
                temperature=0.0)
            if revised and revised.strip():
                answer_attempts.append({"attempt": 3, "text": revised})
                good,clean_answer,validation=evaluate_answer(revised)
                answer_attempts[-1]["validation"] = dict(validation)
            else:
                good = False
                validation = {"grounding_gate": "empty_rewrite", "blocked": True}
        answer_rewrite_success=bool(good)
    # Content validation is observational. A rejected model answer remains
    # genuine experimental data and is submitted with its quality warnings.
    # If all repairs fail content checks, prefer the FIRST non-empty response:
    # the unaltered original utterance best preserves the experimental trace.
    if not good:
        clean_answer = initial_model_answer
    quality_flags = list(dict.fromkeys(
        str(a.get("validation", {}).get("grounding_gate", "unknown"))
        for a in answer_attempts if a.get("validation", {}).get("blocked")
    ))
    grounding_audit=dict(validation,model_called=True,
        source_status="source_grounded" if selected_memory else "current_conversation",
        fallback="rewritten_grounded_answer" if answer_rewrite_count else None,
        consolidated_route=consolidated_route,
        answer_quality_passed=bool(good),
        quality_flags=quality_flags,
        content_validation_nonblocking=True,
        model_answer_attempts=answer_attempts,
        returned_answer_origin=("latest_approved_model" if good else "first_model_output_with_quality_warning"))
    used_source=bool(selected_memory)
    # These are model-reported citations, never deterministic template citations.
    if not used_source:
        cited_units=[]
        citation_complete=True
    citation_method="model_self_report" if used_source else "no_source"
    return clean_answer,dict(prepared, _turn_memory_selection={
        'memory_attribution':{'schema':'memory_citation.v1','complete':citation_complete,'citation_method':citation_method,
            'cited_memory_ids':cited_units,'available_memory_ids':list(dict.fromkeys(
                x['memory_id'] for x in selected_memory if x.get('memory_id')))},
        'question_intent':intent,
        'behavior_mode':('recall' if used_source else action['mode']),
        'behavior_route_reason':('source_evidence_used_after_recall'
                                 if used_source else action['reason']),
        'behavior_hint_after_recall':action['mode'],
        'routing_order':'memory_probe_then_evidence_check_then_behavior',
        'four_stage_routing_order':'retrieve_then_judge;retrieval_miss_to_cache;judgement_fail_to_response',
        'four_stage_branch':four_stage_branch,
        'consolidated_route':consolidated_route,
        'semantic_llm_called':bool(semantic_stats_snapshot['actual_api_calls']),
        'answer_llm_called':True,
        'answer_status':'completed_with_quality_warnings' if quality_flags else 'completed',
        'answer_quality_passed':bool(good),
        'quality_flags':quality_flags,
        'legacy_prompt_version':recall_trace.get('legacy_prompt_version'),
        'fusion_prompt_version':recall_trace.get('fusion_prompt_version'),
        'memory_triggered':True,
        'memory_used_in_answer':bool(
            used_source and any(m.get("fact_category")=="memory" for m in selected_memory)),
        'reality_used_in_answer':bool(
            used_source and any(m.get("fact_category")=="reality" for m in selected_memory)),
        'factbank_temporal_target':recall_trace.get("question_temporal_target"),
        'factbank_selected_count':len(selected_memory),
        'factbank_reality_selected_count':recall_trace.get("reality_visible_count",0),
        'factbank_memory_selected_count':recall_trace.get("memory_visible_count",0),
        'response_source_status':grounding_audit.get(
            "source_status","source_grounded" if used_source else "current_conversation"),

        'selection_status':recall_trace.get("fact_match_status"),
        'factbank_schema':recall_trace.get("factbank_schema"),
        'selected_groups':len(selected_memory),
        'candidate_selected_groups':question_analysis['selected_groups'],
        'total_groups':question_analysis['total_groups'],
        'requested_aspects':question_analysis['aspects'],
        'history_relation':question_analysis['history_relation'],
        'question_sha256':question_digest,
        'model_last_message_sha256':hashlib.sha256(model_last_question.encode('utf-8')).hexdigest(),
        'selected_memory_hashes':selected_digests,
        'knowledge_domain':knowledge_cache.get('domain'),
        'knowledge_prepared_count':knowledge_cache.get('selected_count',0),
        'knowledge_visible_ids':[e['knowledge_id'] for e in selected_knowledge],
        'knowledge_visible_count':len(selected_knowledge),
        'source_recall':recall_trace,
        'runtime_timing':{'retrieval_ms':retrieval_elapsed_ms,
                          'after_retrieval_ms':round((time.monotonic()-turn_started)*1000)-retrieval_elapsed_ms,
                          'total_ms':round((time.monotonic()-turn_started)*1000),
                          'semantic_cache_hits':semantic_stats_snapshot['hits'],
                          'semantic_cache_misses':semantic_stats_snapshot['misses'],
                          'semantic_actual_api_calls':semantic_stats_snapshot['actual_api_calls'],
                          'semantic_model_calls':semantic_stats_snapshot['actual_api_calls'],
                          'semantic_model_calls_planned':(recall_trace.get('v15_semantic_primary') or {}).get('semantic_model_calls',0)
                              if retrieval_strategy in ('new_retrieval','unified_retrieval','fused_retrieval') else
                              (recall_trace.get('legacy_memory_access') or {}).get('semantic_model_calls',0),
                          'semantic_batch_workers':(recall_trace.get('legacy_memory_access') or {}).get('semantic_batch_workers',1)
                              if retrieval_strategy=='legacy_cache' else 1,
                          'semantic_prefilter_ms':(recall_trace.get('v15_semantic_primary') or {}).get('prefilter_elapsed_ms',0),
                          'semantic_candidate_groups_total':(recall_trace.get('v15_semantic_primary') or {}).get('candidate_groups_total',0),
                          'semantic_candidate_groups_sent':(recall_trace.get('v15_semantic_primary') or {}).get('candidate_groups_sent',0),
                          'semantic_prefilter_applied':bool((recall_trace.get('v15_semantic_primary') or {}).get('prefilter_applied',False)),
                          'semantic_full_fallback_attempted':bool((recall_trace.get('v15_semantic_primary') or {}).get('full_retrieval_fallback_attempted',False)),
                          'answer_rewrite_count':answer_rewrite_count,
                          'answer_rewrite_succeeded':(answer_rewrite_success if answer_rewrite_count else None)},
        'question_answer_alignment':{
            "requested_dimension":recall_trace.get("question_dimension"),
            "selected_coverage":recall_trace.get("memory_coverage",[]),
            "evidence_levels":recall_trace.get("memory_evidence_levels",[]),
            "source_question_alignment":recall_trace.get("memory_question_alignment",[]),
            "respondent_answer_alignment":recall_trace.get("memory_answer_alignment",[]),
            "match_protocol":recall_trace.get("memory_match_protocol"),
            "frequency_source_evidence":bool(
                recall_trace.get("question_dimension")=="frequency"
                and any(x in ("source_question_frequency","general_frequency")
                    for x in recall_trace.get("memory_coverage",[]))),
            "fallback_reason":grounding_audit.get("grounding_gate"),
        },
        'grounding_audit':grounding_audit,
        'knowledge_provenance':'general_knowledge_not_personal',
        'history_turns_in_prompt':len(recent),
        'context_chars':len(messages[1]['content']),
        'parameter_memory_budget': {'groups':max_groups,'chars':max_chars},
        'parameter_memory_access_trace': memory_access_trace,
        'parameter_memory_access_level': memory_access_level(behavior['memory']),
        'memory_strength_effective_source_chars':sum(
            len(str(f)) for item in selected_memory
            if item.get("fact_category")=="memory" for f in item.get('facts',[])),
        'reality_source_effective_chars':sum(
            len(str(f)) for item in selected_memory
            if item.get("fact_category")=="reality" for f in item.get('facts',[])),
        'parameter_disclosure_sensitivity': question_sensitivity(question),
        'parameter_max_output_tokens':token_limit,
        'parameter_answer_length_guide':length_guide,
        'parameter_verbosity_value':behavior["verbosity"],
        'parameter_token_limit_applied':bool(grounding_audit.get("model_called")),
        'parameter_model_provider':_REMOTE_RESPONDENT.get(),
        'parameter_rewrite_max_output_tokens':min(token_limit,450)
            if grounding_audit.get("fallback") in ("rewritten_grounded_answer","safe_natural_template")
               and grounding_audit.get("model_called") else None,
        'prompt_protocol':'current_question_last_v0132',
    })
