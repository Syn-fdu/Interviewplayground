"""Request-scoped OpenAI-compatible quality judge; credentials never exported."""
import io,json,math,os,time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

def environment_defaults():
    return {"base_url":os.getenv("BENCHMARK_API_BASE_URL","").strip(),
            "model":os.getenv("BENCHMARK_API_MODEL","").strip(),
            "api_key":os.getenv("BENCHMARK_API_KEY","").strip(),
            "temperature":os.getenv("BENCHMARK_API_TEMPERATURE","0"),
            "max_tokens":os.getenv("BENCHMARK_API_MAX_TOKENS","4096"),
            "timeout":os.getenv("BENCHMARK_API_TIMEOUT","300"),
            "response_format":os.getenv("BENCHMARK_API_RESPONSE_FORMAT","json_object")}

def public_defaults():
    cfg=environment_defaults()
    key=cfg.pop("api_key")
    cfg["key_configured"]=bool(key and cfg["base_url"])
    return cfg

def resolve_config(config):
    env=environment_defaults()
    supplied=config if isinstance(config,dict) else {}
    cfg={k:supplied.get(k) if supplied.get(k) not in (None,"") else v for k,v in env.items()}
    # Never reuse an environment key for an explicitly different endpoint.
    if not supplied.get("api_key") and str(cfg["base_url"]).rstrip("/")!=env["base_url"].rstrip("/"):
        cfg["api_key"]=""
    return cfg

def make_opener(config, transport=urlopen):
    from interview_reportcard_judge import JudgeError
    if not isinstance(config,dict):raise JudgeError("API 配置无效")
    config=resolve_config(config)
    base=str(config.get("base_url","")).strip().rstrip("/")
    parts=urlsplit(base)
    if parts.scheme not in ("http","https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise JudgeError("请输入有效的 API Base URL")
    if parts.scheme=="http" and parts.hostname not in ("localhost","127.0.0.1","::1"):
        raise JudgeError("远程机评 API 必须使用 HTTPS")
    endpoint=base if base.endswith("/chat/completions") else base+"/chat/completions"
    model=str(config.get("model","")).strip()
    key=str(config.get("api_key","")).strip()
    if not model or not key:raise JudgeError("请输入 API 模型和 Key")
    try:
        temperature=float(config.get("temperature",0))
        max_tokens=int(config.get("max_tokens",2500))
        timeout=float(config.get("timeout",120))
    except (TypeError,ValueError):raise JudgeError("API 参数必须为数字")
    if not (math.isfinite(temperature) and 0<=temperature<=2 and 256<=max_tokens<=32768 and math.isfinite(timeout) and 10<=timeout<=600):
        raise JudgeError("API 参数超范围")
    mode=config.get("response_format","json_object")
    if mode not in ("json_object","prompt_only"):raise JudgeError("JSON 输出模式无效")
    def invoke(request,timeout_unused=120,**kwargs):
        original=json.loads(request.data)
        messages=[dict(m) for m in original["messages"]]
        messages[0]["content"]+="\\nReturn one JSON object matching this schema: "+json.dumps(original["format"],ensure_ascii=False)
        payload={"model":model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens,"stream":False}
        if mode=="json_object":payload["response_format"]={"type":"json_object"}
        req=Request(endpoint,data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key},method="POST")
        deadline=time.monotonic()+timeout
        for attempt in range(1,4):
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError()
                with transport(req,timeout=remaining) as response: body=json.load(response)
                break
            except HTTPError as exc:
                status=exc.code
                exc.close()
                transient=status in (408,429,500,502,503,504,520,521,522,523,524)
                if not transient or attempt==3 or deadline-time.monotonic()<=attempt*2+1:
                    raise JudgeError("API 机评 HTTP "+str(status)+"；已保留检查点，可重试") from None
            except (URLError,OSError,TimeoutError):
                if attempt==3 or deadline-time.monotonic()<=attempt*2+1:
                    raise JudgeError("API 机评连接或超时；已保留检查点，可重试") from None
            except ValueError:
                raise JudgeError("API 机评响应格式无效") from None
            time.sleep(attempt*2)
        try:content=body["choices"][0]["message"]["content"]
        except (KeyError,IndexError,TypeError):raise JudgeError("API 返回缺少回答正文") from None
        if not isinstance(content,str):raise JudgeError("API 回答正文不是字符串")
        return io.BytesIO(json.dumps({"message":{"content":content}},ensure_ascii=False).encode())
    return model,invoke,{"endpoint":endpoint,"temperature":temperature,"max_tokens":max_tokens,"timeout":timeout,"response_format":mode}
