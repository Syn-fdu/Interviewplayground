"""Loopback-only two-provider model configuration; keys stay in Python process memory."""
import os,re,threading,ipaddress
from contextvars import ContextVar
from urllib.parse import urlsplit
from playground.interview_agent.config import ModelConfig
# Scoped configuration prevents a batch worker from reading an interactive
# session's model/URL/API key. ContextVars isolate simultaneous threads.
_ACTIVE_MODEL_PROFILE = ContextVar("playground_model_profile", default=None)
PROVIDERS={
 "maas":("https://maas.qianwenaiapi.com/compatible-mode/v1",
         "qwen3.8-2.4t-a95b","QWEN_MAAS_API_KEY","QWEN_MAAS_MODEL"),
 "shubi":("https://api.shubiaobiao.cn/v1",
          "qwen3.5-flash","INTERVIEW_SERVER_API_KEY","INTERVIEW_SERVER_MODEL"),
 "local":("http://127.0.0.1:11434/v1",
          "qwen2.5:7b","PLAYGROUND_LOCAL_API_KEY","PLAYGROUND_LOCAL_MODEL"),
}
_LOCK=threading.RLock()
_OVERRIDES={}
_WEB_KEYS={}  # keyed by (role, provider, normalized base_url); no cross-role key reuse
_MODELS=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
def normalize_base_url(base_url):
    """Validate a user-selected OpenAI-compatible base URL before storing it.

    Arbitrary public HTTPS hosts are supported. HTTP is permitted only for
    local loopback development. Credentials/query/fragment/path traversal are
    rejected, and user keys are scoped to the *exact* normalized URL.
    """
    if not isinstance(base_url,str) or not (8 <= len(base_url.strip()) <= 512):
        raise ValueError("API Base URL is required (max 512 characters).")
    text=base_url.strip().rstrip("/")
    if any(ord(c)<33 or ord(c)==127 for c in text):
        raise ValueError("API URL contains whitespace or control characters.")
    try:
        parsed=urlsplit(text)
        hostname=parsed.hostname
        port=parsed.port
    except ValueError as exc:
        raise ValueError("Invalid API Base URL.") from exc
    if parsed.scheme not in ("https","http") or not hostname:
        raise ValueError("API Base URL must start with https:// (or loopback http://).")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL must not contain credentials, query or fragment.")
    if text.endswith("/chat/completions"):
        raise ValueError("Enter the API base path, without /chat/completions.")
    if any(part in (".","..") for part in parsed.path.split("/")):
        raise ValueError("API URL path traversal is not allowed.")
    try:
        ip=ipaddress.ip_address(hostname)
    except ValueError:
        ip=None
    loopback=hostname in ("localhost","127.0.0.1","::1") or (ip is not None and ip.is_loopback)
    if parsed.scheme=="http" and not loopback:
        raise ValueError("Non-local API endpoints must use HTTPS.")
    if ip is not None and not ip.is_global and not loopback:
        raise ValueError("Private or reserved IP addresses are not API endpoints.")
    if hostname.endswith((".local",".internal")) and not loopback:
        raise ValueError("Private network hostnames are not supported.")
    return text

def _env_value(provider,role="interviewer"):
    default_url,default,key_var,model_var=PROVIDERS[provider]
    if provider=="local":
        return (os.getenv("PLAYGROUND_LOCAL_MODEL","qwen2.5:7b"),
                "ollama", default_url, "local")
    url_var="QWEN_MAAS_BASE_URL" if provider=="maas" else "INTERVIEW_SERVER_BASE_URL"
    role_url_var=("QWEN_MAAS_RESPONDENT_BASE_URL" if provider=="maas"
                  else "INTERVIEW_SERVER_RESPONDENT_BASE_URL")
    role_model_var=("QWEN_MAAS_RESPONDENT_MODEL" if provider=="maas"
                    else "INTERVIEW_SERVER_RESPONDENT_MODEL")
    explicit_url=os.getenv(role_url_var,"").strip() if role=="respondent" else ""
    explicit_url=explicit_url or os.getenv(url_var,"").strip()
    explicit_model=(os.getenv(role_model_var,"").strip() if role=="respondent" else "")
    explicit_model=explicit_model or os.getenv(model_var,"").strip()
    explicit_key=os.getenv(key_var,"").strip()

    # Also recognize the original colleague's LLM_* configuration when its
    # endpoint is the selected provider. Earlier builds used only LLM_*;
    # silently ignoring those is why the webpage appeared unsynchronized.
    legacy_url=os.getenv("LLM_BASE_URL","").strip()
    legacy_model=os.getenv("LLM_MODEL","").strip()
    legacy_key=os.getenv("LLM_API_KEY","").strip()
    use_legacy=False
    if legacy_url:
        try:
            normalized_legacy=normalize_base_url(legacy_url)
            host=(urlsplit(normalized_legacy).hostname or "").lower()
            if provider=="shubi":
                use_legacy=host=="api.shubiaobiao.cn"
            else:
                use_legacy=host=="maas.qianwenaiapi.com"
        except ValueError:
            pass
    # Never interpret "ollama" as a remote API key.
    url=explicit_url or (normalized_legacy if use_legacy else default_url)
    model=explicit_model or (legacy_model if use_legacy else default)
    key=explicit_key or (legacy_key if use_legacy and legacy_key!="ollama" else "")
    source=("environment" if explicit_url or explicit_model or explicit_key else
            "legacy_environment" if use_legacy else "default")
    return model,key,normalize_base_url(url),source

def get(provider,role="interviewer",require_key=True):
    if provider not in PROVIDERS or role not in ("interviewer","respondent"):
        raise ValueError("Provider must be maas or shubi; role interviewer or respondent.")
    scoped=_ACTIVE_MODEL_PROFILE.get()
    if scoped is not None:
        entry=scoped.get((role,provider))
        if entry is None:
            # An unselected provider remains in an independent environment-only
            # baseline. Never fall through into interactive browser overrides.
            model,key,url,_source=_env_value(provider,role)
            entry=ModelConfig(provider="openai_compat",base_url=url,model=model,
                              api_key=key,temperature=0.4,timeout=180)
        if require_key and not entry.api_key:
            raise ValueError("Batch API Key missing for "+role+"/"+provider)
        return entry
    env_model,env_key,env_url,env_source=_env_value(provider,role)
    with _LOCK:
        override=_OVERRIDES.get((role,provider),{})
        model=override.get("model") or env_model
        url=override.get("base_url") or env_url
        web_key=_WEB_KEYS.get((role,provider,url))
        # A shell key belongs to its configured endpoint; never silently send
        # it to a different URL typed into the browser.
        key=web_key or (env_key if url==env_url else "")
    if not isinstance(model,str) or not _MODELS.fullmatch(model):
        raise ValueError("Invalid model ID.")
    if require_key and not key:
        raise ValueError("Missing API Key for "+provider+". Set in webpage or PowerShell.")
    return ModelConfig(provider="openai_compat",base_url=url,model=model,
        api_key=key,temperature=0.4,timeout=180)
def status():
    data={}
    for role in ("interviewer","respondent"):
        data[role]={}
        for provider in PROVIDERS:
            cfg=get(provider,role,False)
            _,_,_,env_source=_env_value(provider,role)
            with _LOCK: web_key=bool(_WEB_KEYS.get((role,provider,cfg.base_url)))
            with _LOCK:
                web_model=bool(_OVERRIDES.get((role,provider)))
            data[role][provider]={
                "model":cfg.model,"base_url":cfg.base_url,"key_configured":bool(cfg.api_key),
                "key_source":"web_session" if web_key else
                             "legacy_environment" if cfg.api_key and env_source=="legacy_environment" else
                             "environment" if cfg.api_key else "missing",
                "config_source":"web_session" if web_model else env_source}
    data["respondent"]["local"]={
        "model":os.getenv("PLAYGROUND_LOCAL_MODEL","qwen2.5:7b"),
        "base_url":"http://127.0.0.1:11434/v1",
        "key_configured":True,"key_source":"local"}
    return data
def set_selection(role,provider,model,api_key=None,base_url=None):
    if role not in ("interviewer","respondent") or provider not in PROVIDERS:
        raise ValueError("Invalid provider or role.")
    if not isinstance(model,str) or not _MODELS.fullmatch(model):
        raise ValueError("Invalid model ID.")
    if provider=="local":
        api_key="ollama"
    if api_key is not None and (not isinstance(api_key,str) or
            not 5<=len(api_key)<=1024 or any(c in api_key for c in "\r\n")):
        raise ValueError("Invalid API Key.")
    current=get(provider,role,False)
    chosen_url=normalize_base_url(base_url if base_url is not None else current.base_url)
    with _LOCK:
        previous=dict(_OVERRIDES.get((role,provider),{}))
        previous["model"]=model
        previous["base_url"]=chosen_url
        if api_key:_WEB_KEYS[(role,provider,chosen_url)]=api_key
        _OVERRIDES[(role,provider)]=previous
    return status()[role][provider]
def reset_for_tests():
    with _LOCK:
        _OVERRIDES.clear()
        _WEB_KEYS.clear()


def batch_config_context(profiles):
    """Use validated per-experiment ModelConfig objects for the current thread."""
    from contextlib import contextmanager
    @contextmanager
    def context():
        token=_ACTIVE_MODEL_PROFILE.set(profiles)
        try:
            yield
        finally:
            _ACTIVE_MODEL_PROFILE.reset(token)
    return context()
