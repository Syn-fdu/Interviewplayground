"""Local-only structured diagnostic journal; NEVER persist prompts or responses."""
import collections, json, os, pathlib, threading, time, uuid
LOCK=threading.RLock()
EVENTS=collections.deque(maxlen=1000)
ACTIVE={}
ROOT=pathlib.Path(__file__).resolve().parent/'diagnostics'
PATH=ROOT/'ollama_calls.jsonl'
SAFE=('attempt','prompt_characters','estimated_input_tokens_not_exact','requested_num_ctx','requested_num_predict','timeout_seconds','memory_mode','stage','failed_phase','done_reason','prompt_eval_count','eval_count','total_duration_ns','prompt_eval_duration_ns','eval_duration_ns','load_duration_ns','response_characters','stable_prefix_sha256','stable_prefix_characters','dynamic_suffix_characters','keep_alive')
def _append(e):
    try:
        ROOT.mkdir(parents=True,exist_ok=True)
        with PATH.open('a',encoding='utf-8') as f:
            f.write(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n')
            f.flush()
    except OSError:
        pass

def begin(role,model,**fields):
    rid=uuid.uuid4().hex[:16]
    e={'event':'begin','id':rid,'timestamp':round(time.time(),3),'role':str(role)[:32],'model':str(model)[:100]}
    e.update({k:fields[k] for k in SAFE if k in fields and type(fields[k]) in (int,float,str,bool)})
    with LOCK:
        ACTIVE[rid]=e
        EVENTS.append(e.copy())
        _append(e)
    return rid

def finish(rid, elapsed, response=None, error=None, **fields):
    response=response if isinstance(response,dict) else {}
    usage=response.get('usage') if isinstance(response.get('usage'),dict) else {}
    native='prompt_eval_count' in response or 'eval_count' in response
    e={'event':'end','id':rid,'timestamp':round(time.time(),3),'status':'error' if error else 'ok',
       'wall_seconds':round(float(elapsed),3), 'error_type':type(error).__name__ if error else None}
    for k in SAFE:
        if k in fields and type(fields[k]) in (int,float,str,bool): e[k]=fields[k]
    e.update({'prompt_tokens':response.get('prompt_eval_count') if native else usage.get('prompt_tokens'),
              'output_tokens':response.get('eval_count') if native else usage.get('completion_tokens'),
              'prompt_seconds':round(response['prompt_eval_duration']/1e9,3) if native and isinstance(response.get('prompt_eval_duration'),(int,float)) else None,
              'output_seconds':round(response['eval_duration']/1e9,3) if native and isinstance(response.get('eval_duration'),(int,float)) else None,
              'load_seconds':round(response['load_duration']/1e9,3) if native and isinstance(response.get('load_duration'),(int,float)) else None,
              'done_reason':str(response.get('done_reason') or e.get('done_reason') or '')[:32]})
    with LOCK:
        original=ACTIVE.pop(rid,None) or {}
        e['role']=original.get('role','unknown')
        e['model']=original.get('model','unknown')
        e['timeout_seconds']=original.get('timeout_seconds')
        e['requested_num_ctx']=original.get('requested_num_ctx')
        e['requested_num_predict']=original.get('requested_num_predict')
        e['estimated_input_tokens_not_exact']=original.get('estimated_input_tokens_not_exact')
        e['prompt_characters']=original.get('prompt_characters')
        EVENTS.append(e.copy())
        _append(e)

def record(role,model,elapsed,response=None,error=None):
    rid=begin(role,model,stage='legacy_wrapper')
    finish(rid,elapsed,response,error)

def snapshot():
    with LOCK:
        return {'ok':True,'events':list(reversed(EVENTS)),'active':list(ACTIVE.values()),
                'max_events':1000,'journal_file':'diagnostics/ollama_calls.jsonl',
                'privacy':'metadata_only_local_disk_no_prompts_no_answers'}
