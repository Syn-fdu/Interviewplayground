"""InterviewReportCard experimental local LLM interview-quality judge.

Eight interview-related measures, two participant experience measures N/A.
LLM is a fallible annotator, not a validated human coder. Never infer ground truth.
"""
from __future__ import annotations
import json, re, time, hashlib, threading, copy, socket
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.error import URLError
from playground.evaluation.benchmark_llm_judge import _sanitize_turns, _parse_model_json, JudgeError
from playground.interview.respondent_llm import validate_model
from playground.common.performance_metrics import record as perf_record

ENDPOINT="http://127.0.0.1:11434/api/chat"
ALLOWED_MODELS=("gemma3:4b","qwen2.5:7b","llama3.1:8b")
BATCH=2
# In-memory, per-session progress only. No private transcripts are written to disk.
CHECKPOINT_TTL_SECONDS=6*3600
MAX_CHECKPOINTS=16
LOCAL_JUDGE_TIMEOUT_SECONDS=480
API_JUDGE_TIMEOUT_SECONDS=120
_CHECKPOINTS={}
_CHECKPOINT_LOCK=threading.RLock()

def _fingerprint(run,guide,model,config):
    raw=json.dumps({"run":run,"guide":guide,"model":model,"config":config,
       "protocol":"reportcard_v121_2turns"},sort_keys=True,ensure_ascii=False,default=str)
    return hashlib.sha256(raw.encode('utf8')).hexdigest()

def _load_checkpoint(store,key):
    with _CHECKPOINT_LOCK:
        now=time.monotonic()
        for k,v in list(store.items()):
            if not isinstance(v,dict) or now-v.get('at',0)>CHECKPOINT_TTL_SECONDS:
                store.pop(k,None)
        value=store.get(key)
        return copy.deepcopy(value['rows']) if value else []

def _save_checkpoint(store,key,rows):
    with _CHECKPOINT_LOCK:
        store[key]={"at":time.monotonic(),"rows":copy.deepcopy(rows)}
        if len(store)>MAX_CHECKPOINTS:
            oldest=min(store,key=lambda x:store[x]['at'])
            store.pop(oldest,None)

def _clear_checkpoint(store,key):
    with _CHECKPOINT_LOCK:store.pop(key,None)

def _validate_batch_rounds(got,batch,batch_number):
    expected=[t['round'] for t in batch]
    actual=[x.get('round') if isinstance(x,dict) else None for x in got] if isinstance(got,list) else []
    valid=[x for x in actual if type(x) is int]
    missing=[x for x in expected if x not in valid]
    unexpected=[x for x in valid if x not in expected]
    duplicates=sorted({x for x in valid if valid.count(x)>1})
    malformed=len(actual)-len(valid)
    if (not isinstance(got,list) or len(actual)!=len(expected) or missing or unexpected or duplicates or malformed):
        raise JudgeError(
          f"InterviewReportCard 第 {batch_number} 批轮次不完整；"
          f"预期轮次={expected}；实际轮次={actual[:12]}；"
          f"缺失={missing}；额外={unexpected}；重复={duplicates}；无效条目={malformed}；"
          "未通过轮次完整性校验。此前成功批次已保留。"
        )
    return {r['round']:r for r in got}



def _guide_topics(guide):
    if not isinstance(guide,dict) or not isinstance(guide.get("topics"),list):
        raise JudgeError("必须上传正常的研究 Guide JSON（包含 topics）")
    subs=[]
    for topic in guide["topics"]:
        if not isinstance(topic,dict): continue
        for item in topic.get("subtopics",[]):
            if not isinstance(item,dict): continue
            sid=item.get("subtopic_id")
            name=item.get("name",item.get("title",item.get("label","")))
            if isinstance(sid,str) and sid and isinstance(name,str) and name:
                subs.append({"id":sid,"name":name[:1200]})
    if not subs or len(subs)>1000 or len({s['id'] for s in subs})!=len(subs):
        raise JudgeError("Guide 必须包含 1–1000 个不同 ID 的研究子主题")
    return subs


def _ask(model, system, task, schema, opener):
    """Structured judge request with actionable Ollama diagnostics.

    Never silently replace failed model judgments with synthetic scores.
    API proxy requests retain the shorter remote timeout; local inference has
    a separate, longer window and keeps the model resident between batches.
    """
    is_local = opener is urlopen
    payload={"model":model,"stream":False,"format":schema,
      "options":{"temperature":0,"num_predict":2500},
      "messages":[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(task,ensure_ascii=False)}]}
    if is_local:
        payload["keep_alive"]="10m"
    request=Request(ENDPOINT,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
      headers={"Content-Type":"application/json"},method="POST")
    started=time.monotonic()
    try:
        timeout = LOCAL_JUDGE_TIMEOUT_SECONDS if is_local else API_JUDGE_TIMEOUT_SECONDS
        with opener(request,timeout=timeout) as resp:
            response=json.load(resp)
    except HTTPError as exc:
        perf_record("judge",model,time.monotonic()-started,error=exc)
        try:
            detail=exc.read(2048).decode("utf-8","replace")
            detail=json.loads(detail).get("error",detail)
        except (UnicodeError,ValueError,AttributeError,TypeError):
            detail=""
        detail=re.sub(r"[\r\n]+"," ",str(detail))[:260]
        if is_local and exc.code==404:
            raise JudgeError(f"Ollama 未找到评审模型 {model}（HTTP 404）。请先运行 ollama list，必要时执行 ollama pull {model}。{detail}") from exc
        if is_local and exc.code in (400,422):
            raise JudgeError(f"Ollama 不接受此次结构化机评请求（HTTP {exc.code}）：{detail}。请确认 Ollama 版本支持 JSON Schema format。") from exc
        raise JudgeError(f"机评模型返回 HTTP {exc.code}：{detail or '请查看 Ollama 服务日志'}") from exc
    except (socket.timeout, TimeoutError) as exc:
        perf_record("judge",model,time.monotonic()-started,error=exc)
        raise JudgeError(f"本地机评模型 {model} 在 {LOCAL_JUDGE_TIMEOUT_SECONDS if is_local else API_JUDGE_TIMEOUT_SECONDS} 秒内未返回结果。请查看 Ollama 是否仍在加载/推理，并检查可用内存；已成功的逐轮批次会保留以便重试。") from exc
    except (URLError,OSError) as exc:
        perf_record("judge",model,time.monotonic()-started,error=exc)
        reason=getattr(exc,"reason",exc)
        if isinstance(reason,(socket.timeout,TimeoutError)):
            raise JudgeError(f"评审模型 {model} 调用超时（每次本地请求上限 {LOCAL_JUDGE_TIMEOUT_SECONDS} 秒）；已成功批次可在同一会话重试。") from exc
        raise JudgeError(f"无法连接评审模型 {model}：{str(reason)[:180]}。请执行 ollama serve，检查 http://127.0.0.1:11434/api/tags 和 ollama list。") from exc
    except (ValueError,UnicodeError) as exc:
        perf_record("judge",model,time.monotonic()-started,error=exc)
        raise JudgeError("Ollama 返回了无法解析的 HTTP JSON 响应；请检查服务版本与模型日志。") from exc
    perf_record("judge",model,time.monotonic()-started,response)
    if not isinstance(response,dict):
        raise JudgeError("Ollama 返回内容不是 JSON 对象")
    if response.get("error"):
        raise JudgeError("Ollama 机评错误："+str(response["error"])[:260])
    raw=(response.get("message") or {}).get("content")
    if not isinstance(raw,str) or not raw.strip():
        raise JudgeError("评审模型未返回文本；可能是模型加载、上下文或输出 token 限制导致。请查看 Ollama 日志。")
    result=_parse_model_json(raw)
    if not isinstance(result,dict):raise JudgeError("模型必须返回结构化 JSON 对象")
    return result


def _text_norm(s):
    return re.sub(r'\s+',' ',s).strip().casefold()


def _quote_in_answer(quote,answer):
    return bool(isinstance(quote,str) and quote.strip() and _text_norm(quote) in _text_norm(answer))


def _parse_holistic_score(value, name):
    """Normalize defensible model representations; never manufacture a rating.

    A real out-of-range score is deliberately N/A with an audit warning, not
    clamped to 1..4 or treated as a valid measured score.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{name} 返回布尔值 {value!r}，已标记 N/A，待人工复核"
    if isinstance(value, str):
        cleaned=value.strip()
        if cleaned.casefold() in ('', 'n/a', 'na', 'null', 'none', 'not applicable',
                                   'unknown', '无法评价', '无法评估', '不适用', '未评估', '无依据'):
            return None, None
        # Support only unambiguous numeric renderings, not free-form explanations.
        match=re.fullmatch(r'([1-4])(?:\s*/\s*4|\s*(?:分|\s*of\s*4))?',cleaned,re.I)
        if match:
            return int(match.group(1)), None
        return None, f"{name} 返回无法识别的分数 {value[:60]!r}，已标记 N/A，待人工复核"
    if type(value) is int and 1<=value<=4:
        return value, None
    if type(value) is float and value.is_integer() and 1<=value<=4:
        return int(value), None
    return None, f"{name} 返回超出1–4范围或无效的分数 {str(value)[:60]!r}，已标记 N/A，待人工复核"


def _reportcard_turns(run):
    turns=run.get("turns")
    if not isinstance(turns,list) or not 1<=len(turns)<=1000:
        raise JudgeError("机评需要 1–1000 轮完整 Transcript 问答")
    result=[]
    for i,t in enumerate(turns,1):
        if not isinstance(t,dict):
            raise JudgeError(f"第 {i} 轮格式不正确")
        q=t.get("interviewer",t.get("question"))
        a=t.get("respondent",t.get("answer"))
        if not isinstance(q,str) or not q.strip() or not isinstance(a,str) or not a.strip():
            raise JudgeError(f"第 {i} 轮缺少问答正文，请导入 Transcript JSONL 而非仅含诊断的 Run JSON")
        result.append({"round":i,"question":q,"answer":a})
    return result

def _normalize_coverage_ids(raw, allowed_ids):
    """Reject IDs outside the canonical Outline slot schema, never score them as 0%."""
    if not isinstance(raw, list):
        raise JudgeError("机评 covered_subtopic_ids 不是数组；本次覆盖率无效")
    valid, seen, warnings = [], set(), []
    for item in raw:
        if not isinstance(item, str):
            raise JudgeError("机评 covered_subtopic_ids 含非字符串 ID；覆盖率无效")
        item = item.strip()
        if item not in allowed_ids:
            raise JudgeError("机评返回未知必需 Slot ID: " + item[:120] +
              "。预期 question_id:slot_id，当前覆盖率无效；不会把映射错误计算为0%。")
        elif item in seen:
            warnings.append("重复子主题 ID 已去重: " + item)
        else:
            valid.append(item)
            seen.add(item)
    return valid, warnings


def outline_reference(outline):
    if not isinstance(outline,dict) or not isinstance(outline.get("topics"),list):
        raise JudgeError("请提供预处理 Outline JSON")
    subtopics=[]; seen=set()
    for topic in outline["topics"]:
        for q in topic.get("questions",[]):
            qid=q.get("id")
            for slot in q.get("required_slots",[]):
                sid=slot.get("id") or slot.get("slot_id")
                desc=slot.get("description")
                if not qid or not sid or not isinstance(desc,str) or not desc.strip():
                    raise JudgeError("Outline 必需 Slot 缺少 ID 或描述")
                ident=str(qid)+":"+str(sid)
                if ident in seen:raise JudgeError("Outline Slot ID 重复")
                seen.add(ident)
                subtopics.append({"subtopic_id":ident,"name":str(q.get("text",""))+" | "+desc})
    if not subtopics:raise JudgeError("Outline 没有必需 Slots")
    return {"goal":outline.get("objective",""),"topics":[{"subtopics":subtopics}],"research_questions":[]}

def evaluate_reportcard(run,guide,model,config=None,opener=urlopen,checkpoint_store=None,outline=None,api_config=None):
    turns=_reportcard_turns(run)
    topics=_guide_topics(guide)
    if api_config:
        from playground.evaluation.benchmark_api_judge import make_opener
        model,opener,api_identity=make_opener(api_config)
    else:
        api_identity=None
        model=validate_model(model,'gemma3:4b')
        if model not in ALLOWED_MODELS:raise JudgeError("评审模型必须选择下拉框中三种模型之一")
    outline_questions=[]
    for topic in (outline or {}).get("topics",[]) if isinstance(outline,dict) else []:
        if not isinstance(topic,dict):continue
        for q in topic.get("questions",[]):
            if not isinstance(q,dict):continue
            outline_questions.append({"id":q.get("id"),"question":str(q.get("text",""))[:500],
                "required_slots":[str(x.get("description","")) for x in q.get("required_slots",[]) if isinstance(x,dict)],
                "optional_slots":[str(x.get("description","")) for x in q.get("optional_deepening_slots",[]) if isinstance(x,dict)]})
    ids={x['id'] for x in topics}
    store = _CHECKPOINTS if checkpoint_store is None else checkpoint_store
    key = _fingerprint(run,guide,model,{"config":config,"outline":outline,"rubric_revision":"canonical_slot_contract_v175","judge_api":api_identity})
    rows = _load_checkpoint(store,key)
    expected_rounds = [t['round'] for t in turns]
    # Only fully validated contiguous batches are reusable.
    valid_sizes = {i for i in range(0,len(turns)+1,BATCH)} | {len(turns)}
    if (len(rows) not in valid_sizes or
        [r.get('round') for r in rows] != expected_rounds[:len(rows)]):
        _clear_checkpoint(store,key)
        rows=[]
    resumed=len(rows)
    sys=("你是独立质性访谈研究编码员，评估的是采访本身及采访者的表现，而非受访者记忆力。"
      "只凭本次对话与公开 Outline，不能声称核实真人事实或受访者实际体验。"
      "逐轮研究相关性:0=无关/空泛,1=部分相关,2=直接为研究问题提供信息。"
      "covered_subtopic_ids 只能写回答实质满足的必需 Slot，不能因仅提及关键词算覆盖。"
      "受访者拒答、无信息的客套话、机械重复问题、'不方便说'不能作为研究主题已被回答的证据；相关性通常为0。"
      "判断覆盖时必须检查引文中是否有回答必需 Slot 所需的新事实、经历、观点或理由；仅回答'是/不知道'通常不足以覆盖。"
      "若列任何覆盖主题，support_quote 必须是受访者回答中的连续逐字引用，能够证明这轮覆盖。"
      "novel 仅当回答含研究目标相关、Outline 未涵盖的实质性新信息，且有 novel_quote 原文。"
      "leading、unclear、support_rapport 只能从采访者发言判断，受访者回答空泛或拒绝不代表问题有缺陷。"
      "普通开放式问题、礼貌询问、正常澄清追问都不算 support_rapport；只有采访者明确表达鼓励、共情、认可或关系建立时才标 true。"
      "只有问题含答案暗示、未经根据的前提或施压才判 leading；复合提问不必然不清晰，须解释具体歧义。"
      "重复相同问题而未加入新澄清目的应降低全场适应性和连贯性，而不是随意标记诱导或模糊。"
      "采用保守证据标准：相关性2必须有具体经历、事实、理由或可分析观点，泛泛描述最多1；不要因回答长或语气流畅给高分。"
      "covered_subtopic_ids 只允许复制 allowed_subtopics 中的 id；必须使用列表中的完整 question_id:slot_id。没有匹配项时返回空数组。"
      "Outline 必需 Slots 是覆盖率分母；可选 Slots 仅用于理解追问与新发现，不计入必需覆盖率。不要因为采访者问到了某个 Slot 就算覆盖。"
      "新发现必须未在此前访谈出现，不能把同一信息的换句话说算新发现。前文仅用于判断重复。"
      "每批必须恰好返回所有输入轮次，一轮不缺，唯一编号。仅返回 JSON。")
    row_schema={"type":"object","properties":{
      "round":{"type":"integer"},"relevance":{"type":"integer","enum":[0,1,2]},
      "covered_subtopic_ids":{"type":"array","items":{"type":"string","enum":sorted(ids)}},
      "support_quote":{"type":"string"},"novel":{"type":"boolean"},"novel_quote":{"type":"string"},
      "leading":{"type":"boolean"},"unclear":{"type":"boolean"},"support_rapport":{"type":"boolean"},
      "rationale":{"type":"string"}},"required":["round","relevance","covered_subtopic_ids","support_quote",
      "novel","novel_quote","leading","unclear","support_rapport","rationale"]}
    recovered_batches=[]
    for offset in range(resumed,len(turns),BATCH):
        batch=turns[offset:offset+BATCH]
        batch_ids=[t['round'] for t in batch]
        # Enforce count and identity through Ollama structured output, then verify again server-side.
        local_row_schema=copy.deepcopy(row_schema)
        local_row_schema['properties']['round']['enum']=batch_ids
        schema={"type":"object","properties":{"turns":{"type":"array",
          "minItems":len(batch),"maxItems":len(batch),"items":local_row_schema}},"required":["turns"]}
        task={"study_goal":str(guide.get('goal',''))[:500],
          "research_questions":guide.get('research_questions',[])[:15],
          "allowed_subtopics":topics,"outline_questions":outline_questions,"prior_turns":turns[:offset],"turns":batch,
          "require_exact_round_ids":batch_ids}
        result=_ask(model,sys,task,schema,opener)
        try:
            mapping=_validate_batch_rounds(result.get('turns'),batch,offset//BATCH+1)
        except JudgeError:
            # Never guess which duplicate belongs to which round. Rescore each
            # round from the real question/answer with singleton schema.
            mapping={}
            for pos,turn in enumerate(batch):
                rid=turn["round"]
                single_schema=copy.deepcopy(schema)
                spec=single_schema["properties"]["turns"]
                spec["minItems"]=spec["maxItems"]=1
                spec["items"]["properties"]["round"]["enum"]=[rid]
                single_task=dict(task,turns=[turn],require_exact_round_ids=[rid],
                    prior_turns=turns[:offset+pos])
                for recovery_attempt in range(2):
                    single=_ask(model,sys+" 本次只评价指定轮次 "+str(rid)+"，必须原样返回该编号，禁止评价历史轮次。",single_task,single_schema,opener)
                    try:
                        mapping.update(_validate_batch_rounds(single.get("turns"),[turn],offset//BATCH+1))
                        break
                    except JudgeError:
                        if recovery_attempt:raise
            recovered_batches.append(offset//BATCH+1)
        batch_rows=[]
        for turn in batch:
            t=mapping[turn['round']]
            if type(t.get('relevance'))!=int or t['relevance'] not in (0,1,2):
                raise JudgeError(f'第 {offset//BATCH+1} 批第 {turn["round"]} 轮：相关性评分越界')
            for flag in ('novel','leading','unclear','support_rapport'):
                if type(t.get(flag)) is not bool: raise JudgeError(f'第 {offset//BATCH+1} 批第 {turn["round"]} 轮：{flag} 非布尔类型')
            picked, coverage_warnings = _normalize_coverage_ids(t.get('covered_subtopic_ids'), ids)
            # Narrow, auditable correction: an explicit standalone refusal is not
            # research evidence. Do not infer that the interviewer asked poorly.
            answer_norm=_text_norm(turn['answer'])
            is_short_refusal=(len(answer_norm)<160 and bool(re.match(
                r"^(?:i(?:'d| would)? rather not|i (?:prefer|don't want) to|no comment|"
                r"cannot disclose|can't discuss|不方便(?:说|透露|讨论)?|不愿(?:讨论|透露)?|"
                r"不想(?:讨论|回答)?|无可奉告|拒绝回答)",answer_norm)))
            raw_relevance=t['relevance']
            if is_short_refusal:
                picked=[]
                t['relevance']=0
                t['novel']=False
            quote=t.get('support_quote','')
            if not isinstance(quote,str):quote=''
            if picked and (t['relevance']==0 or not _quote_in_answer(quote,turn['answer'])):
                raise JudgeError(f'第 {turn["round"]} 轮覆盖证据缺失或非受访者原文。'
                    ' 不计算虚假0%，需要模型修正或人工复核。')
            else:note=''
            if coverage_warnings:
                note=(note+'；' if note else '')+'；'.join(coverage_warnings)
            if is_short_refusal:
                note=(note+'；' if note else '')+'明确拒答不计入研究相关性或 Guide 覆盖（模型原相关性='+str(raw_relevance)+'）'
            novel=t['novel']
            if any(_text_norm(turn['answer'])==_text_norm(prev['answer']) for prev in turns[:turn['round']-1]):
                novel=False
                note=(note+'；' if note else '')+'与前轮回答完全重复，不计新发现'
            novel_quote=t.get('novel_quote','')
            if not isinstance(novel_quote,str):novel_quote=''
            if novel and (t['relevance']!=2 or not _quote_in_answer(novel_quote,turn['answer'])):
                novel=False
                note=(note+'；' if note else '')+'新发现证据不足，未计入'
            batch_rows.append({"round":turn['round'],"question":turn['question'],"answer":turn['answer'],
              "relevance":t['relevance'],"covered_subtopic_ids":picked,
              "support_quote":quote if picked else '',"novel":novel,
              "novel_quote":novel_quote if novel else '',"leading":t['leading'],"unclear":t['unclear'],
              "support_rapport":t['support_rapport'],"rationale":str(t.get('rationale',''))[:500],
              "audit_note":note,"coverage_validation_warnings":coverage_warnings})
        rows.extend(batch_rows)
        _save_checkpoint(store,key,rows)
    rows.sort(key=lambda r:r['round'])
    overall_sys=("你是访谈质量审查员，评估采访者的整体提问连贯性与根据回答调整策略的能力。"
       "评分1-4；不可把受访者记忆准确性算作采访者能力。"
       "Planner等固定题库如果不能体现自主策略调整，adaptiveness 应为 null，说明理由。"
       "评价前区分采访者提问与受访者回答；不能把受访者拒答误写成采访者拒答。"
       "若已明确重复原问题且没有新追问，连贯性和自主适应性需体现无效重复。"
       "连贯性和适应性评分锚点：1=频繁失焦或机械重复；2=基本相关但错失多次明确追问机会；3=多数衔接合理且有针对性追问；4=持续依据实际回答调整并有具体证据，几乎无无效重复。没有明确证据不得给4。"
       "不要把完成大纲或自然措辞等同于适应性。参考 Outline 的必需 Slots，但允许合理跳题。"
       "不要推断真人访谈体验。仅输出 JSON。")
    overall_schema={"type":"object","properties":{
      "coherence":{"type":["integer","null"],"enum":[1,2,3,4,None]},"adaptiveness":{"type":["integer","null"],"enum":[1,2,3,4,None]},
      "coherence_comment":{"type":"string"},"adaptiveness_comment":{"type":"string"},
      "overall_comment":{"type":"string"}},"required":["coherence","adaptiveness","coherence_comment","adaptiveness_comment","overall_comment"]}
    # A single dedicated holistic judgment preserves context across batches.
    global_result=_ask(model,overall_sys,{"interviewer_mode":run.get('interviewer_mode',run.get('interviewer_policy')),
       "study_goal":guide.get('goal',''),"outline_questions":outline_questions,"turns":[{"round":t['round'],"question":t['question'][:600],
       "answer":t['answer'][:600]} for t in turns]},overall_schema,opener)
    # A fixed Planner question list cannot demonstrate autonomous strategy adaptation.
    mode=str(run.get('interviewer_mode',run.get('interviewer_policy',''))).lower()
    if 'planner' in mode:
        global_result['adaptiveness']=None
        global_result['adaptiveness_comment']='Planner 固定题目回放，无法评价模型的自主情境适应性'
    holistic_warnings=[f"第 {r['round']} 轮：{w}" for r in rows for w in r.get("coverage_validation_warnings",[])]
    holistic_raw_scores={}
    for name in ('coherence','adaptiveness'):
        raw_value=global_result.get(name)
        normalized, warning=_parse_holistic_score(raw_value,name)
        global_result[name]=normalized
        if warning:
            holistic_warnings.append(warning)
            holistic_raw_scores[name]=raw_value
            comment_key=name+'_comment'
            previous=str(global_result.get(comment_key,'')).strip()
            global_result[comment_key]=(previous+'；' if previous else '')+warning
    seen=set(x for r in rows for x in r['covered_subtopic_ids'])
    relevance_volume=sum(len(r['answer'].strip())*r['relevance'] for r in rows)
    metric=lambda value,unit,note: {"value":value,"unit":unit,"note":note}
    metrics={
       "relevant_response_volume":metric(relevance_volume,"weighted_characters","逐轮回答字符数 × 研究相关性(0/1/2)；不是纯粹质量得分"),
       "guide_coverage":metric(round(len(seen)/len(topics)*100,2),"percent",
           f"已验证 {len(seen)}/{len(topics)} 个必需 Slot；原始证据为逐轮受访者引文。未完成访谈时是截至中断的累计覆盖率。"),
       "novel_responses":metric(sum(r['novel'] for r in rows),"count","大纲外但相关的新信息回答次数，须经原文引用核对"),
       "coherence":metric(global_result.get('coherence'),"rating_1_to_4",str(global_result.get('coherence_comment',''))[:800]),
       "adaptiveness":metric(global_result.get('adaptiveness'),"rating_1_to_4",str(global_result.get('adaptiveness_comment',''))[:800]),
       "leading_questions":metric(sum(r['leading'] for r in rows),"count","诱导性提问次数"),
       "unclear_questions":metric(sum(r['unclear'] for r in rows),"count","模糊提问次数"),
       "support_rapport":metric(sum(r['support_rapport'] for r in rows),"count","支持、共情与关系建立表达次数"),
       "overall_experience":metric(None,"rating_1_to_5","等待真人受访者评价；机评不填"),
       "comfort_level":metric(None,"rating_1_to_4","等待真人受访者评价；机评不填")}
    # Auxiliary audit, never mixed into the ten published ReportCard measures.
    question_seen=set()
    repeats=[]
    for row in rows:
        key_question=_text_norm(row['question'])
        if key_question and key_question in question_seen:
            repeats.append(row['round'])
        question_seen.add(key_question)
    flagged_support=[r['round'] for r in rows if r['support_rapport']]
    auxiliary={"exact_duplicate_question_rounds":repeats,
       "exact_duplicate_question_count":len(repeats),
       "support_rapport_flagged_rounds":flagged_support,
       "refusal_audited_rounds":[r['round'] for r in rows if '明确拒答' in r.get('audit_note','')]}
    if len(flagged_support)>len(rows)*0.6:
        holistic_warnings.append('支持共情标记覆盖超过60%的采访轮次，请核实是否误把正常提问当成共情表达')
    _clear_checkpoint(store,key)
    return {"schema":"interview_reportcard_judge.v1","status":"llm_suggestion_unverified",
      "judge_model":model,"judge_version":"v1.7.5_canonical_slot_contract","coverage_basis":"outline_required_slots","partial_run":run.get('completed') is False,"recovered_round_batches":recovered_batches,"judge_provider":"api" if api_config else "ollama","outline_used":bool(outline_questions),"paper":"InterviewPlayground InterviewReportCard",
      "metrics":metrics,"turn_annotations":rows,"auxiliary_diagnostics":auxiliary,"resume_diagnostics":{"reused_rounds":resumed,
       "batch_size":BATCH,"auto_retries":0,"checkpoint_storage":"local_server_memory_only"},"scoring_warnings":holistic_warnings,"invalid_raw_scores":holistic_raw_scores,"covered_subtopic_ids":sorted(seen),
      "total_subtopics":len(topics),"overall_comment":str(global_result.get('overall_comment',''))[:1500],
      "participant_experience":{"overall_experience":None,"comfort_level":None,"source":"not_collected"},
      "warning":"本地LLM评分建议；研究者必须核验逐轮依据。真人体验未收集。不得据此声称已验证访谈信效度。"}
