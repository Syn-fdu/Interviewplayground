"""Spoken Truth: source-fact elicitation, local semantic judge, evidence-audited."""
import re, json, hashlib
from urllib.request import urlopen
from interview_reportcard_judge import _ask, _reportcard_turns, JudgeError, _text_norm
from agent_memory_export import eligible_source_facts

def facts_from_memory(value):
    if not isinstance(value,dict): return []
    rows=value.get("source_facts")
    if not isinstance(rows,list):
        if isinstance(value.get("respondent"),dict):
            rows=eligible_source_facts(value)
        else: return []
    result=[]; seen=set()
    for f in rows:
        if not isinstance(f,dict):continue
        ident=f.get("id"); text=f.get("text")
        if not isinstance(ident,str) or not isinstance(text,str) or not text.strip():continue
        if ident in seen:raise JudgeError("Memory source_facts 存在重复 ID: "+ident)
        seen.add(ident)
        result.append({"id":ident,"dataset_memory_id":str(f.get("dataset_memory_id") or ident),
            "text":text,"source_question":str(f.get("source_question") or "")})
    return result

def evaluate_semantic(run,facts,model,opener=urlopen,checkpoint_store=None):
    turns=_reportcard_turns(run)
    answers={t["round"]:t["answer"] for t in turns}
    if not facts:
        return {"value":None,"unit":"percent","status":"not_evaluable","note":"缺少原始 source_facts；请选择当前访谈或上传对应 Agent Memory JSON。","denominator":0}
    # Full transcript for cross-language and cross-turn evidence; no hidden facts in ordinary quality judge.
    key=hashlib.sha256(json.dumps({"protocol":"spoken_truth_semantic_v166","facts":facts,"turns":turns,"model":model},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    from interview_reportcard_judge import _load_checkpoint,_save_checkpoint,_clear_checkpoint
    store=checkpoint_store if checkpoint_store is not None else {}
    rows=_load_checkpoint(store,key)
    if [x["fact_id"] for x in rows]!=[x["id"] for x in facts[:len(rows)]]:
        rows=[]
    system=("你是独立事实引出评审员。对每条原始事实，判断模拟受访者是否在本次访谈的回答中明确表达了同一事实。"
        "允许中英文翻译、同义改写、跨轮上下文；必须保持主体、时间、数量、否定、因果及限定条件一致。"
        "来源问题仅用于解释短事实，不是证据；采访者提问、推测、常识、事实库本身不能证明受访者说过。"
        "matched=完整语义支持；partial=仅部分支持；contradicted=明确冲突；not_found=未引出；uncertain=无法判断。"
        "matched 必须提供受访者原文的连续非空引文和对应轮次，禁止编造。只返回 JSON，每个输入事实恰好一次。")
    for offset in range(len(rows),len(facts),6):
        batch=facts[offset:offset+6]; ids=[f["id"] for f in batch]
        row_schema={"type":"object","properties":{"fact_id":{"type":"string","enum":ids},"status":{"type":"string","enum":["matched","partial","contradicted","not_found","uncertain"]},"evidence":{"type":"array","items":{"type":"object","properties":{"round":{"type":"integer"},"quote":{"type":"string"}},"required":["round","quote"]}},"reason":{"type":"string"}},"required":["fact_id","status","evidence","reason"]}
        schema={"type":"object","properties":{"facts":{"type":"array","minItems":len(batch),"maxItems":len(batch),"items":row_schema}},"required":["facts"]}
        task={"source_facts":batch,"transcript":turns}
        raw=_ask(model,system,task,schema,opener)
        got=raw.get("facts")
        if not isinstance(got,list) or len(got)!=len(ids) or any(not isinstance(x,dict) for x in got) or sorted(str(x.get("fact_id")) for x in got)!=sorted(ids):
            raise JudgeError("Spoken Truth 第 "+str(offset//6+1)+" 批事实 ID 不完整；已保留成功批次，重试可继续。")
        byid={x["fact_id"]:x for x in got}
        for f in batch:
            item=byid[f["id"]]; status=item.get("status")
            if status not in ("matched","partial","contradicted","not_found","uncertain"):status="uncertain"
            evidence=[]
            for e in item.get("evidence",[]) if isinstance(item.get("evidence"),list) else []:
                if not isinstance(e,dict):continue
                n=e.get("round"); q=e.get("quote")
                if type(n) is int and n in answers and isinstance(q,str) and q.strip() and _text_norm(q) in _text_norm(answers[n]):
                    evidence.append({"round":n,"quote":q})
            if status in ("matched","partial","contradicted") and not evidence:status="uncertain"
            rows.append({"fact_id":f["id"],"status":status,"evidence":evidence,"reason":str(item.get("reason",""))[:500]})
        _save_checkpoint(store,key,rows)
    counts={k:sum(x["status"]==k for x in rows) for k in ("matched","partial","contradicted","not_found","uncertain")}
    total=len(facts)
    _clear_checkpoint(store,key)
    return {"value":round(100*counts["matched"]/total,2),"unit":"percent","status":"semantic_judge_unverified","reference_basis":"source_facts","method":"semantic_fact_elicitation_v166","numerator":counts["matched"],"denominator":total,"counts":counts,"fact_annotations":rows,"note":"严格完整语义命中 "+str(counts["matched"])+"/"+str(total)+"；部分支持和不确定不计命中；需人工复核。"}
