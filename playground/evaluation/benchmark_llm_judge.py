"""Opt-in local Ollama judge. Suggestions only; not calibrated scores or ground truth."""
import json
import time
from playground.common.performance_metrics import record as perf_record
import re
from urllib.request import Request, urlopen
from urllib.error import URLError
from playground.interview.respondent_llm import validate_model

JUDGE_ENDPOINT = "http://127.0.0.1:11434/api/chat"
MAX_TURNS = 20
MAX_CHARS = 1200

class JudgeError(ValueError):
    pass

def _parse_model_json(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.I)
        if match: return json.loads(match.group(1))
        raise JudgeError("评审模型未返回完整 JSON；请换用支持结构化输出的本地模型")

def _sanitize_turns(run):
    turns = run.get("turns", run.get("session_state", {}).get("history", []))
    if not isinstance(turns,list) or not 1<=len(turns)<=MAX_TURNS:
        raise JudgeError("评审需要 1–20 轮问答")
    result=[]
    for i, t in enumerate(turns,1):
        if not isinstance(t,dict): raise JudgeError("逐轮数据格式错误")
        q=t.get("interviewer",t.get("question"))
        a=t.get("respondent",t.get("answer"))
        if not isinstance(q,str) or not isinstance(a,str) or not q.strip() or not a.strip():
            raise JudgeError("LLM 评审需要完整的逐轮问答正文")
        result.append({"round":i,"question":q[:MAX_CHARS],"answer":a[:MAX_CHARS]})
    return result

def evaluate_llm(run, guide, model, opener=urlopen):
    if not isinstance(run,dict): raise JudgeError("缺少运行记录")
    turns=_sanitize_turns(run)
    model=validate_model(model,"gemma3:4b")
    subtopics=[]
    if isinstance(guide,dict):
        for topic in guide.get("topics",[])[:20]:
            if isinstance(topic,dict):
                for sub in topic.get("subtopics",[])[:20]:
                    if isinstance(sub,dict):
                        subtopics.append({"id":str(sub.get("subtopic_id",""))[:64],
                                          "name":str(sub.get("name",sub.get("title",sub.get("label",""))))[:120]})
    system=("你是独立访谈质量审查员。仅依据下方明确提供的采访问题、回答和公开研究大纲评分；"
            "不是受访者，不可假装看到隐藏记忆、真实事实或未提供的材料。"
            "仅返回 JSON，不要 Markdown。逐轮 research_relevance 只能为 0/1/2 或 null；"
            "leading/unclear/support_rapport/redundant 只能为 true/false/null；"
            "coherence/adaptiveness 只能为 1/2/3/4 或 null。"
            "uncertainty 和 rationale 需简短；不可自行声称已验证事实或体验。"
            "未能判断的项目必须返回 null。逐轮输出必须与输入轮次数一致。"
            "covered_subtopic_ids 只可选自已提供的大纲 ID；无大纲时必须为空数组。")
    instruction={
       "task":"逐轮独立评分建议。只判断有回答支持的主题覆盖。不要填真人体验。不可作科研金标准。",
       "allowed_subtopics":subtopics[:100],
       "turns":turns,
       "required_output":{"turns":[{"round":1,"research_relevance":2,"covered_subtopic_ids":[],
                        "leading":False,"unclear":False,"support_rapport":False,
                        "redundant":False,"rationale":"简短理由"}],
                "overall":{"coherence":3,"adaptiveness":3,"rationale":"简短理由"},
                "uncertainty":"局限"}
    }
    schema={
      "type":"object","properties":{
       "turns":{"type":"array","items":{"type":"object","properties":{
          "round":{"type":"integer"},"research_relevance":{"type":["integer","null"]},
          "covered_subtopic_ids":{"type":"array","items":{"type":"string"}},
          "leading":{"type":["boolean","null"]},"unclear":{"type":["boolean","null"]},
          "support_rapport":{"type":["boolean","null"]},"redundant":{"type":["boolean","null"]},
          "rationale":{"type":"string"}},"required":["round","research_relevance","covered_subtopic_ids","leading","unclear","support_rapport","redundant","rationale"]}},
       "overall":{"type":"object","properties":{
          "coherence":{"type":["integer","null"]},"adaptiveness":{"type":["integer","null"]},"rationale":{"type":"string"}},
          "required":["coherence","adaptiveness","rationale"]},
       "uncertainty":{"type":"string"}},"required":["turns","overall","uncertainty"]}
    payload={"model":model,"stream":False,"format":schema,
             "options":{"temperature":0,"num_predict":2300},
             "messages":[{"role":"system","content":system},
                         {"role":"user","content":json.dumps(instruction,ensure_ascii=False)}]}
    req=Request(JUDGE_ENDPOINT, data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type":"application/json"},method="POST")
    perf_start = time.monotonic()
    try:
        with opener(req,timeout=120) as response: body=json.load(response)
    except (URLError,OSError,TimeoutError,ValueError) as exc:
        perf_record("judge", model, time.monotonic()-perf_start, error=exc)
        raise JudgeError("本机评审模型请求失败，请检查 Ollama 和模型是否已下载") from exc
    perf_record("judge", model, time.monotonic()-perf_start, body)
    raw=(body.get("message") or {}).get("content")
    if not isinstance(raw,str): raise JudgeError("评审模型未返回文本")
    result=_parse_model_json(raw)
    if not isinstance(result,dict) or not isinstance(result.get("turns"),list) or len(result["turns"])!=len(turns):
        raise JudgeError("评审结果轮数不匹配")
    allowed={x["id"] for x in subtopics if x["id"]}
    expected={t["round"] for t in turns}; seen=set()
    for row in result["turns"]:
        if not isinstance(row,dict) or type(row.get("round"))!=int or row["round"] not in expected or row["round"] in seen:
            raise JudgeError("评审轮次无效或重复")
        seen.add(row["round"])
        rel=row.get("research_relevance")
        if rel is not None and (type(rel)!=int or rel not in (0,1,2)):raise JudgeError("相关性分数越界")
        for key in ("leading","unclear","support_rapport","redundant"):
            if row.get(key) is not None and type(row.get(key)) is not bool:raise JudgeError(key+" 类型错误")
        subs=row.get("covered_subtopic_ids")
        if not isinstance(subs,list) or len(subs)>100 or any(s not in allowed for s in subs):
            raise JudgeError("模型使用了不在公开大纲中的子主题编号")
        row["rationale"]=str(row.get("rationale",""))[:350]
    overall=result.get("overall")
    if not isinstance(overall,dict):raise JudgeError("缺少整体评价")
    for k in ("coherence","adaptiveness"):
        x=overall.get(k)
        if x is not None and (type(x)!=int or x not in (1,2,3,4)):
            raise JudgeError("整体评分越界")
    overall["rationale"]=str(overall.get("rationale",""))[:600]
    return {"status":"llm_suggestion_unverified","model":model,
            "judge_version":"0.1_local_opt_in","turns":sorted(result["turns"],key=lambda v:v["round"]),
            "overall":overall,"uncertainty":str(result.get("uncertainty",""))[:800],
            "warning":"LLM 建议未经人工审核、未验证评分信效度；不替代 Benchmark 客观统计及人工 Rubrics。"}
