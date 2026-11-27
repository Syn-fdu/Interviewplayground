"""Safe integration of v0.8 respondent controls with v0.7.2 Scenario 2.0.

Restricted source remains in the existing local-only STORE; the interviewer
never sees respondent memories. Researcher exports are explicit and local only.
"""
from __future__ import annotations
import copy
import json
from playground.interview.respondent_core import (
    StudioError, SCHEMA, VERSION, TRAITS, migrate, validate_bundle, default_state,
    plan_answer, answer_and_commit, export_scope, public_guide, now_iso
)
from playground.interview.respondent_llm import (
    _select_memories_json, _semantic_memory_retrieve, validate_model,
    FIXED_LOCAL_ENDPOINT, LocalModelError
)
from urllib.request import urlopen
from playground.common.interruptible_http import interruptible_urlopen
from playground.research.guide_schema import standardize_guide, validate_guide, with_validation_snapshot

MAX_EDIT_BYTES = 6 * 1024 * 1024

def make_bundle(scenario, model="qwen2.5:7b", public_pool=None, public_pool_source="default"):
    if scenario.get("source_type") != "v2_restricted_local":
        raise StudioError("升级研究工作台仅管理服务端注册的 V2 Scenario")
    b = migrate(scenario)
    from playground.memory.memory_system import default_pool, pool_summary
    pool = public_pool if public_pool is not None else default_pool()
    b["respondent"]["public_memory_pool"] = copy.deepcopy(pool)
    b["respondent"]["public_pool_summary"] = pool_summary(pool,public_pool_source)
    b["source_scenario"] = {
        "scenario_id": scenario["scenario_id"],
        "source_type": scenario["source_type"],
        "provenance": copy.deepcopy(scenario.get("provenance", {})),
        "usage": copy.deepcopy(scenario.get("usage", {})),
        "note": "仅保存来源元数据；研究包不包含自动提取的 Benchmark 隐藏标准答案"
    }
    b["interview_guide"] = with_validation_snapshot(scenario["public"]["interview_guide"])
    persona = scenario["private"].get("respondent_persona") or {}
    base = scenario["public"].get("background") or ""
    from playground.research.participant_profile import sanitize_public_briefing
    b["respondent"]["persona"] = {
        "interviewer_briefing": sanitize_public_briefing(scenario["public"].get("participant_briefing")),
        "source_linked_profile_attributes": copy.deepcopy(persona.get("source_linked_profile_attributes", [])),
        "profile_review_status": persona.get("profile_review_status", "requires_human_review"),
        "profile_construction": persona.get("profile_construction", "unreviewed_source"),
        "public_background": str(base)[:500],
        "speaking_style": "使用自然口语；根据有来源的事实答题，不主动汇总完整人生经历。",
        "verified_identity_attributes": copy.deepcopy(persona.get("verified_identity_attributes",{})),
        "source_context_note": persona.get("source_context_provenance","unverified"),
        "review_status": persona.get("review_status","pending")
    }
    old_traits = persona.get("behavioral_traits") or {}
    for name in TRAITS:
        setting=old_traits.get(name)
        if isinstance(setting,dict):
            b["respondent"]["behavior_parameters"][name] = setting.get("value",0.5)
    b["generation"]["model"]=validate_model(model,"qwen2.5:7b")
    b["generation"]["language"]="en" if scenario.get("language")=="en" else "zh"
    b["generation"]["provider"]="ollama"
    b["audit"]["notes"]="v0.7.2 来源场景转换为 v0.8.1 受访者研究工作台；句子事实候选需人工审核"
    b["migration_warnings"].append(
        "原访谈已做保守的第三人称重述和低信息片段筛查；不等于通过语义蕴含校验。"
        "公共池仅提供非事实表达线索，绝不构成个人经历或引用证据。请人工复核。")
    if scenario.get("provenance",{}).get("memory_stats",{}).get("memory_source")=="memorybank.json":
        b["audit"]["notes"]="预构建 MemoryBank；语义复核与原子事实拆分尚未全部完成。"
    b["session_state"]=default_state()
    validate_bundle(b)
    return b

def _semantic_selection(bundle, question, history, model, opener=interruptible_urlopen):
    """Always select locally across languages, including when <= 10 memories.

    On very large banks, retain the legacy batchwise local semantic selector.
    The original v0.7.2 <=10 all_memory shortcut is intentionally bypassed.
    """
    bank = bundle["respondent"]["memory_bank"]
    if not bank:
        return [], "no_memory_bank"
    snippets = [{
        "memory_id":m["memory_id"],
        "source_question":m.get("source_question", ""),
        "text":m.get("content","")
    } for m in bank]
    if len(snippets)<=38:
        selected = _select_memories_json({
            "question":question,
            "last_respondent_response":str(history[-1].get("answer",""))[-200:] if history else "",
            "candidate_memories":[{
                "id":m["memory_id"],
                "original_question":str(m["source_question"])[:150],
                "respondent_excerpt":m["text"][:260]
            } for m in snippets]
        },model,opener)
        known={m["memory_id"] for m in snippets}
        return list(dict.fromkeys(i for i in selected.get("selected_ids",[]) if i in known))[:5], "ollama_multilingual_semantic"
    selected,_ = _semantic_memory_retrieve(snippets,question,history,model,opener,max_selected=5)
    return [m["memory_id"] for m in selected], "ollama_multilingual_semantic_batched"

def plan(bundle, question, history=None, model=None, retrieval="semantic", opener=interruptible_urlopen):
    history=history or []
    if retrieval=="semantic":
        ids, status = _semantic_selection(bundle,question,history,
                                          validate_model(model,bundle["generation"]["model"]),opener)
        p=plan_answer(bundle, question, semantic_memory_ids=ids)
    elif retrieval=="lexical_preview":
        status="lexical_preview_only_not_equivalent_to_live_semantic"
        p=plan_answer(bundle,question)
    else:
        raise StudioError("retrieval 必须是 semantic 或 lexical_preview")
    p["retrieval_status"]=status
    return p

def respond(bundle, question, history, model, language, behavioral_traits=None,
            client=None, opener=interruptible_urlopen, memory_mode="retrieval"):
    """Return a new, verified bundle + browser-compatible respondent answer."""
    b=copy.deepcopy(bundle)
    if behavioral_traits is not None:
        if not isinstance(behavioral_traits,dict) or set(behavioral_traits)!=set(TRAITS):
            raise StudioError("受访者参数必须完整包含六项")
        b["respondent"]["behavior_parameters"]=copy.deepcopy(behavioral_traits)
    if language not in ("zh","en","bi"):
        raise StudioError("受访者语言格式无效")
    b["generation"]["language"]="en" if language=="en" else "zh"
    b["generation"]["model"]=validate_model(model,b["generation"]["model"])
    validate_bundle(b)
    if memory_mode not in ("retrieval", "full", "evidence"):
        raise StudioError("memory_mode 必须为 retrieval、evidence 或 full", "invalid_memory_mode", 400)
    b["memory_mode"] = memory_mode
    if memory_mode == "full":
        candidate_ids, retrieval_status = None, "full_memory_all_provided"
    elif memory_mode == "evidence":
        candidate_ids, retrieval_status = None, "local_evidence_router"
    else:
        candidate_ids, retrieval_status = _semantic_selection(b,question,history,b["generation"]["model"],opener)
    committed,result=answer_and_commit(b,question,client=client,semantic_memory_ids=candidate_ids,
                                       _validated=True)
    turn=result["turn"]
    fact_mem={f["fact_id"]:m["memory_id"]
              for m in b["respondent"]["memory_bank"] for f in m["facts"]}
    # Old UI's E/M references remain compatible, but fact-level IDs are
    # recorded separately rather than pretending one M contains one fact.
    used_mem=list(dict.fromkeys(fact_mem[fid] for fid in turn["used_fact_ids"] if fid in fact_mem))
    ans={
        "answer":turn["answer"],
        "text":turn["answer"],
        "supported":bool(used_mem),
        "evidence_ids":used_mem,
        "used_fact_ids":turn["used_fact_ids"],
        "new_fact_ids":turn["new_fact_ids"],
        "memory_total":len(b["respondent"]["memory_bank"]),
        "public_memory_sample_count":len(turn.get("public_memory_sample_ids") or []),
        "public_memory_pool_sha256":turn.get("public_pool_sha256"),
        "retrieval_status":retrieval_status,
        "memory_mode":memory_mode,
        "memory_bank_count":len(b["respondent"]["memory_bank"]),
        "recalled_memory_ids":turn["recalled_memory_ids"],
        "behavioral_traits":copy.deepcopy(b["respondent"]["behavior_parameters"]),
        "grounding_status":"model_fact_ids_unverified" if used_mem else "insufficient_source",
        "model":b["generation"]["model"],
        "json_format_retry_count":turn.get("generation_diagnostics",{}).get("json_format_retry_count",0),
        "generation_diagnostics":copy.deepcopy(turn.get("generation_diagnostics",{})),
        "diagnostic_note":"事实编号合法不等于语义蕴含已验证；研究数据不应直接用作评分金标准。"
    }
    committed["generation"]["language"]=bundle["generation"]["language"]
    committed["generation"]["model"]=bundle["generation"]["model"]
    committed["respondent"]["behavior_parameters"]=copy.deepcopy(bundle["respondent"]["behavior_parameters"])
    return committed,ans

def guard_guide_edit(old_guide,new_guide):
    if not isinstance(new_guide,dict):
        raise StudioError("采访大纲必须是 JSON 对象")
    if "topics" not in new_guide or not isinstance(new_guide["topics"],list):
        raise StudioError("大纲必须包含 topics 数组")
    if len(new_guide["topics"])>60:
        raise StudioError("topics 条数过多")
    # Source provenance cannot be promoted from reconstruction to original.
    if old_guide.get("original_guide_provided") is not True and (
        new_guide.get("original_guide_provided") is True or
        new_guide.get("source_kind")=="original_guide_provided"):
        raise StudioError("不能把重建大纲重标记为原始大纲；请保留原始 provenance")
    if new_guide.get("source_kind") != old_guide.get("source_kind"):
        raise StudioError("不可更改原始大纲来源标识 source_kind（可修改具体内容）")
    report = validate_guide(new_guide)
    if not report["ok"]:
        issue = report["errors"][0]
        raise StudioError("采访大纲不符合 Guide Schema v1.0：" + issue["path"] + "：" + issue["message"])


def apply_update(current, section, value):
    """Pure update: no source-metadata edits; caller commits atomically."""
    b=copy.deepcopy(current)
    if section=="guide":
        guard_guide_edit(b["interview_guide"],value)
        b["interview_guide"]=with_validation_snapshot(value)
    elif section=="behavior":
        if not isinstance(value,dict) or set(value)!=set(TRAITS):
            raise StudioError("六项参数必须完整存在")
        b["respondent"]["behavior_parameters"]=copy.deepcopy(value)
    elif section=="persona":
        if not isinstance(value,dict):
            raise StudioError("persona 必须是对象")
        b["respondent"]["persona"]=copy.deepcopy(value)
    elif section=="memory":
        if not isinstance(value,list):
            raise StudioError("Memory Bank 必须是数组")
        b["respondent"]["memory_bank"]=copy.deepcopy(value)
    elif section=="generation":
        if not isinstance(value,dict):
            raise StudioError("generation 必须是对象")
        value=copy.deepcopy(value)
        if value.get("provider")!="ollama":
            raise StudioError("实际访谈仅支持本地 Ollama；演示模式不能接入真实访谈")
        value["model"]=validate_model(value.get("model"),b["generation"]["model"])
        b["generation"]=value
    else:
        raise StudioError("可修改 guide / behavior / persona / memory / generation / state")
    b["session_state"]=default_state()
    validate_bundle(b)
    b["audit"]["revision"]=int(b["audit"].get("revision",0))+1
    b["audit"]["updated_at"]=now_iso()
    return b

def apply_import(current,scope,data):
    if not isinstance(data,dict):
        raise StudioError("导入必须是 JSON 对象")
    if scope=="full":
        if data.get("schema_version")!=SCHEMA:
            raise StudioError("完整研究包 schema_version 不一致")
        source=data.get("source_scenario") or {}
        original=current.get("source_scenario") or {}
        if source.get("scenario_id")!=original.get("scenario_id") or (
                source.get("provenance") or {}).get("source_fingerprint_sha256") != (
                original.get("provenance") or {}).get("source_fingerprint_sha256"):
            raise StudioError("导入包不属于当前 Scenario / V2 来源，拒绝跨场景覆盖")
        b=copy.deepcopy(data)
        guard_guide_edit(current["interview_guide"],b.get("interview_guide"))
        b["source_scenario"]=copy.deepcopy(current["source_scenario"])
        b["interview_guide"]=with_validation_snapshot(b["interview_guide"])
        b["session_state"]=default_state()
        validate_bundle(b)
        if b["generation"]["provider"]!="ollama":
            raise StudioError("完整研究包不能启用 demo 生成器")
        b["generation"]["model"]=validate_model(b["generation"]["model"],"qwen2.5:7b")
        b["audit"]=copy.deepcopy(current["audit"])
        b["audit"]["revision"]+=1
        b["audit"]["updated_at"]=now_iso()
        return b
    if scope=="guide":
        return apply_update(current,"guide",data.get("interview_guide"))
    if scope=="respondent":
        if "respondent" not in data:
            raise StudioError("受访者配置缺少 respondent")
        b=copy.deepcopy(current)
        b["respondent"]=copy.deepcopy(data["respondent"])
        if isinstance(data.get("generation"),dict):
            b["generation"]=copy.deepcopy(data["generation"])
        b["session_state"]=default_state()
        guard_guide_edit(current["interview_guide"],b["interview_guide"])
        validate_bundle(b)
        if b["generation"]["provider"]!="ollama":
            raise StudioError("受访者配置不能启用 demo")
        b["audit"]["revision"]+=1
        b["audit"]["updated_at"]=now_iso()
        return b
    raise StudioError("只允许 full / guide / respondent 范围导入；会话状态通过 session 编辑")

def export_bundle(bundle, scope, state=None):
    b=copy.deepcopy(bundle)
    if state is not None:
        b["session_state"]=copy.deepcopy(state)
    if scope in ("full","guide","respondent","state","diagnostics"):
        result=export_scope(b,scope)
        if scope in ("state","diagnostics"):
            # Enables the Benchmark to bind Session State to the exact guide
            # without exposing the hidden respondent memory or gold answers.
            from playground.evaluation.benchmark_foundation import digest
            result["scenario_id"]=(b.get("source_scenario") or {}).get("scenario_id")
            result["source_fingerprint_sha256"]=((b.get("source_scenario") or {})
                .get("provenance") or {}).get("source_fingerprint_sha256")
            result["guide_sha256"]=digest(b["interview_guide"])
            result["guide_schema_version"]=b["interview_guide"].get("guide_schema_version")
        return result
    raise StudioError("导出范围无效")
