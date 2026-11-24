"""Benchmark Foundation v0.1 — deterministic descriptive metrics + manual rubrics.

No pseudo-LLM judge. Do not treat model self-reported reference IDs, coverage
heuristics, or question-word overlap as validated response quality. Any measure
requiring unavailable transcript, guide, or human coding is null, never zero.
All data is processed in memory on the local-only Python server.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from benchmark_config import validate_config, fingerprint, BenchmarkConfigError

BENCHMARK_VERSION = "0.2"
RUBRIC_VERSION = "benchmark_foundation.v0.1"

class BenchmarkError(ValueError):
    pass


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")).hexdigest()


def measure(value, unit, status="computed", method="", note=""):
    return {"value": value, "unit": unit, "status": status, "method": method, "note": note}


def missing(unit, cause):
    return measure(None, unit, "not_evaluable", note=cause)


def normalized_text(text):
    # Exact repeats under case/spacing/punctuation normalization — NOT semantic similarity.
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", text).casefold())


def _extract_turns(run):
    if "session_state" in run:
        state = run.get("session_state")
        if not isinstance(state, dict): raise BenchmarkError("session_state 必须是对象")
        raw = state.get("history", [])
        origin = "researcher_session_state"
    elif isinstance(run.get("turns"), list):
        raw = run["turns"]
        origin = run.get("report_type", "run_json")
    else:
        raise BenchmarkError("JSON 须包含 turns 或 session_state.history")
    if not isinstance(raw, list):
        raise BenchmarkError("轮次必须是列表")
    turns=[]
    seen=set()
    for i, row in enumerate(raw, 1):
        if not isinstance(row, dict): raise BenchmarkError("第 %s 轮格式不正确"%i)
        number=row.get("round",i)
        if type(number)!=int or number<1 or number in seen:
            raise BenchmarkError("重复或无效的 round 编号")
        seen.add(number)
        question=row.get("interviewer",row.get("question"))
        answer=row.get("respondent",row.get("answer"))
        refs=row.get("used_fact_ids")
        if refs is None: refs=row.get("evidence_ids")
        turns.append({
            "round":number,
            "question":question if isinstance(question,str) and question.strip() else None,
            "answer":answer if isinstance(answer,str) and answer.strip() else None,
            "reference_ids":refs if isinstance(refs,list) else None,
        })
    return sorted(turns,key=lambda x:x["round"]),origin


def _guide_subtopics(guide):
    if guide is None:return None
    if not isinstance(guide,dict):raise BenchmarkError("guide 必须是 JSON 对象")
    topics=guide.get("topics")
    if not isinstance(topics,list):raise BenchmarkError("guide 缺少 topics 列表")
    ids=[]
    for t in topics:
        if not isinstance(t,dict):continue
        for sub in t.get("subtopics",[]) if isinstance(t.get("subtopics"),list) else []:
            if isinstance(sub,dict) and isinstance(sub.get("subtopic_id"),str):
                ids.append(sub["subtopic_id"])
    if len(set(ids))!=len(ids):raise BenchmarkError("guide 子主题编号重复")
    return ids


def empty_annotation(turns):
    return {
        "rubric_version":RUBRIC_VERSION,
        "reviewer_id":"",
        "overall":{"coherence":None,"adaptiveness":None},
        "turn_annotations":[{
            "round":t["round"],"research_relevance":None,"covered_subtopic_ids":[],
            "subtopics_reviewed":False,"leading":None,"unclear":None,"support_rapport":None,"redundant":None,"note":""
        } for t in turns],
        "participant_experience":{
            "source":"not_collected", "overall_experience":None, "comfort_level":None
        }
    }


def make_annotation_template(run):
    if not isinstance(run,dict):raise BenchmarkError("run 必须是对象")
    turns,_=_extract_turns(run)
    return empty_annotation(turns)


def validate_annotation(annotation,turns,subtopics):
    if annotation is None:return None
    if not isinstance(annotation,dict) or annotation.get("rubric_version")!=RUBRIC_VERSION:
        raise BenchmarkError("标注必须使用 benchmark_foundation.v0.1")
    reviewer=annotation.get("reviewer_id","")
    if not isinstance(reviewer,str) or len(reviewer)>120:
        raise BenchmarkError("reviewer_id 无效")
    score=annotation.get("overall") or {}
    if not isinstance(score,dict):raise BenchmarkError("overall 必须是对象")
    for dim in ("coherence","adaptiveness"):
        v=score.get(dim)
        if v is not None and (type(v)!=int or not 1<=v<=4):
            raise BenchmarkError("%s 评分须为 1–4 或 null"%dim)
    rows=annotation.get("turn_annotations",[])
    if not isinstance(rows,list):raise BenchmarkError("turn_annotations 必须是数组")
    valid={r["round"] for r in turns}
    seen=set()
    for row in rows:
        if not isinstance(row,dict):raise BenchmarkError("逐轮标注须为对象")
        rd=row.get("round")
        if type(rd)!=int or rd not in valid or rd in seen:
            raise BenchmarkError("逐轮标注的 round 不在访谈中或重复")
        seen.add(rd)
        rel=row.get("research_relevance")
        if rel is not None and (type(rel)!=int or not 0<=rel<=2):
            raise BenchmarkError("research_relevance 必须为 0、1、2 或 null")
        subs=row.get("covered_subtopic_ids")
        if not isinstance(subs,list) or len(set(subs))!=len(subs):
            raise BenchmarkError("covered_subtopic_ids 必须是无重复列表")
        covered_reviewed=row.get("subtopics_reviewed",False)
        if type(covered_reviewed)!=bool:
            raise BenchmarkError("subtopics_reviewed 必须为 true/false")
        if subs and not covered_reviewed:
            raise BenchmarkError("标注子主题时必须先确认本轮主题已完成核对")
        if subs and subtopics is None:
            raise BenchmarkError("标注子主题前必须提供匹配的采访大纲")
        if any((not isinstance(x,str)) or x not in subtopics for x in subs) if subtopics is not None else False:
            raise BenchmarkError("标注使用了不存在的 subtopic_id")
        for key in ("leading","unclear","support_rapport","redundant"):
            value=row.get(key)
            if value is not None and type(value)!=bool:
                raise BenchmarkError("%s 必须为 true/false/null"%key)
        if len(str(row.get("note", "")))>2000:
            raise BenchmarkError("标注 note 不应超过 2000 字")
    experience=annotation.get("participant_experience") or {}
    if not isinstance(experience,dict):raise BenchmarkError("participant_experience 必须是对象")
    source=experience.get("source","not_collected")
    if source not in ("not_collected","human_self_report"):
        raise BenchmarkError("体验评分仅支持 human_self_report 来源")
    for k,limit in (("overall_experience",5),("comfort_level",4)):
        value=experience.get(k)
        if value is not None and (source!="human_self_report" or type(value)!=int or not 1<=value<=limit):
            raise BenchmarkError("体验分数仅允许直接取得的真人自评")
    return annotation


def runtime_audit(run, turns):
    """Diagnostic observations only; never a substitute for answer-quality judgments.

    Exported restricted V2 records contain no Q/A text. Selection metadata is
    counted as runtime evidence, not evidence that the respondent remembered
    accurately or that a question was relevant.
    """
    conditions = run.get("experiment_conditions") if isinstance(run.get("experiment_conditions"), dict) else {}
    mode = conditions.get("interviewer_policy", run.get("policy"))
    rows = run.get("turns") if isinstance(run.get("turns"),list) else []
    if "session_state" in run:
        state=run["session_state"]
        rows=state.get("history",[]) if isinstance(state,dict) else []
    total=len(turns)
    audited=[]
    known=0; hits=0; no_hits=0; visible_chars=0; no_relevance=0
    for i, t in enumerate(turns):
        row=rows[i] if i<len(rows) and isinstance(rows[i],dict) else {}
        selection=row.get("memory_selection")
        if not isinstance(selection,dict):
            audited.append({"round":t["round"],"status":"unknown","retrieved_groups":None,"visible_chars":None,"reason":"未导出逐轮记忆选择诊断"})
            continue
        raw=selection.get("selected_groups")
        access=selection.get("parameter_memory_access_trace")
        if not isinstance(access,dict):access={}
        groups=access.get("visible_groups",raw)
        chars=access.get("visible_chars")
        if type(groups)!=int or groups<0:
            audited.append({"round":t["round"],"status":"unknown","retrieved_groups":None,"visible_chars":None,"reason":"记忆组数字段不可用"})
            continue
        known+=1
        if groups: hits+=1
        else:no_hits+=1
        if type(chars)==int and chars>=0: visible_chars+=chars
        status="memory_selected" if groups else "no_memory_selected"
        if selection.get("selection_status")=="no_relevant_memory":no_relevance+=1
        audited.append({"round":t["round"],"status":status,"retrieved_groups":groups,
                        "visible_chars":chars if type(chars)==int and chars>=0 else None,
                        "reason":str(selection.get("selection_status") or "")[:100]})
    diag=run.get("runtime_diagnostics") if isinstance(run.get("runtime_diagnostics"),dict) else {}
    meta={
      "version":"benchmark_runtime_audit.v1",
      "interviewer_policy":mode,
      "turns_recorded":total,
      "turns_with_selection_diagnostics":known,
      "turns_with_memory":hits if known else None,
      "turns_without_memory":no_hits if known else None,
      "memory_selection_rate":round(hits/known,6) if known else None,
      "no_relevant_memory_reports":no_relevance if known else None,
      "visible_memory_characters_total":visible_chars if known and all(
          r["visible_chars"] is not None for r in audited if r["status"]!="unknown") else None,
      "complete_transcript":bool(total) and all(t["question"] is not None and t["answer"] is not None for t in turns),
      "interviewer_followup_metric_applicable":mode not in ("planner","baseline"),
      "runtime_completed":diag.get("completed") if type(diag.get("completed"))==bool else None,
      "runtime_stopped_early":diag.get("stopped_early") if type(diag.get("stopped_early"))==bool else None,
      "per_turn":audited,
      "notes":[
        "记忆选择率只代表检索器找到并选出资料，不代表事实准确、回答有据或受访者真正记住。",
        "缺少问答正文时，不能对自然度、相关性、事实真实性和回答一致性打分。",
      ],
    }
    if mode in ("planner","baseline"):
        meta["notes"].append("固定问题模式不适合将动态追问率作为采访者适应性成绩。")
    if known<total:
        meta["notes"].append("部分轮次缺失记忆诊断，比例分母仅包含可检查轮次。")
    return meta


def evaluate(run, guide=None, annotation=None, benchmark_config=None):
    try:
        applied_config = validate_config(benchmark_config)
    except BenchmarkConfigError as exc:
        raise BenchmarkError(str(exc)) from exc
    active_rules = {m["id"]: m for m in applied_config["metrics"] if m["enabled"]}
    def params(metric_id):
        return active_rules.get(metric_id, {}).get("parameters", {})
    if not isinstance(run,dict):raise BenchmarkError("run JSON 必须是对象")
    turns,run_type=_extract_turns(run)
    pinned=run.get("guide_sha256")
    if pinned and guide is not None and pinned!=digest(guide):
        raise BenchmarkError("运行记录的 Guide 指纹与提供的大纲不匹配，请使用访谈当时的 Guide JSON")
    subtopic_ids=_guide_subtopics(guide)
    annotation=validate_annotation(annotation,turns,subtopic_ids)
    n=len(turns)
    complete_text=bool(turns) and all(t["question"] is not None and t["answer"] is not None for t in turns)
    questions_ready=bool(turns) and all(t["question"] is not None for t in turns)
    answers_ready=bool(turns) and all(t["answer"] is not None for t in turns)
    diagnosis=run.get("runtime_diagnostics") if isinstance(run.get("runtime_diagnostics"),dict) else {}
    summary=run.get("summary") if isinstance(run.get("summary"),dict) else {}
    turn_limit=diagnosis.get("turns_requested", run.get("turn_limit", summary.get("turn_limit")))
    if type(turn_limit)!=int or turn_limit<=0:turn_limit=None
    metrics={
      "turns_recorded":measure(n,"turns",method="count_turn_objects"),
      "interview_completion_ratio": measure(round(min(n/turn_limit,1),6),"ratio",method="recorded_turns_div_requested_turns") if turn_limit else missing("ratio","未记录计划轮数"),
      "interviewer_question_characters":measure(sum(len(t["question"].strip()) for t in turns),"characters",method="literal_string_length") if questions_ready else missing("characters","没有完整采访问题正文"),
      "participant_response_characters":measure(sum(len(t["answer"].strip()) for t in turns),"characters",method="literal_string_length") if answers_ready else missing("characters","没有完整受访者回答正文"),
      "exact_duplicate_question_count":missing("questions","没有完整采访问题正文"),
      "exact_duplicate_question_rate":missing("ratio","没有完整采访问题正文"),
      "distinct_model_reference_ids":missing("IDs","没有每轮完整的引用 ID 列表"),
      "model_reference_mentions":missing("mentions","没有每轮完整的引用 ID 列表"),
      "interruptions":missing("events","未导出 runtime_diagnostics.interruptions"),
      "transient_retries":missing("retries","未导出 runtime_diagnostics.transient_retries"),
      "spoken_truth":missing("percent","需要已关联的原始 Record；常规机评会自动计算，独立离线 Foundation 评价为 N/A"),
    }
    if questions_ready:
        seen=set(); repeats_by_mode={}
        for mode in set(params(m).get("normalization", "nfkc_strip_punctuation")
                        for m in ("exact_duplicate_question_count", "exact_duplicate_question_rate")):
            seen=set(); repeats=0
            for t in turns:
                norm=(normalized_text(t["question"]) if mode=="nfkc_strip_punctuation"
                      else t["question"].strip())
                if norm in seen:repeats+=1
                seen.add(norm)
            repeats_by_mode[mode]=repeats
        count_mode=params("exact_duplicate_question_count").get("normalization", "nfkc_strip_punctuation")
        rate_mode=params("exact_duplicate_question_rate").get("normalization", "nfkc_strip_punctuation")
        metrics["exact_duplicate_question_count"]=measure(repeats_by_mode[count_mode],"questions",
            method="normalized_exact_repeat:"+count_mode,note="不检测语义重复")
        metrics["exact_duplicate_question_rate"]=measure(round(repeats_by_mode[rate_mode]/n,6),"ratio",
            method="normalized_exact_repeat_ratio:"+rate_mode,note="不检测语义重复")
    if turns and all(t["reference_ids"] is not None for t in turns):
        ids=[str(x) for t in turns for x in t["reference_ids"] if isinstance(x,str)]
        metrics["distinct_model_reference_ids"]=measure(len(set(ids)),"IDs",method="unique_model_reported_reference_ids",note="模型自报引用，不能推断事实真实性")
        metrics["model_reference_mentions"]=measure(len(ids),"mentions",method="count_model_reported_reference_ids",note="包括重复引用；不代表新信息")
    if isinstance(diagnosis.get("interruptions"),list):
        metrics["interruptions"]=measure(len(diagnosis["interruptions"]),"events",method="runtime_diagnostics")
    if type(diagnosis.get("transient_retries"))==int:
        metrics["transient_retries"]=measure(diagnosis["transient_retries"],"retries",method="runtime_diagnostics")
    # No automatic inference of guide coverage, relevant response, interviewer
    # quality, or participant experience. Requires supplied full transcripts and
    # complete manual coding to avoid misleading zero scores.
    assessed={
      "guide_coverage":missing("ratio","需要 Guide + 完整受访者回答 + 每轮主题人工标注"),
      "relevant_response_volume_characters":missing("weighted_characters","需要逐轮研究问题相关性标注（0–2）和完整回答"),
      "novel_response_count":missing("responses","需要相关性=2且未覆盖任何大纲子主题的人工标注"),
      "leading_question_count":missing("utterances","需要逐轮人工判定"),
      "unclear_question_count":missing("utterances","需要逐轮人工判定"),
      "support_rapport_count":missing("utterances","需要逐轮人工判定"),
      "redundant_question_count_manual":missing("questions","需要逐轮人工判断是否重复索取已回答信息、且无新追问目的"),
      "redundant_question_rate_manual":missing("ratio","需要逐轮人工判断问题是否构成无效重复"),
      "coherence_manual":missing("1–4","需要人工对整体访谈评分"),
      "adaptiveness_manual":missing("1–4","需要人工对整体访谈评分"),
      "overall_experience_self_report":missing("1–5","缺少真人受访者自评"),
      "comfort_self_report":missing("1–4","缺少真人受访者自评"),
    }
    rows=annotation.get("turn_annotations",[]) if annotation is not None else []
    by_round={row["round"]:row for row in rows}
    full_rows=(n>0 and len(rows)==n)
    relevance_complete=answers_ready and full_rows and all(
        by_round[t["round"]].get("research_relevance") is not None for t in turns)
    coverage_complete=answers_ready and full_rows and bool(subtopic_ids) and all(
        by_round[t["round"]].get("subtopics_reviewed") is True for t in turns)
    if relevance_complete:
        weights=params("relevant_response_volume_characters").get("relevance_weights",[0,1,2])
        assessed["relevant_response_volume_characters"]=measure(
            sum(len(t["answer"].strip())*weights[by_round[t["round"]]["research_relevance"]] for t in turns),
            "weighted_characters",method="sum(response_characters_times_configured_relevance_weights)",
            note="人工相关度 0–2 的权重由 Benchmark Config 决定；不是独立事实数量")
    for event, metric in (("leading","leading_question_count"),("unclear","unclear_question_count"),
                          ("support_rapport","support_rapport_count")):
        if questions_ready and full_rows and all(by_round[t["round"]].get(event) is not None for t in turns):
            assessed[metric]=measure(sum(1 for t in turns if by_round[t["round"]][event]),
                "utterances",method="count_manual_binary_annotations",
                note="人工编码，不是自动 LLM Judge")
    redundant_complete=(questions_ready and full_rows and all(
        by_round[t["round"]].get("redundant") is not None for t in turns))
    if redundant_complete:
        redundant_count=sum(1 for t in turns if by_round[t["round"]]["redundant"] is True)
        assessed["redundant_question_count_manual"]=measure(
            redundant_count,"questions",method="count_human_judged_redundancy",
            note="人工判断重复索取已回答内容、且没有新的澄清或深挖目的；不同于精确文本重复")
        assessed["redundant_question_rate_manual"]=measure(
            round(redundant_count/n,6),"ratio",method="human_judged_redundancy_div_questions",
            note="全部问题逐轮人工审核后计算；不通过词面相似度或 LLM 自行推断")
    if coverage_complete:
        hit={tag for a in rows for tag in a.get("covered_subtopic_ids",[])}
        assessed["guide_coverage"]=measure(round(len(hit)/len(subtopic_ids),6),"ratio",
            method="manual_response_topic_tags_div_all_guide_subtopics",
            note="只有受访者回答直接涉及的子主题才计入，不能按提问或模型自报覆盖")
        if relevance_complete:
            threshold=params("novel_response_count").get("minimum_relevance",2)
            assessed["novel_response_count"]=measure(sum(1 for t in turns
                if by_round[t["round"]]["research_relevance"]>=threshold and
                   not by_round[t["round"]].get("covered_subtopic_ids")),"responses",
                method=f"manual_relevance_ge_{threshold}_and_zero_guide_subtopics")
    if annotation and complete_text:
        overall=annotation.get("overall") or {}
        for field,key in (("coherence","coherence_manual"),("adaptiveness","adaptiveness_manual")):
            if overall.get(field) is not None:
                assessed[key]=measure(overall[field],"1–4",method="single_interview_manual_rating",
                    note="v0.1 人工整场评分；不等于论文十轮片段平均的 LLM Judge")
        experience=annotation.get("participant_experience") or {}
        if experience.get("source")=="human_self_report":
            for source_field,target,unit in (("overall_experience","overall_experience_self_report","1–5"),
                                             ("comfort_level","comfort_self_report","1–4")):
                if experience.get(source_field) is not None:
                    assessed[target]=measure(experience[source_field],unit,"self_reported",method="human_participant_self_report")
    metrics = {m["id"]:metrics[m["id"]] for m in applied_config["metrics"]
               if m["enabled"] and m["group"]=="objective"}
    assessed = {m["id"]:assessed[m["id"]] for m in applied_config["metrics"]
                if m["enabled"] and m["group"]=="rubric"}
    evaluation={
      "benchmark_config_schema":applied_config["schema"],
      "benchmark_config_sha256":fingerprint(applied_config),
      "benchmark_config_snapshot":applied_config,
      "benchmark_config_name":applied_config["name"],
      "enabled_metric_count":len(metrics)+len(assessed),
      "foundation_version":BENCHMARK_VERSION,
      "rubric_version":RUBRIC_VERSION,
      "report_kind":"descriptive_foundation_not_validated_composite_score",
      "scenario_id":run.get("scenario_id",run.get("source_scenario",{}).get("scenario_id") if isinstance(run.get("source_scenario"),dict) else None),
      "source_type":run_type,
      "input_sha256":digest(run),
      "guide_sha256":digest(guide) if guide else None,
      "reviewer_id":annotation.get("reviewer_id") if annotation else None,
      "input_status":{"turn_count":n,"complete_transcript":complete_text,"guide_present":guide is not None,
                      "subtopic_count":len(subtopic_ids) if subtopic_ids is not None else None,
                      "complete_manual_turn_coding":bool(relevance_complete and coverage_complete and
                         questions_ready and all(assessed[name]["status"]=="computed" for name in
                         ("leading_question_count","unclear_question_count","support_rapport_count",
                          "redundant_question_count_manual","redundant_question_rate_manual") if name in assessed))},
      "objective_metrics":metrics,
      "rubric_metrics":assessed,
      "runtime_audit":runtime_audit(run,turns),
      "annotation_progress":{"annotation_rows":len(rows),"total_turns":n,
            "relevance_complete":bool(relevance_complete),"coverage_complete":bool(coverage_complete),"redundancy_complete":bool(redundant_complete)},
      "warnings":[
        "无综合评分：未验证维度权重或人工—模型评分一致性。",
        "模型引用编号不等于已核实事实，旧版 rubric 词面启发式不计入正式量表。",
      ],
    }
    if not complete_text:
        evaluation["warnings"].append("只有受限元数据或不完整正文；所有依赖回答内容的指标均显示不可评价。")
    if not guide:
        evaluation["warnings"].append("缺少关联大纲；无法评价主题覆盖率。")
    return evaluation
