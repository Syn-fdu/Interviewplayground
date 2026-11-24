"""Interview Guide Schema v1.0 — public, source-preserving, reviewable.

This is a semi-structured qualitative guide, NOT a deterministic question script,
NOT an automatic interview-plan controller, and NOT a source of respondent facts.
Normalization never upgrades original provenance or pretends completion has been
observed; it proposes only explicit, review-needed design fields.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

GUIDE_SCHEMA_VERSION = "1.0"
GUIDE_METHOD = "semi_structured_topic_subtopic_with_probe_intents"
VALID_INTENTS = ("explore", "clarify", "concrete_example", "reason", "consequence", "contrast", "transition")
VALID_SOURCE_KINDS = (
    "original_guide_provided", "source_annotation_plus_transcript",
    "reconstructed_from_transcript", "synthetic_demo", "unknown",
)


def _text(value):
    return str(value or "").strip()


def _hash_obj(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def standardize_guide(source: dict) -> dict:
    """Idempotently enrich a legacy V2 guide; preserves source and unrecognized keys.

    Does not replace source labels with guessed intent, nor dynamically mark a
    subtopic covered. Fields proposed by standardization are clearly labelled.
    """
    if not isinstance(source, dict):
        raise ValueError("Interview Guide 必须是对象")
    guide = copy.deepcopy(source)
    guide["guide_schema_version"] = GUIDE_SCHEMA_VERSION
    guide["guide_paradigm"] = GUIDE_METHOD
    guide.setdefault("schema_version", "2.0")  # legacy V2 context is retained
    guide.setdefault("review_status", "needs_review")
    guide.setdefault("evaluation_policy", {
        "coverage_unit": "subtopic", "evidence_role": "respondent_answer_only",
        "review_required": True, "no_automatic_coverage_claim": True,
        "free_emergent_topics_allowed": True,
    })
    guide.setdefault("transition_policy", {
        "method": "reviewable_guidance_not_automatic_agent_override",
        "description": "有具体信息后可转到下一个子主题；遇到新线索或需澄清时可继续深入。",
        "decision_owner": "interviewer_or_explicit_experiment_planner",
    })
    rqs = guide.get("research_questions")
    if isinstance(rqs, list):
        for n, rq in enumerate(rqs, 1):
            if isinstance(rq, dict):
                rq.setdefault("id", "RQ%d" % n)
                rq.setdefault("review_status", "requires_human_review")
    topics = guide.get("topics")
    if not isinstance(topics, list):
        return guide
    for ti, topic in enumerate(topics, 1):
        if not isinstance(topic, dict):
            continue
        topic.setdefault("topic_id", "T%02d" % ti)
        topic.setdefault("guide_role", "broad_topic")
        topic.setdefault("linked_research_question_ids", [])
        topic.setdefault("review_status", "requires_human_review")
        subs = topic.get("subtopics")
        if not isinstance(subs, list):
            continue
        for si, sub in enumerate(subs, 1):
            if not isinstance(sub, dict):
                continue
            sub.setdefault("subtopic_id", f"{topic['topic_id']}-S{si:02d}")
            sub.setdefault("probe_intents", [
                {"intent": "concrete_example", "purpose": "获取与主题有关的具体经历"},
                {"intent": "reason", "purpose": "了解原因、解释或意义"},
                {"intent": "clarify", "purpose": "澄清尚不明确的回答"},
            ])
            sub.setdefault("completion_signals", [
                {"signal": "substantive_response", "meaning": "受访者的回答直接涉及这一子主题"},
                {"signal": "specific_detail", "meaning": "回答包含一个相关经历、理由或具体细节"},
            ])
            sub.setdefault("completion_policy", {
                "decision": "manual_or_validated_judge",
                "no_information_gain_hint": "连续追问无实质新增信息时考虑切换，但不自动强制切换",
                "completion_is_not_questions_asked": True,
            })
            sub.setdefault("linked_research_question_ids", [])
            sub.setdefault("review_status", "requires_human_review")
            sub.setdefault("source_kind", topic.get("source_kind", guide.get("source_kind", "unknown")))
            sub.setdefault("information_gap", "needs_interview_discovery")
    return guide


def validate_guide(source, *, normalize=True) -> dict:
    """Return structural errors, review warnings, counts and canonical snapshot.

    Severity distinction: errors prevent saving; review warnings do NOT pretend
    that a reconstructed guide has been professionally approved.
    """
    errors, warnings = [], []
    if not isinstance(source, dict):
        return {"ok": False, "errors": [{"path": "$", "message": "必须是 JSON 对象"}],
                "warnings": [], "normalized": None, "stats": {}}
    g = standardize_guide(source) if normalize else copy.deepcopy(source)
    goal = g.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        errors.append({"path": "goal", "message": "请填写非空研究目标"})
    elif len(goal) > 1200:
        errors.append({"path": "goal", "message": "研究目标过长"})
    rqs = g.get("research_questions")
    if not isinstance(rqs, list) or not 1 <= len(rqs) <= 12:
        errors.append({"path": "research_questions", "message": "需要 1–12 项研究问题"})
        rqs = []
    rqids = set()
    for i, rq in enumerate(rqs):
        p = f"research_questions[{i}]"
        if not isinstance(rq, dict) or not _text(rq.get("text")):
            errors.append({"path": p, "message": "研究问题须包含 text"})
            continue
        rid = rq.get("id")
        if not isinstance(rid, str) or not re.fullmatch(r"RQ[A-Za-z0-9_-]{1,22}", rid):
            errors.append({"path": p + ".id", "message": "研究问题 ID 格式应为 RQ..."})
        elif rid in rqids:
            errors.append({"path": p + ".id", "message": "研究问题 ID 重复"})
        else:
            rqids.add(rid)
    topics = g.get("topics")
    if not isinstance(topics, list) or not 1 <= len(topics) <= 60:
        errors.append({"path": "topics", "message": "需要 1–60 个主题"})
        topics = []
    tids, sids, nsubs = set(), set(), 0
    for ti, t in enumerate(topics):
        p = f"topics[{ti}]"
        if not isinstance(t, dict):
            errors.append({"path": p, "message": "主题必须是对象"})
            continue
        tid = t.get("topic_id")
        if not isinstance(tid, str) or not re.fullmatch(r"T[A-Za-z0-9_-]{1,22}", tid):
            errors.append({"path": p + ".topic_id", "message": "主题 ID 格式应为 T..."})
        elif tid in tids:
            errors.append({"path": p + ".topic_id", "message": "主题 ID 重复"})
        else:
            tids.add(tid)
        if not _text(t.get("title")):
            errors.append({"path": p + ".title", "message": "主题名称不能为空"})
        links = t.get("linked_research_question_ids", [])
        if not isinstance(links, list) or any(x not in rqids for x in links):
            errors.append({"path": p + ".linked_research_question_ids", "message": "RQ 引用不存在"})
        elif not links:
            warnings.append({"path": p, "message": "尚未人工确认与研究问题的映射"})
        subs = t.get("subtopics")
        if not isinstance(subs, list) or not 1 <= len(subs) <= 30:
            errors.append({"path": p + ".subtopics", "message": "每主题需要 1–30 个子主题"})
            continue
        nsubs += len(subs)
        for si, st in enumerate(subs):
            sp = f"{p}.subtopics[{si}]"
            if not isinstance(st, dict):
                errors.append({"path": sp, "message": "子主题必须是对象"})
                continue
            sid = st.get("subtopic_id")
            if not isinstance(sid, str) or not isinstance(tid, str) or not sid.startswith(tid + "-S"):
                errors.append({"path": sp + ".subtopic_id", "message": "子主题 ID 必须属于父主题"})
            elif sid in sids:
                errors.append({"path": sp + ".subtopic_id", "message": "子主题 ID 重复"})
            else:
                sids.add(sid)
            if not _text(st.get("label")):
                errors.append({"path": sp + ".label", "message": "子主题名称不能为空"})
            probes = st.get("suggested_probes", [])
            if not isinstance(probes, list) or not all(isinstance(q, str) and q.strip() for q in probes):
                errors.append({"path": sp + ".suggested_probes", "message": "建议追问必须是字符串列表"})
            intents = st.get("probe_intents")
            if not isinstance(intents, list) or not intents:
                errors.append({"path": sp + ".probe_intents", "message": "至少需要一个追问目的"})
            else:
                for k, item in enumerate(intents):
                    if not isinstance(item, dict) or item.get("intent") not in VALID_INTENTS:
                        errors.append({"path": sp + f".probe_intents[{k}]", "message": "无效追问目的"})
            signals = st.get("completion_signals")
            if not isinstance(signals, list) or not signals or not all(
                isinstance(x, dict) and _text(x.get("signal")) for x in signals):
                errors.append({"path": sp + ".completion_signals", "message": "至少需要一个完成信号"})
            links = st.get("linked_research_question_ids", [])
            if not isinstance(links, list) or any(x not in rqids for x in links):
                errors.append({"path": sp + ".linked_research_question_ids", "message": "RQ 引用不存在"})
            if not st.get("source_interviewer_turn_indices") and g.get("source_kind") != "synthetic_demo":
                warnings.append({"path": sp, "message": "没有来源提问轮次，需人工核查主题来源"})
            if st.get("review_status", "requires_human_review") != "approved":
                warnings.append({"path": sp, "message": "主题设计尚待研究者审核"})
    if g.get("source_kind") != "original_guide_provided":
        warnings.append({"path": "source_kind", "message": "当前不是已证明的原始访谈大纲；不得改写来源标签"})
    if g.get("review_status") not in ("approved", "human_reviewed"):
        warnings.append({"path": "review_status", "message": "整个采访大纲尚未完成审核"})
    g["guide_schema_version"] = GUIDE_SCHEMA_VERSION
    stats = {"research_questions": len(rqs), "topics": len(topics), "subtopics": nsubs,
             "source_kind": g.get("source_kind", "unknown"), "review_status": g.get("review_status", "unknown")}
    return {"ok": not errors, "schema_version": GUIDE_SCHEMA_VERSION,
            "errors": errors, "warnings": warnings, "stats": stats,
            "normalized": g, "fingerprint_sha256": _hash_obj(g)}


def with_validation_snapshot(guide):
    """Return a standardized guide with a fresh structural audit snapshot."""
    g=standardize_guide(guide)
    assessment=validate_guide(g,normalize=False)
    g["guide_validation"]={"ok":assessment["ok"],"stats":assessment["stats"],
        "errors":assessment["errors"],"warnings":assessment["warnings"][:40]}
    return g
