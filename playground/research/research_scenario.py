"""Evidence-auditable, **non-oracle** research guide and respondent memory builders.

The input is a V2 triple: outline, transcript and respondent profile. A source
annotation is not a provided original interview guide. Transcript-derived
questions/claims always remain clearly marked as reconstructions. All output
is local-only research preview and requires independent consent/privacy review.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import hashlib
import re

from playground.research.scenario_builder import scrub, tokens
from playground.research.guide_schema import with_validation_snapshot

TRAIT_NAMES = ("knowledge", "understanding", "reflexivity", "memory", "verbosity", "disclosure")
TRAIT_DESCRIPTIONS = {
    "knowledge": "对主题知识的熟悉程度",
    "understanding": "理解抽象或复杂问题的能力",
    "reflexivity": "反思并解释自己经历的倾向",
    "memory": "回忆具体经历细节的容易程度",
    "verbosity": "单次回答详细程度",
    "disclosure": "在被适当询问时透露个人信息的意愿",
}

STOP = set("""the this that these those with where which why what when how whose would could should can was were from into about over after before during yourself your my mine their his her them then there thing things being been based interview interviewer respondent please tell something anything something whether different part parts do did does is are i we you have has had of or for an to a and in on at by it as if but our they also very such more much one two three make made given like know think talk say something question questions""".split())


def _safe_text(value, name=None, size=180):
    return scrub(value, name, size) if value is not None else ""


def _list_strings(value, name, limit=16, size=260):
    if isinstance(value, str):
        value = [value]
    return [_safe_text(v, name, size) for v in (value or [])[:limit]
            if isinstance(v, (str, int, float)) and str(v).strip()] if isinstance(value, list) else []


def _keywords(text, n=4):
    toks = [t for t in tokens(text) if len(t) >= 3 and t not in STOP and not t.isnumeric()]
    counts = Counter(toks)
    # Prefer first few distinctive terms rather than assigning hidden answer facts.
    seen = []
    for t in toks:
        if counts[t] and t not in seen:
            seen.append(t)
    return seen[:n]


def _text_from_source_row(x, name, size=360):
    if isinstance(x, str):
        return _safe_text(x, name, size)
    if isinstance(x, dict):
        for key in ("text", "verbatim", "content", "summary", "description", "value"):
            if isinstance(x.get(key), str):
                return _safe_text(x[key], name, size)
    return ""


def _source_questions(outline, record, name):
    """Only public source outline questions; never reconstruct guide from transcript."""
    found = []
    seen = set()
    for topic in outline.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        for question in topic.get("questions") or []:
            text = question.get("text") if isinstance(question, dict) else question
            if not isinstance(text, str):
                continue
            text = _safe_text(text, name, 300)
            if text and text not in seen:
                seen.add(text)
                found.append({"text": text, "turn_index": None, "origin": "original_guide"})
    return found[:180]


def make_interview_guide(outline, record, topic, name):
    """Create guide with explicit provenance. No answers, fact cards, or hidden labels.

    Reconstructed subtopics use question-language clues. They are proposals,
    not a claim to have recovered the original interviewer's intentions.
    """
    source_qs = _source_questions(outline, record, name)
    original = bool(outline.get("topics"))
    genuine_goals = _list_strings(outline.get("goal"), name, 5, 300)
    genuine_targets = _list_strings(
        [q.get("text", "") if isinstance(q, dict) else q
         for t in outline.get("topics") or []
         if isinstance(t, dict) for q in t.get("questions") or []], name, 12, 240)
    annotations = _list_strings(outline.get("objective_lines_extracted_from_annotation"), name, 6, 260)
    # Source annotations are metadata / research objectives; unlike gold answer keys,
    # source_info_items_dict may contain participant facts and MUST remain private.
    source_kind = ("original_guide_provided" if original and (genuine_goals or genuine_targets)
                   else "source_annotation_plus_transcript" if annotations
                   else "reconstructed_from_transcript")
    source_label = {
        "original_guide_provided": "原研究提供的采访大纲（有明确来源标记）",
        "source_annotation_plus_transcript": "来源研究目标注释 + 逐字稿重建（非原始大纲）",
        "reconstructed_from_transcript": "从逐字稿重建（非原始大纲）",
    }[source_kind]
    main_goal = genuine_goals[0] if original and genuine_goals else (
        annotations[0] if annotations else f"围绕{topic}了解受访者的经历、观点和具体例子")
    rq_raw = genuine_goals + genuine_targets if original and (genuine_goals or genuine_targets) else annotations
    if not rq_raw:
        rq_raw = [f"关于{topic}，受访者有哪些相关经历和观点？"]
    rqs = [{"id": f"RQ{i+1}", "text": s,
            "source_kind": "original_guide" if original and (genuine_goals or genuine_targets)
                else "source_annotation" if annotations else "transcript_reconstruction",
            "review_status": "requires_human_review"} for i,s in enumerate(rq_raw[:8])]

    # Themes are constructed from ASKED questions, never respondent answers.
    # Repeated/substantially overlapping question lexical topics are merged.
    clusters = []
    for q in source_qs:
        words = _keywords(q["text"], 4)
        if not words:
            continue
        qs = set(words)
        match = next((g for g in clusters if len(qs & g["words"]) >= 2), None)
        if match is None and len(clusters) >= 5:
            match = max(clusters, key=lambda g: len(qs & g["words"]))
        if match is None:
            match = {"words": set(), "questions": [], "terms": []}
            clusters.append(match)
        match["words"].update(qs)
        match["questions"].append(q)
        match["terms"].extend(words)
    if not clusters:
        clusters = [{"words": set(), "questions": [], "terms": _keywords(topic,3)}]

    topic_items = []
    for i, group in enumerate(clusters[:6], 1):
        tid = f"T{i:02d}"
        counts = Counter(group["terms"])
        terms = [x for x,_ in counts.most_common(4)]
        label = " / ".join(terms[:3]) if terms else _safe_text(topic,name,60)
        subtopics = []
        for j,q in enumerate(group["questions"][:6], 1):
            sid = f"{tid}-S{j:02d}"
            focus_terms = _keywords(q["text"],3)
            focus = " / ".join(focus_terms) if focus_terms else label
            subtopics.append({
                "subtopic_id": sid, "label": focus,
                "source_kind": "transcript_reconstruction",
                "source_interviewer_turn_indices": [q["turn_index"]] if q["turn_index"] is not None else [],
                "suggested_probes": [
                    f"围绕「{focus}」，您能描述一个具体经历吗？",
                    "当时有哪些因素让您这样看待这件事？",
                    "有没有什么细节是我还没有问到的？",
                ],
                "information_gap": "needs_interview_discovery",
                "confidence": "heuristic_unreviewed"
            })
        if not subtopics:
            subtopics = [{"subtopic_id": tid+"-S01", "label": label,
                          "source_kind": "transcript_reconstruction", "source_interviewer_turn_indices":[],
                          "suggested_probes": ["您可以介绍一个相关的具体例子吗？"],
                          "information_gap": "needs_interview_discovery", "confidence":"heuristic_unreviewed"}]
        topic_items.append({
            "topic_id": tid,
            "title": label,
            "source_kind": "transcript_reconstruction",
            "subtopics": subtopics,
            "suggested_probes": ["请从您的经历谈起。", "有没有特别重要、尚未提到的细节？"],
            "status": "not_interviewed",
        })

    # When source objectives are explicitly available, build meaningful themes
    # from THOSE annotations rather than opaque keyword combinations. They are
    # NOT original interview guides and remain labelled as such.
    source_theme_seeds = (genuine_targets if original and genuine_targets else
                          annotations if annotations else [])
    if source_theme_seeds:
        objectives=[]
        for i,seed in enumerate(source_theme_seeds[:6],1):
            title=re.sub(r"^(?:objective|goal|question|information item)\s*#?\d*\s*[:.\-]?\s*", "",seed,flags=re.I).strip(" -.;: ")[:110]
            if not title:continue
            seed_terms=set(_keywords(title,8))
            # Source citations are collected only when an interviewer turn
            # genuinely overlaps the objective; empty citation stays review-needed.
            scored=[]
            for q in source_qs:
                common=len(seed_terms & set(_keywords(q['text'],10)))
                if common and isinstance(q.get('turn_index'),int):
                    scored.append((common,q['turn_index']))
            refs=[idx for score,idx in sorted(scored,reverse=True)[:6]]
            tid=f"T{i:02d}"
            # One original objective -> one auditable scoring target.
            # Never manufacture a second "views/reasoning" requirement solely to
            # increase the denominator of Guide coverage. Source annotations
            # remain explicitly flagged as unreviewed reconstructions.
            subtopics=[{
                'subtopic_id':tid+'-S01', 'label':title,
                'source_kind':'original_guide' if original and genuine_targets else 'source_annotation',
                'source_interviewer_turn_indices':refs,
                'suggested_probes':[f'关于「{title}」，您能介绍一个具体的过程或例子吗？',
                                    '还有哪些原因或观点值得补充？'],
                'information_gap':'needs_interview_discovery',
                'confidence':'objective_based_unreviewed'}]
            objectives.append({
                'topic_id':tid,'title':title,
                'source_kind':'original_guide' if original and genuine_targets else 'source_annotation',
                'subtopics':subtopics,'suggested_probes':['您能谈谈这方面的经历吗？'],
                'status':'not_interviewed'})
        if objectives:
            topic_items=objectives

    answered = set()
    for turn in record.get("turns") or []:
        if turn.get("role") == "respondent":
            answered.add(turn.get("turn_index"))
    # A missing adjacent respondent may reflect a transcript boundary or role
    # anomaly; report only as an audit lead, never proof that guest refused.
    gaps = []
    turns = record.get("turns") or []
    for i,t in enumerate(turns):
        if t.get("role") != "interviewer" or len(str(t.get("text") or "")) < 12:
            continue
        next_turn = turns[i+1] if i+1 < len(turns) else None
        if not next_turn or next_turn.get("role") != "respondent" or not str(next_turn.get("text") or "").strip():
            gaps.append({"type":"unverified_answer_boundary", "source_turn_index":t.get("turn_index", i),
                         "status":"review_needed", "note":"下一轮不是可识别的受访者发言；不等于受访者真实拒答。"})
        if len(gaps)>=12:
            break
    phases = [
        {"phase_id":"opening", "name":"开场与知情引导", "goal":"确认主题、营造非诱导性交流氛围", "recommended_action":"open"},
        {"phase_id":"exploration", "name":"主题探索", "goal":"按主题获得第一层经历与观点", "recommended_action":"explore"},
        {"phase_id":"deepening", "name":"深度追问", "goal":"询问具体例子、原因和矛盾之处", "recommended_action":"follow_up"},
        {"phase_id":"closing", "name":"回顾与收尾", "goal":"确认理解、询问遗漏和受访者是否愿意补充", "recommended_action":"conclude"},
    ]
    output = {
        "schema_version":"2.0", "source_kind":source_kind, "source_label":source_label,
        "original_guide_provided":bool(original and (genuine_goals or genuine_targets)),
        "research_questions":rqs, "goal":main_goal[:450],
        "topics":topic_items, "phases":phases,
        "information_gaps":gaps, "source_question_count":len(source_qs),
        "source_questions_for_local_refinement":source_qs[:60],
        "provenance_note":"此大纲可能包含从逐字稿重建的研究设计线索，不等于原采访的真实计划；需要人工复核。",
        "review_status":"unreviewed",
    }
    # Public guide only; no respondent answers enter this normalizer.
    return with_validation_snapshot(output)


def build_memory_bank(record, name, max_memories=400):
    """Build one source-grounded, quality-screened memory per useful response.

    Compact attributed information units replace raw transcript passages.
    Rejected filler/duplicate units are counted; no semantic entailment is
    claimed without an independent human review.
    """
    from playground.memory.memory_system import normalize_source_memory
    # Raw transcript is the only personal memory source.
    # Derived memories are created at Scenario preparation, not read from dataset.
    memories=[]; preceding=""; prior_index=None; total=0; seen=set()
    reasons=Counter(); source_units=0; retained_units=0
    for i, turn in enumerate(record.get("turns") or []):
        role=turn.get("role"); original=str(turn.get("text") or "").strip()
        idx=turn.get("turn_index",i)
        if role=="interviewer":
            preceding=_safe_text(original,name,360);prior_index=idx
        elif role=="respondent":
            total+=1
            if len(original)<8:
                reasons["short_source_turn"]+=1
                continue
            digest=hashlib.sha256(original.encode("utf-8")).hexdigest()
            if digest in seen:
                reasons["duplicate_source_turn"]+=1
                continue
            seen.add(digest)
            if len(memories)>=max_memories:
                reasons["memory_limit_exceeded"]+=1
                continue
            memory,issues=normalize_source_memory(
                original, preceding, name=name, turn_index=idx,
                question_index=prior_index, memory_id=f"M{len(memories)+1:04d}")
            reasons.update(issues)
            if memory is None:
                reasons["rejected_low_information_turn"]+=1
                continue
            source_units+=max(len(memory["facts"])+len(issues), 0)
            retained_units+=len(memory["facts"])
            memories.append(memory)
    return memories,{
        "source_respondent_turn_count":total,
        "stored_memory_count":len(memories),
        "stored_memory_record_count":retained_units,
        "rejected_or_skipped_unit_count":sum(reasons.values()),
        "quality_rejection_reasons":dict(reasons),
        "truncated_due_to_limit":bool(reasons["memory_limit_exceeded"]),
        "selection_method":"source_grounded_information_units_quality_screen_v0.9",
        "semantic_entailment_checked":False,
        "review_status":"auto_normalized_requires_human_review"
    }


def build_persona(profile, memories, name):
    identity={"participant_code":profile.get("participant_code"),"name_verified_by_human":False}
    if not isinstance(identity,dict):identity={}
    verified=identity.get("name_verified_by_human") is True
    allowed={}
    # Do not infer demographic attributes from unverified field values.
    if verified:
        for key in ("age","gender","education","occupation"):
            value=identity.get(key)
            if isinstance(value,(str,int,float)) and str(value).strip():
                allowed[key]=_safe_text(value,name,95)
    raw_context=[]  # Profile v3.4 is independent of transcript; no outline-derived bio.
    source_attributes=profile.get("attributes") if isinstance(profile.get("attributes"),dict) else {}
    context=[]
    if isinstance(raw_context,str):context=[_safe_text(raw_context,name,400)]
    elif isinstance(raw_context,list):
        for entry in raw_context[:3]:
            text=_text_from_source_row(entry,name,280)
            if text:context.append(text)
    behavioral_traits={key:{"value":1.0,"source":"experimental_default_not_observed",
                            "description":TRAIT_DESCRIPTIONS[key]}
                       for key in TRAIT_NAMES}
    # We do not synthesize nonexistent life events or observed personality scores.
    return {
        "role":"single_anonymized_respondent", "verified_identity_attributes":allowed,
        "source_context_unverified":context,
        "independent_profile_attributes":{
            str(k):_safe_text(str(v),name,160)
            for k,v in source_attributes.items()
            if isinstance(v,(str,int,float,bool)) and v is not None},
        "source_context_provenance":"original_metadata_not_independently_verified" if context else "not_available",
        "behavioral_traits":behavioral_traits,
        "persona_construction":"evidence_bound_no_fabricated_biography",
        "review_status":"privacy_and_identity_review_pending",
        "memory_count":len(memories),
    }


def build_evaluation_reference(outline, memory_bank, guide):
    # Gold information remains private/server-only. The presence of a memory is
    # *not* a human-validated benchmark ground truth.
    return {
        "schema_version":"2.0", "reference_status":"auto_extracted_not_gold",
        "target_information":[{
            "target_id":m["memory_id"],"evidence_source":m["source"],
            "answer_status":"source_excerpt_present_unverified",
            "human_review":"pending",
        } for m in memory_bank],
        "unresolved_source_gaps":guide.get("information_gaps",[]),
        "annotation_info_items_available":bool(outline.get("source_info_items_dict")),
        "disclosure_to_interviewer":False,
    }
