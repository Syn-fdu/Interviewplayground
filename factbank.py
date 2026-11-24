"""FactBank v1: classify permitted interview statements as remembered events
or reality-at-interview states. No new autobiographical facts are invented here.

"Reality" means habits, roles and ongoing opinions AT THE TIME OF THE SOURCE
INTERVIEW, not verified real-time facts about the actual person in 2026.
"""
from __future__ import annotations
import hashlib
import json
import re
from agent_memory_export import eligible_source_facts

SCHEMA="factbank_classification_v1"
# Prioritize events when a concrete incident is described. Common English past
# participles ("used") alone are NOT enough: "I used to" is historical but
# "How did you use" may ask about a regular workflow in a past interview.
EVENT=re.compile(
    r"第一次|最早|曾经|过去|以前|当时|那次|有一次|上次|去年|之前|"
    r"大学期间|某次|某个项目|一个项目里|\d{4}年|"
    r"\b(?:first time|when did you start|how long ago|last time|"
    r"last year|last task|last project|the task i did|back when|once when|"
    r"in college|at university|"
    r"before (?:i|we) started|i remember when|on a project)\b",
    re.I)
PRESENT=re.compile(
    r"目前|现在|平时|日常|通常|一般|经常|每周|每天|习惯|主要|"
    r"倾向|我觉得|我认为|我喜欢|我使用|我会用|"
    r"\b(?:currently|now|usually|typically|normally|often|"
    r"sometimes|regularly|every day|in general|"
    r"i use|i mainly|i prefer|i think|i feel|we use|"
    r"for now|my job|i work)\b",
    re.I)
QUESTION_HISTORIC=re.compile(
    r"首次|最早|第一次|回忆|经历|过去|以前|当时|某一次|"
    r"什么时候开始|发生了什么|最近一次|举个具体例子|"
    r"\b(?:when did|first use|first start|past experience|"
    r"last time|specific example|tell me about a time)\b",re.I)
QUESTION_REALITY=re.compile(
    r"日常|现在|目前|平时|习惯|通常|一般|主要|经常|频率|"
    r"哪种工具|哪些工具|怎样看待|怎么看|"
    r"\b(?:currently|now|nowadays|usually|typically|"
    r"how often|what tools|in your work|your daily)\b",re.I)

def classify(fact):
    answer=str(fact.get("text") or "")
    question=str(fact.get("source_question") or "")
    # A question can clarify a fragment but cannot manufacture a date or
    # claim. Mixed/ambiguous text stays conservative and explicitly flagged.
    history=bool(QUESTION_HISTORIC.search(question))
    ongoing=bool(QUESTION_REALITY.search(question))
    explicit_past=bool(EVENT.search(answer))
    # Most transcripts split one event over several tiny "Mhm" follow-up
    # turns. Event-specific past actions must not become ongoing habits merely
    # because their source question lacks temporal wording.
    past_action=bool(re.search(
        r"\b(?:i|we) (?:had to|went|opened|checked|tried|realized|realised|"
        r"decided|started|finished|implemented|added|removed|built|made|"
        r"was supposed to|were supposed to)\b|"
        r"\b(?:then i|then we|after that|so i had to|when i was)\b",
        answer,re.I))
    if past_action:
        explicit_past=True
    explicit_present=bool(PRESENT.search(answer))
    if history and not ongoing:return "memory","historical_source_question",0.92
    if explicit_past and not explicit_present:return "memory","respondent_event",0.90
    if explicit_present and not explicit_past:return "reality","respondent_state_or_habit",0.9
    if ongoing and not history:return "reality","ongoing_source_question",0.84
    if explicit_past and explicit_present:
        if past_action:
            return "memory","retrospective_action_with_current_context",0.75
        return "memory","mixed_temporality_review",0.55
    return "reality","unspecified_time_review",0.50

def classified_facts(bundle):
    results=[]
    for f in eligible_source_facts(bundle):
        category,reason,confidence=classify(f)
        results.append({**f,"fact_category":category,
                        "classification_reason":reason,
                        "classification_confidence":confidence,
                        "time_reference":"original_interview",
                        "provenance_status":"source_derived",
                        "classification_review_required":confidence<.8})
    return results

def summary(bundle):
    facts=classified_facts(bundle)
    n_memory=sum(f["fact_category"]=="memory" for f in facts)
    n_reality=len(facts)-n_memory
    return {"schema":SCHEMA,"total":len(facts),
            "memory_count":n_memory,"reality_count":n_reality,
            "review_count":sum(f["classification_review_required"] for f in facts),
            "original_source_only":True,"facts":facts}

def preparation_view(bundle):
    d=summary(bundle)
    return {k:v for k,v in d.items() if k!="facts"}

def question_target(question,history=None):
    """Interpret demanded temporal frame, not whether to skip retrieval."""
    from question_intent import focus_question
    q=focus_question(question)
    past=bool(QUESTION_HISTORIC.search(q))
    present=bool(QUESTION_REALITY.search(q))
    if past and not present:return "memory"
    if present and not past:return "reality"
    # "哪些类型的任务" asks for regular activity, not a dated event.
    if re.search(r"哪些.{0,16}(?:任务|工作|类型|用途|工具)|"
                 r"主要.{0,16}(?:利用|使用|协助|做)|"
                 r"一般.{0,10}(?:如何|怎么)|"
                 r"how (?:do|does) .* usually",q,re.I):
        return "reality"
    if re.search(r"具体.{0,14}(?:案例|事件|经历|哪次)|"
                 r"那次|当年|那时|以前|回想",q,re.I):
        return "memory"
    return "mixed"
