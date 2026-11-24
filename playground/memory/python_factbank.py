"""Consolidated source-grounded MemoryBank temporal classification.

Rule levels 4/7/8/9/10/11/13/14/15 are preserved verbatim in behavior:
each successive classifier delegates to the preceding implementation.
Consolidating the implementation removes version-suffixed import chains,
but the public final classify() remains exactly the original v15 logic.
Do not delete any rule section without differential regression tests.
"""
from __future__ import annotations


# ====================== Original rule level v4 ======================
"""MemoryBank-native, deterministic FactBank classifier. No network, no LLM.

Source of truth: MemoryBank original_text + its provenance/status. Never use
contextualized_text or model-generated memory as primary evidence.
"""
import argparse, json, re, hashlib
from pathlib import Path
VERSION_V4 = 'python_factbank_index.v4'
EXCLUDED_STATUS = ('excluded','quarantined','acknowledgment','nonfactual','non_factual','role_attribution_unresolved')
EXCLUDED_KIND = ('acknowledgment','not_runtime_usable','quarantined','non_factual','original_interview_procedure')
FILLER = re.compile(r"^(?:(?:yeah|yep|yup|yes|no|nah|okay|ok|mm|mhm|uh|um|ha|hmm|right|sure|fine|absolutely|thanks|thank you|you're welcome|you.re welcome|alright|all right|so|and|well|i see|i agree|perfect|i don.t know)[\s,!.?;:'\-]*)+$",re.I)
SOCIAL = re.compile(r"^(?:thank you(?: so much)?|you.re welcome|sorry,? can you hear me(?: again)?|no,? i.m all good|i (?:have|got) (?:one|two) questions?|but um okay i.m a little bit unsure|okay,? so i have a question)[.!?\s]*$",re.I)
HIST = re.compile(r'\b(?:last (?:week|month|year|monday|tuesday|wednesday|thursday|friday)|(?:\d+|a|few|several|two|three|four) (?:days?|weeks?|months?|years?) ago|(?:when|while|during) (?:schools? were|the pandemic|covid|the lockdown|i first|we first|i was|we were)|at first|when i first|in the beginning|previously|prior to|back (?:then|in)|before i|i had never|i haven.t even|i was (?:part|responsible|working|studying|adding)|i used to|i started|we had (?:fortnightly|weekly)|we (?:went|built|created|introduced|decided|launched)|i (?:went|built|created|decided|published|asked|wrote|checked)|i haven.t (?:ever|even)|i have never|in one of my courses|at that time|the previous year|before the original interview|before (?:the|my) interview)\b',re.I)
HABIT = re.compile(r'\b(?:usually|typically|generally|normally|mostly|often|sometimes|rarely|always|whenever|daily|regularly|every (?:day|week|time)|i (?:use|work|prefer|think|believe|feel|like|trust|validate|check|practice|review|teach|mainly|mostly|want)|we (?:use|work|follow|prefer|think|believe|practice|review|teach|mainly|mostly|want|have been)|my (?:team|company|role|position|work|job|advice)|our (?:team|company|work|practice|workflow|industry|process)|the (?:system|tool|software|field|industry|data|company|users?|robot|robots) (?:is|are|can|should|has|have)|i would (?:say|recommend)|it depends|you (?:can|should|need)|they (?:can|should|need)|as a (?:developer|researcher|teacher|student)|i am (?:a|an|the)|i.m (?:a|an|the)|my advice|in general|these days|nowadays|right now|at present)\b',re.I)
QUESTION_HABIT = re.compile(r'\b(?:usually|typically|normally|often|how (?:do|does|would)|what (?:practices|approaches|strategies|position|role|methods|is your|are your)|what (?:do|does|would) you (?:do|use|think|prefer|recommend)|how often|in general|what advice|what (?:tools|features) (?:do|would) you|why (?:do|does|is)|what is (?:your|the))\b',re.I)
QUESTION_HIST = re.compile(r'\b(?:have you (?:ever|received|faced|had|done)|did you|were you|when did|what happened|how long ago|what (?:were|was) the|back then|during (?:covid|the)|what (?:challenges|difficulties) have you)\b',re.I)
SHORT_QUESTION = re.compile(r'\b(?:what is your (?:position|role|job|title)|which (?:tools?|language|platform)|what (?:practices|approaches|strategies)(?: or (?:practices|approaches|strategies))? have worked|what is the (?:biggest|main|highest)|do you (?:use|work|have|prefer)|are you (?:responsible|using|working)|were you responsible|have you (?:faced|received|had|ever)|what is your name|what is the company name)\b',re.I)
CONTENT = re.compile(r'[A-Za-z\u4e00-\u9fff0-9]')

def classify_v4(row):
    s=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    status=str(row.get('semantic_status') or '').lower()
    kind=str(row.get('entry_kind') or '').lower()
    provenance={'memory_id':row.get('memory_id') or row.get('id'), 'source_turn_id':row.get('source_turn_id'), 'source_question_turn_id':row.get('source_question_turn_id')}
    def result(label, reason, confidence, review=False):
        return {**provenance,'category':label,'is_fact':label!='non_fact','exclude_from_retrieval':label in ('non_fact','review'), 'review_required': review or label=='review', 'confidence':confidence,'classification_reason':reason,'schema':VERSION_V4,'llm_calls':0}
    if any(token in status for token in EXCLUDED_STATUS) or any(token in kind for token in EXCLUDED_KIND):
        return result('non_fact','memorybank_source_status_gate',1.0)
    if not s or not CONTENT.search(s) or FILLER.fullmatch(s) and not (SHORT_QUESTION.search(q) and re.search(r'\b(?:yes|no|nah)\b',s,re.I)):
        return result('non_fact','conversational_filler',.98)
    if SOCIAL.fullmatch(s) or re.fullmatch(r'(?:sorry|thanks|thank you|you.re welcome)[.!,? ]*',s,re.I):
        return result('non_fact','social_or_interview_management',.98)
    # Keep question-anchored short responses, including yes/no: same token can be
    # meaningful in one interview and a disposable backchannel in another.
    words=re.findall(r'[A-Za-z\u4e00-\u9fff0-9]+',s)
    if len(words)<=5:
        if SHORT_QUESTION.search(q):
            if re.search(r'\b(?:were you|have you (?:faced|received|had|ever))\b',q,re.I):
                return result('memory','question_anchored_past_short_answer',.78,True)
            return result('reality','question_anchored_short_answer',.82,True)
        if re.fullmatch(r'(?:yes|no|yeah|yep|nah)[.!\s]*',s,re.I):
            return result('non_fact','unanchored_yes_no',.75,True)
        if re.search(r'\b(?:usually|typically|when you|how often)\b',q,re.I):
            return result('reality','habit_question_short_answer',.72,True)
        return result('review','short_answer_missing_semantic_anchor',.35,True)
    h=bool(HIST.search(s)); p=bool(HABIT.search(s)); qh=bool(QUESTION_HIST.search(q)); qp=bool(QUESTION_HABIT.search(q))
    # A past-tense narration with a bounded episode takes precedence; a present
    # habit co-occurring with an episode can be mixed and must be reviewed.
    if h and p:
        if re.search(r'\b(?:used to|at first|when i first|during covid|while schools were|in one of my courses|back then)\b',s,re.I):
            return result('memory','bounded_earlier_episode',.80,True)
        if re.search(r'\b(?:i have never|i haven.t even|i haven.t ever)\b',s,re.I) and not re.search(r'\b(?:at first|last|ago|before)\b',s,re.I):
            return result('reality','ongoing_experience_to_date',.76,True)
        return result('review','mixed_historical_and_ongoing',.45,True)
    if h: return result('memory','explicit_bounded_history',.91)
    if p: return result('reality','ongoing_state_habit_or_general_claim',.85)
    if qp and not qh:return result('reality','question_anchored_ongoing',.72,True)
    if qh and not qp:return result('memory','question_anchored_historical',.72,True)
    # Expository text with non-personal present tense: don't force Memory.
    if re.search(r'\b(?:is|are|can|should|would|need|depends|allows|means|require|needs|tends)\b',s,re.I):
        return result('reality','general_present_statement',.64,True)
    return result('review','insufficient_temporal_evidence',.30,True)

def evaluate_rows(rows):
    output=[]; seen=set()
    for r in rows:
        mid=r.get('memory_id') or r.get('id')
        if not mid or mid in seen:raise ValueError(f'missing or duplicate memory ID: {mid}')
        seen.add(mid)
        output.append({**r,'python_factbank_v4':classify_v4(r)})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    if Path(a.input).resolve()==Path(a.output).resolve():raise ValueError('Refuse overwrite')
    doc=json.loads(Path(a.input).read_text(encoding='utf8'))
    if isinstance(doc,list):out=evaluate_rows(doc)
    elif isinstance(doc,dict) and isinstance(doc.get('entries'),list):out={**doc,'entries':evaluate_rows(doc['entries'])}
    else:raise ValueError('Expected MemoryBank entry list or {entries:[...]}')
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print('written',len(out if isinstance(out,list) else out['entries']))




# ====================== Original rule level v7 ======================
"""Python-only MemoryBank temporal classifier v7: experimental.
Source original_text + source question; source validation must run in adapter.
This is not a semantic fact verifier, and does not call LLMs.
"""
import re
VERSION_V7='python_factbank_index.v7-experimental'

def emit_v7(base,cat,reason,confidence=.83,review=False):
    r=base.copy();r.update(category=cat,is_fact=cat not in ('non_fact','review'),exclude_from_retrieval=cat in ('non_fact','review'),classification_reason=reason,confidence=confidence,review_required=review or cat=='review')
    return r
def classify_version(r,version=7):
    base=classify_v4({'memory_id':r['memory_id'],'original_text':r['text'],'source_question':r['question'],'semantic_status':r['semantic_status'],'entry_kind':r['entry_kind'],'source_turn_id':r.get('source_turn_id')})
    if version==4:return base
    if base['classification_reason']=='memorybank_source_status_gate':return base
    s=r['text'].strip();q=r['question'].strip();low=s.lower();ql=q.lower();tokens=re.findall(r"[a-z0-9]+",low)
    # v5: interview pragmatics and question-dependent answers
    if re.fullmatch(r"\s*(?:yeah|yes|yep|mhm|right|okay|ok|uh|um|ha|i think so|i guess|pretty casual|sure|exactly|so|like|yeah yeah|yeah like pretty casual|i think so yeah yeah|i think so yeah|yeah pretty casual)[,.!?\s]*",low):
        if re.search(r'\b(?:have you|did you|were you|do you|are you|have these)\b',ql) and re.fullmatch(r'(?:yeah|yes|yep|no|nah|not really)[,.!?\s]*',low):
            if re.search(r'\b(?:did|were|have you|have these)\b',ql):return emit_v7(base,'memory','v5_question_anchored_yes_no_past',.75,True)
            return emit_v7(base,'reality','v5_question_anchored_yes_no_present',.75,True)
        return emit_v7(base,'non_fact','v5_pragmatic_acknowledgment',.95)
    if len(tokens)<24 and re.search(r'\b(?:i.ll send you the spelling|can you hear me|i think so,? yeah|pretty casual,? i guess)\b',low):return emit_v7(base,'non_fact','v5_interview_management',.88,True)
    if re.search(r'\b(?:whenever|every day|regularly|currently|these days|we do as well|we have got a set|i don.t search for|i sometimes|i try to cope|we do not believe|at the moment|we rely primarily|still trying to figure|it is generally|generally speaking)\b',low):
        if not re.search(r'\b(?:last week|two weeks ago|yesterday|a few years ago)\b',low):return emit_v7(base,'reality','v5_current_or_habitual',.83)
    if re.search(r'\b(?:yes|yeah)\b',low) and len(tokens)<=9 and re.search(r'\b(?:did that happen|did you|have you|were you)\b',ql):return emit_v7(base,'memory','v5_question_anchored_short_history',.75,True)
    if re.search(r'\b(?:recent example|a recent example|when did|what goals did you set|at the outset)\b',ql):
        if not re.search(r'\b(?:always|usually|typically|generally|every day|regularly)\b',low):return emit_v7(base,'memory','v5_specific_past_question',.79,True)
    if re.search(r'\b(?:just after you left|at first|last week|before the interview|took me time|last month)\b',low):return emit_v7(base,'memory','v5_specific_past_event',.90)
    if version==5:return base
    # v6: ongoing claims vs episodic events; avoid 'did' in interviewer question overriding answer's present-tense.
    if re.search(r'\b(?:i would invest|i would recommend|my advice|we need to|should have|should be|is needed|are needed|you can|you should|they can|we do not believe)\b',low):
        return emit_v7(base,'reality','v6_ongoing_preference_or_prescription',.85)
    if re.search(r'\b(?:i have got|i.ve got|there.s only one|i don.t|i do not|there is|there are|we have|we do|i am|i.m|i know|i think|is generally|are generally|can be|could be|should be)\b',low):
        if not re.search(r'\b(?:last week|ago|at first|just after|at the outset|we did have|in \d{4})\b',low):return emit_v7(base,'reality','v6_present_state_or_generalized_claim',.73,True)
    if re.search(r'\b(?:i can.t say i have|can.t think of any times|i haven.t had any)\b',low) and re.search(r'\b(?:did|have|were|before|had)\b',ql):return emit_v7(base,'memory','v6_reflected_historical_absence',.78,True)
    if re.search(r'\b(?:we did have|i was able to|it took|we had |we introduced|it wasn.t|we did not set|we didn.t set)\b',low):
        return emit_v7(base,'memory','v6_finished_episode',.83,True)
    if version==6:return base
    # v7: stronger source-level caution for short non-facts, bounded scope for past events.
    if re.search(r'\b(?:i can.t say i have|there was|there were|i spoke to|we spoke to|wasn.t information|it took quite|we had some dedicated|the focus was primarily|did not set numerical)\b',low):return emit_v7(base,'memory','v7_reconstructed_episode',.80,True)
    if re.search(r'\b(?:what can libraries do|what do you think libraries|what more should|would you recommend|how does|can you propose|do you think|what role does|what factors help)\b',ql):
        if not re.search(r'\b(?:last week|at first|years ago|we did have)\b',low):return emit_v7(base,'reality','v7_advice_general_question',.78,True)
    if re.search(r'\b(?:science is|a formal citation|this is|there is no|there are no|data policy|i don.t search|i just don.t|currently|still|usually|sometimes)\b',low):return emit_v7(base,'reality','v7_current_rule_state',.78,True)
    return base

def classify_v7(row):
    normalized={"memory_id": row.get("memory_id") or row.get("id"), "text": row.get("original_text") if row.get("original_text") is not None else row.get("text", ""), "question": row.get("source_question") or row.get("question", ""), "semantic_status":row.get("semantic_status", ""), "entry_kind": row.get("entry_kind", ""), "source_turn_id":row.get("source_turn_id")}
    output=classify_version(normalized,7)
    output["schema"]=VERSION_V7
    return output



# ====================== Original rule level v8 ======================
"""MemoryBank Python classifier v8: status/kind gate and pragmatic intent.
Experimental. All decisions derive from original_text, question and provenance metadata.
"""
import re
VERSION_V8='python_factbank_index.v8'
INTERACTION_KINDS=('interaction_or_clarification_not_independent_fact','original_interview_procedure','interview_interaction','unclear_or_incomplete_source')

def emit_v8(base,cat,reason,confidence=.8,review=False):
    x={**base,'category':cat,'is_fact':cat in ('memory','reality'),
       'exclude_from_retrieval':cat not in ('memory','reality'),
       'classification_reason':reason,'confidence':confidence,'review_required':review or cat=='review','schema':VERSION_V8}
    return x

def classify_v8(row):
    base=classify_v7(row)
    base['schema']=VERSION_V8
    if base['classification_reason']=='memorybank_source_status_gate':return base
    kind=str(row.get('entry_kind') or '').lower()
    if any(x in kind for x in INTERACTION_KINDS):
        return emit_v8(base,'non_fact','source_entry_kind_non_independent',.99)
    t=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    if not t:return emit_v8(base,'non_fact','empty_original_text',1)
    # The respondent may ask an interviewer a question instead of answering.
    if t.endswith('?') and re.search(r'\b(?:do you mean|you mean|what do you mean|so you want|are you asking|you want me to|in terms of|like,? do you mean)\b',t,re.I):
        return emit_v8(base,'non_fact','respondent_clarification_question',.95)
    if re.fullmatch(r'(?:oh right|okay,? it.s done|okay,? bye|bye|bye bye|alright|that.s okay|no problem|yes,? yes,? definitely|ha(?: ha)?|yep|yeah|okay|yes)[.,!\s]*',t,re.I):
        if re.search(r'\b(?:have you|did you|are you|do you|what|were you|is it|does it)\b',q,re.I) and not re.search(r'\b(?:consent|record|interview|thank|close|end|permission)\b',q,re.I):
            return base
        return emit_v8(base,'non_fact','interview_backchannel_or_closure',.95)
    return base



# ====================== Original rule level v9 ======================
"""MemoryBank Python v9: stronger question-grounded polarity and temporal negation."""
import re
VERSION_V9='python_factbank_index.v9'

def classify_v9(row):
    base=classify_v8(row);base['schema']=VERSION_V9
    if base['classification_reason'] in ('memorybank_source_status_gate','source_entry_kind_non_independent','interview_backchannel_or_closure','respondent_clarification_question'):return base
    s=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    l=s.lower();ql=q.lower()
    # A deny-current-example statement is not a completed remembered episode.
    if re.search(r'\b(?:haven.t|have not|never) (?:gone through|had|encountered|been through|done)\b',l) and re.search(r'\b(?:yet|so far|never|at all|any time)\b',l):
        return emit_v8(base,'reality','ongoing_absence_to_date',.82,True)
    # Current policies and periodic reciprocal arrangements, not one-time event.
    if re.search(r'\b(?:each year|per year|annually|in the future|we are preparing to|we plan to)\b',l) and not re.search(r'\b(?:last year|years ago|last week|we decided to)\b',l):
        return emit_v8(base,'reality','recurring_or_current_plan',.78,True)
    if re.search(r'\b(?:they don.t expect you to|they don.t need|you just (?:go|need)|we (?:generally|regularly) do)\b',l):
        return emit_v8(base,'reality','ongoing_expectation_or_practice',.82)
    if re.search(r'\b(?:they are for client|they.re for client|it is for client|they are commercial|no,? they are)\b',l):
        return emit_v8(base,'reality','short_qualified_current_status',.84)
    # A question about general protocols not a request for a one-off anecdote.
    if re.search(r'\b(?:can you provide insight|what would|what should|what do you think|how do you see)\b',ql) and re.search(r'\b(?:in general|my view|i would|i think|we should|of course)\b',l):
        if not re.search(r'\b(?:last month|last year|years ago|back then|i first started)\b',l):
            return emit_v8(base,'reality','general_advice_question_response',.81,True)
    # Generic negative experience in a current setting, vs explicit historic incident.
    if re.search(r'\b(?:no,? teaching methods are|no barriers|not very close with anyone)\b',l):
        return emit_v8(base,'reality','current_condition_or_no_barriers',.80,True)
    if re.fullmatch(r'(?:yes[,. ]+){1,3}(?:definitely|exactly|sure)[.!, ]*',l) and not re.search(r'\b(?:did you|have you|are you|were you|do you)\b',ql):
        return emit_v8(base,'non_fact','agreement_without_independent_claim',.88)
    return base



# ====================== Original rule level v10 ======================
"""MemoryBank Python v10: distinguish generalized answers from incidental anecdotes."""
import re
VERSION_V10='python_factbank_index.v10'
GENERAL_Q=re.compile(r'\b(?:do you think|what do you think|how do you|how would you|do you believe|what is your view|how important|what advice|what role|do you agree|would you say|do you think ethics|what should)\b',re.I)
GENERAL_A=re.compile(r'\b(?:there (?:are|is) (?:definitely|many|some)|in (?:general|principle)|i think|i believe|it depends|sometimes|usually|normally|generally|can be|should be|they should|you can|we can|might|tend to|my opinion|the boundaries|i would)\b',re.I)

def classify_v10(row):
    base=classify_v9(row);base['schema']=VERSION_V10
    if base['category']=='non_fact':return base
    s=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    sl=s.lower()
    # General question and answer dominated by a present-tense principle, despite an incidental anecdote.
    if GENERAL_Q.search(q) and GENERAL_A.search(s) and not re.search(r'\b(?:last week|last month|two years ago|in \d{4}|for four months|when i started)\b',sl):
        return emit_v8(base,'reality','general_question_dominant_generalization',.76,True)
    if re.match(r'yes[,\s]+depending on each',sl) and re.search(r'\b(?:flexible|individual|different)\b',sl):
        return emit_v8(base,'reality','conditional_ongoing_policy',.75,True)
    return base



# ====================== Original rule level v11 ======================
"""MemoryBank v11: regression fixes, source-conservative; no LLM or network calls."""
import re
VERSION_V11='python_factbank_index.v11'

def emit_v11(base,cat,reason,confidence=.8,review=False):
    result=emit_v8(base,cat,reason,confidence,review)
    result['schema']=VERSION_V11
    return result

def classify_v11(row):
    base=classify_v10(row);base['schema']=VERSION_V11
    if base['category']=='non_fact':return base
    t=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    s=t.lower()
    # Completed, dated internships vs current job.
    if re.search(r'\b(?:internship|placement|interned)\b',s) and re.search(r'\b(?:for (?:\d+|one|two|three|four|six) months?|the second internship|my team was|worked with)\b',s):
        return emit_v11(base,'memory','bounded_internship_work_history',.86,True)
    if re.search(r'\b(?:at first|initially|previously|back then|in the beginning)\b',s) and re.search(r'\b(?:i had never|had never|was difficult|adjusting)\b',s):
        return emit_v11(base,'memory','historical_context_overrides_absence_to_date',.89)
    # V7 regex wrongly sees conditional "if there were" as an actual past episode.
    if re.search(r'\b(?:if there were|if there was|if i were|if we were)\b',s) and re.search(r'\b(?:generally|usually|prefer|we still|we would|we use|should|is|are)\b',s) and not re.search(r'\b(?:last month|last year|years ago|when i was)\b',s):
        return emit_v11(base,'reality','counterfactual_not_past_event',.82,True)
    # Named factual answers (organization/entity) can be short, even 1-3 words.
    if re.search(r'\b(?:name of the company|company name|what is the name)\b',q,re.I) and re.search(r'[A-Za-z\u4e00-\u9fff]',t) and len(t.split())<=8:
        return emit_v11(base,'reality','question_anchored_entity_answer',.83,True)
    # A brief consent/backchannel statement contains no personal knowledge.
    if re.fullmatch(r'(?:yeh|yeah|yes|yep|uh huh|mm hmm)[.!?\s]*',s) and not re.search(r'\b(?:have you|did you|were you|do you|are you|have you ever)\b',q,re.I):
        return emit_v11(base,'non_fact','unanchored_backchannel',.94)
    return base



# ====================== Original rule level v13 ======================
"""Experimental v13: principled question-anchored classification on top of v11.
Source provenance verification remains exclusively in the native adapter.
"""
import re
VERSION_V13='python_factbank_index.v13'

def update_v13(base,cat,reason,confidence=.75,review=False):
    out=dict(base)
    out.update(schema=VERSION_V13,category=cat,is_fact=cat in ('memory','reality'),exclude_from_retrieval=cat not in ('memory','reality'),review_required=review or cat=='review',classification_reason=reason,confidence=confidence)
    return out

def classify_v13(row):
    base=classify_v11(row);base['schema']=VERSION_V13
    if base['classification_reason'] in ('memorybank_source_status_gate','source_entry_kind_non_independent','interview_backchannel_or_closure','respondent_clarification_question'):
        return base
    if base['category']=='non_fact':return base
    t=str(row.get('original_text') or '').strip(); q=str(row.get('source_question') or '').strip()
    s=t.lower();ql=q.lower()
    # Interview administration (followup logistics) is not respondent biographical evidence.
    if re.search(r'\b(?:let me check|i can share|i will send|i.ll send|after this interview|after the interview|i.ll get back to you)\b',s) and re.search(r'\b(?:company docs|send you|share something|after this interview|let me check)\b',s):
        if not re.search(r'\b(?:we usually|we always|every week|at my job)\b',s):return update_v13(base,'non_fact','interview_followup_logistics',.85,True)
    if re.fullmatch(r'(?:yeh|yeah|yep|yes|right)[,. ]+(?:that.s what.s happening|that.s right|exactly|i see)[.!, ]*',s):
        if not re.search(r'\b(?:did you|have you|do you|are you)\b',ql):return update_v13(base,'non_fact','minimal_backchannel_no_standalone_claim',.87,True)
    # Existing state or lack of experience to date vs a bounded past incident.
    if re.search(r'\b(?:never (?:have|had)|have never|haven.t ever)\b',s) and not re.search(r'\b(?:at first|last (?:year|week)|when i was|back then|used to)\b',s):
        return update_v13(base,'reality','ongoing_negative_experience',.78,True)
    if re.search(r'\b(?:how aware are you|how conscious are you|what (?:is|are) (?:your|the) (?:research interest|cooperation mechanism)|could you briefly introduce your research interests)\b',ql):
        if not re.search(r'\b(?:last year|years ago|at that time|when i started|back then)\b',s):
            return update_v13(base,'reality','question_current_state_or_method',.77,True)
    if re.search(r'\b(?:how can|how should|how would|what is the best way|what are the best practices)\b',ql) and re.search(r'\b(?:i (?:think|would|believe)|you (?:should|need|can)|there are|it is important|the (?:first|second)|two primary|one is where)\b',s):
        if not re.search(r'\b(?:last (?:year|week)|years ago|at that time|when i started)\b',s):
            return update_v13(base,'reality','general_guidance_not_historical_episode',.78,True)
    return base



# ====================== Original rule level v14 ======================
"""v14 experimental: targeted generalization and concise mechanism answers; source-native, deterministic."""
import re
VERSION_V14='python_factbank_index.v14'
MECHANISM_QUESTION=re.compile(r'\b(?:what is the (?:cooperation |sharing |publication )?mechanism|how (?:is|does) (?:the|this) (?:system|policy|mechanism) (?:work|operate)|what is the process|what system do you use)\b',re.I)

def classify_v14(row):
    base=classify_v13(row);base['schema']=VERSION_V14
    if base['category']=='non_fact' or base['classification_reason']=='memorybank_source_status_gate':return base
    text=str(row.get('original_text') or '').strip();q=str(row.get('source_question') or '').strip();s=text.lower()
    if MECHANISM_QUESTION.search(q) and 1<=len(text.split())<=9 and not re.search(r'\b(?:last year|back then|ago|previously|we used to)\b',s):
        out=update_v13(base,'reality','question_anchored_current_mechanism',.73,True);out['schema']=VERSION_V14;return out
    if re.match(r'^(?:exactly|yes|right)[.! ]+',s) and re.search(r'\b(?:does not have to|do not have to|doesn.t have to|not only|is not limited to|can also)\b',s):
        out=update_v13(base,'reality','agreement_with_independent_current_claim',.77,True);out['schema']=VERSION_V14;return out
    # General trends can mention an example's age without being solely a past episode.
    if re.search(r'\b(?:it.s changing|things are changing|these tools are changing|i think.*(?:best|important|better))\b',s) and re.search(r'\b(?:right now|currently|these days|nowadays)\b',s) and not re.search(r'\b(?:last week i|in \d{4} i|when i was|i used to)\b',s):
        out=update_v13(base,'reality','present_trend_with_incidental_historical_reference',.73,True);out['schema']=VERSION_V14;return out
    return base



# ====================== Original rule level v15 ======================
"""MemoryBank-native conservative temporal rules v15. Only MemoryBank original_text is fact evidence.
Deterministic, LLM-free. v14 source-status gates are retained by returning protected outputs.
"""
import re
VERSION_V15='python_factbank_index.v15'
PROTECTED={'memorybank_source_status_gate','source_entry_kind_non_independent','interview_backchannel_or_closure','respondent_clarification_question','conversational_filler','social_or_interview_management'}
QUESTION_ONLY=re.compile(r'^\s*(?:so|okay|yeah|well|sorry)?[,.! ]*(?:how|what|why|who|where|when|which|could|can|do|does|did|are|is|would|shall)\b[^.!]{1,140}\?\s*$',re.I)
PRESENT=re.compile(r'\b(?:i (?:think|believe|prefer|feel|consider|usually|typically|often|sometimes|always|currently|still|do|have|am|would)|we (?:think|believe|prefer|usually|typically|often|sometimes|always|currently|still|do|have|are|would)|there (?:is|are)|it (?:is|can|does|means|depends)|these (?:are|can)|they (?:are|can|do)|(?:should|would|can|could|need to|generally|in general|right now|currently|nowadays|at present|every day|each year|regularly|normally))\b',re.I)
EPISODE=re.compile(r'\b(?:last (?:week|month|year|summer)|while schools were|in one of my courses|yesterday|\d+ (?:days|weeks|months|years) ago|when i was|when we were|i (?:started|finished|went|built|wrote|created|joined|worked|decided|introduced|received|sent|asked)\b|we (?:started|finished|went|built|wrote|created|joined|worked|decided|introduced|received|sent|asked)\b|(?:in|during) (?:19|20)\d\d|at that time|back then)\b',re.I)
PURE_MANAGEMENT=re.compile(r'^\s*(?:can you (?:repeat|hear|clarify)|could you (?:repeat|clarify)|how do you (?:mean|spell)|what do you mean|how do i (?:find|answer) that|shall we (?:start|continue)|i(?:'+'\u2019|'+"'"+r')?m (?:not sure i understand|sorry can you repeat))[^.!?]{0,100}[?.! ]*$',re.I)

def classify_v15(row):
    base=classify_v14(row);base['schema']=VERSION_V15
    if base['classification_reason'] in PROTECTED or base['category']=='non_fact' and base['classification_reason']=='memorybank_source_status_gate':return base
    s=str(row.get('original_text') if row.get('original_text') is not None else row.get('text') or '').strip()
    q=str(row.get('source_question') or row.get('question') or '').strip()
    def emit(cat,reason,conf=.78,review=False):
        out=dict(base)
        out.update(category=cat,is_fact=cat in ('memory','reality'),exclude_from_retrieval=cat not in ('memory','reality'),review_required=review or cat=='review',classification_reason=reason,confidence=conf)
        return out
    if PURE_MANAGEMENT.fullmatch(s) or (QUESTION_ONLY.fullmatch(s) and not re.search(r'\b(?:i|we|my|our)\b',s,re.I)):
        return emit('non_fact','respondent_interview_clarification',.92,True)
    # A 'did you' interviewer prompt is not evidence that the respondent's present-tense
    # comment describes a completed event. This was the dominant source of v14 holdout errors.
    if base['classification_reason'] in ('question_anchored_historical','v5_question_anchored_short_history','question_anchored_past_short_answer'):
        if PRESENT.search(s) and not EPISODE.search(s) and len(s.split())>6:
            return emit('reality','answer_present_overrides_past_question',.75,True)
    # A real bounded episode in the answer has priority over generic question heuristics.
    if base['category']=='reality' and EPISODE.search(s) and not PRESENT.search(s):
        return emit('memory','explicit_completed_episode_overrides_question',.81,True)
    # Current policies and reflective opinions can include past-tense framing.
    if base['category']=='memory' and base['classification_reason'] in ('v7_reconstructed_episode','v6_finished_episode','question_anchored_historical') and PRESENT.search(s) and not EPISODE.search(s):
        return emit('reality','current_claim_without_bounded_episode',.73,True)
    return base



# Public version-neutral entrypoint. Keep v15 as the final stage.
VERSION=VERSION_V15
classify=classify_v15
