"""Validated JSON IDs selected via the respondent model, inspired by v0.9.1."""
from __future__ import annotations
import json,re

PROMPT=("You are a multilingual EVIDENCE RETRIEVER, not an interviewer or answerer. "
 "Match the meaning across Chinese and English, not keywords alone. "
 "Compare the current question with EACH memory's ORIGINAL interview question "
 "and respondent answer excerpt, preserving event boundaries. "
 "Choose up to the requested limit of directly relevant IDs. "
 "For time questions prefer original time answers; for specific cases prefer "
 "one coherent episode; for frequency require frequency evidence. "
 "Do not invent IDs or facts. Return ONLY JSON {\"selected_ids\": [\"G0000\"]}.")

def select_with_respondent_model(question, previous, candidates, limit, call):
    if not isinstance(candidates,list) or len(candidates)>120:
        raise ValueError('semantic_candidates_invalid')
    allowed={str(c['id']) for c in candidates}
    payload={'question':question,'last_respondent_response':previous,
             'max_selected':limit,'candidate_memories':candidates}
    raw=call([{'role':'system','content':PROMPT},
              {'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
    raw=re.sub(r'<think>.*?</think>','',str(raw),flags=re.S|re.I).strip()
    if raw.startswith('```'):
        raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw,flags=re.I).strip()
    obj=json.loads(raw)
    ids=obj.get('selected_ids') if isinstance(obj,dict) else None
    if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids):
        raise ValueError('semantic_selected_ids_invalid')
    return list(dict.fromkeys(x for x in ids if x in allowed))[:limit]
