"""Source-only metadata: no synthetic secondary fact identities.

Every locator resolves directly to a permitted original MemoryBank/source record.
"""
from __future__ import annotations
import re

_TIME_PATTERNS = (
    re.compile(r"\b20\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b", re.I),
    re.compile(r"\b(?:last year|last month|a few weeks ago|several weeks ago|"
               r"a few months ago|several months ago|\d{1,2}\s*(?:-|to|–)\s*\d{1,2}\s*"
               r"(?:months?|years?) ago|\d{1,2}\s*(?:months?|years?) before)\b", re.I),
    re.compile(r"\b(?:during (?:college|university|my studies)|in college|at university|"
               r"when i was in college|before the interview)\b", re.I),
    re.compile(r"去年|前年|今年|上个月|去年年底|几周前|几星期前|几个月前|"
               r"几年前|大学期间|上大学时|访谈前|当时|那时候|以前", re.I),
)



def _event_id(source_id: str):
    match = re.match(r"^F(\d{4})_", str(source_id))
    return "G" + match.group(1) if match else None


def _time_expressions(text: str):
    value = str(text or "")
    found = []
    for pattern in _TIME_PATTERNS:
        for match in pattern.finditer(value):
            phrase = match.group(0).strip()
            if phrase and phrase not in found:
                found.append(phrase[:64])
    return found[:8]



SCHEMA='source_memory_locator.v1'

def build_locators(records):
    rows=[]
    for record in records or []:
        if not isinstance(record,dict):continue
        source_id=str(record.get('id') or '').strip()
        text=str(record.get('text') or '').strip()
        if not source_id or not text:continue
        memory_id=str(record.get('dataset_memory_id') or source_id).strip()
        importance=str(record.get('llm_importance') or 'background')
        rows.append({'source_id':source_id,'memory_id':memory_id,
          'event_id':_event_id(source_id),
          'time_expressions':_time_expressions(text+' '+str(record.get('source_question') or '')),
          'temporal_scope':'as_of_original_interview',
          'access_scope':'eligible_source_candidate',
          'fact_temporal_scope':str(record.get('llm_temporal_scope') or 'unknown'),
          'fact_type':str(record.get('llm_fact_type') or 'other'),
          'semantic_tags':list(record.get('llm_semantic_tags') or []),
          'importance':importance,
          'importance_weight':{'key':3,'supporting':2,'background':1}.get(importance,1),
          'review_pending':bool(record.get('review_pending') or record.get('classification_review_required'))})
    if len({x['source_id'] for x in rows})!=len(rows):
        raise ValueError('Duplicate source locator IDs')
    if len({x['memory_id'] for x in rows})!=len(rows):
        raise ValueError('Duplicate memory IDs within candidate')
    return rows

def annotate_selected(records,locators_by_source):
    result=[]
    for record in records or []:
        item=dict(record)
        locator=locators_by_source.get(str(item.get('source_id') or ''))
        if locator:
            for key,value in locator.items():
                if key!='source_id':item[key]=value
        result.append(item)
    return result
