"""HR12: v0.9.1-inspired cross-lingual selection of permission-filtered MemoryBank groups.

Uses the original interview question AND respondent excerpt, returns only authorized
source IDs. Remote semantic selection is optional; deterministic recall is a
separately labelled degradation, never an invented evidence source.
"""
from __future__ import annotations
from playground.interview.respondent_core import StudioError
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from playground.memory.high_recall_fallback import _score as evidence_score, _intent

SEMANTIC_VERSION = "v091_question_answer_batched_semantic_hr12"


def _raise_semantic_failure(exc, operation):
    """Preserve safe upstream metadata instead of falsely labelling every failure HTTP 502.

    This adapter must not degrade to lexical/Python recall on failure.
    Never surface raw exception text, provider responses, prompts, or keys.
    """
    if isinstance(exc, StudioError):
        status = exc.status
        detail = exc.detail_code or exc.code
        retryable = bool(exc.retryable)
        diagnostic_id = getattr(exc, "diagnostic_id", None)
        structure = getattr(exc, "response_structure", None)
    elif isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        status, detail, retryable = 502, "semantic_invalid_selection", False
        diagnostic_id, structure = None, None
    else:
        status, detail, retryable = 502, "semantic_selector_unclassified_failure", False
        diagnostic_id, structure = None, None
    label = "语义重排" if operation == "rerank" else "语义检索"
    error = StudioError(
        f"{label} LLM 请求未成功；本轮停止，之前成功的访谈轮次不受影响。",
        "semantic_llm_request_failed", status,
        detail_code=detail, retryable=retryable)
    error.error_detail = f"失败阶段：受访者{label}；原因代码：{detail}"
    if diagnostic_id:
        error.diagnostic_id = diagnostic_id
    if isinstance(structure, dict):
        error.response_structure = {**structure, "failed_stage": "semantic_"+operation}
    raise error from exc


def grouped_records(bundle, allowed_records):
    permitted = {str(r['id']): r for r in allowed_records if isinstance(r, dict) and r.get('id') and r.get('text')}
    groups=[]
    for i, group in enumerate(bundle['respondent']['memory_bank']):
        facts=[]
        for j, _ in enumerate(group.get('facts') or []):
            sid=f'F{i:04d}_{j:03d}'
            if sid in permitted:
                facts.append(permitted[sid])
        if facts:
            groups.append({'id':f'G{i:04d}', 'facts':facts,
                           'original_question': ' '.join(dict.fromkeys(str(f.get('source_question') or '') for f in facts))[:320],
                           'respondent_excerpt':' '.join(str(f.get('text') or '')[:180] for f in facts[:3])[:420]})
    return groups


def _seed_score(question, group):
    # Only source-derived metadata is a retrieval hint; no question text is
    # promoted to a respondent fact. The score is for model-free degradation.
    score=max((evidence_score(question, r) for r in group['facts']),default=0.)
    if score: return score
    # Exact distinctive Chinese/English overlap is an explicit low-confidence
    # model-free fallback for questions not covered by predefined intent tags.
    q=str(question).lower()
    text=(group['original_question']+' '+group['respondent_excerpt']).lower()
    chinese=re.findall(r'[\u4e00-\u9fff]{2,}',q)
    overlap=any(part in text for word in chinese for n in (min(4,len(word)),2)
                for part in (word[i:i+n] for i in range(max(1,len(word)-n+1))))
    if overlap and not _intent(q): return 2.0
    return 0.0


def select_groups(question, history, groups, semantic_selector=None, max_groups=3, parallel_batches=False):
    """Chunk/rerank as in v0.9.1; never accept invented group IDs."""
    if not groups:
        return [], {'selection_method':'empty_group_bank','semantic_status':'no_groups', 'semantic_model_calls':0}
    candidates=[]
    failures=[]
    calls=0
    started=time.monotonic()
    previous=str((history or [])[-1].get('answer') or '')[-200:] if history and isinstance(history[-1],dict) else ''
    batch_count=(len(groups)+37)//38 if semantic_selector else 0
    workers=min(2,batch_count) if parallel_batches and batch_count>1 and os.getenv("INTERVIEW_LEGACY_SELECTOR_WORKERS","2")!="1" else 1
    if semantic_selector:
        chunks=[groups[start:start+38] for start in range(0,len(groups),38)]
        def process_chunk(chunk):
            choices=[{'id':g['id'],'original_question':g['original_question'][:150],
                      'respondent_excerpt':g['respondent_excerpt'][:240]} for g in chunk]
            try:
                picked=semantic_selector(question,previous,choices,5)
                if not isinstance(picked,list):raise ValueError('invalid_semantic_selection')
                ids={x for x in picked if isinstance(x,str) and x in {g['id'] for g in chunk}}
                return [g for g in chunk if g['id'] in ids],None
            except Exception as exc:
                # Never silently replace a failed LLM with Python keyword matching.
                _raise_semantic_failure(exc, "retrieval")
        if workers>1:
            # Preserve respondent provider ContextVars in worker threads. Each task
            # receives an independent copied context; original chunk order is kept.
            results=[None]*len(chunks)
            with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='legacy-memory') as pool:
                future_map={pool.submit(copy_context().run,process_chunk,chunk):idx
                            for idx,chunk in enumerate(chunks)}
                for future in as_completed(future_map):
                    results[future_map[future]]=future.result()
        else:
            results=[process_chunk(chunk) for chunk in chunks]
        calls+=len(chunks)
        for chosen,error_name in results:
            candidates.extend(chosen)
            if error_name:failures.append(error_name)
        if len(candidates)>max_groups:
            choices=[{'id':g['id'],'original_question':g['original_question'][:150],
                      'respondent_excerpt':g['respondent_excerpt'][:300]} for g in candidates[:120]]
            try:
                calls+=1
                picked=semantic_selector(question,previous,choices,max_groups)
                if not isinstance(picked,list):raise ValueError('invalid_rerank')
                ids=set(picked[:max_groups]) & {g['id'] for g in candidates}
                if ids:candidates=[g for g in candidates if g['id'] in ids]
            except Exception as exc:
                _raise_semantic_failure(exc, "rerank")
        if not candidates and not failures:
            # An explicit model-selected empty set must remain empty.
            return [], {'selection_method':'llm_batched','semantic_status':'no_relevant_memory',
                         'semantic_model_calls':calls,'semantic_errors':[],
                         'semantic_batch_workers':workers,'semantic_batch_count':batch_count,
                         'semantic_elapsed_ms':round((time.monotonic()-started)*1000)}
        method='llm_batched' if not failures else 'llm_partial_with_deterministic_degradation'
    else:
        candidates=[g for g in groups if _seed_score(question,g)>=2]
        method='deterministic_source_question_fallback'
    # Semantic model chooses group membership; fallback/rerank order uses
    # source-question alignment, avoiding response-only semantic contamination.
    candidates=sorted({g['id']:g for g in candidates}.values(), key=lambda g:(-_seed_score(question,g),g['id']))
    return candidates[:max_groups], {'selection_method':method,
        'semantic_status':'selected' if candidates else 'no_relevant_memory',
        'semantic_model_calls':calls,'semantic_errors':failures,
        'semantic_batch_workers':workers,'semantic_batch_count':batch_count,
        'semantic_elapsed_ms':round((time.monotonic()-started)*1000),
        'semantic_selected_groups':[g['id'] for g in candidates[:max_groups]]}


def selected_sources(question, groups, max_facts_per_group=3):
    records=[]
    for group in groups:
        facts=sorted(group['facts'],key=lambda f:(-_score_with_question(question,f),str(f['id'])))
        for fact in facts[:max_facts_per_group]:
            records.append({'source_id':str(fact['id']), 'text':str(fact['text']),
              'source_question':str(fact.get('source_question') or ''),
              'fact_category':fact.get('fact_category','reality'),
              'cache_source':'v091_grouped_semantic_original',
              'event_group':group['id']})
    return records


def _score_with_question(question,fact):
    return evidence_score(question,fact)
