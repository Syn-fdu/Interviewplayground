"""Researcher-only local export: completed prepared memory + original permitted inputs.
No inference: source IDs prove inputs, not semantic coverage in the model output.
"""
from datetime import datetime, timezone
from playground.memory.integrated_memory import lookup, SAFE

def eligible_source_facts(bundle):
    source = bundle.get('source_scenario') or {}
    local_v2 = source.get('source_type') == 'v2_restricted_local'
    eligible = []
    for mi, memory in enumerate(bundle['respondent']['memory_bank']):
        for fi, fact in enumerate(memory.get('facts') or []):
            sensitivity = str(fact.get('sensitivity', 'unknown')).strip().lower()
            if sensitivity not in SAFE and not (sensitivity == 'unknown' and local_v2):
                continue
            text = str(fact.get('text') or '').strip()
            if text:
                # Source question is retrieval CONTEXT, not evidence for a
                # personal assertion. A short answer ("6-12 months") can
                # only be understood as an AI-onset estimate together with
                # its actual preceding interview question.
                source_question=str(memory.get('source_question') or '').strip()
                eligible.append({'id':f'F{mi:04d}_{fi:03d}', 'text':text,
                                 'source_question':source_question[:500],
                                 'sensitivity':sensitivity, 'review_pending':sensitivity=='unknown',
                                 **({'dataset_memory_id':fact['dataset_memory_id'],
                                     'dataset_source':memory.get('source',{}),
                                     'prebuilt_semantic_tags':fact.get('semantic_tags',[])}
                                    if fact.get('dataset_memory_id') else {})})
    return eligible

def export_prepared_memory(bundle, model, preparation_mode, prepared_snapshot=None):
    # A validated in-memory snapshot is the authoritative preparation when the
    # active UI model/provider/trait settings differ from the original prepare.
    # Export must never trigger new preparation or use a different fact pool.
    prepared = prepared_snapshot if prepared_snapshot is not None else lookup(bundle, model, preparation_mode)

    source = bundle.get('source_scenario') or {}
    local_v2 = source.get('source_type') == 'v2_restricted_local'
    eligible = eligible_source_facts(bundle)
    if prepared.get("factbank_llm_index"):
        from playground.memory.llm_factbank import apply_classification, summary as llm_summary
        category_rows=apply_classification(bundle,prepared.get("factbank_llm_index"))
        classification=llm_summary(prepared.get("factbank_llm_index"))
    else:
        from playground.memory.factbank import classified_facts, preparation_view
        category_rows=classified_facts(bundle)
        classification=preparation_view(bundle)
    batches=[];current=[];chars=0
    for item in eligible:
        n=len(item['text'])
        if current and (chars+n>2200 or len(current)>=12):
            batches.append(current);current=[];chars=0
        current.append(item);chars+=n
    if current:batches.append(current)
    method=prepared.get('preparation_method','')
    llm_factbank = method.startswith('llm_factbank_') or method.startswith('python_first_selective_llm_')
    unit_map={x.get("source_id"):x for x in prepared.get("factbank_llm_index",[])
              if isinstance(x,dict)}
    summaries = []
    for idx,group in enumerate(prepared.get('card', [])):
        if not isinstance(group,dict):continue
        facts = group.get('facts') or []
        content='\n'.join(str(f) for f in facts) if isinstance(facts,list) else str(facts)
        summaries.append({'batch':idx+1, 'theme':group.get('theme',''),
                          'agent_summary':content,
                          'source_fact_ids':([sid for sid,row in unit_map.items()
                              if row.get('event_id')==str(group.get('theme','')).replace('SourceEvent ','')]
                              if llm_factbank else []),
                          'provenance_note':'Source IDs are source-derived event membership; semantic coverage is separately reported'})
    from playground.memory.source_evidence import build_locators
    categories={r["id"]:r for r in category_rows}
    unit_records=[]
    for f in eligible:
        category=categories.get(f["id"],{})
        semantic=unit_map.get(f["id"],{})
        unit_records.append(dict(f,
            fact_category=category.get("fact_category",semantic.get("fact_category","reality")),
            llm_fact_type=semantic.get("fact_type","other"),
            llm_temporal_scope=semantic.get("temporal_scope","unknown"),
            llm_importance=semantic.get("importance","background"),
            llm_semantic_tags=semantic.get("semantic_tags",[])))
    return {'schema':'interview-playground-agent-memory-export-v1',
            'source_locator_schema':'source_memory_locator.v1',
            'source_locators':build_locators(unit_records),
            'created_at':datetime.now(timezone.utc).isoformat(),
            'scenario_id':source.get('scenario_id'), 'model':model,
            'source_fingerprint_sha256':(source.get('provenance') or {}).get('source_fingerprint_sha256'),
            'source_dataset':(source.get('provenance') or {}).get('dataset'),
            'source_candidate_id':(source.get('provenance') or {}).get('candidate_id'),
            'preparation_mode':preparation_mode,'preparation_method':method,
            'cache_version':prepared.get('cache_version'),
            'source_revision':bundle['audit']['revision'],
            'completed_cache_only':True,
            'quality_warning':'Source IDs indicate inputs to integration, not semantic coverage. Check text manually.',
            'preparation_seconds_original':prepared.get('preparation_seconds'),
            'agent_batch_count':prepared.get('agent_batch_count'),
            'source_facts':eligible,'integrated_memory':summaries,
             'memory_source':(source.get('provenance') or {}).get('memory_stats',{}).get('memory_source','legacy_records'),
             'memory_source_stats':(source.get('provenance') or {}).get('memory_stats',{}),
            'factbank_schema':classification['schema'],
            'factbank_preparation':classification,
            'factbank_categories':[{
                "source_id":row["id"],
                "category":row["fact_category"],
                "classification_reason":row["classification_reason"],
                "confidence":row["classification_confidence"],
                "needs_review":row["classification_review_required"]}
                for row in category_rows],
            'factbank_llm_index':prepared.get("factbank_llm_index",[]),
            'factbank_llm_summary':prepared.get("factbank_llm_summary"),
            'multilingual_index_status':prepared.get('multilingual_index_status','not_reported'),
            'multilingual_vector_ready':bool(prepared.get('multilingual_vector_index')),
            'llm_api_call_count_this_run':prepared.get('llm_api_call_count_this_run'),
            'llm_enrichment_error_code':prepared.get('llm_enrichment_error_code'),
            'semantic_coverage':prepared.get('semantic_coverage','not_reported'),
            'full_bilingual_semantic_complete':prepared.get('full_bilingual_semantic_complete',False),
            'fast_preparation_quality_status':prepared.get('fast_preparation_quality_status','not_reported'),
            'public_distractors_are_not_personal_facts':True}
