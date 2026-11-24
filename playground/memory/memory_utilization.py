"""Memory-ID utilization; clearly distinct from Spoken Truth or factual accuracy."""

def evaluate(run,source_facts):
    ids={str(f.get('dataset_memory_id') or f.get('id')) for f in source_facts
         if isinstance(f,dict) and (f.get('dataset_memory_id') or f.get('id'))}
    cited=set(); invalid=set(); incomplete=[]; rows=[]; observed=0
    for i,turn in enumerate(run.get('turns') or [],1):
        if not isinstance(turn,dict):continue
        att=turn.get('memory_attribution')
        if not isinstance(att,dict) or att.get('schema')!='memory_citation.v1':
            incomplete.append(i);continue
        observed+=1
        declared=att.get('cited_memory_ids')
        available=att.get('available_memory_ids')
        if not isinstance(declared,list) or not isinstance(available,list) or any(
            not isinstance(x,str) for x in declared+available):
            incomplete.append(i);continue
        if att.get('complete') is not True:incomplete.append(i)
        permitted=set(available)&ids
        cited.update(set(declared)&permitted)
        invalid.update(set(declared)-permitted)
        rows.append({'round':i,'cited_memory_ids':sorted(set(declared)&permitted),
                     'citation_method':att.get('citation_method')})
    value=round(100*len(cited)/len(ids),2) if ids and observed else None
    complete=bool(ids and observed and not incomplete and not invalid)
    return {'value':value,'unit':'percent','status':(
        'self_reported_memory_usage' if complete else
        'incomplete_attribution' if observed else 'no_memory_attribution'),
        'method':'unique_memory_id_utilization_v175',
        'numerator':len(cited),'denominator':len(ids),
        'used_memory_ids':sorted(cited),'invalid_memory_ids':sorted(invalid),
        'missing_rounds':sorted(set(incomplete)),'round_citations':rows,
        'independently_verified':False,
        'note':'不同 Memory ID 自报引用数 / 全部可用 Memory ID；不是 Spoken Truth 或准确率。'
         '未经逐句独立证据验证，不能称为真实使用率。'}
