"""Question-level information elicitation, separate from Memory ID utilization.

Reference source questions form Q_target. A source question counts as spoken
only if ALL of its eligible gold answer records were semantically matched in
respondent speech; this is an explicit original-interview-question proxy, not
an automatically asserted gold mapping to the new research outline.
"""
from collections import defaultdict
from playground.evaluation.interview_reportcard_judge import JudgeError
from playground.evaluation.spoken_truth_core import evaluate_semantic
from playground.memory.memory_utilization import evaluate as memory_utilization

def evaluate(run,facts,model,checkpoint_store=None,opener=None):
    if not facts:
        raise JudgeError('Spoken Truth 缺少来源 MemoryBank，不能计算')
    kwargs={'checkpoint_store':checkpoint_store}
    if opener is not None:kwargs['opener']=opener
    matched=evaluate_semantic(run,facts,model,**kwargs)
    statuses={str(x['fact_id']):x for x in matched.get('fact_annotations',[])}
    groups=defaultdict(list)
    for f in facts:
        q=str(f.get('source_question') or '').strip()
        if q:groups[q].append(str(f['id']))
    if not groups:
        return {'value':None,'unit':'percent','status':'no_reference_questions',
                'method':'reference_question_elicitation_v175','numerator':0,'denominator':0,
                'note':'没有原始访谈目标问题与标准答案映射，Spoken Truth 不可计算。'}
    details=[]; counted=0
    for question,ids in groups.items():
        found=[i for i in ids if statuses.get(i,{}).get('status')=='matched']
        complete=len(found)==len(ids)
        if complete:counted+=1
        details.append({'question':question,'gold_record_count':len(ids),
                        'matched_record_count':len(found),'spoken':complete,
                        'source_ids':ids,
                        'evidence':sum((statuses.get(i,{}).get('evidence',[]) for i in found),[])})
    return {'value':round(100*counted/len(groups),2),'unit':'percent',
            'status':'semantic_judge_unverified','method':'reference_question_elicitation_v175',
            'reference_basis':'original_interview_source_questions',
            'numerator':counted,'denominator':len(groups),
            'reference_questions':details,
            'memory_record_elicitation':{'matched':matched.get('numerator'),
                                         'total':matched.get('denominator')},
            'note':'模拟访谈说出的完整原访谈标准答案问题数 / 可核验原访谈问题数。'
                   '采用 InterviewBench 问题级信息引出定义的来源问题代理；'
                   '尚未将研究提纲和参与者金标准答案一对一匹配，不能称为论文原实验复现。'
                   '评分器输出须抽样人工核验。'}
