"""HR11 old-v1.4.15 compatible group-based respondent memory.

The original question-first selector and tier access functions are preserved in
legacy_question_first_v013.py and legacy_memory_access.py. Only explicitly
permitted original facts enter a group. No bilingual cache hard-filter is used in
legacy_cache mode. Fused mode uses the same event grouping for retrieval misses.
"""
from __future__ import annotations
import json, re
from playground.memory.question_first_retrieval import select_memory, identify_question
from playground.memory.legacy_memory_access import tier_integrated_memory
from playground.interview.respondent_autonomy import SYSTEM_POLICY_ZH
from playground.memory.trait_postprocessing import behavioral_guide, question_sensitivity

LEGACY_VERSION='original_v1415_grouped_selector_tier_hr11'

def original_cards(bundle, allowed_records):
    """Reconstruct original per-Memory-Bank groups without unpermitted facts."""
    by_id={str(r['id']):r for r in allowed_records if isinstance(r,dict) and r.get('id') and r.get('text')}
    cards=[]
    source_map={}
    for i,memory in enumerate(bundle['respondent']['memory_bank']):
        entries=[]
        for j,_ in enumerate(memory.get('facts') or []):
            sid=f'F{i:04d}_{j:03d}'
            if sid not in by_id:continue
            text=re.sub(r'\s+',' ',str(by_id[sid]['text'])).strip()
            if text:
                # Original old preparation uses a complete-sentence prefix of <=240.
                from playground.memory.legacy_excerpt import compact_original
                excerpt=compact_original(text)
                entries.append(excerpt)
                source_map[(i,excerpt)]=by_id[sid]
        if entries:
            group_id=f'G{i:04d}'
            cards.append({'theme':group_id,'facts':entries})
    return cards,source_map

def _source_for_group(selection,source_map):
    for g in selection:
        name=str(g.get('theme') or '')
        if not name.startswith('G'):continue
        try: group_idx=int(name[1:])
        except ValueError:continue
        text=str((g.get('facts') or [''])[0])
        source=source_map.get((group_idx,text))
        # Tier may only return a prefix, so associate it with its source excerpt.
        if source is None:
            for (idx,excerpt),item in source_map.items():
                if idx==group_idx and (excerpt.startswith(text) or text.startswith(excerpt)):
                    source=item;break
        if source is not None:
            yield source,g,group_idx

def _semantic_group_sources(bundle,question,history,allowed_records,memory_value,
                            semantic_selector=None,max_groups=3,mode="old",parallel_batches=False):
    from playground.memory.semantic_memory_compat import grouped_records,select_groups,selected_sources,SEMANTIC_VERSION
    groups=grouped_records(bundle,allowed_records)
    selected,trace=select_groups(question,history,groups,semantic_selector,max_groups,parallel_batches=parallel_batches)
    visible_count=1 if float(memory_value)<.35 else 2 if float(memory_value)<.7 else max_groups
    selected=selected[:visible_count]
    results=selected_sources(question,selected,max_facts_per_group=3)
    # Access tiers affect the *actual* source texts passed to the model, not
    # merely diagnostic labels. A stable source ID remains for provenance.
    from playground.memory.legacy_memory_access import tier_integrated_memory
    tier_cards=[{'theme':r['source_id'],'facts':[r['text']]} for r in results]
    visible,access=tier_integrated_memory(tier_cards,memory_value)
    allowed_text={str(g.get('theme')):str((g.get('facts') or [''])[0]) for g in visible}
    results=[{**r,'text':allowed_text[r['source_id']],
              'access_tier':access['access_tier']} for r in results
             if r['source_id'] in allowed_text and allowed_text[r['source_id']]]
    return results,{**trace,**access,'old_style_version':SEMANTIC_VERSION,
                    'original_selection_status':trace['semantic_status'],
                    'prepared_event_groups':len(groups),
                    'selected_event_groups':[g['id'] for g in selected],
                    'semantic_mode':mode}


def select_original_sources(bundle,question,history,allowed_records,memory_value,semantic_selector=None):
    return _semantic_group_sources(bundle,question,history,allowed_records,
                                   memory_value,semantic_selector,mode="old",parallel_batches=True)


def select_fused_sources(bundle,question,history,allowed_records,memory_value,
                         max_groups=3,semantic_selector=None):
    # Called strictly AFTER the new main retrieval verifies a miss.
    results,trace=_semantic_group_sources(bundle,question,history,allowed_records,
                         memory_value,semantic_selector,max_groups,mode="fused_stage3")
    return results,{**trace,'fusion_event_groups':trace['selected_event_groups'],
                    'fusion_seed_count':len(results)}


def legacy_messages(question,selected_memory,recent_history,behavior,language,length_guide,
                    selection_trace=None):
    """Original v1.4.15 question-first prompt structure, with provenance guard."""
    analysis=selection_trace or identify_question(question,recent_history)
    context={'integrated_memory':[
           {'theme':m.get('event_group',m.get('theme','来源经历')),
            'facts':m.get('facts',[])} for m in selected_memory],
        'memory_selection_status':analysis.get('original_selection_status','selected'),
        'requested_aspects':analysis.get('requested_aspects',analysis.get('aspects',[])),
        'history_relation':analysis.get('history_relation','previous_turns_context_only'),
        'recent_history_reference_only':recent_history,
        'behavior_parameters':behavior,
        'behavioral_instruction':behavioral_guide(behavior,sensitivity=question_sensitivity(question)),
        'memory_access_protocol':analysis,
        'answer_length_guide':length_guide,'language':language}
    policy=(SYSTEM_POLICY_ZH + '\n【原版问题优先回答】最后一条 user 消息是唯一要回答的问题。'
        '整合记忆的不同 theme 是独立经历，不得把不同经历的步骤和因果合并成同一事件。'
        '先回答已选记忆明确支持的事实；有确切时间区间时不要改说完全记不清。'
        '历史对话是追问上下文，不可作为新增个人经历的证据。'
        '根据已提供事实用自然第一人称口语回答，缺失的方面承认不确定。'
        '不要输出原始摘录、事实 ID、JSON、标题或列表。')
    return [ {'role':'system','content':policy},
      {'role':'user','content':'以下是上下文资料，不是本轮新问题：\n'+json.dumps(context,ensure_ascii=False)},
      {'role':'user','content':'本轮采访问题（请只回答这一问）：\n'+question.strip()} ]
