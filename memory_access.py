"""Stable post-preparation, fact-level accessibility for the synthetic lab.

The master fact fixture is immutable. This module never touches Agent preparation
or saved memory. For real Agent summaries that lack annotated atomic facts,
retain the original question-first retrieval rather than heuristically rewriting.
"""
from __future__ import annotations
from trait_postprocessing import memory_access_level, memory_budget

# Verified, shortened *gist* from the synthetic source material (no invented facts).
# Full text and exact numbers remain in the fixture, not modified here.
GISTS={
    'L01':'我负责产品协调与用户反馈。',
    'L02':'去年我负责过一个改进试用用户注册流程的项目。',
    'L03':'项目是由一个小团队共同完成的。',
    'L04':'项目中访谈过试用用户，也调整过表单流程。',
    'L05':'注册完成率后来有所提高，但不能据此断定只有界面改动起作用。',
    'L06':'项目中期进行过评审和可用性测试。',
    'L07':'我们讨论过手机号输入栏，最后做了部分调整。',
    'L08':'我曾漏记部分访谈反馈，后来进行了补充。',
    'L09':'那次遗漏让我有些难堪和担心，我向团队说明过。',
    'L10':'之后我改进了访谈记录核对方法。',
    'L11':'项目繁忙时我有过加班压力，也调整过工作分工。',
    'L12':'有些自身经历可以谈，但同事和客户的私人资料不对外透露。',
    'L13':'项目没有继续追踪长期效果的数据。',
    'L14':'团队评估过工作量与技术边界。',
}

def choose_synthetic_facts(facts, question_fact_ids, memory):
    """Choose complete fact units, returning explicit visible facts and trace.

    At low memory, preserve a topic-specific gist of *every* relevant fact;
    never arbitrarily discard the only event behind a question. At high memory,
    keep the full original fact text. Mid level uses a mix of full high-salience
    fact text and gists for the rest. All conditions use deterministic rules.
    """
    level=memory_access_level(memory)
    ids=list(dict.fromkeys(question_fact_ids))
    if level=='precise_details':
        chosen=[(fid,facts[fid]['answer']) for fid in ids]
    elif level=='key_facts':
        chosen=[(fid,facts[fid]['answer'] if i==0 else GISTS.get(fid,facts[fid]['answer']))
                for i,fid in enumerate(ids)]
    else:
        chosen=[(fid,GISTS.get(fid,facts[fid]['answer'])) for fid in ids]
    # The complete units are intentionally not cut mid-sentence.
    return [f'{fid}：{content}' for fid,content in chosen],level


# General source-grounded integrated-memory policy (formerly a separate module).
"""Post-preparation memory access for arbitrary completed integrated cards.

Never edits source cards or preparation caches. Operates on source excerpts
returned by question-first retrieval and only creates verbatim sentence excerpts.
No invented 'gists', numeric substitutions, or new LLM calls.
"""

import re
from trait_postprocessing import memory_access_level

_SENTENCE = re.compile(r'(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[^\d])')


def _verbatim_sentences(source: str, number: int) -> str:
    raw = str(source or '').strip()
    if not raw:
        return ''
    sentences = [s.strip() for s in _SENTENCE.split(raw) if s.strip()]
    # Only cut at explicit sentence boundaries. If no natural boundary exists,
    # return the entire verbatim unit rather than a misleading mid-sentence cut.
    if len(sentences) <= number:
        return raw
    prefix = raw[:raw.find(sentences[number])].rstrip() if sentences[number] in raw else ''.join(sentences[:number])
    return prefix if prefix and raw.startswith(prefix) else raw


def tier_integrated_memory(selected, value):
    """Render selected excerpts at a deterministic accessibility tier.

    The original candidate retrieval must already have finished. Low memory
    retains one top-ranked relevant unit, mid two, high three. Source excerpts
    are either verbatim or sentence-boundary prefixes of source excerpts.
    """
    tier = memory_access_level(value)
    count = {'gist': 1, 'key_facts': 2, 'precise_details': 3}[tier]
    sentences = {'gist': 1, 'key_facts': 2, 'precise_details': 999999}[tier]
    out = []
    for group in selected[:count]:
        if not isinstance(group, dict):
            continue
        src = group.get('facts', [])
        if not isinstance(src, list) or not src:
            continue
        item = _verbatim_sentences(src[0], sentences)
        if item:
            out.append({'theme': str(group.get('theme') or ''), 'facts': [item]})
    return out, {
        'access_tier': tier,
        'candidate_groups': len(selected),
        'visible_groups': len(out),
        'visible_chars': sum(len(g['facts'][0]) for g in out),
        'precision_policy': 'verbatim_sentence_prefix_no_inference_v114',
        'preparation_unchanged': True,
    }
