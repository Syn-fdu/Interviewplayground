"""Post-preparation memory access for arbitrary completed integrated cards.

Never edits source cards or preparation caches. Operates on source excerpts
returned by question-first retrieval and only creates verbatim sentence excerpts.
No invented 'gists', numeric substitutions, or new LLM calls.
"""
from __future__ import annotations
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
