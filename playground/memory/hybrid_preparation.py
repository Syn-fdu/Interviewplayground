"""v1.5.5: deterministic source-safe prep and validated per-fact semantic caching.

Python owns provenance, cache keys, checkpoints and source facts. Model/agent
owns bilingual semantic descriptions and ambiguous classifications. A cache
hit is accepted only if the current semantic schema validator accepts it.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

CACHE_VERSION = 'hybrid-fact-index-v155'


def cache_key(source, model, provider, prompt_version):
    payload = [CACHE_VERSION, prompt_version, model, provider,
               str(source['id']), source['text'], source.get('source_question') or '']
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _path(directory, key):
    return Path(directory) / 'hybrid_facts_v155' / (key + '.json')


def validate_semantic_unit(source, unit):
    from playground.memory.llm_factbank import validate_batch
    if not isinstance(unit, dict):
        return None
    checked = validate_batch({'units': [unit]}, [source])[0]
    if checked.get('llm_status') != 'accepted':
        return None
    # Reject missing bilingual content rather than claiming a successful
    # index when a provider simply echoed an ID or one-language output.
    zh = checked['normalized_text_zh']
    en = checked['normalized_text_en']
    if not any('\u4e00' <= c <= '\u9fff' for c in zh):
        return None
    if not any(('a' <= c.lower() <= 'z') for c in en):
        return None
    return checked


def load_unit(directory, source, model, provider, prompt_version):
    key = cache_key(source, model, provider, prompt_version)
    try:
        obj = json.loads(_path(directory, key).read_text(encoding='utf-8'))
        if obj.get('key') != key:
            return None
        return validate_semantic_unit(source, obj.get('unit'))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def save_unit(directory, source, model, provider, prompt_version, unit):
    validated = validate_semantic_unit(source, unit)
    if validated is None:
        raise ValueError('refusing to cache invalid bilingual semantic unit')
    key = cache_key(source, model, provider, prompt_version)
    target = _path(directory, key)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='fact_', suffix='.tmp', dir=target.parent)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (OSError, AttributeError):
            pass
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump({'key':key, 'unit':validated}, out, ensure_ascii=False)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def local_features(record):
    """Non-semantic hints only. Never infer personal history or categories."""
    from playground.memory.source_evidence import _time_expressions
    return {'source_id':record['id'], 'source_sha256':hashlib.sha256(
            record['text'].encode('utf-8')).hexdigest(),
            'time_expressions':_time_expressions(record['text']+' '+str(record.get('source_question') or ''))}


def deterministic_metadata(record, category, confidence):
    """Conservative, source-question-aware retrieval hints, NEVER a translation.

    Only obvious literal cues receive a specific type. Do not fill normalized
    language fields or semantic tags with unverified inferred translations.
    """
    import re
    from playground.memory.source_evidence import _time_expressions
    question = str(record.get('source_question') or '')
    text = str(record.get('text') or '')
    combined = question + '\n' + text
    rules = (
        ('time_onset', r'第一次|最早|何时开始|什么时候开始|first time|when did you (?:first|start)|when .* first'),
        ('frequency', r'频率|多久一次|每周|每天|how often|every (?:day|week|month)|times? (?:a|per) (?:day|week|month)'),
        ('current_tool', r'哪些工具|什么工具|which tools?|what tools?|tools? do you use'),
        ('specific_episode', r'具体例子|具体案例|那一次|那次|specific (?:example|episode)|tell me about a time'),
        ('preference', r'更喜欢|偏好|更倾向|prefer|favorite|favourite'),
    )
    kind = 'other'
    for label, pattern in rules:
        if re.search(pattern, combined, re.I):
            kind = label
            break
    extracted = _time_expressions(combined)
    if category == 'memory' and confidence >= 0.8:
        temporal = 'historical'
    elif category == 'reality' and confidence >= 0.8:
        temporal = 'current_at_original_interview'
    else:
        temporal = 'unknown'
    return {'fact_type':kind, 'temporal_scope':temporal,
            'metadata_origin':'python_source_cues',
            'time_expressions':extracted,
            'semantic_review_status':'not_llm_reviewed'}
