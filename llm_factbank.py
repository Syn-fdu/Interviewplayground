"""LLM-led FactBank preparation metadata with immutable original-source evidence.

LLM output is a semantic index only. Original quotes and source questions are
always materialized from the authorized source bundle by source ID.
"""
from __future__ import annotations
import hashlib
import json
import re

SCHEMA_VERSION = "llm_factbank_index.v154"
PROMPT_VERSION = "llm_factbank_prepare.strict_wire_v4"
FACT_TYPES = {
    "time_onset", "current_tool", "frequency", "method", "specific_episode",
    "reason", "impact", "preference", "challenge", "team_context", "other",
}
TEMPORAL_SCOPES = {"historical", "current_at_original_interview", "unknown"}
IMPORTANCE = {"key", "supporting", "background"}
SEMANTIC_TAGS = {
    "ai_tools", "coding", "programming", "tool_integration", "frequency",
    "onset_time", "specific_episode", "method", "reason", "impact",
    "preference", "challenge", "teamwork", "debugging", "testing",
    "documentation", "learning", "privacy", "recording", "workflow",
    "current_state", "historical_event",
}


def _hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def response_schema():
    return {
        "type": "object",
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "normalized_text_zh": {"type": "string"},
                        "normalized_text_en": {"type": "string"},
                        "fact_category": {"type": "string", "enum": ["memory", "reality"]},
                        "fact_type": {"type": "string", "enum": sorted(FACT_TYPES)},
                        "temporal_scope": {"type": "string", "enum": sorted(TEMPORAL_SCOPES)},
                        "semantic_tags": {"type": "array", "items": {"type": "string", "enum": sorted(SEMANTIC_TAGS)}},
                        "importance": {"type": "string", "enum": sorted(IMPORTANCE)},
                        "confidence": {"type": "number"},
                        "review_required": {"type": "boolean"},
                        "classification_reason": {"type": "string"},
                    },
                    "required": ["source_id", "normalized_text_zh", "normalized_text_en",
                                 "fact_category", "fact_type", "temporal_scope",
                                 "semantic_tags", "importance", "confidence",
                                 "review_required", "classification_reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["units"],
        "additionalProperties": False,
    }


def build_prompt(batch):
    instruction = (
        "You prepare a source-grounded interview FactBank. For each input fact, "
        "understand the source question and answer together, then classify and "
        "create short bilingual search descriptions. The source question is context, "
        "not proof of an event. Preserve uncertainty and distinguish past events "
        "from the respondent's state at the original interview. Never convert relative "
        "time into a calendar date. Do not merge different events. Do not add motives, "
        "causes, people, dates, tools, or actions absent from the source. "
        "normalized_text_zh/en are retrieval hints only, not replacement evidence. "
        "Return EXACTLY one unit for EACH input source_id, with an exact matching source_id. "
        "Be concise: normalized_text_zh and normalized_text_en should each be one short sentence; "
        "classification_reason should be one short clause. No lengthy explanations. "
        "The top level JSON must be {\"units\": [ ... ]}. Each unit MUST contain: "
        "source_id, normalized_text_zh, normalized_text_en, fact_category, fact_type, "
        "temporal_scope, semantic_tags, importance, confidence (0 to 1), review_required "
        "(boolean), classification_reason. Use only enum values defined in the JSON schema. "
        "If a fact is ambiguous, set "
        "temporal_scope=unknown, lower confidence, and review_required=true. "
        "Classify historical personal episodes as memory and state/habit at the "
        "original interview as reality. Output only the required JSON object."
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({"source_facts": batch}, ensure_ascii=False)},
    ]


def _unresolved(record, reason="missing_or_invalid_llm_row"):
    return {
        "source_id": str(record["id"]),
        "normalized_text_zh": "",
        "normalized_text_en": "",
        "fact_category": "memory",  # fail closed: do not make an unclassified fact more accessible
        "fact_type": "other",
        "temporal_scope": "unknown",
        "semantic_tags": [],
        "importance": "background",
        "confidence": 0.0,
        "review_required": True,
        "classification_reason": reason,
        "llm_status": "unresolved",
    }


def validate_batch(result, batch):
    """Validate IDs and metadata; never trust model-authored source quotations."""
    # Accept common provider wrappers, but always verify exact source IDs.
    rows = None
    if isinstance(result, dict):
        rows = result.get("units")
        if not isinstance(rows, list):
            for name in ("facts", "results", "items", "classifications"):
                if isinstance(result.get(name), list):
                    rows = result[name]
                    break
        if not isinstance(rows, list) and isinstance(result.get("data"), dict):
            rows = result["data"].get("units")
    if not isinstance(rows, list):
        rows = []
    allowed = {str(x["id"]): x for x in batch}
    grouped = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or row.get("id") or row.get("fact_id") or "")
        if source_id in allowed:
            grouped.setdefault(source_id, []).append(row)
    output = []
    for source_id, original in allowed.items():
        matches = grouped.get(source_id, [])
        if len(matches) != 1:
            output.append(_unresolved(original, "missing_or_duplicate_llm_row"))
            continue
        row = matches[0]
        zh = str(row.get("normalized_text_zh") or "").strip()[:500]
        en = str(row.get("normalized_text_en") or "").strip()[:500]
        category = row.get("fact_category")
        fact_type = row.get("fact_type")
        temporal = row.get("temporal_scope")
        importance = row.get("importance")
        tags = row.get("semantic_tags")
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        review = bool(row.get("review_required", True))
        reason = str(row.get("classification_reason") or "")[:240]
        if not zh or not en:
            output.append(_unresolved(original, "empty_semantic_index_row"))
            continue
        if (category not in ("memory", "reality") or fact_type not in FACT_TYPES
                or temporal not in TEMPORAL_SCOPES or importance not in IMPORTANCE
                or not isinstance(tags, list) or any(x not in SEMANTIC_TAGS for x in tags)
                or not isinstance(row.get("review_required"), bool)
                or not isinstance(row.get("confidence"), (int,float))
                or not 0 <= row["confidence"] <= 1 or not reason.strip()):
            output.append(_unresolved(original, "invalid_semantic_fields"))
            continue
        if temporal == "historical" and category == "reality":
            category = "memory"
            review = True
            reason = (reason + " temporal_category_conflict").strip()
        if temporal == "current_at_original_interview" and category == "memory":
            review = True
            reason = (reason + " current_state_review").strip()
        if confidence < 0.55:
            review = True
        output.append({
            "source_id": source_id,
            "normalized_text_zh": zh,
            "normalized_text_en": en,
            "fact_category": category,
            "fact_type": fact_type,
            "temporal_scope": temporal,
            "semantic_tags": sorted(set(tags)),
            "importance": importance,
            "confidence": confidence,
            "review_required": review,
            "classification_reason": reason,
            "llm_status": "accepted",
        })
    return output


def apply_classification(bundle, index):
    """Apply LLM categories to authorized original facts without replacing quotes."""
    from agent_memory_export import eligible_source_facts
    originals = eligible_source_facts(bundle)
    mapping = {str(x.get("source_id")): x for x in (index or []) if isinstance(x, dict)}
    out = []
    for source in originals:
        row = mapping.get(str(source["id"]))
        if row is None:
            row = _unresolved(source, "missing_from_prepared_index")
        item = dict(source)
        item.update({
            "fact_category": row.get("fact_category", "memory"),
            "classification_reason": row.get("classification_reason", "llm_unresolved"),
            "classification_confidence": row.get("confidence", 0.0),
            "classification_review_required": bool(row.get("review_required", True)),
            "provenance_status": "source_derived",
            "time_reference": "original_interview",
            "llm_fact_type": row.get("fact_type", "other"),
            "llm_temporal_scope": row.get("temporal_scope", "unknown"),
            "llm_semantic_tags": row.get("semantic_tags", []),
            "llm_normalized_text_zh": row.get("normalized_text_zh", ""),
            "llm_normalized_text_en": row.get("normalized_text_en", ""),
            "llm_importance": row.get("importance", "background"),
            "llm_status": row.get("llm_status", "unresolved"),
        })
        out.append(item)
    return out


def summary(index):
    rows = list(index or [])
    return {
        "schema": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "total": len(rows),
        "memory_count": sum(x.get("fact_category") == "memory" for x in rows),
        "reality_count": sum(x.get("fact_category") == "reality" for x in rows),
        "unresolved_count": sum(x.get("llm_status") != "accepted" for x in rows),
        "review_count": sum(bool(x.get("review_required")) for x in rows),
        "importance_counts": {k: sum(x.get("importance") == k for x in rows)
                               for k in sorted(IMPORTANCE)},
    }


def signature(index):
    return _hash(json.dumps(index or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# One-shot STRICT wire protocol: enforce field types and enum codes in the
# upstream request, not merely in a post-hoc prompt/validator. A numeric
# codebook keeps a large response short while eliminating free-text enums.
# All canonical FactBank keys and original source evidence remain unchanged.
WIRE_FACT_TYPES = tuple(sorted(FACT_TYPES))
WIRE_TEMPORALS = tuple(sorted(TEMPORAL_SCOPES))
WIRE_TAGS = tuple(sorted(SEMANTIC_TAGS))
WIRE_IMPORTANCE = tuple(sorted(IMPORTANCE))
WIRE_CATEGORIES = ('memory', 'reality')
WIRE_PROTOCOL = 'factbank_strict_wire.v4'


def compact_schema(n=None):
    """Provider-enforceable JSON Schema for the exact compact row structure."""
    row_properties = {
        'i': {'type': 'integer', 'minimum': 0},
        'z': {'type': 'string'},  # Chinese semantic search hint
        'e': {'type': 'string'},  # English semantic search hint
        'c': {'type': 'integer', 'enum': list(range(len(WIRE_CATEGORIES)))},
        't': {'type': 'integer', 'enum': list(range(len(WIRE_FACT_TYPES)))},
        's': {'type': 'integer', 'enum': list(range(len(WIRE_TEMPORALS)))},
        'g': {'type': 'array', 'items': {'type': 'integer', 'enum': list(range(len(WIRE_TAGS)))}},
        'p': {'type': 'integer', 'enum': list(range(len(WIRE_IMPORTANCE)))},
        'f': {'type': 'number'},
        'v': {'type': 'boolean'},
        'r': {'type': 'string'},
    }
    fields=list(row_properties)
    schema={'type':'object','properties':{
        'rows':{'type':'array','items':{
            'type':'object','properties':row_properties,
            'required':fields,'additionalProperties':False,
        }}},'required':['rows'],'additionalProperties':False}
    # Some OpenAI-compatible backends accept only the basic strict schema
    # subset, so we deliberately avoid prefixItems and conditional schemas.
    return schema


def compact_prompt(records):
    inputs=[{'i':i,'source_id':r['id'],'q':r.get('source_question') or '',
             'text':r['text']} for i,r in enumerate(records)]
    codebooks={
        'c':{i:x for i,x in enumerate(WIRE_CATEGORIES)},
        't':{i:x for i,x in enumerate(WIRE_FACT_TYPES)},
        's':{i:x for i,x in enumerate(WIRE_TEMPORALS)},
        'g':{i:x for i,x in enumerate(WIRE_TAGS)},
        'p':{i:x for i,x in enumerate(WIRE_IMPORTANCE)},
    }
    system=(
        'Prepare an evidence-grounded bilingual semantic index for every input fact. '
        'The server enforces a JSON Schema. Return ONLY {"rows":[{...}]} with exactly '
        'one object for each input i, in ascending order 0..N-1. '
        'Each object MUST have exactly these typed fields: '
        'i=integer input index; z=short Chinese source-faithful description; '
        'e=short English source-faithful description; c=integer category code; '
        't=integer fact type code; s=integer temporal scope code; '
        'g=array of integer semantic tag codes; p=integer importance code; '
        'f=number between 0 and 1; v=JSON boolean review-needed; '
        'r=short source-grounded classification reason. '
        'NEVER return free-text for c,t,s,p or g. Example: fact_type=3 is INVALID; '
        'write the actual JSON property t with its NUMERIC code. '
        'These are the ONLY permitted integer meanings: '
        +json.dumps(codebooks,ensure_ascii=False,separators=(',',':'))+'. '
        'Use the question only to interpret the quoted source fact, not to invent an event. '
        'Personal events occurring in the past are memory; habits, opinions and '
        'states at the original interview are reality. When timing is ambiguous, '
        'choose unknown temporal code, f<=0.54, v=true. '
        'Do not invent dates, people, motivations or tools. No explanations outside JSON. '
        'Write short but meaningful bilingual retrieval hints; never rewrite source evidence.'
    )
    return [{'role':'system','content':system},
            {'role':'user','content':json.dumps({'N':len(inputs),'facts':inputs},ensure_ascii=False)}]


def expand_compact_rows(result, records):
    """Check provider-constrained response and expand to original FactBank schema.

    Reject missing / duplicate identities and invalid semantic evidence. Do not
    infer categories or fill missing bilingual descriptions after the request.
    """
    rows=result.get('rows') if isinstance(result,dict) else None
    if not isinstance(rows,list):
        return [],'模型未返回 rows 数组'
    if len(rows)!=len(records):
        return [],f'模型返回 {len(rows)}/{len(records)} 条记录'
    seen=set();converted=[];invalid={}
    for expected,row in enumerate(rows):
        if not isinstance(row,dict):
            return [],f'第 {expected} 条不是结构化对象'
        if set(row)!=set(('i','z','e','c','t','s','g','p','f','v','r')):
            return [],f'第 {expected} 条存在缺失或额外字段'
        idx=row['i']
        if type(idx) is not int or idx<0 or idx>=len(records) or idx in seen:
            return [],'模型返回无效或重复的事实序号'
        seen.add(idx)
        if idx!=expected:
            return [],'模型返回的事实序号与原文顺序不一致'
        def coded(value,table):
            return table[value] if type(value) is int and 0<=value<len(table) else None
        category=coded(row['c'],WIRE_CATEGORIES)
        fact_type=coded(row['t'],WIRE_FACT_TYPES)
        temporal=coded(row['s'],WIRE_TEMPORALS)
        importance=coded(row['p'],WIRE_IMPORTANCE)
        tags=row['g']
        tag_values=([coded(tag,WIRE_TAGS) for tag in tags] if isinstance(tags,list) else None)
        zh=row['z'].strip() if isinstance(row['z'],str) else ''
        en=row['e'].strip() if isinstance(row['e'],str) else ''
        reason=row['r'].strip() if isinstance(row['r'],str) else ''
        confidence=row['f']
        if (not zh or not any('\u4e00'<=ch<='\u9fff' for ch in zh)
                or not en or not any('a'<=ch.lower()<='z' for ch in en)
                or not reason or not category or not fact_type or not temporal
                or not importance or tag_values is None or None in tag_values
                or type(confidence) not in (int,float) or not 0<=confidence<=1
                or type(row['v']) is not bool):
            invalid['invalid_semantic_fields']=invalid.get('invalid_semantic_fields',0)+1
            continue
        converted.append({'source_id':str(records[idx]['id']),
            'normalized_text_zh':zh,'normalized_text_en':en,
            'fact_category':category,'fact_type':fact_type,
            'temporal_scope':temporal,'semantic_tags':sorted(set(tag_values)),
            'importance':importance,'confidence':float(confidence),
            'review_required':row['v'],'classification_reason':reason})
    if invalid:
        return [],'语义字段未通过严格协议（%s）；未写入成功缓存'%str(invalid)
    return converted,None


