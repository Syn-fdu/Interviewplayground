"""v1.6.1 interviewer preparation. Public original Outline only; never takes respondent records.

The project v1.5.9 PDF pipeline accepts only a PDF path; this adapter feeds its
existing `Outline` model and `enrich_slots()` algorithm from the normalized JSON.
"""
from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
import io
import json
import threading
import zipfile
from playground.interview_agent.outline.models import Outline
from playground.interview_agent.outline.preprocess import enrich_slots
from playground.research.data_intake import discover_sources, candidate_id
from playground.research.group_interviewer_bridge import _OUTLINES, _LOCK, remote_configuration
from playground.interview_agent.llm.openai_compat import OpenAICompatClient

_DETAILS={}
# Public interviewer Outline only. No respondent answers, MemoryBank or API keys are cached.
_OUTLINE_CACHE_DIR = Path(__file__).resolve().parents[2] / '.runtime_cache' / 'interviewer_outlines'
_OUTLINE_CACHE_VERSION = 'native-slot-cache-v1'

def _content_cache_key(outline_data, provider, config):
    """Content-addressed key: identical Outline and LLM configuration reuse Slots."""
    agent_source = Path(__file__).resolve().parents[2] / 'playground' / 'interview_agent' / 'outline' / 'preprocess.py'
    agent_digest = hashlib.sha256(agent_source.read_bytes()).hexdigest()
    material = {'schema': _OUTLINE_CACHE_VERSION, 'outline': outline_data,
                'provider': provider, 'model': config.model,
                'base_url': config.base_url, 'preprocess_sha256': agent_digest}
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def _read_content_cache(digest):
    path = _OUTLINE_CACHE_DIR / (digest + '.json')
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or raw.get('cache_version') != _OUTLINE_CACHE_VERSION:
            return None
        data = raw.get('outline')
        if not isinstance(data, dict):
            return None
        candidate = Outline.from_dict(data)
        questions = candidate.questions()
        if not questions or any(not q.required_slots or all(
                slot.source == 'fallback' for slot in q.required_slots) for q in questions):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None

def _write_content_cache(digest, outline_data):
    _OUTLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _OUTLINE_CACHE_DIR / (digest + '.json')
    payload = {'cache_version': _OUTLINE_CACHE_VERSION, 'outline': outline_data}
    tmp = _OUTLINE_CACHE_DIR / (digest + '.' + os.urandom(6).hex() + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, separators=(',', ':'))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

# Prevent two preparation workers from submitting the same Outline concurrently.
_CONTENT_CACHE_LOCK = threading.RLock()

def _original_outline(data:bytes, dataset:str, candidate:str):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root=discover_sources(z.namelist()).get(dataset)
        if not root or "interview_outlines" not in root["files"]:
            raise ValueError('当前数据集没有原始 Outline 文件')
        rows=json.loads(z.read(root['files']['interview_outlines'])).get('records') or []
        for row in rows:
            if candidate_id(dataset,row.get('dialogue_id') or row.get('outline_id'))==candidate:
                return row
    raise ValueError('找不到该场景对应的原始 Outline')

def _question_text(value):
    """Normalize the V3 and legacy question representations without any answer fields."""
    if isinstance(value, dict):
        value = value.get("text") or value.get("source_text") or value.get("verbatim_interviewer_text")
    return " ".join(value.split()).strip()[:800] if isinstance(value, str) else ""


def _validated_topics(source):
    """Preserve interviewer-native fields; never strip slots from source questions.

    Rich source Outlines can contain IDs, goals, probes, branch rules and real
    required_slots. Plain legacy question strings are also supported unchanged.
    """
    topics=[]
    for item in source.get("topics") or []:
        if not isinstance(item,dict):
            continue
        questions=[]
        for q in item.get("questions") or []:
            text=_question_text(q)
            if not text:continue
            if isinstance(q,dict):
                # Only known public interviewer schema fields. Do not copy
                # respondent/gold/source-answer fields into the interviewer.
                allowed=("id","goal","intent","probes","required_slots",
                         "optional_deepening_slots","branch_rules")
                question={"text":text,**{key:copy.deepcopy(q[key]) for key in allowed if key in q}}
            else:
                question={"text":text}
            questions.append(question)
        if questions:
            topics.append({"id":item.get("id") or "T%d"%(len(topics)+1),
                           "title":str(item.get("title") or "Interview topic"),
                           "questions":questions})
    if topics:
        return topics,"source_outline_topics"
    prompts=[]
    if source.get("original_interview_outline_provided") is True:
        for row in source.get("original_guide_question_preview") or []:
            q=_question_text(row)
            if q:prompts.append(q)
    if prompts:
        return [{"title":"Original interview guide",
                 "questions":[{"text":p} for p in dict.fromkeys(prompts)]}], "legacy_public_guide_questions"
    return [],"none"


def _reconstruct_interviewer_topics(z, info, source, dataset, candidate):
    """Compatibility for an old uploaded dataset with empty Outline.

    Only take explicit interviewer-role text. Do not include any respondent turn or
    answer/gold/reference field. This mirrors the v3.4 source reconstruction policy.
    """
    path = info.get("files", {}).get("interview_records")
    if not path:
        return []
    rows = json.loads(z.read(path)).get("records") or []
    record = next((r for r in rows if
                   candidate_id(dataset, r.get("dialogue_id")) == candidate), None)
    if not record:
        return []
    import re
    starts = re.compile(
        r"^(?:what|when|where|why|how|who|which|can|could|would|will|do|does|did|"
        r"is|are|was|were|have|has|had|should|tell me|describe|please describe)\b",
        re.IGNORECASE,
    )
    questions, seen = [], set()
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict) or turn.get("role") != "interviewer":
            continue
        text = _question_text(turn.get("text"))
        if not text:
            continue
        # Keep only direct interviewer questions, trimming leading remarks or
        # paraphrases of what the respondent said whenever possible.
        segments = re.split(r"(?<=[?？])\\s*", text)
        for part in segments:
            q = part.strip()
            if not 12 <= len(q) <= 280:
                continue
            if ("?" not in q and "？" not in q and not starts.match(q)
                    and not re.search(r"(什么|如何|怎么|为什么|能否|请描述|请介绍|谈谈)", q)):
                continue
            # Only a question, not a whole turn with possible respondent recap.
            if "?" in q and len(q)>160:
                q = q[q.rfind(".",0,q.find("?"))+1:].strip()
            key = re.sub(r"\\W+","",q.casefold())
            if key in seen:
                continue
            seen.add(key)
            questions.append({"text":q})
            if len(questions)>=28:
                break
        if len(questions)>=28:
            break
    if not questions:
        return []
    title = str(source.get("source_topic_label") or
                record.get("topic") or dataset.replace("_"," ").title())
    return [{"title":title[:120],"questions":questions}]


def _load_source_outline(source_bytes, dataset, candidate):
    """Return the matching outline and the canonical ZIP index for safe recovery."""
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as z:
        info = discover_sources(z.namelist()).get(dataset)
        if not info or "interview_outlines" not in info["files"]:
            raise ValueError("当前数据集未提供 interview_outlines.json，请更新至 Dataset v3.4")
        rows = json.loads(z.read(info["files"]["interview_outlines"])).get("records") or []
        row = next((r for r in rows if
                    candidate_id(dataset,r.get("dialogue_id") or r.get("outline_id")) == candidate), None)
        if row is None:
            raise ValueError("找不到场景对应的 Outline：dialogue_id / outline_id 关联失败")
        topics, method = _validated_topics(row)
        if not topics:
            topics = _reconstruct_interviewer_topics(z,info,row,dataset,candidate)
            if topics:
                method="recovered_from_interviewer_turns"
        return row, topics, method


def prepare_public_outline(source_bytes,scenario,revision,provider='shubi',llm=None,outline_level=None):
    provenance=scenario.get('provenance') or {}
    dataset=provenance.get('dataset')
    candidate=provenance.get('candidate_id')
    fingerprint=provenance.get('source_fingerprint_sha256')
    if not all(isinstance(v,str) and v for v in (dataset,candidate,fingerprint)):
        raise ValueError('场景缺少原始资料关联信息')
    # In-memory binding belongs to the active scenario; the durable cache is keyed
    # by actual interviewer Outline content and model configuration, not scenario ID.
    from playground.common.model_services import get as get_model_config
    effective_cfg=get_model_config(provider,"interviewer",False)
    level=outline_level or provenance.get('guide_level') or 'level1'
    if level not in ('level0','level1','level2'):raise ValueError('Invalid selected guide level')
    key=(fingerprint,dataset,candidate,revision,level)
    with _LOCK:
        cached=copy.deepcopy(_DETAILS.get(key))
        if (key in _OUTLINES and cached
                and cached.get("preparation_method")=="native_llm_slot_agent_v1"
                and cached.get("provider")==provider
                and cached.get("interviewer_model")==effective_cfg.model
                and cached.get("interviewer_base_url")==effective_cfg.base_url
                and not cached.get("native_fallback_question_ids")):
            return {'ok':True,'cached':True,**cached}
    from playground.research.graded_guides import match, question_view, _dataset
    from playground.research.outline_levels import has_refined_guide, build_native_outline
    graded_row=match(dataset,candidate,provenance.get('guide_category'))
    if (provenance.get('graded_guide_status')=="matched" and
            level not in (provenance.get('available_guide_levels') or [])):
        raise ValueError("This level was not bound to the selected scenario")
    projection=(scenario.get("public") or {}).get("interview_guide",{}).get("graded_guide_source") or {}
    if projection.get("origin")=="outline_levels" or (
            not projection and has_refined_guide(dataset)):
        native=build_native_outline(dataset,level)
        outline=Outline.from_dict(native)
        topics=native["topics"]
        source={"source_topic_label":native["title"],"goal":native.get("objective","")}
        question_source="verified_dataset_local_outline_levels_"+level
    elif graded_row is not None:
        topics=question_view(graded_row,level)
        source={'source_topic_label':graded_row.get('category') or dataset,'goal':[]}
        question_source='verified_user_supplied_graded_guide_'+level
    elif _dataset(dataset) is not None:
        raise ValueError('No matching graded guide for this interview; cross-interview fallback is prohibited')
    else:
        source,topics,question_source=_load_source_outline(source_bytes,dataset,candidate)
    if not topics:
        raise ValueError(
            "此场景的 Outline 为空，且原始 Record 中未发现可恢复的采访者提问。"
            "请导入 Dataset v3.4；若已使用 v3.4，请重新导入数据包并创建场景。"
        )
    goals=source.get('goal') or source.get('interview_goal_original') or []
    if isinstance(goals,str):
        goals=[goals]
    title=str(source.get('source_topic_label') or
              next((t['title'] for t in topics if t.get('title')),dataset))
    outline=Outline.from_dict({
        'title':title,
        'objective':'；'.join(str(x) for x in goals if isinstance(x,str) and x.strip()),
        'topics':topics,
    })
    if not outline.questions():raise ValueError('Outline 结构无可用问题')
    if provider not in ("maas","shubi","local"):
        raise ValueError("无效的采访者提供商")

    # Serialize same-content workers. Persistent entries survive service restarts,
    # while the native per-scenario _OUTLINES remains the only interview execution input.
    content_key=_content_cache_key(outline.to_dict(), provider, effective_cfg)
    with _CONTENT_CACHE_LOCK:
        saved=_read_content_cache(content_key)
        if saved is not None:
            outline=Outline.from_dict(copy.deepcopy(saved))
            enriched=len(outline.questions())
            reused=True
        else:
            # The native Interviewer requires real Slot enrichment; no fallback
            # is promoted into a successful cache entry.
            if llm is None:
                llm=OpenAICompatClient(remote_configuration(provider),retries=2)
            enriched=enrich_slots(outline,llm,batch=6)
            reused=False
        fallback=[q.id for q in outline.questions()
                  if not q.required_slots or all(slot.source=="fallback" for slot in q.required_slots)]
        if fallback:
            raise ValueError(
                "采访员 Slot Agent 预处理未完成：%d/%d 道题仍为 fallback。"
                "示例题号：%s。未生成合格的 outline.json；请检查模型 JSON 输出或 API。" %
                (len(fallback),len(outline.questions()),",".join(fallback[:5])))
        data=outline.to_dict()
        if not reused:
            try:
                _write_content_cache(content_key, data)
            except OSError:
                # Disk caching is an optimization; successful native preparation
                # must remain usable even on read-only installations.
                pass
    source_slots=len(outline.questions())
    benchmark_slot_eligible=True
    details={'scenario_id':scenario['scenario_id'], 'revision':revision,
             'source_fingerprint_sha256':fingerprint,
             'source_dataset':dataset,'source_candidate_id':candidate,'guide_level':level,
             'outline_level':level,
             'provider':provider,
             'interviewer_model':effective_cfg.model,
             'interviewer_base_url':effective_cfg.base_url,
             'question_count':len(outline.questions()),
             'topics':len(outline.topics),'enriched_count':enriched,
             'ready':True, 'schema':'native_interviewer_agent_outline_v1',
             'preparation_method':'native_llm_slot_agent_v1',
             'outline_cache_hit':reused, 'outline_cache_key':content_key[:12],
             'benchmark_slot_eligible':benchmark_slot_eligible,
             'native_fallback_question_ids':fallback,
             'source_slot_questions':source_slots,
             'source':'normalized_original_outline_v3_4',
             'question_source':question_source,
             'requires_source_review':True,
             'review_status':'model_processed_not_human_reviewed'}
    with _LOCK:
        _OUTLINES[key]=copy.deepcopy(data)
        _DETAILS[key]=copy.deepcopy(details)
    return {'ok':True,'cached':reused,**details}

def read_prepared(scenario,revision,outline_level=None):
    p=scenario.get('provenance') or {}
    key=(p.get('source_fingerprint_sha256'),p.get('dataset'),p.get('candidate_id'),revision,outline_level or p.get('guide_level') or 'level1')
    with _LOCK:
        entry=copy.deepcopy(_OUTLINES.get(key))
        details=copy.deepcopy(_DETAILS.get(key))
    return {'ok':True,'ready':bool(entry),'metadata':details,'agent_outline':entry}

def clear():
    with _LOCK:
        _OUTLINES.clear()
        _DETAILS.clear()
