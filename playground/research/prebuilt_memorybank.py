"""Load a versioned, dialogue-bound prebuilt bank without constructing facts from Records."""
from collections import Counter
from hashlib import sha256
import json
from playground.research.data_intake import IntakeError

SCHEMA = "memorybank.context.v1"
NORMALIZED = {"source_reviewed_example", "rule_expanded_unreviewed"}
NON_FACT = {"clarification_question_not_personal_fact", "superseded_preliminary_rating",
            "unresolved_category", "unclear_or_incomplete_source",
            "interaction_or_clarification_not_independent_fact",
            "acknowledgment_not_independent_fact", "original_interview_procedure"}

def read_selected(z, info, dataset, record):
    path=info["files"].get("memorybank")
    if not path:
        return None
    manifest=json.loads(z.read(info["manifest"]))
    raw=z.read(path)
    digest=sha256(raw).hexdigest()
    if digest != str((manifest.get("output_sha256") or {}).get("memorybank.json","")).lower():
        raise IntakeError("MemoryBank SHA-256 校验失败。")
    doc=json.loads(raw)
    if not isinstance(doc,dict) or doc.get("schema_version")!=SCHEMA or doc.get("dataset_id")!=dataset:
        raise IntakeError("MemoryBank schema 或 dataset_id 不匹配。")
    if doc.get("source_records_sha256") != (manifest.get("output_sha256") or {}).get("interview_records.json"):
        raise IntakeError("MemoryBank 与当前 Records 版本不一致。")
    rows=doc.get("records")
    if not isinstance(rows,list):
        raise IntakeError("MemoryBank records 缺失。")
    did=str(record["dialogue_id"])
    hits=[r for r in rows if isinstance(r,dict) and str(r.get("dialogue_id"))==did]
    if len(hits)!=1:
        raise IntakeError("MemoryBank 无法唯一关联该 dialogue_id；拒绝回退到 Records。")
    row=hits[0]
    for k in ("profile_id","outline_id"):
        if row.get(k)!=record.get(k):
            raise IntakeError("MemoryBank "+k+" 与原始记录不一致。")
    return row, digest, did in (doc.get("reviewed_dialogue_ids") or [])

def materialize(selected, record, name=None):
    from playground.research.scenario_builder import scrub
    row,digest,dialogue_reviewed=selected
    turns={t["turn_index"]:t for t in record["turns"]}
    expected={i for i,t in turns.items() if t.get("role")=="respondent"}
    contexts=row.get("context_turns")
    if not isinstance(contexts,list) or len(contexts)!=len(turns):
        raise IntakeError("MemoryBank 上下文不完整。")
    context_ids=set()
    for t in contexts:
        i=t.get("turn_index")
        if i in context_ids or i not in turns or t.get("text")!=turns[i].get("text") or t.get("role")!=turns[i].get("role"):
            raise IntakeError("MemoryBank 上下文与原始来源不一致。")
        context_ids.add(i)
    entries=row.get("entries")
    if not isinstance(entries,list):
        raise IntakeError("MemoryBank entries 缺失。")
    observed=set(); mids=set(); memories=[]; counts=Counter()
    for e in entries:
        i=e.get("source_turn_id");mid=e.get("memory_id")
        if i not in expected or i in observed or not isinstance(mid,str) or not mid or mid in mids:
            raise IntakeError("MemoryBank 存在重复或非法来源 ID。")
        if str(e.get("dialogue_id"))!=str(record["dialogue_id"]) or e.get("original_text")!=turns[i].get("text"):
            raise IntakeError("MemoryBank 原文或 dialogue_id 不匹配。")
        refs=e.get("source_turn_ids")
        if not isinstance(refs,list) or i not in refs or any(k not in turns for k in refs):
            raise IntakeError("MemoryBank 来源 Turn ID 无效。")
        observed.add(i);mids.add(mid)
        status=e.get("semantic_status","context_linked_unreviewed")
        kind=e.get("entry_kind","")
        raw=e["original_text"]
        if status in ("empty_source","acknowledgment_only") or kind in NON_FACT or not raw.strip():
            counts["excluded_nonfact"]+=1
            continue
        # Do not convert respondent questions into claims. Leave uncertain
        # answers available as their actual words, without generated assertions.
        if status not in NORMALIZED and raw.strip().endswith(("?", "？")):
            counts["excluded_unreviewed_question"]+=1
            continue
        normalized=e.get("normalized_text")
        enhanced=status in NORMALIZED and isinstance(normalized,str) and bool(normalized.strip())
        text=scrub(normalized if enhanced else raw,name,50000)
        if not text:
            counts["excluded_empty"]+=1
            continue
        # Prefer the actual earlier substantive question to an intervening
        # "Mhm" or "yes"; immediate prompt remains in provenance.
        qi=e.get("source_question_turn_id"); ai=e.get("anchor_question_turn_id")
        for j in (qi,ai):
            if j is not None and (j not in turns or turns[j].get("role")!="interviewer"):
                raise IntakeError("MemoryBank 问题来源 ID 不是采访者轮次。")
        immediate=turns.get(qi,{}).get("text","")
        anchor=turns.get(ai,{}).get("text","")
        # Tail of long introduction often holds the actual question.
        anchor=anchor[-320:] if len(anchor)>320 else anchor
        immediate=immediate[-180:] if len(immediate)>180 else immediate
        q=scrub(" ".join(dict.fromkeys(x for x in (anchor,immediate) if x)),name,500)
        tags=e.get("semantic_tags") or []
        tags=[scrub(x,name,120) for x in tags if isinstance(x,str)][:32]
        source={"dataset_memory_id":mid,"dialogue_id":str(record["dialogue_id"]),
                "respondent_turn_index":i,"interviewer_turn_index":qi,
                "anchor_question_turn_index":ai,"source_turn_ids":refs,
                "original_answer_sha256":sha256(raw.encode()).hexdigest(),
                "original_excerpt_preview":scrub(raw,name,360),
                "semantic_status":status,"memorybank_sha256":digest}
        if e.get("source_consistency_note"):
            source["source_consistency_note"]=scrub(e["source_consistency_note"],name,2000)
        facts=[{"fact_id":mid+"_01","text":text,"sensitivity":"unknown",
                "verification_status":status,"granularity":"prebuilt_answer_level",
                "dataset_memory_id":mid,"semantic_tags":tags}]
        memories.append({"memory_id":mid,"source_question":q,"text":text,
                         "type":"episodic","facts":facts,"source":source,
                         "importance":0.5,"source_type":"prebuilt_memorybank"})
        counts["enhanced" if enhanced else "unreviewed"]+=1
    if observed!=expected:
        raise IntakeError("MemoryBank 未覆盖全部受访者轮次；拒绝自动回退。")
    if len(memories)>10000:
        raise IntakeError("当前场景超过 10000 条记忆的运行上限。")
    stats={"source_respondent_turn_count":len(expected),"stored_memory_count":len(memories),
           "stored_memory_record_count":len(memories),"selection_method":"prebuilt_memorybank_context_v1",
           "memory_source":"memorybank.json","memorybank_sha256":digest,
           "enhanced_memory_count":counts["enhanced"],"unreviewed_memory_count":counts["unreviewed"],
           "excluded_entry_count":len(expected)-len(memories),"dialogue_reviewed":dialogue_reviewed,
           "semantic_entailment_checked":False,"atomic_fact_split_complete":False,
           "quality_rejection_reasons":dict(counts),"review_status":"prebuilt_partial_semantic_review"}
    return memories,stats
