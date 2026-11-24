"""Source-verified two-stage recall for the respondent.

Preparation creates a compact locator cache. Answering selects relevant cached
cues, samples one reproducible access roll per memory/knowledge dimension and
only then resolves matching full source records. No interviewer access.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
import re
from pathlib import Path

from playground.interview.respondent_core import StudioError
from playground.memory.agent_memory_export import eligible_source_facts
from playground.memory.knowledge_bank import prepare as knowledge_prepare, resolve_domain, _bank
from playground.memory.question_first_retrieval import select_memory, identify_question, _terms, _score, _alias_overlap
from playground.memory.memory_access import tier_integrated_memory

VERSION="indexed_source_memory_recall_v175"
BASE=Path(__file__).resolve().parents[2]/"prepared_memory"
MAX_MEMORIES=3
MAX_REALITY=3
MAX_KNOWLEDGE=4

def _json_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                        separators=(",",":")).encode("utf-8")).hexdigest()

def _signature(bundle, prepared):
    # The source index is reusable after restart only for the exact source
    # records and prepared knowledge subset used by this experiment.
    source=eligible_source_facts(bundle)
    kb=prepared.get("domain_knowledge") or knowledge_prepare(bundle)
    return _json_hash({"algorithm":VERSION,"facts":source,
         "knowledge": [(e["knowledge_id"],e["text"]) for e in kb.get("entries",[])],
        "prepared_card":prepared.get("card",[]),
        "llm_factbank_index_signature":prepared.get("factbank_llm_index_signature"),
        "scenario":(bundle.get("source_scenario") or {}).get("scenario_id"),
        "revision":bundle["audit"]["revision"]})

def _path(key):
    return BASE/("indexed_recall_"+key+".json")

def build_cache(bundle,prepared):
    """Prepare source locators and Source locator metadata without runtime LLM calls."""
    key=_signature(bundle,prepared)
    path=_path(key)
    if path.is_file():
        try:
            stored=json.loads(path.read_text(encoding="utf-8"))
            if stored.get("version")==VERSION and stored.get("signature")==key:
                return stored
        except (OSError,ValueError,TypeError):
            pass
    from playground.memory.factbank import classified_facts
    if prepared.get("factbank_llm_index"):
        from playground.memory.llm_factbank import apply_classification
        eligible=apply_classification(bundle,prepared.get("factbank_llm_index"))
    else:
        eligible=classified_facts(bundle)
    from playground.memory.source_evidence import build_locators, SCHEMA as SOURCE_EVIDENCE_SCHEMA
    source_locators=build_locators(eligible)
    units_by_source={u["source_id"]:u for u in source_locators}
    domain=prepared.get("domain_knowledge") or knowledge_prepare(bundle)
    memory=[]
    for fact in eligible:
        text=fact["text"].strip()
        # The cache contains a brief verbatim cue, not a generated claim.
        first_sentence=re.split(r"(?<=[。！？!?；;])",text,maxsplit=1)[0].strip()
        # Failed retrieval must still leave a useful, *verbatim* memory gist.
        # 42 Chinese characters were often too little to give a normal answer.
        cue=(first_sentence or text)[:min(110,len(text))]
        if len(text)>len(cue):cue=cue.rstrip("，,；;。 ")+"…"
        # Only derived matching terms are added to the locator cache.
        # The answer model still sees only `cue` unless source access succeeds.
        source_question=str(fact.get("source_question") or "").strip()
        unit=units_by_source.get(str(fact["id"]),{})
        llm_text=" ".join(x for x in (fact.get("llm_normalized_text_zh"),
                                      fact.get("llm_normalized_text_en")) if x)
        search_terms=set(_terms(text)).union(_terms(source_question)).union(_terms(llm_text))
        semantic_tags=set(_semantic_tags(text)).union(_semantic_tags(source_question))
        semantic_tags.update(fact.get("llm_semantic_tags") or [])
        semantic_tags.update(fact.get("prebuilt_semantic_tags") or [])
        memory.append({"source_id":fact["id"],"cue":cue,
             "llm_semantic_index":llm_text[:1000],
             "search_terms":sorted(search_terms)[:300],
             "semantic_tags":sorted(semantic_tags),
             "llm_importance":fact.get("llm_importance"),
             "llm_status":fact.get("llm_status"),
             "memory_id":unit.get("memory_id"),
             "event_id":unit.get("event_id"),
             "time_expressions":unit.get("time_expressions",[]),
             "temporal_scope":unit.get("temporal_scope","as_of_original_interview"),
             "access_scope":unit.get("access_scope","eligible_source_candidate"),
             "source_question_hash":hashlib.sha256(source_question.encode()).hexdigest()[:16]
                 if source_question else None,
             "theme":"来源事实","review_pending":bool(fact.get("review_pending") or fact.get("classification_review_required")),
             "fact_category":fact["fact_category"],
             "classification_reason":fact["classification_reason"],
             "classification_confidence":fact["classification_confidence"]})
    # The failed-recall state exposes a topic cue, never its full definition.
    # Prepared Agent summaries can help *locate* the original record even
    # when wording differs, but the summary is never treated as authoritative
    # source text. A summary cue is associated only if a permitted original
    # fact shares sufficient concrete source terms with it.
    summary_facts=[
        str(f).strip()[:500]
        for group in (prepared.get("card") or [])
        if isinstance(group,dict)
        for f in (group.get("facts") or [])
        if isinstance(f,str) and f.strip()
    ]
    for entry in memory:
        original_terms=set(entry["search_terms"])
        expanded=set(original_terms)
        for summary in summary_facts:
            terms=_terms(summary)
            common=original_terms.intersection(terms)
            if len(common)>=4 or (len(common)>=2 and
                                 _semantic_tags(summary).intersection(entry["semantic_tags"])):
                expanded.update(terms)
        entry["search_terms"]=sorted(expanded)[:550]
    # Index expansion remains only a retrieval aid; each recalled sentence
    # comes from `allowed[source_id]`, never from an LLM summary.
    knowledge=[{"source_id":e["knowledge_id"],"topic":e["topic"],
                "cue":str(e["topic"])[:60]} for e in domain.get("entries",[])]
    memory_count=sum(x["fact_category"]=="memory" for x in memory)
    reality_count=len(memory)-memory_count
    cache={"version":VERSION,"signature":key,"memory":memory,
           "source_evidence_schema":SOURCE_EVIDENCE_SCHEMA,"source_locators":source_locators,
           "event_group_count":len({u["event_id"] for u in source_locators if u.get("event_id")}),
           "knowledge":knowledge,"domain":domain.get("domain"),
           "prepared_memory_count":memory_count,
           "prepared_reality_count":reality_count,
           "prepared_fact_count":len(memory),
           "prepared_knowledge_count":len(knowledge)}
    BASE.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".indexed_recall_",dir=str(BASE))
    try:
        try:os.fchmod(fd,0o600)
        except (OSError,AttributeError):pass
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(cache,stream,ensure_ascii=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return cache

def _probability(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise StudioError("Memory/Knowledge 参数无效","invalid_recall_trait",400)
    return 0.5+0.5*max(0.0,min(1.0,float(value)))

def _roll(session_id,turn_index,question,kind):
    # Stable by question/turn across independent sessions: controlled Memory=0
    # and Memory=1 experiments see the same draw for an identical question.
    # Retries remain idempotent; unrelated questions still vary.
    data=f"{VERSION}|{turn_index}|{kind}|{question.strip()}"
    return int(hashlib.sha256(data.encode()).hexdigest()[:15],16)/float(0xFFFFFFFFFFFFFFF)

def _semantic_tags(text):
    """Small bilingual concept clusters, derived from source text only.

    This is an index aid for paraphrased interview questions (e.g. "AI 工具"
    vs a source discussing Copilot). It never creates autobiographical facts.
    """
    value=str(text or "").lower()
    tags=set()
    ai=("人工智能","ai","chatgpt","gpt","copilot","大模型","模型助手","智能工具","生成式","llm","deepseek")
    coding=("编程","代码","程序","开发","软件","ide","vscode","vs code","pycharm",
            "github","调试","debug","coding","programming","javascript","python","插件")
    work=("工作","项目","任务","团队","岗位","职业","公司","实习","work","project")
    learning=("大学","课程","学校","学习","入门","接触","开始","最早","第一次","初次","课堂",
              "college","university","start","first","learn")
    if any(x in value for x in ai):tags.add("ai_tools")
    if any(x in value for x in coding):tags.add("coding")
    if "ai_tools" in tags and "coding" in tags:tags.add("ai_coding")
    if any(x in value for x in work):tags.add("work")
    if any(x in value for x in learning):tags.add("learning")
    return tags



def _question_dimension(question):
    """Requested *answer type*, kept independent of the broad subject.

    More specific communicative intents precede generic "how/why" words.
    Every dimension controls evidence matching; no dimension may create facts.
    """
    from playground.memory.question_intent import focus_question
    q=focus_question(question).lower()
    if re.search(
        r"最早|第一次|初次|什么时候开始|何时开始|从什么时候|哪个阶段开始|"
        r"when (?:did|was).{0,28}(?:start|begin|first)|"
        r"first (?:use|try|start|begin)|how long ago",q,re.I):
        return "onset_time"
    if re.search(
        r"最近.{0,12}(?:例子|案例|经历|项目|任务|一次)|"
        r"具体.{0,10}(?:例子|案例|经历)|举.{0,4}例|"
        r"recent.{0,16}(?:example|project|case|task)",q,re.I):
        return "specific_episode"
    if re.search(
        r"频率|多久.{0,6}(?:一次|用一次)|多长时间.{0,8}(?:用|一次)|"
        r"多(?:久|频繁|常).{0,10}(?:用|使用)|"
        r"(?:一天|每天|每周|每个月|一周|几天).{0,12}(?:几次|一次|用)|"
        r"用.{0,6}(?:几次|一次)|高还是低|经常.{0,5}(?:用|使用)|"
        r"how often|frequency|times? (?:a|per) (?:day|week|month)|"
        r"daily|weekly",q,re.I):
        return "frequency"
    if re.search(r"相比|对比|比较|区别|差别|有何不同|和以前.{0,10}不同|"
                 r"优缺点|利弊|好处.{0,8}坏处|"
                 r"compare|compared|versus|difference between|"
                 r"pros and cons|advantages and disadvantages",q,re.I):
        return "comparison"
    if re.search(r"困难|挑战|障碍|遇到.{0,14}(?:问题|麻烦)|"
                 r"哪里难|哪里卡住|难点|challenge|difficult|problem|obstacle",q,re.I):
        return "challenges"
    if re.search(r"影响|改变|效果|效率|节省|快了|慢了|提升|"
                 r"effect|impact|efficiency|productivity|faster|slower",q,re.I):
        return "impact"
    if re.search(r"感受|心情|感觉|担心|顾虑|满意|喜欢|"
                 r"how did you feel|how do you feel|emotion|worried",q,re.I):
        return "feeling"
    if re.search(r"为什么|原因|动机|为何|why did|because|reason",q,re.I):
        return "reason"
    if re.search(r"如何|怎么|通过什么|怎样|哪(?:些|个)(?:工具|平台)|"
                 r"where did you use|how did",q,re.I):
        return "method"
    return "general"


# "first time" by itself can describe *any* first-time task, not first
# use of AI. In the uploaded experiment, "one of the first times" referred
# to adding other components BEFORE using AI, yet was mislabelled as the
# user's first AI use. Require an explicit AI-adoption anchor instead.
_FREQUENCY_EVIDENCE=re.compile(
    r"\b(?:very often|often|sometimes|rarely|seldom|never|almost never|"
    r"frequently|regularly|occasionally|daily|weekly|every day|"
    r"once a (?:day|week|month)|twice a (?:day|week|month))\b|"
    r"(?:经常|频繁|偶尔|很少|每天|每周|几乎不用|从不|有时|通常|基本每天)",re.I)
_FREQUENCY_SOURCE_QUESTION=re.compile(
    r"how often|frequency|how many times|each of the following|"
    r"多久用一次|使用频率|多频繁",re.I)

_FIRST_USE=re.compile(
    r"最早.{0,30}(?:使用|尝试|接触).{0,30}(?:AI|人工智能|代码助手|编程助手|Copilot)|"
    r"(?:AI|人工智能|代码助手|编程助手|Copilot).{0,30}(?:最早|开始|第一次|初次)|"
    r"(?:开始|第一次|最初).{0,35}(?:用AI|用人工智能|用代码助手|用Copilot)|"
    r"(?:在大学|大学期间|上学时).{0,50}(?:AI|人工智能|编程助手|Copilot)|"
    r"(?:first|initially|initial|started|began).{0,45}"
    r"(?:use|used|using|try|tried|adopt).{0,30}"
    r"(?:AI|ChatGPT|Copilot|coding assistant|AI tools|AI for programming)|"
    r"(?:started|began) using (?:AI|ChatGPT|Copilot|coding assistant|AI tools)|"
    r"when i (?:began|started) (?:to )?(?:use|try) (?:AI|ChatGPT|Copilot)|"
    r"(?:AI|ChatGPT|Copilot|coding assistant).{0,25}(?:first use|first tried|started using)|"
    r"since (?:college|university|20\d\d).{0,40}(?:AI|ChatGPT|Copilot)",
    re.I)
_PRE_AI_CONTEXT=re.compile(
    r"before (?:i|we|they)?\s*(?:started )?(?:using|used|trying|tried)\s+AI|"
    r"在(?:使用|接触|尝试)(?:AI|人工智能)(?:之前|以前)",re.I)
_FIRST_EVENT_CONTEXT=re.compile(
    r"(?:first (?:time|times)|第一次).{0,85}"
    r"(?:add(?:ing)?|integrat(?:ing|ed)|implement(?:ing|ed)|"
    r"写|添加|集成|接入).{0,45}(?:component|组件|cloud|云)|"
    r"(?:add(?:ing)?|integrat(?:ing|ed)|implement(?:ing|ed)).{0,35}"
    r"(?:component|cloud).{0,100}(?:first (?:time|times))",re.I)

_EPISODE=re.compile(
    r"最近|上次|那次|有一次|当时|具体项目|具体任务|"
    r"这次任务|那个项目|recently|last time|"
    r"on (?:one|a) project|once when|in a recent",re.I)
_TOOL_USAGE=re.compile(
    r"(?:use|using|uses|used|利用|采用|安装|通过|接入|集成|调用|借助).{0,65}"
    r"(?:GitHub\s*Copilot|ChatGPT|IDE|VS Code|PyCharm|代码助手|AI工具|人工智能工具)|"
    r"(?:GitHub\s*Copilot|ChatGPT|IDE|VS Code|PyCharm).{0,65}"
    r"(?:use|using|uses|used|integration|集成|安装|使用)",re.I)
_AI_USE=re.compile(
    r"人工智能|ai\b|chatgpt|copilot|智能助手|大模型|"
    r"自动补全|代码助手|编程工具|ide\b|coding assistant|ai tools",re.I)


def _dimension_quality(question,original_text,source_question=""):
    """Interpret elliptical answers using their VERIFIED source question.

    The question tells us what a short answer refers to; it is not evidence
    that a fact occurred. The answer itself must still provide the asserted
    time, stage or explicit use-history.
    """
    dim=_question_dimension(question)
    text=str(original_text or "")
    origin_q=str(source_question or "")
    if dim=="onset_time":
        # "Developers start to become operators" is NOT when this individual
        # first began using AI tools, even if it shares AI/coding vocabulary.
        explicit=bool(_FIRST_USE.search(text))
        thematic=bool(_AI_USE.search(text))
        # "6-12 months" is a valid relative period when explicitly answering
        # a preceding source question about starting to use AI. Preserve its
        # historical relativity: do not invent an absolute calendar date.
        duration_answer=bool(re.search(
            r"\b\d{1,2}\s*(?:-|–|to)\s*\d{1,2}\s*"
            r"(?:months?|years?)\b|"
            r"\b\d{1,2}\s*(?:months?|years?)\b|"
            r"\b(?:six|twelve|one|two|three)\s+(?:months?|years?)\b|"
            r"\b(?:last year|two years ago)\b|"
            r"\d{1,2}到\d{1,2}个月|\d{1,2}个月(?:前|左右)",text,re.I))
        source_onset=bool(_question_dimension(origin_q)=="onset_time"
                          and _AI_USE.search(origin_q))
        # A contrast such as "Before using AI, I once had to add another
        # component; it was one of the first times..." is about component
        # work, not the participant's first adoption of AI.
        misleading_task_first=(
            bool(_PRE_AI_CONTEXT.search(text)) and
            bool(_FIRST_EVENT_CONTEXT.search(text)) and
            not re.search(r"started using AI|began using AI|首次使用AI|第一次用AI",text,re.I))
        if misleading_task_first:
            return 0.0,"unrelated_onset_time"
        if explicit and thematic:
            return 1.45,"direct"
        if source_onset and duration_answer:
            return 1.65,"direct_source_question_context"
        return 0.0,"unrelated_onset_time"
    if dim=="frequency":
        # A tool merely being used is not evidence of HOW OFTEN it is used.
        # A source questionnaire can supply the question context for an
        # elliptical answer ("Often for all of them").
        if not _FREQUENCY_EVIDENCE.search(text):
            return 0.0,"no_frequency_claim"
        source_q_freq=bool(_FREQUENCY_SOURCE_QUESTION.search(origin_q))
        if source_q_freq:
            return 1.8,"source_question_frequency"
        if _AI_USE.search(text) or _TOOL_USAGE.search(text):
            return 1.15,"general_frequency"
        # Unanchored "sometimes" may concern some other activity.
        return 0.0,"unrelated_frequency"
    if dim=="specific_episode":
        if _EPISODE.search(text) and _AI_USE.search(text):return 1.5,"direct"
        if _AI_USE.search(text) and _TOOL_USAGE.search(text):
            return .42,"partial_general_method_only"
        return 0.0,"unrelated_specific_episode"
    return 1.0,"general"


def _match_cached(entries,question,history,max_items,source_texts=None,
                  source_questions=None):
    """Score the full source-derived *index*, never just its 42-char gist.

    The previous implementation compared a Chinese question against a short
    prefix plus the generic theme "个人经历". This produced zero hits even for
    relevant original records and made Memory=1 look amnesiac. Candidate ranking
    now uses original-source-derived index terms, with full source text kept
    outside the answer context until the recall probability succeeds.
    """
    analysis=identify_question(question,history)
    from playground.memory.question_intent import focus_question
    # Discard interviewer prefaces and avoid anchoring a mismatched old topic.
    query=focus_question(question)
    analysis["requested_dimension"]=_question_dimension(question)
    query_terms=_terms(query)
    query_tags=_semantic_tags(query)
    matched=[]
    for entry in entries:
        candidate_terms=set(entry.get("search_terms",[]))
        title=entry.get("topic") or entry.get("theme") or ""
        overlap=query_terms & candidate_terms
        term_score=sum(1.9 if len(t)>=3 else .8 for t in overlap)
        # Actual question-theme overlap, not the generic "个人经历" label.
        relevance=term_score/max(2.8,len(query_terms)**.48) if query_terms else 0
        relevance+=_alias_overlap(query,entry.get("cue",""))*2.3
        relevance+=_score(query,entry.get("cue",""),entry.get("topic") or "")
        # Cross-language frequency questions can be answered by an English
        # frequency interview question + elliptical English answer.
        if analysis["requested_dimension"]=="frequency" and source_questions:
            source_q=source_questions.get(entry.get("source_id"),"")
            if _FREQUENCY_SOURCE_QUESTION.search(source_q):
                relevance+=5.0
        shared_tags=query_tags.intersection(entry.get("semantic_tags",[]))
        if "ai_coding" in shared_tags:
            relevance+=3.2
        elif "ai_tools" in shared_tags and "coding" in query_tags:
            relevance+=1.4
        elif "coding" in shared_tags and "ai_tools" in query_tags:
            relevance+=1.0
        elif "coding" in shared_tags:
            relevance+=1.0
        elif "ai_tools" in shared_tags:
            relevance+=.8
        elif "learning" in shared_tags and "learning" in query_tags:
            relevance+=.7
        elif "work" in shared_tags and "work" in query_tags:
            relevance+=.6
        coverage="general"
        if source_texts is not None:
            factor,coverage=_dimension_quality(
                question,source_texts.get(entry["source_id"],""),
                (source_questions or {}).get(entry["source_id"],""))
            relevance*=factor
        if relevance>0:
            matched.append((relevance,entry,coverage))
    matched.sort(key=lambda row:(-row[0],row[1]["source_id"]))
    if not matched:
        return [],dict(analysis,selection_status="no_relevant_memory",
                       selected_groups=0,total_groups=len(entries))
    top=matched[0][0]
    # A weak but source-grounded tool-use record can answer the *general*
    # part of a recent-case question; it must never be promoted to evidence
    # of an actual recent incident.
    cutoff=max(.18 if _question_dimension(question)=="specific_episode" else .58,
               top*.36)
    selection=[(e,coverage) for score,e,coverage in matched if score>=cutoff][:max_items]
    return [e for e,_ in selection],dict(
        analysis,selection_status="selected" if selection else "no_relevant_memory",
        selected_groups=len(selection),total_groups=len(entries),
        selected_coverage=[coverage for _,coverage in selection],
        top_score=round(top,3))

def recall(bundle,prepared,question,history,traits,session_id,turn_index,
           retrieval_strategy="new_retrieval",semantic_selector=None,
           semantic_skip_reason=None):
    """Return only permitted cue or full source text, with no private text in telemetry."""
    from playground.memory.factbank import classified_facts, question_target
    from playground.memory.llm_factbank import apply_classification
    from playground.memory.memory_matcher import ranked_match
    if retrieval_strategy not in ("new_retrieval","unified_retrieval","python_binary_retrieval","python_retrieval","legacy_cache","fused_retrieval"):
        raise ValueError("invalid_retrieval_strategy")
    unified_pool = retrieval_strategy == "unified_retrieval"
    binary_mode = retrieval_strategy == "python_binary_retrieval"
    effective_strategy = "new_retrieval" if (unified_pool or binary_mode) else retrieval_strategy
    cache=build_cache(bundle,prepared)
    available=(apply_classification(bundle,prepared.get("factbank_llm_index"))
               if prepared.get("factbank_llm_index") else classified_facts(bundle))
    if unified_pool:
        # Compare only classification: all evidence uses the same searchable pool.
        # Retain source IDs, text and original provenance; other gates stay unchanged.
        available=[dict(f, fact_category="memory") for f in available]
    allowed={f["id"]:f for f in available}
    kb=prepared.get("domain_knowledge") or knowledge_prepare(bundle)
    current_ids={e["knowledge_id"] for e in kb.get("entries",[])}
    by_category={
        category:[e for e in cache["memory"] if unified_pool or binary_mode or e.get("fact_category")==category] if category=="memory" else ([] if (unified_pool or binary_mode) else [e for e in cache["memory"] if e.get("fact_category")==category])
        for category in ("memory","reality")
    }
    answer_texts={fid:item["text"] for fid,item in allowed.items()}
    original_questions={fid:item.get("source_question","") for fid,item in allowed.items()}
    target=question_target(question,history)
    match_results={}
    # Temporal target determines bank priority, not permission to invent.
    # Both banks remain independently searchable for useful contextual backup.
    for category in (() if effective_strategy=="legacy_cache" or (effective_strategy in ("new_retrieval","fused_retrieval") and semantic_selector is not None) else ("memory","reality")):
        match_results[category]=ranked_match(
            by_category[category],question,history,
            MAX_MEMORIES if category=="memory" else MAX_REALITY,
            answer_texts,original_questions)
    # v1.5: LLM judges event groups using original question + answer evidence.
    # The Python lexical retriever is retained as an explicit baseline mode.
    # Never promote a model-authored quote into a source fact; only authorized IDs
    # are materialized from the original FactBank.
    semantic_primary_trace=None
    if effective_strategy in ("new_retrieval","fused_retrieval") and semantic_selector is not None:
        from playground.memory.semantic_memory_compat import grouped_records, select_groups, selected_sources
        from playground.memory.hybrid_retrieval import prefilter_groups
        from playground.memory.source_evidence import annotate_selected
        groups=grouped_records(bundle,available)
        llm_index={str(x.get("source_id")):x for x in prepared.get("factbank_llm_index",[])
                   if isinstance(x,dict)}
        for group in groups:
            index_rows=[llm_index.get(str(f.get("id")),{}) for f in group.get("facts",[])]
            group["semantic_index_text"]=" ".join(
                str(v) for row in index_rows
                for v in (row.get("normalized_text_zh"),row.get("normalized_text_en")) if v)
            group["semantic_index_tags"]=sorted({tag for row in index_rows
                for tag in (row.get("semantic_tags") or [])})
            group["semantic_index_importance"]=max(
                (int({"key":3,"supporting":2,"background":1}.get(row.get("importance"),1))
                 for row in index_rows), default=1)
        candidates,prefilter_trace=prefilter_groups(question,history,groups,vector_index=prepared.get('multilingual_vector_index'))
        chosen,semantic_primary_trace=select_groups(
            question,history,candidates,semantic_selector=semantic_selector,
            max_groups=6,parallel_batches=True)
        full_fallback_attempted=False
        # An explicit empty selection after a narrowed candidate set is the
        # only automatic quality fallback. Model errors retain the existing
        # deterministic degradation rather than multiplying failed API calls.
        if (prefilter_trace.get("prefilter_applied") and
                semantic_primary_trace.get("semantic_status")=="no_relevant_memory" and
                not semantic_primary_trace.get("semantic_errors")):
            full_fallback_attempted=True
            full_chosen,full_trace=select_groups(
                question,history,groups,semantic_selector=semantic_selector,
                max_groups=6,parallel_batches=True)
            full_trace["full_retrieval_fallback_attempted"]=True
            full_trace["full_retrieval_fallback_selected"]=(
                full_trace.get("semantic_status")=="selected")
            semantic_primary_trace=full_trace
            if full_trace.get("semantic_status")=="selected":
                chosen=full_chosen
        validated=selected_sources(question,chosen,max_facts_per_group=3)
        # Source locator metadata is only a locator. Source text still comes from
        # the permission-filtered original FactBank below.
        unit_map={u["source_id"]:u for u in cache.get("source_locators",[])}
        validated=annotate_selected(validated,unit_map)
        dimension=_question_dimension(question)
        rejected_dimension_ids=[]
        if dimension in ("onset_time","frequency","specific_episode"):
            dimension_valid=[]
            for item in validated:
                factor,coverage=_dimension_quality(
                    question,item.get("text",""),item.get("source_question",""))
                minimum=1.0
                if factor>=minimum:
                    item["dimension_coverage"]=coverage
                    dimension_valid.append(item)
                else:
                    rejected_dimension_ids.append(str(item.get("source_id") or ""))
            validated=dimension_valid
            # If a narrowed candidate set selected only a wrong event, search
            # the complete authorized group bank once before declaring no hit.
            if (not validated and prefilter_trace.get("prefilter_applied") and
                    not full_fallback_attempted):
                full_fallback_attempted=True
                full_chosen,full_trace=select_groups(
                    question,history,groups,semantic_selector=semantic_selector,
                    max_groups=6,parallel_batches=True)
                full_trace["full_retrieval_fallback_attempted"]=True
                full_validated=annotate_selected(
                    selected_sources(question,full_chosen,max_facts_per_group=3),unit_map)
                full_validated=[item for item in full_validated
                    if _dimension_quality(question,item.get("text",""),
                        item.get("source_question",""))[0]>=minimum]
                full_trace["full_retrieval_fallback_selected"]=bool(full_validated)
                semantic_primary_trace=full_trace
                validated=full_validated
        semantic_primary_trace.update(prefilter_trace)
        semantic_primary_trace["full_retrieval_fallback_attempted"]=full_fallback_attempted
        semantic_primary_trace["dimension_guard_rejected_count"]=len(rejected_dimension_ids)
        semantic_primary_trace["dimension_guard_rejected_source_ids"]=rejected_dimension_ids[:20]
        if (not validated and rejected_dimension_ids and
                dimension in ("onset_time","frequency","specific_episode")):
            semantic_primary_trace["semantic_status"]="no_supported_dimension_evidence"
        candidates_by_category={"memory":[],"reality":[]}
        seen=set()
        for item in validated:
            sid=item.get("source_id")
            if sid in allowed and sid not in seen:
                seen.add(sid)
                category=("memory" if binary_mode else allowed[sid].get("fact_category","reality"))
                if category not in candidates_by_category: category="reality"
                candidates_by_category[category].append({"source_id":sid,
                    "event_group":item.get("event_group") or item.get("event_id"),
                    "memory_id":item.get("memory_id"),
                    "time_expressions":item.get("time_expressions",[]),
                    "semantic_selected":True})
        for category in ("memory","reality"):
            match_results[category]=(candidates_by_category[category],{
                "selection_status":"semantic_selected" if candidates_by_category[category] else "no_relevant_memory",
                "match_protocol":"v175_source_memory_selection",
                "requested_dimension":dimension,"selected_coverage":[
                    item.get("dimension_coverage","general") for item in validated],
                "evidence_levels":[],"selected_topic_coverage":[],
                "selected_question_alignment":[],"selected_answer_alignment":[]})
    elif effective_strategy in ("new_retrieval","fused_retrieval"):
        if semantic_skip_reason:
            semantic_primary_trace={"semantic_status":"skipped_non_personal_route",
                "selection_method":"procedural_fast_route","semantic_model_calls":0,
                "semantic_errors":[],"semantic_skip_reason":str(semantic_skip_reason)}
        else:
            semantic_primary_trace={"semantic_status":"selector_unavailable",
                "selection_method":"python_baseline_degradation","semantic_model_calls":0,
                "semantic_errors":["semantic_selector_unavailable"]}
    memory_candidates,memory_trace=match_results.get("memory",([], {"selection_status":"legacy_cache_only"}))
    reality_candidates,reality_trace=match_results.get("reality",([], {"selection_status":"legacy_cache_only"}))
    if binary_mode:
        # Stage 1: retrieval works over ALL eligible facts irrespective of labels.
        # Stage 2: only after source IDs have been retrieved, read their labels
        # to apply the Memory probability judgement and preserve Reality access.
        selected_memory=[];selected_reality=[]
        for candidate in memory_candidates:
            src=allowed.get(candidate.get("source_id"))
            if not src: continue
            if src.get("fact_category")=="memory": selected_memory.append(candidate)
            else: selected_reality.append(candidate)
    elif unified_pool:
        selected_memory=memory_candidates
        selected_reality=[]
    elif target=="memory":
        selected_memory=memory_candidates
        selected_reality=reality_candidates if not memory_candidates else []
    elif target=="reality":
        selected_reality=reality_candidates
        selected_memory=memory_candidates if not reality_candidates else []
    else:
        selected_memory=memory_candidates[:2]
        selected_reality=reality_candidates[:2]
    # Retrieval is a separate stage from cache answering. Confirm primary
    # hits against independently source-grounded dimension/domain evidence.
    # A weak, wrong-dimension hit must not prevent the cache-answer path.
    from playground.memory.high_recall_fallback import lookup as cached_source_lookup, _intent as cache_intent
    cache_matches=([] if effective_strategy=="legacy_cache" else
                   cached_source_lookup(question,available,max_items=16))
    legacy_access=None
    if effective_strategy=="legacy_cache":
        from playground.memory.legacy_memory_compat import select_original_sources
        cache_matches,legacy_access=select_original_sources(
            bundle,question,history,available,traits["memory"],semantic_selector=semantic_selector)
    fused_access=None
    confirmed_ids={m["source_id"] for m in cache_matches}
    has_known_intent=bool(cache_intent(question))
    unfiltered_count=len(selected_memory)+len(selected_reality)
    def evidence_valid(entry):
        fid=entry["source_id"]
        if fid in confirmed_ids or not has_known_intent:
            return True
        from playground.memory.high_recall_fallback import _intent
        intents=_intent(question)
        # For ordinary queries, retain exact distinctive subject overlap even
        # when the bilingual cache vocabulary does not name that subject.
        if intents.intersection({"frequency","start","example","recording","opinion"}):
            return False
        text=(str(allowed[fid].get("text") or "")+" "+
              str(allowed[fid].get("source_question") or "")).lower()
        distinctive=re.findall(r"[\u4e00-\u9fff]{4,}|[a-z]{5,}",question.lower())
        spans=[part[i:i+4] for part in distinctive for i in range(max(1,len(part)-3))]
        stop={"你在代码","发生了什","什么工具","如何使用","具体说说","请问你在"}
        return any(term not in stop and term in text for term in spans)
    if has_known_intent and semantic_primary_trace is None:
        selected_memory=[e for e in selected_memory if evidence_valid(e)]
        selected_reality=[e for e in selected_reality if evidence_valid(e)]
    primary_count=len(selected_memory)+len(selected_reality)
    if effective_strategy=="fused_retrieval":
        # Only compute the old-style cache path after the NEW verifier declares
        # a genuine miss; a failed judgement must never access legacy facts.
        if primary_count==0:
            from playground.memory.legacy_memory_compat import select_fused_sources
            cache_matches,fused_access=select_fused_sources(
                bundle,question,history,available,traits["memory"],semantic_selector=semantic_selector)
        else:
            cache_matches=[]
    # v1.6.0 dual-gate reconstruction:
    #   Primary hit -> memory judgement only; cache is forbidden.
    #   Primary miss -> independent cache-access draw; no second memory draw.
    cache_candidate_available = (primary_count == 0 and bool(cache_matches))
    p_cache = _probability(traits["memory"])
    roll_cache = _roll(session_id, turn_index, question, "cache_fallback")
    cache_gate_passed = bool(cache_candidate_available and roll_cache < p_cache)
    cache_fallback = cache_gate_passed
    high_recall_status = (
        "cache_answer_ready" if cache_fallback
        else "cache_probability_rejected" if cache_candidate_available
        else "retrieval_verified" if primary_count
        else "no_source_evidence")
    appended_ids=[]
    matched_knowledge,know_trace=_match_cached(cache["knowledge"],question,history,MAX_KNOWLEDGE)
    p_memory=_probability(traits["memory"])
    p_knowledge=_probability(traits["knowledge"])
    roll_memory=_roll(session_id,turn_index,question,"memory")
    roll_knowledge=_roll(session_id,turn_index,question,"knowledge")
    passed_memory=roll_memory<p_memory
    passed_knowledge=roll_knowledge<p_knowledge

    # Memory probability gates historical Memory facts only. Reality facts
    # describe the respondent's state at the original interview and are not
    # randomized by the Memory slider.
    judgement_passed=passed_memory
    person=[]
    if primary_count:
        for category,candidates in (("memory",selected_memory),("reality",selected_reality)):
            if category=="memory" and not judgement_passed:
                continue
            for e in candidates:
                src=allowed.get(e["source_id"])
                if src is None:continue
                unit=next((u for u in cache.get("source_locators",[])
                           if u.get("source_id")==e["source_id"]),{})
                person.append({
                    "theme":"历史经历" if category=="memory" else "访谈时的现实状态",
                    "facts":[src["text"][:4000]],
                    "source_question_context":str(src.get("source_question") or "")[:350],
                    "fact_category":category,
                    "temporal_scope":"as_of_original_interview",
                    "source_id":e["source_id"],"source_recovered":True,
                    "memory_id":unit.get("memory_id"),
                    "event_id":unit.get("event_id"),
                    "time_expressions":unit.get("time_expressions",[]),
                    "fact_temporal_scope":unit.get("fact_temporal_scope","unknown"),
                    "fact_type":unit.get("fact_type","other"),
                    "semantic_tags":unit.get("semantic_tags",[]),
                    "importance":unit.get("importance","background"),
                    "importance_weight":unit.get("importance_weight",1),
                    "review_pending":unit.get("review_pending",False),
                    "access_scope":"observable_selected_source",
                })
    # Stage 3 receives only source-verified cached original answers, never
    # generated summaries. Cache branch is selected by retrieval miss ALONE;
    # no memory judgement outcome can activate it.
    if cache_fallback:
        unit_map={u["source_id"]:u for u in cache.get("source_locators",[])}
        for match in cache_matches[:8]:
            src=allowed[match["source_id"]]
            unit=unit_map.get(src["id"],{})
            person.append({
                "theme":(match.get("event_group") or "来源缓存回答"), "facts":[(match["text"] if effective_strategy in ("legacy_cache","fused_retrieval") else src["text"])[:4000]],
                "event_group":match.get("event_group"),
                "source_question_context":str(src.get("source_question") or "")[:350],
                "fact_category":src.get("fact_category","reality"),
                "temporal_scope":"as_of_original_interview",
                "source_id":src["id"],"source_recovered":True,
                "memory_id":unit.get("memory_id"),
                "event_id":unit.get("event_id"),
                "time_expressions":unit.get("time_expressions",[]),
                "fact_temporal_scope":unit.get("fact_temporal_scope","unknown"),
                "fact_type":unit.get("fact_type","other"),
                "semantic_tags":unit.get("semantic_tags",[]),
                "importance":unit.get("importance","background"),
                "importance_weight":unit.get("importance_weight",1),
                "review_pending":unit.get("review_pending",False),
                "access_scope":"observable_verified_cache_source",
                "cache_answer":True,
            })
    response_branch=("cache_answer" if cache_fallback else
                     "grounded_answer" if person else
                     "memory_judgement_failed" if primary_count else "no_cache_evidence")
    # Full general knowledge originates in KnowledgeBank, not personal memory.
    bank_map={}
    row=resolve_domain(bundle)
    if row:
        full=_bank(row["domain"])
        if full:
            bank_map={e["knowledge_id"]:e for e in full.get("entries",[])}
    knowledge=[]
    for e in matched_knowledge:
        if e["source_id"] not in current_ids:continue
        original=bank_map.get(e["source_id"])
        if original is None:continue
        detail=original["text"] if passed_knowledge else e["cue"]
        knowledge.append({"knowledge_id":e["source_id"],"topic":e["topic"],
                          "text":detail})
    memory_person=[p for p in person if p["fact_category"]=="memory"]
    reality_person=[p for p in person if p["fact_category"]=="reality"]
    # Match diagnostics must describe the bank actually used, including
    # fallback to the other bank when the temporal priority had no candidate.
    chosen_trace=(memory_trace if memory_person and not reality_person
                  else reality_trace if reality_person and not memory_person
                  else memory_trace if target=="memory" else reality_trace)
    trace={
        "retrieval_strategy":retrieval_strategy,
        "recall_protocol":VERSION,
        "factbank_schema":prepared.get("factbank_schema") or cache.get("source_evidence_schema"),
        "factbank_preparation_method":prepared.get("preparation_method","deterministic_source_excerpts"),
        "factbank_llm_summary":prepared.get("factbank_llm_summary"),
        "question_temporal_target":target,
        "memory_prepared_count":cache["prepared_memory_count"],
        "reality_prepared_count":cache["prepared_reality_count"],
        "fact_prepared_count":cache["prepared_fact_count"],
        "memory_candidate_count":len(selected_memory),
        "reality_candidate_count":len(selected_reality),
        "memory_source_recovered":bool(memory_person) and (judgement_passed or cache_fallback),
        "reality_source_recovered":bool(reality_person),
        "memory_material":"verified_source_cache" if cache_fallback and memory_person else
                          "original_source" if memory_person else "none",
        "reality_material":"original_source" if reality_person else "none",
        "memory_access_probability":p_memory,"memory_roll":round(roll_memory,6),
        "memory_visible_count":len(memory_person),
        "reality_visible_count":len(reality_person),
        "fact_visible_count":len(person),
        "memory_source_ids":[p["source_id"] for p in memory_person],
        "reality_source_ids":[p["source_id"] for p in reality_person],
        "knowledge_prepared_count":cache["prepared_knowledge_count"],
        "knowledge_candidate_count":len(matched_knowledge),
        "knowledge_source_recovered":passed_knowledge and bool(knowledge),
        "knowledge_access_probability":p_knowledge,"knowledge_roll":round(roll_knowledge,6),
        "knowledge_visible_ids":[e["knowledge_id"] for e in knowledge],
        "knowledge_visible_count":len(knowledge),
        "cache_signature_sha256":cache["signature"],
        "retrieval_rule":("v091_semantic_grouped_old_hr12" if effective_strategy=="legacy_cache" else "v175_source_memory_primary_plus_grouped_semantic_fallback" if effective_strategy=="fused_retrieval" else "factbank_dual_retrieval_plus_verified_source_cache" if effective_strategy=="python_retrieval" else "v175_source_memory_primary_plus_verified_cache"), 
        "primary_candidate_count":primary_count,
        "unverified_primary_count":unfiltered_count-primary_count,
        "retrieval_succeeded":bool(primary_count),
        "judgement_passed":judgement_passed,
        "memory_judgement_passed":judgement_passed,
        "reality_judgement_passed":True,
        "response_branch":response_branch,
        "cache_answer_source_ids":[m["source_id"] for m in cache_matches[:8]] if cache_fallback else [],
        "high_recall_status":high_recall_status,
        "high_recall_added_ids":appended_ids,
        "cache_fallback_triggered":cache_fallback,
        "cache_candidate_available":cache_candidate_available,
        "cache_access_probability":p_cache,
        "cache_roll":round(roll_cache,6),
        "cache_gate_passed":cache_gate_passed,
        "cache_gate_rejected":bool(cache_candidate_available and not cache_gate_passed),
        "memory_access_path":("primary_memory" if primary_count else
                              "cache_fallback" if cache_fallback else "none"),
        "recall_pairing":"identical_round_question_share_draw_across_sessions",
        "memory_match_status":memory_trace.get("selection_status"),
        "reality_match_status":reality_trace.get("selection_status"),
        "fact_match_status":("source_cache_fallback" if cache_fallback
                             else chosen_trace.get("selection_status") if primary_count
                             else "no_relevant_memory"),
        "question_dimension":chosen_trace.get("requested_dimension"),
        "memory_coverage":chosen_trace.get("selected_coverage",[]),
        "memory_evidence_levels":chosen_trace.get("evidence_levels",[]),
        "memory_topic_coverage":chosen_trace.get("selected_topic_coverage",[]),
        "memory_match_protocol":chosen_trace.get("match_protocol"),
        "memory_question_alignment":chosen_trace.get("selected_question_alignment",[]),
        "memory_answer_alignment":chosen_trace.get("selected_answer_alignment",[]),
        "knowledge_match_status":know_trace.get("selection_status")
    }
    if effective_strategy in ("new_retrieval","fused_retrieval"):
        trace["v15_semantic_primary"]=semantic_primary_trace
        trace["primary_selection_method"]=(semantic_primary_trace or {}).get("selection_method","python_baseline")
        trace["primary_source_ids"]=[e["source_id"] for e in selected_memory+selected_reality]
    trace["source_evidence_schema"]=cache.get("source_evidence_schema")
    trace["source_locator_count"]=len(cache.get("source_locators",[]))
    trace["event_group_count"]=cache.get("event_group_count",0)
    trace["selected_memory_ids"]=[p.get("memory_id") for p in person if p.get("memory_id")]
    trace["selected_event_ids"]=[p.get("event_id") for p in person if p.get("event_id")]
    trace["selected_time_expressions"]=[x for p in person for x in p.get("time_expressions",[])]
    trace["selected_fact_temporal_scopes"]=[p.get("fact_temporal_scope","unknown") for p in person]
    trace["selected_fact_types"]=[p.get("fact_type","other") for p in person]
    trace["selected_fact_importance"]=[p.get("importance","background") for p in person]
    trace["temporal_scope_policy"]="relative_time_is_anchored_to_original_interview"
    trace["information_layers"]={
        "latent_layer":"full source records remain server-side and are not passed to the answer model",
        "observable_layer":"only selected, permission-checked source records are materialized",
    }
    trace["temporal_event_spread"]=(
        _question_dimension(question)=="onset_time" and
        len(set(trace["selected_event_ids"]))>1)
    if effective_strategy=="python_retrieval":
        trace["primary_selection_method"]="python_baseline_hr13"
    if effective_strategy=="legacy_cache":
        trace["legacy_memory_access"]=legacy_access
        trace["legacy_prompt_version"]="v091_semantic_grouped_hr12"
    if effective_strategy=="fused_retrieval":
        trace["fusion_memory_access"]=fused_access if cache_fallback else None
        trace["fusion_fallback_used"]=cache_fallback
        trace["fusion_prompt_version"]=("v091_semantic_grouped_hr12" if cache_fallback else None)
    return person,knowledge,trace
