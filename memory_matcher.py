"""Evidence-aware bilingual memory matcher (v1).

Pure local ranking: independent query facets + source question/answer relevance
+ corpus IDF + topic cohesion. Source questions contextualize short answers but
do not become biographical evidence. No network/embedding or private text log.

Match states distinguish direct evidence, partial context and no evidence; they
must not be confused with stochastic Memory access permission.
"""
from __future__ import annotations
import math
import re
from collections import Counter

# Broad, language-neutral concepts, not fixes for any one question category.
# Full source text remains the only authority for autobiographical claims.
FACETS={
 "tools":r"(?<![a-z])ai(?![a-z])|\b(?:copilot|chatgpt|cursor|claude|ai tools?|coding assistants?|ide|"
         r"vscode|vs code|pycharm|editor|plugin|plugins?)\b|"
         r"工具|助手|编辑器|插件|开发环境|集成",
 "coding":r"\b(?:code|coding|programming|developer|software)\b|"
          r"编程|代码|开发|软件|写程序",
 "debugging":r"\b(?:debug|debugging|bug|fixing|troubleshoot)\b|"
             r"调试|修复|报错|问题定位",
 "testing":r"\b(?:test|tests|testing|unit test)\b|测试|单测",
 "documentation":r"\b(?:document|documentation|docs|specification)\b|文档|说明|规范",
 "learning":r"\b(?:learn|learning|study|understand|explain|concept)\b|"
            r"学习|理解|解释|概念|模块",
 "features":r"\b(?:feature|features|component|components|implement|implementation)\b|"
            r"功能|组件|实现|扩展|需求",
 "workflow":r"\b(?:process|workflow|steps|integration|integrated|approach|method)\b|"
            r"流程|步骤|方式|过程|协作|路径",
 "experience":r"\b(?:experience|experienced|project|task|example|case)\b|"
              r"经历|项目|任务|案例|例子",
 "frequency":r"\b(?:often|rarely|sometimes|frequently|daily|weekly|frequency|"
             r"every day|once|twice|how often)\b|频率|经常|频繁|偶尔|很少|"
             r"每天|每周|多久用|几次",
 "time":r"\b(?:when|first|started|begin|began|months?|years?|since)\b|"
        r"最早|第一次|开始|多久前|哪年|什么时候",
 "reason":r"\b(?:because|why|reason|motivation|decided)\b|"
          r"为什么|原因|理由|动机",
 "impact":r"\b(?:impact|affect|effect|efficiency|faster|slower|productive|change)\b|"
          r"影响|效率|变化|更快|更慢|节省|效果",
 "feeling":r"\b(?:feel|feeling|felt|worried|concerned|comfortable|like|prefer)\b|"
           r"感受|担心|顾虑|喜欢|偏好|满意|体验",
 "comparison":r"\b(?:compare|compared|difference|versus|without|before|after)\b|"
              r"相比|对比|区别|不同|没有AI|不用AI|之前之后",
 "challenges":r"\\b(?:challenge|challenges|difficult|difficulty|hard|error|"
              r"problem|problems|issue|issues|struggle|cumbersome)\\b|"
              r"困难|难点|挑战|麻烦|棘手|费劲|不方便|卡住",
 "people":r"\b(?:colleague|colleagues|team|coworker|manager)\b|"
          r"同事|团队|领导|组员",
}
COMPILED={k:re.compile(v,re.I) for k,v in FACETS.items()}
GENERIC={"experience"}
SPECIAL={"debugging","testing","documentation","learning","features",
         "challenges","impact","feeling","comparison","frequency","time",
         "reason","people","workflow"}
WEIGHTS={k:(2.8 if k in SPECIAL else 1.0) for k in FACETS}

INTENT_FACETS={
 "onset_time":{"time"},
 "frequency":{"frequency"},
 "specific_episode":{"experience"},
 "reason":{"reason"},
 "method":{"workflow"},
 "comparison":{"comparison"},
 "challenges":{"challenges"},
 "impact":{"impact"},
 "feeling":{"feeling"},
 "general":set(),
}
# Facets co-occurring with this query topic are checked separately from the
# required answer type (e.g. frequency vs AI tools).
STOPWORDS=frozenset(
 "the and you your for with from when how what where why which that this "
 "some have used using use can did does does not there about into while "
 "were was are they their they them then like very also just actually"
 .split())

def facets(text):
    value=str(text or "")
    return {k for k,pattern in COMPILED.items() if pattern.search(value)}

def tokens(text):
    from question_first_retrieval import _terms
    out=_terms(text)
    return {w for w in out if w not in STOPWORDS}

def idf_stats(entries):
    df=Counter()
    for ent in entries:
        df.update(set(ent.get("search_terms",[])))
    n=len(entries)
    return df,n

def lexical(query_terms,candidate_terms,df,n):
    if not query_terms:return 0.0
    matched=query_terms & candidate_terms
    if not matched:return 0.0
    weight=lambda term: min(4.5,math.log(1+(n+1)/(df.get(term,0)+1)))
    overlap=sum(weight(term) for term in matched)
    total=sum(weight(term) for term in query_terms)
    return 3.3*overlap/max(total,0.001)

def ranked_match(entries,question,history,max_items,source_texts,source_questions):
    from source_recall import _question_dimension,_dimension_quality
    from question_first_retrieval import identify_question
    from question_intent import focus_question
    from question_first_retrieval import _alias_overlap

    analysis=identify_question(question,history)
    focus=focus_question(question)
    requested=_question_dimension(question)
    query_facets=facets(focus)
    # "不同活动中的使用频率" is not a question comparing two
    # alternatives. Avoid an accidental comparison-facet veto.
    if "comparison" in query_facets and not re.search(
        r"相比|对比|比较|差异|区别|不同于|之间.*不同|versus|compared|difference",
        focus,re.I):
        query_facets.discard("comparison")
    topic_facets=query_facets-GENERIC-INTENT_FACETS.get(requested,set())
    # If the question only specifies a requested answer dimension and no
    # concrete topic ("每种情况多久用一次"), the original source question
    # is a legitimate anchor for finding an elliptical answer.
    unspecified_topic=not topic_facets
    # No fixed category dominates: lexical matches in the respondent's
    # own answer and original question both help find source evidence.
    query_terms=tokens(focus)
    df,n=idf_stats(entries)
    candidates=[]
    for entry in entries:
        fid=entry["source_id"]
        text=str(source_texts.get(fid,""))
        original_q=str(source_questions.get(fid,""))
        answer_facets=facets(text)
        question_facets=facets(original_q)
        source_facets=answer_facets|question_facets
        direct_factor,dim_coverage=_dimension_quality(question,text,original_q)
        if direct_factor<=0:
            continue
        source_terms=set(entry.get("search_terms",[]))
        answer_terms=tokens(text)
        q_terms=tokens(original_q)
        direct_lex=lexical(query_terms,answer_terms,df,n)
        question_lex=lexical(query_terms,q_terms,df,n)
        whole_lex=lexical(query_terms,source_terms,df,n)
        topic_intersection=topic_facets & source_facets
        # If the question has a concrete topic, a generic coding/AI match
        # alone must not override it.
        topic_total=len(topic_facets)
        if topic_total:
            topic_coverage=(
                sum(WEIGHTS.get(k,1.0) for k in topic_intersection) /
                sum(WEIGHTS.get(k,1.0) for k in topic_facets)
            )
        else:
            topic_coverage=1.0
        # Concrete topic requirements outweigh generic AI/coding overlap.
        concrete=topic_facets & SPECIAL
        if concrete and not (concrete & source_facets):
            continue
        main_dimension=INTENT_FACETS.get(requested,set())
        answer_dim=bool(main_dimension & answer_facets)
        question_dim=bool(main_dimension & question_facets)
        # A "frequently" answer, or a period interval, requires its
        # original question to establish what activity it describes.
        # Source question can disambiguate elliptical "6–12 months" or
        # "often", but cannot by itself assert the factual answer.
        answer_support=(
            direct_factor>=1.0
            and (answer_dim or (question_dim and dim_coverage.startswith(
                ("direct","source_question","general_frequency"))))
        )
        if requested in ("challenges","impact","feeling",
                         "comparison","reason"):
            # Don't interpret a tool name as proof of its effect or someone's
            # motivation. Require evidence in the RESPONDENT'S answer.
            answer_support=bool(main_dimension & answer_facets) and (
                direct_lex>=.12 or bool(topic_intersection)
                or question_lex>=.30)
        elif requested=="method":
            # "How do you use AI for testing?" often has an elliptical
            # questionnaire answer ("For writing tests, very often"). The
            # activity is meaningful without an explicit "workflow" word.
            answer_support=bool(
                "workflow" in answer_facets or
                (topic_facets & answer_facets & SPECIAL) or
                (question_lex>=.22 and direct_lex>=.15)
            )
        elif requested=="general":
            # Source question alone cannot confirm the respondent's answer.
            # Match meaningful content facets, not the mere mention of AI.
            informative=(query_facets & answer_facets)-{"tools","coding","experience"}
            answer_support=bool(
                (direct_lex>=.42 and topic_coverage>=.65) or informative or
                (len(query_facets & answer_facets)>=2 and topic_coverage>=.65)
            )
        # A strong source-question match can rescue bilingual short answers.
        score=(1.7*direct_lex+1.4*question_lex+.5*whole_lex+
               2.6*topic_coverage+.6*len(topic_intersection) +
               .4*_alias_overlap(focus,original_q+" "+text))
        score*=max(0.2,min(1.8,direct_factor))
        # Answer-form fit: for "which tools" seek named tools; for "which
        # tasks" seek actual activities. Topic similarity alone often
        # ranks vague discourse about AI agents above the direct answer.
        tool_question=bool(re.search(
            r"哪些?(?:工具|平台|软件)|什么(?:工具|平台|软件)|"
            r"哪种(?:工具|平台)|通过什么|"
            r"what (?:tools|platforms|software)|which (?:tools|platforms)",
            focus,re.I))
        activity_question=bool(re.search(
            r"哪些.{0,16}(?:任务|工作|活动|用途|类型)|"
            r"主要.{0,16}(?:完成|协助|做|用来|用途)|"
            r"what (?:tasks|activities)|what do you use .* for",
            focus,re.I))
        tool_named=bool(re.search(
            r"\b(?:github\s+copilot|chatgpt|cursor|claude|vs\s*code|"
            r"pycharm|jetbrains|ide|visual\s+studio)\b|"
            r"插件|编辑器|开发环境",text,re.I))
        actions_stated=bool(re.search(
            r"\b(?:debug(?:ging)?|bug fixing|writing (?:code|tests)|"
            r"generat(?:e|ing) (?:code|tests)|explain(?:ing)? (?:code|concepts)|"
            r"document(?:ation)?|refactor(?:ing)?|test(?:ing)?)\b|"
            r"写代码|补全|调试|排查|修复|测试|解释|编写文档|重构",
            text,re.I))
        if tool_question:
            score+=3.8 if tool_named else -1.5
            # An editor/platform alone does not identify the AI assistant.
            # When AI-tool names are asked for, explicit assistant products
            # are stronger evidence than a generic IDE or Teams mention.
            if "tools" in query_facets:
                named_assistant=bool(re.search(
                    r"\b(?:github\s+copilot|chatgpt|cursor|claude|"
                    r"deepseek|qwen|codeium|tabnine)\b",text,re.I))
                score+=3.0 if named_assistant else -0.8
        if activity_question:
            score+=3.3 if actions_stated else -1.4
        if requested!="general" and answer_support:
            score+=1.4
        # Require at least one substantive link: either a source question
        # involving the user's topic or a relevant respondent statement.
        topical=bool(direct_lex>=.22 or question_lex>=.22 or
                     topic_intersection or
                     (unspecified_topic and answer_support and question_dim))
        if not topical:continue
        if requested!="general" and topic_total and topic_coverage==0:
            continue
        if requested in ("impact","feeling","challenges","comparison") and not answer_support:
            # A related topic without the asked effect/feeling/problem is not
            # a useful autobiographical match, even as "partial" evidence.
            continue
        evidence="direct" if answer_support else "partial"
        # A generic word like "faster" must not make a source responsive to
        # someone's FEELINGS; facet matches are supporting evidence only.

        if requested=="specific_episode" and dim_coverage=="partial_general_method_only":
            evidence="partial"
        # Very short remarks ("Mhm.", "another approach") are weak context
        # even when the preceding interview question shares keywords.
        if len(text.strip())<12 and not answer_support:continue
        if score<1.05:continue
        candidates.append((score,evidence,dim_coverage,entry,
                           round(topic_coverage,3),round(question_lex,3),
                           round(direct_lex,3)))
    candidates.sort(key=lambda v:(v[1]!="direct",-v[0],v[3]["source_id"]))
    if not candidates:
        return [],dict(analysis,requested_dimension=requested,
                       selection_status="no_relevant_memory",
                       selected_groups=0,total_groups=len(entries),
                       match_protocol="evidence_ranker_v1",top_score=0.0,
                       selected_coverage=[],evidence_levels=[],
                       selected_topic_coverage=[])
    top=candidates[0][0]
    strong=[v for v in candidates if v[1]=="direct"]
    # Use partial source context ONLY when direct support is not present.
    pool=strong if strong else candidates
    minimum=max(.92, pool[0][0]*(.24 if len(topic_facets)>=2 else .43))
    qualified=[v for v in pool if v[0]>=minimum]
    # Greedy topical coverage: answers about "testing AND documentation"
    # should not return three almost-identical testing details.
    selected=[]
    covered=set()
    remaining=list(qualified)
    while remaining and len(selected)<max_items:
        best=max(
            remaining,
            key=lambda v:(v[0]+sum(1.8*WEIGHTS.get(k,1.0)
                              for k in ((facets(source_texts.get(v[3]["source_id"],""))
                                        |facets(source_questions.get(v[3]["source_id"],"")))
                                        & topic_facets-covered)),
                          v[0],v[3]["source_id"]))
        selected.append(best);remaining.remove(best)
        covered.update(facets(source_texts.get(best[3]["source_id"],""))
                       |facets(source_questions.get(best[3]["source_id"],"")))
    status="direct_evidence" if strong else "partial_evidence_only"
    return [v[3] for v in selected],dict(
        analysis,requested_dimension=requested,selection_status=status,
        selected_groups=len(selected),total_groups=len(entries),
        selected_coverage=[v[2] for v in selected],
        evidence_levels=[v[1] for v in selected],
        selected_topic_coverage=[v[4] for v in selected],
        top_score=round(top,3),match_protocol="evidence_ranker_v1",
        selected_question_alignment=[v[5] for v in selected],
        selected_answer_alignment=[v[6] for v in selected])
