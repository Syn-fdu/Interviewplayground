"""Low-cost bilingual candidate prefilter for the LLM semantic selector.

This layer ranks source groups; it never returns autobiographical facts and
never replaces the LLM selector. Low-confidence queries keep the full bank.
"""
from __future__ import annotations
import math
import os
import re
import time
from collections import Counter

VERSION = "prefilter.v180_fixed44_recall_first"
DEFAULT_CANDIDATES = 44

# Query-time multilingual *retrieval hints*, never extra FactBank categories.
# Source evidence remains untouched; these names are only ranking equivalents.
# The first tuple contains the user-question triggers, the second source cues.
# Restrict terms to relatively distinctive lexical entities/concepts so generic
# "AI" or "code" does not swamp the ranker. No network/LLM dependency.
_BILINGUAL_BRIDGES = (
    (("编辑器", "代码编辑器", "开发环境", "集成开发环境", "editor", "ide"),
     ("vs code", "vscode", "visual studio code", "code editor", "editor", "ide", "pycharm", "intellij", "webstorm", "sublime text", "vim", "emacs")),
    (("调试", "排查错误", "修复bug", "修复 bug", "debug", "debugging", "bug fix"),
     ("debug", "debugging", "bug fixing", "troubleshoot", "finding a bug", "bug originates")),
    (("测试", "单元测试", "集成测试", "test", "testing"),
     ("test", "testing", "unit test", "integration test", "typescript", "eslint")),
    (("验证", "核查", "检查生成", "审查ai", "审核ai", "validate", "verify ai", "review ai"),
     ("validat", "verify", "review ai", "read the code", "check whether", "safety nets")),
    (("设计稿", "设计文件", "figma", "design mockup"),
     ("figma", "design document", "design mockup", "design files")),
    (("文档", "说明文档", "documentation", "markdown"),
     ("documentation", "markdown", "documenting", "docs")),
    (("组件", "颜色变体", "进度条", "component", "color variant", "progress bar"),
     ("component", "color variant", "progress bar", "storybook")),
    (("需求", "需求规格", "用户故事", "规格说明", "specification", "requirements", "user story"),
     ("specification", "requirements", "user story", "ticket", "technical review")),
    (("代码审查", "代码评审", "拉取请求", "pull request", "code review"),
     ("pull request", "code review", "review comment", "reviewed", "approved")),
    (("代理式编程", "智能体编程", "自主编码", "agentic coding", "open code"),
     ("agentic coding", "open code", "opencode", "robot-like", "agent coding")),
)


def _bridge_features(question, source):
    """(matched concept count, direct named-entity count); no factual inference."""
    q = str(question or "").casefold()
    doc = str(source or "").casefold()
    # Normalized spaces make 'VS  Code' and 'VS-Code' searchable together.
    q = q.replace("-", " ")
    doc = doc.replace("-", " ")
    def contains(text, cue):
        if cue == "validat":
            return bool(re.search(r"\bvalidat[a-z]*\b", text))
        if cue and cue[0].isascii() and cue[0].isalpha():
            return bool(re.search(r"(?<![a-z0-9])" + re.escape(cue) + r"(?![a-z0-9])", text))
        return cue in text
    bridge_hits = 0
    entity_hits = 0
    for query_cues, source_cues in _BILINGUAL_BRIDGES:
        if any(contains(q, cue) for cue in query_cues) and any(contains(doc, cue) for cue in source_cues):
            bridge_hits += 1
            if any(contains(doc, cue) for cue in ("vs code", "vscode", "visual studio code", "pycharm", "intellij", "figma", "open code")):
                entity_hits += 1
    return bridge_hits, entity_hits


def _group_text(group):
    facts = group.get("facts") or []
    answer = " ".join(str(f.get("text") or "") for f in facts if isinstance(f, dict))
    source_q = str(group.get("original_question") or "")
    excerpt = str(group.get("respondent_excerpt") or "")
    semantic_index = str(group.get("semantic_index_text") or "")
    return source_q, excerpt, answer, (source_q + " " + excerpt + " " + answer + " " + semantic_index).strip()


def prefilter_groups(question, history, groups, max_candidates=None, vector_index=None):
    """Return (candidate_groups, safe telemetry). No raw source text in telemetry."""
    started = time.monotonic()
    groups = list(groups or [])
    try:
        limit = int(max_candidates or os.getenv("INTERVIEW_HYBRID_CANDIDATES", str(DEFAULT_CANDIDATES)))
    except (TypeError, ValueError):
        limit = DEFAULT_CANDIDATES
    limit = max(12, min(64, limit))
    enabled = os.getenv("INTERVIEW_HYBRID_PREFILTER", "1").strip().lower() not in {"0", "false", "off"}
    base = {
        "prefilter_version": VERSION,
        "prefilter_enabled": enabled,
        "candidate_groups_total": len(groups),
        "candidate_groups_sent": len(groups),
        "prefilter_applied": False,
        "prefilter_fallback_reason": None,
        "prefilter_top_score": 0.0,
    }
    if not enabled:
        base["prefilter_fallback_reason"] = "disabled_by_environment"
        base["prefilter_elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return groups, base
    if len(groups) <= limit:
        base["prefilter_fallback_reason"] = "small_bank_no_prefilter_needed"
        base["prefilter_elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return groups, base

    from playground.memory.question_first_retrieval import _terms, _alias_overlap, identify_question
    from playground.memory.question_intent import focus_question
    from playground.memory.source_recall import _semantic_tags, _question_dimension, _dimension_quality

    focus = focus_question(question)
    analysis = identify_question(question, history or [])
    query = str(analysis.get("retrieval_query") or focus)
    q_terms = _terms(query)
    q_tags = _semantic_tags(query)
    dimension = _question_dimension(question)

    docs = []
    document_frequency = Counter()
    for group in groups:
        source_q, excerpt, answer, combined = _group_text(group)
        terms = _terms(combined)
        document_frequency.update(terms)
        tags=set(_semantic_tags(combined))
        tags.update(group.get("semantic_index_tags") or [])
        docs.append((group, source_q, excerpt, answer, combined, terms, tags))

    n_docs = max(1, len(docs))
    query_weights = {term: min(4.5, math.log(1 + (n_docs + 1) /
                       (document_frequency.get(term, 0) + 1))) for term in q_terms}
    total_query_weight = sum(query_weights.values()) or 1.0
    ranked = []
    dimension_sensitive = dimension in {"onset_time", "frequency", "specific_episode"}
    for group, source_q, excerpt, answer, combined, terms, tags in docs:
        overlap = q_terms.intersection(terms)
        lexical = sum(query_weights.get(term, 0.0) for term in overlap) / total_query_weight
        source_overlap = q_terms.intersection(_terms(source_q))
        source_question_score = sum(query_weights.get(term, 0.0) for term in source_overlap) / total_query_weight
        alias = _alias_overlap(focus, combined)
        alias_score = min(1.0, alias / 3.0)
        bridge_hits, bridge_entity_hits = _bridge_features(focus, combined)
        shared_tags = q_tags.intersection(tags)
        tag_score = len(shared_tags) / max(1, len(q_tags)) if q_tags else 0.0
        direct_dimension = 0.0
        if dimension_sensitive:
            factors = []
            facts = group.get("facts") or []
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                factor, _ = _dimension_quality(
                    question, str(fact.get("text") or ""),
                    str(fact.get("source_question") or source_q))
                factors.append(float(factor))
            direct_dimension = max(factors, default=0.0)
            # Soft dimension weighting only; never drop evidence before LLM selection.
            if direct_dimension < 1.0:
                direct_dimension = 0.0
        importance_weight=int(group.get("semantic_index_importance",1) or 1)
        importance_boost={3:0.18,2:0.09}.get(importance_weight,0.0)
        score = (3.2 * lexical + 0.8 * source_question_score +
                 1.0 * alias_score + 1.5 * tag_score + importance_boost +
                 (1.4 * min(1.0, direct_dimension / 1.8) if dimension_sensitive else 0.0) +
                 3.2 * min(2, bridge_hits) + 0.45 * min(1, bridge_entity_hits))
        ranked.append((score, str(group.get("id") or ""), group))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    top_score = ranked[0][0] if ranked else 0.0
    base["prefilter_top_score"] = round(top_score, 4)
    # Stable budget: low confidence is a diagnostic, not permission to send
    # the complete bank. The LLM may still select zero relevant sources.
    base["prefilter_low_confidence"] = bool(top_score < 0.42)

    selected = [row[2] for row in ranked[:limit]]
    if len(selected) < limit:
        used = {str(g.get("id")) for g in selected}
        selected.extend(g for g in groups if str(g.get("id")) not in used)
        selected = selected[:limit]
    # Add locally embedded cross-language candidates without excluding strong
    # lexical matches. For time-sensitive questions retain the original guard.
    if vector_index is not None and not dimension_sensitive:
        from playground.memory.local_multilingual_index import rank as vector_rank
        scores = vector_rank(question, groups, vector_index)
        if scores and scores[0][0] >= 0.45:
            by_id = {str(g.get('id')):g for g in groups}
            embedding_ids = [gid for score,gid in scores if score >= 0.45][:max(1,limit//2)]
            merged = [by_id[gid] for gid in embedding_ids if gid in by_id]
            seen_ids = {str(g.get('id')) for g in merged}
            merged.extend(g for g in selected if str(g.get('id')) not in seen_ids)
            selected = merged[:limit]
            base['multilingual_vector_applied'] = True
        else:
            base['multilingual_vector_applied'] = False
    # Preserve at least the top-ranked groups; only source IDs from the
    # original authorized group list can pass this boundary.
    allowed_ids = {str(g.get("id")) for g in groups}
    selected = [g for g in selected if str(g.get("id")) in allowed_ids]
    if not selected:
        base["prefilter_fallback_reason"] = "empty_authorized_bank"
        base["prefilter_elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return [], base
    base.update({
        "candidate_groups_sent": len(selected),
        "prefilter_applied": len(selected) < len(groups),
        "prefilter_fallback_reason": None if len(selected) < len(groups) else "no_reduction",
        "prefilter_elapsed_ms": round((time.monotonic() - started) * 1000),
    })
    return selected, base
