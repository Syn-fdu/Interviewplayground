"""v0.13: deterministic question-first, selective memory routing for local interviews.

No network calls, embeddings, inferred autobiographical facts or retries.  The
completed integrated card stays unchanged on disk; routing is per turn only.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any

# Recognition is about the question's *answer dimension*, not a topic detector.
ASPECTS = {
    'reason': ('为什么', '为何', '原因', '理由', '动机', '怎么会', '出于什么', '考虑', 'why', 'reason'),
    'process': ('怎么做', '如何做', '怎样做', '过程', '经过', '步骤', '当时怎么', '怎么处理', 'how did'),
    'feeling': ('感受', '心情', '感觉', '情绪', '害怕', '担心', '开心', '难过', 'feeling', 'feel'),
    'time': ('什么时候', '哪一年', '几岁', '多久', '何时', '时间', 'when'),
    'place': ('在哪里', '在哪儿', '什么地方', '地点', 'where'),
    'person': ('谁', '哪个人', '什么人', '和谁', 'who'),
    'impact': ('影响', '结果', '后果', '改变', '后来怎么样', '之后怎样', 'impact', 'result'),
    'example': ('举例', '例子', '具体哪件', '具体什么事', '比如', 'example'),
    'opinion': ('怎么看', '如何看待', '认为', '觉得', '评价', '看法', 'what do you think'),
    'comparison': ('相比', '比较', '不同', '区别', '变化', '以前和现在', 'difference', 'compare'),
    'choice': ('选择', '决定', '取舍', '为什么选', 'decision', 'choose'),
}
STOP = frozenset('的了呢吗吧啊呀是不是我你他她它们这那什么一个一些关于请能可以有没有是否又还再说谈讲告诉介绍具体当时现在之前之后已经就是觉得事情经历情况问题时候如何怎么为什么'.split())
# Normalize near-synonyms only for *retrieval*. Never inject new claims.
ALIASES = (
    ('工作', '职业', '岗位', '职场', '就业', '上班', '事业'),
    ('学校', '上学', '读书', '学业', '学习', '大学', '教育'),
    ('父亲', '爸爸', '父母', '家长', '家庭'),
    ('母亲', '妈妈', '父母', '家长', '家庭'),
    ('朋友', '同学', '伙伴', '友人'),
    ('结婚', '婚姻', '配偶', '夫妻'),
    ('孩子', '子女', '儿子', '女儿'),
    ('搬家', '迁居', '搬迁', '迁移'),
    ('病', '生病', '疾病', '健康', '身体'),
    ('钱', '收入', '经济', '工资', '薪水'),
    ('离开', '辞职', '退出', '离职'),
    ('选择', '决定', '决策', '考虑'),
)
FOLLOWUP = ('那后来', '接下来', '再具体', '具体一点', '那时候呢', '这件事', '刚才', '你说的', '然后呢', '还有呢', '为什么这么', '那你')


def _clean(text: Any) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip().lower()


def _terms(text: str) -> set[str]:
    s = re.sub(r'[^\u4e00-\u9fffa-z0-9]+', ' ', _clean(text))
    terms = set(re.findall(r'[a-z][a-z0-9]{2,}', s))
    for chinese in re.findall(r'[\u4e00-\u9fff]+', s):
        for n in (2, 3):
            terms.update(chinese[i:i+n] for i in range(max(0, len(chinese)-n+1)))
    terms.difference_update({'什么','怎么','你们','我们','他们','一个','这个','那个','可以','是否','没有','还是','还有','因为','所以','当时','现在','之前','之后','具体','觉得','时候','事情','经历','问题','回答','已经','后来','一下','一些','如何','自己','这样','那样','知道','告诉','对于'})
    return terms


def _alias_overlap(question: str, candidate: str) -> int:
    return sum(bool(any(x in question for x in group)) and bool(any(x in candidate for x in group)) for group in ALIASES)


def identify_question(question: str, history: list[dict] | None = None) -> dict:
    q = _clean(question)
    aspects = [name for name, cues in ASPECTS.items() if any(c in q for c in cues)]
    turns = [t for t in (history or []) if isinstance(t, dict)]
    followup = any(c in q for c in FOLLOWUP) or (len(_terms(q)) <= 2 and bool(turns))
    # Last question is used only as a retrieval anchor for elliptical follow-ups.
    anchor = _clean(turns[-1].get('question', '')) if followup and turns else ''
    return {'question':question, 'aspects':aspects or ['open'],
            'history_relation':'contextual_followup' if followup and turns else 'independent_or_explicit',
            'retrieval_query':(q + ' ' + anchor[:180]).strip(),
            'current_focus':'Answer the CURRENT question and its requested angle; do not narrate other angles.',
            'anchor_from_previous_question':bool(anchor)}


def _score(query: str, text: str, theme: str = '') -> float:
    qt = _terms(query)
    content = _clean(text + ' ' + theme)
    ct = _terms(content)
    if not qt or not ct:
        return 0.0
    shared = qt & ct
    # 3-grams are more specific than 2-grams. Long summaries must not win by length.
    strong = sum(1.9 if len(t) >= 3 else 0.8 for t in shared)
    alias = _alias_overlap(query, content) * 2.3
    return round(strong / max(2.8, len(qt) ** 0.48) + alias, 3)


def select_memory(card: list[dict], question: str, history: list[dict] | None = None,
                  *, max_groups: int = 3, max_chars: int = 2300) -> tuple[list[dict], dict]:
    """Returns subset of completed card, non-sensitive selection telemetry.

    Zero matches => ZERO story snippets, not a random top-N fallback.  A broad
    follow-up may use the previous question as a disambiguation anchor.
    """
    analysis = identify_question(question, history)
    candidates = []
    for index, group in enumerate(card):
        if not isinstance(group, dict):
            continue
        facts = [str(v).strip() for v in (group.get('facts') or []) if str(v).strip()]
        theme = str(group.get('theme') or '')
        if not facts:
            continue
        # Rank individual facts for Python-prepared groups too, not just entire chapters.
        for fact_index, fact in enumerate(facts):
            score = _score(analysis['retrieval_query'], fact, theme)
            if score > 0:
                candidates.append((score, index, fact_index, fact, theme))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not candidates:
        return [], dict(analysis, selection_status='no_relevant_memory', selected_groups=0,
                        selected_chars=0, total_groups=len(card), scores=[])
    top_score = candidates[0][0]
    # Strict enough to block superficial overlaps, tolerant of meaningful short Chinese questions.
    min_score = max(0.95, top_score * .42)
    selected = []
    used_group_ids = set()
    length = 0
    for score, index, _, fact, theme in candidates:
        if score < min_score or len(used_group_ids) >= max_groups:
            break
        if index in used_group_ids:
            continue
        # Do not silently truncate the source: if too long, skip it completely.
        if len(fact) > max_chars or length + len(fact) > max_chars:
            continue
        selected.append({'theme':theme, 'facts':[fact]})
        used_group_ids.add(index)
        length += len(fact)
    # If even the highest candidate cannot fit, don't inject an arbitrary fragment.
    return selected, dict(analysis, selection_status=('selected' if selected else 'no_fitting_relevant_memory'),
                          selected_groups=len(selected), selected_chars=length,
                          total_groups=len(card), scores=[p[0] for p in candidates[:3]])
