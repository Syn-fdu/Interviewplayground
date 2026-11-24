"""v0.9 public-cue sampling and evidence-grounded transcript memory normalization.

PUBLIC_POOL is non-factual, scenario-independent narrative *scaffolding*, NOT a
replacement for personal experience. All source-derived units keep traceable
turn/hash evidence and remain marked as requiring human semantic review.
No network, cloud, or model-based claims of validated entailment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path

from scenario_builder import scrub, tokens

POOL_SCHEMA = "public_memory_pool.v1"
POOL_USE = "non_factual_prompting_only"
DEFAULT_PATH = Path(__file__).resolve().parent / "memory_pools" / "default_public_memory_pool.json"
MIN_PUBLIC_CARDS = 50
PUBLIC_DRAW = 50
MAX_POOL = 3000
MAX_POOL_BYTES = 2 * 1024 * 1024

class PoolError(ValueError):
    pass


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_pool(raw):
    if not isinstance(raw, dict) or raw.get("schema_version") != POOL_SCHEMA:
        raise PoolError("公共池需要 schema_version=public_memory_pool.v1")
    if raw.get("usage") != POOL_USE:
        raise PoolError("公共池只支持 non_factual_prompting_only，不能把外部素材冒充本人事实")
    cards = raw.get("memories")
    if not isinstance(cards, list) or not MIN_PUBLIC_CARDS <= len(cards) <= MAX_POOL:
        raise PoolError("公共池必须包含 50–3000 条候选，以确保每轮能抽取 50 条")
    result = []
    seen_ids = set()
    seen_phrases = set()
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            raise PoolError(f"第 {idx + 1} 条公共池素材必须是对象")
        mid, category = card.get("memory_id"), card.get("category")
        zh, en = card.get("cue_zh"), card.get("cue_en")
        if not isinstance(mid, str) or not re.fullmatch(r"P[A-Za-z0-9_-]{2,35}", mid) or mid in seen_ids:
            raise PoolError(f"第 {idx + 1} 条 ID 无效或重复；使用 P 开头的唯一 ID")
        if not isinstance(category, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,35}", category):
            raise PoolError(f"{mid} 的 category 无效")
        if any(not isinstance(t, str) or not 24 <= len(t.strip()) <= 380 for t in (zh, en)):
            raise PoolError(f"{mid} 缺少有效的中英文非事实线索（各 24–380 字符）")
        zh, en = zh.strip(), en.strip()
        key = (zh.lower(), en.lower())
        if key in seen_phrases:
            raise PoolError(f"公共池包含重复内容：{mid}")
        seen_ids.add(mid)
        seen_phrases.add(key)
        result.append({"memory_id": mid, "category": category,
                       "cue_zh": zh, "cue_en": en, "factual_evidence": False})
    draw=raw.get("sample_per_turn",PUBLIC_DRAW)
    if type(draw) is not int or not 1 <= draw <= min(len(cards),200):
        raise PoolError("每轮抽取数量须为 1–200 之间的整数，且不能超过池容量")
    normalized = {"schema_version": POOL_SCHEMA, "usage": POOL_USE,"sample_per_turn":draw,
                  "name": str(raw.get("name") or "External public memory pool")[:120],
                  "memories": result}
    if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_POOL_BYTES:
        raise PoolError("公共池总量超过 2 MiB 限制")
    return normalized


def default_pool():
    return validate_pool(json.loads(DEFAULT_PATH.read_text(encoding="utf-8")))


def pool_summary(pool, source="default"):
    return {"schema_version":POOL_SCHEMA, "usage":POOL_USE,
            "source":source, "name":pool["name"],
            "total":len(pool["memories"]), "sample_per_turn": pool.get("sample_per_turn",PUBLIC_DRAW),
            "category_count":len({x["category"] for x in pool["memories"]}),
            "sha256":_hash(pool)}


def sample_public_cues(pool, *, seed, scenario_id, turn, question, language="zh"):
    """Diversified, repeatable 50-card sample for each answer attempt.

    A retry with the same scene, seed, turn and question gets identical cards.
    Sampling does not depend on global RNG or change disclosure state.
    """
    cards = pool.get("memories", [])
    draw=pool.get("sample_per_turn",PUBLIC_DRAW)
    if type(draw) is not int or not 1 <= draw <= min(len(cards),200):
        raise PoolError("公共池抽取数量超过有效容量")
    salt = f"{seed}\0{scenario_id}\0{turn}\0{question}"
    rnd = random.Random(int(hashlib.sha256(salt.encode("utf-8")).hexdigest(),16))
    groups = {}
    for card in cards:
        groups.setdefault(card["category"], []).append(card)
    for group in groups.values():
        rnd.shuffle(group)
    # Spread across categories; avoid one topic consuming all context slots.
    chosen = []
    categories = sorted(groups)
    rnd.shuffle(categories)
    while len(chosen) < draw:
        progressed = False
        for category in categories:
            if groups[category] and len(chosen) < draw:
                chosen.append(groups[category].pop())
                progressed = True
        if not progressed:
            break
    assert len(chosen) == draw
    return [{"memory_id":c["memory_id"], "category":c["category"],
             "cue":c["cue_en"] if language=="en" else c["cue_zh"],
             "factual_evidence": False} for c in chosen]

# V0.9.1: compact PROMPT-ONLY representations. The original 50 sampled
# cards and all IDs remain intact in the private trace. Shortening common
# disclaimers and serialisation overhead does not change the random draw.
_DEFAULT_ZH_LEAD = re.compile(r"^一般情境线索：在「[^」]{1,110}」中[，,:：\s]*")
_DEFAULT_ZH_TAIL = re.compile(r"[；;]?这不说明受访者本人经历过这种情况[。.!！\s]*$")
_DEFAULT_EN_LEAD = re.compile(r"^(?:(?:Generic|General) (?:context|scenario|narrative) cue|General context for [^:]{1,180})\s*[:：]\s*", re.I)
_DEFAULT_EN_TAIL = re.compile(r"[;.]?\s*(?:This|It) does not (?:mean|imply|indicate|establish) (?:that )?(?:the )?(?:participant|respondent).*?$", re.I)


def compact_public_cues(sample, language='zh'):
    """Preserve all 50 sampled cue IDs with one short nonfactual idea each.

    This is presentation compression, not retrieval or an evidence filter.
    Truncation is explicit and never used to assert facts about a person.
    """
    compact=[]
    for item in sample:
        text=str(item.get('cue','')).strip()
        if language=='en':
            text=_DEFAULT_EN_TAIL.sub('', _DEFAULT_EN_LEAD.sub('',text)).strip()
            limit=98
        else:
            text=_DEFAULT_ZH_TAIL.sub('',_DEFAULT_ZH_LEAD.sub('',text)).strip()
            limit=64
        text=re.sub(r'\s+', ' ',text).strip(' ，,；;。 .')
        if len(text)>limit:
            # Text prefix only; do not fabricate continuations or personal facts.
            cut=max(text.rfind(c,0,limit) for c in ('，',',','；',';','。','.',' '))
            if cut<limit//2: cut=limit
            text=text[:cut].rstrip(' ，,；;。 .')+'…'
        compact.append(f"{item['memory_id']}|{item['category']}|{text}")
    return compact


# Conservative transcript cleanup: never assert facts absent from the supplied span.
_FILLERS_EN = re.compile(r"^(?:yeah|um+|uh+|erm+|well|okay|ok|right|you know|i mean|sort of|kind of)[,.:;\s]+",re.I)
_FILLERS_ZH = re.compile(r"^(?:嗯+|呃+|啊+|那个|就是|然后|对+|是的|其实|我觉得吧)[，,、。\s]+")
_LOW = re.compile(r"^(?:ok(?:ay)?|yeah|yes|no|right|fine|great|thank\s+you|i\s+don't\s+know|i\s+am\s+not\s+sure|嗯+|啊+|好的|对|是|不是|谢谢|不知道|没印象|记不清)[.!?\s。！？]*$",re.I)
_ROLE = re.compile(r"^\s*(?:respondent|participant|interviewee|speaker\s*\d+|答|受访者)\s*[:：-]\s*",re.I)


def _clean_span(text, name=None):
    s = scrub(text, name, 2600)
    s = _ROLE.sub("", s)
    s = re.sub(r"\[(?:laughter|laughs|inaudible|crosstalk|pause|silence|noise)\]", " ", s, flags=re.I)
    s = re.sub(r"\((?:laughs?|inaudible|pause)\)", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" \t\r\n-—，,;；")
    for _ in range(3):
        s2 = _FILLERS_EN.sub("", _FILLERS_ZH.sub("", s))
        if s2 == s: break
        s = s2
    return s.strip()


def _segments(text):
    # Strong sentence boundaries, without splitting decimal numbers or every comma.
    pieces = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[\"“A-Z])|\n{2,}",text)
    # Don't cap this to 4 units; preserve information from long responses.
    return [p.strip() for p in pieces if p.strip()]


def _quality_issue(span):
    plain = span.strip(" .,!?:;，。！？、:：;；\"'“” ")
    if not plain or _LOW.fullmatch(plain):
        return "low_information_acknowledgement"
    has_chinese=bool(re.search(r"[\u3400-\u9fff]",plain))
    words=tokens(plain)
    if has_chinese:
        # Concise Chinese statements can hold meaningful facts in 8–12 chars.
        if len(plain)<8 or len(words)<3:
            return "too_short_or_generic"
    else:
        concrete_first_person=bool(re.search(
            r"^(?:I|We)\s+(?:am|was|were|work|worked|live|lived|have|had|feel|felt|prefer|preferred|study|studied|did|used)\b",plain,re.I))
        if len(plain)<(12 if concrete_first_person else 16) or (len(words)<3 and not concrete_first_person):
            return "too_short_or_generic"
    if len(plain)>700:
        return "overlong_unsegmented"
    if len(set(plain))<=5:
        return "low_character_diversity"
    if re.search(r"(?:\b(?:a|ha|hm)\b\W*){8,}",plain,re.I):
        return "repeated_filler"
    return ""


def _restatement(span):
    """Conservative, auditable third-person rewrite; NOT semantic entailment proof."""
    s=span.strip()
    if re.search(r"[\u3400-\u9fff]", s):
        # Third-person, explicit epistemic attribution; no invented causality.
        s = re.sub(r"^我(?=在|当|曾|从|对|会|有|没|很|觉|认|要|想|做|用|通|学|工|经|以|后|选|找|跟|和|与|不|也|就|因|还|希)","受访者",s)
        if s.startswith("我们"):
            s = "受访者提到，其所在群体" + s[2:]
        if not s.startswith(("受访者", "据受访者")):
            s = "受访者谈到：" + s
    else:
        # Preserve tense, grammatical subject and level of certainty when possible.
        s = re.sub(r"\bI\s+am\b","the participant is",s,flags=re.I)
        s = re.sub(r"\bI\s+was\b","the participant was",s,flags=re.I)
        s = re.sub(r"\bI\s+have\b","the participant has",s,flags=re.I)
        s = re.sub(r"\bI\s+had\b","the participant had",s,flags=re.I)
        s = re.sub(r"\bI'm\b","the participant is",s,flags=re.I)
        s = re.sub(r"\bI've\b","the participant has",s,flags=re.I)
        s = re.sub(r"\bmy\b","the participant's",s,flags=re.I)
        s = re.sub(r"\bme\b","the participant",s,flags=re.I)
        s = re.sub(r"\bI\b","the participant",s,flags=re.I)
        if not s.lower().startswith(("the participant", "according to the participant")):
            s = "The participant reported: " + s[0].lower() + s[1:] if s else ""
    return s.strip()


def normalize_source_memory(raw, question, *, name=None, turn_index=None,
                            question_index=None, memory_id="M0001"):
    """Return compact units with source hashes or rejection reasons, never made-up facts.

    Original answer is *not* carried into prompting; raw source snippet is kept in
    researcher-only `source.original_excerpt_preview` for audit (redacted best effort).
    """
    if not isinstance(raw,str):
        return None, ["not_text"]
    cleaned = _clean_span(raw, name)
    if not cleaned:
        return None, ["empty"]
    parts = _segments(cleaned)
    # Keep 1 coherent memory per respondent turn, with several independent units.
    # Ignore repeated and vacuous subspans; never mechanically split into invented claims.
    items, reasons, seen = [], [], set()
    for span in parts[:36]:
        cleaned_part = _clean_span(span,name)
        issue = _quality_issue(cleaned_part)
        if issue:
            reasons.append(issue)
            continue
        normalized = _restatement(cleaned_part)
        digest = hashlib.sha256(cleaned_part.casefold().encode("utf-8")).hexdigest()
        if digest in seen:
            reasons.append("duplicate_span")
            continue
        seen.add(digest)
        items.append({"text": normalized, "source_excerpt_sha256":digest,
                      "verification_status":"auto_normalized_needs_review",
                      "normalization":"speaker_attribution_and_filler_removal_no_invented_claims",
                      "sensitivity":"unknown"})
    if not items:
        return None, reasons or ["no_usable_information"]
    # Prevent primary memory from becoming a pasted transcript.
    memory_content = " ".join(x["text"] for x in items)[:1400]
    content_type = "reflective" if any(re.search(r"(?:I (?:feel|think|realized)|我觉得|我认为|我意识到)",s,re.I) for s in parts) else "episodic"
    source = {"respondent_turn_index":turn_index,
              "interviewer_turn_index":question_index,
              "original_answer_sha256":hashlib.sha256(raw.encode("utf-8")).hexdigest(),
              "original_excerpt_preview":scrub(raw,name,360),
              "source_question_sha256":hashlib.sha256(str(question).encode("utf-8")).hexdigest(),
              "normalization_review":"pending_human_semantic_review"}
    facts=[]
    for idx,unit in enumerate(items,1):
        facts.append({"fact_id":f"F{memory_id[1:]}_{idx:02d}",**unit})
    return {"memory_id":memory_id,"source_question":scrub(question,name,300),
            "text":memory_content,"type":content_type,
            "facts":facts,"source":source,"importance":min(1.,.4+.07*len(items)),
            "source_type":"source_grounded_normalized_not_gold",
            "confidence":"provisional_requires_semantic_review"}, reasons
