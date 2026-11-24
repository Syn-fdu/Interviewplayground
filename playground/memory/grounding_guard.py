"""Conservative per-answer grounding gate for personal interview narratives.

This does NOT prove arbitrary natural-language entailment. It prevents the most
serious class of fabrication: inventing biographical details when nothing was
recalled, and introducing named tools, numbers or dates unsupported by the
personal-memory context supplied for that turn.
"""
import re

ABSTAIN_ZH="这方面我确实想不太起来了，没法准确说出具体情况。"
ABSTAIN_EN="I can't reliably remember those details, so I don't want to guess."

# Identifiable details frequently hallucinated in AI programming interviews.
_ENTITIES=re.compile(
    r"(?<![A-Za-z0-9])(?:GitHub\s*Copilot|ChatGPT|Swagger\s*UI|VS\s*Code|"
    r"PyCharm|Jupyter\s*Notebook|scikit[- ]learn|TensorFlow|PyTorch|"
    r"Python|JavaScript|Java|IDE|API|OpenAI|Ollama|DeepSeek|"
    r"Copilot|Docker|Kubernetes)(?![A-Za-z0-9])",re.I)
_NUMERIC=re.compile(r"(?<![A-Za-z0-9])(?:\d{4}年|\d+(?:\.\d+)?\s*(?:年|个月|岁|次|个|%|％)|大[一二三四]年)(?![A-Za-z0-9])")

def _canonical(value):
    return re.sub(r"[\s_\-]+","",str(value).lower())

def verify(answer,personal_memory,language="zh"):
    """Return (safe_text, telemetry); never treat question/history as evidence."""
    sources=" ".join(str(f) for item in personal_memory for f in item.get("facts",[]))
    if not sources.strip():
        return (ABSTAIN_EN if language=="en" else ABSTAIN_ZH,
                {"grounding_gate":"no_personal_source","blocked":True,
                 "unsupported_named_details":[]})
    unsupported=[]
    norm=_canonical(sources)
    for match in list(_ENTITIES.finditer(answer))+list(_NUMERIC.finditer(answer)):
        term=match.group()
        if _canonical(term) not in norm:
            unsupported.append(term)
    if unsupported:
        # Never log the unsupported claims themselves to the exported report:
        # only count and reason. The raw answer is discarded.
        return (ABSTAIN_EN if language=="en" else ABSTAIN_ZH,
                {"grounding_gate":"unsupported_specific_detail","blocked":True,
                 "unsupported_named_detail_count":len(set(map(str.lower,unsupported)))})
    # An answer can still invent a *generic* project or career story without
    # mentioning named tools. Compare content-bearing bigrams with the actual
    # recovered personal source, excluding ordinary functional words.
    from playground.memory.question_first_retrieval import _terms
    source_terms=_terms(sources)
    answer_terms=_terms(answer)
    neutral={"使用","工作","进行","完成","之后","当时","因为","开始","通过",
             "比如","可以","可能","还有","需要","这个","那个","比较","一些",
             "具体","实际","情况","主要","通常","负责","帮助","自己","我们",
             "感觉","觉得","问题","没有","是的","目前","之前","回答"}
    novel={t for t in answer_terms-source_terms if t not in neutral}
    # Absolute and relative thresholds avoid vetoing short harmless answers.
    novelty=len(novel)/max(1,len(answer_terms))
    # Source interviews may be English while respondent replies in Chinese.
    # Character n-gram "novelty" is meaningless across languages: the old
    # gate rejected correct natural translations and forced source pasting.
    source_latin=len(re.findall(r"[A-Za-z]",sources))
    source_hanzi=len(re.findall(r"[一-鿿]",sources))
    answer_latin=len(re.findall(r"[A-Za-z]",answer))
    answer_hanzi=len(re.findall(r"[一-鿿]",answer))
    cross_language=(source_latin>source_hanzi*2 and
                    source_latin>=25 and answer_hanzi>=8)
    if (not cross_language and len(answer_terms)>=16 and
            len(novel)>=14 and novelty>=.86):
        return (ABSTAIN_EN if language=="en" else ABSTAIN_ZH,
                {"grounding_gate":"novel_unverified_narrative","blocked":True,
                 "unsupported_term_count":len(novel)})
    return answer,{"grounding_gate":"limited_lexical_audit","blocked":False,
                   "unsupported_named_detail_count":0,
                   "lexical_novelty_ratio":round(novelty,3),
                   "cross_language_paraphrase":cross_language}
