"""Source-only, high-recall fallback for factbank retrieval misses.

No model call and no invented autobiographical facts.  Read only permission-filtered
source records.  A retrieval miss is NOT a failure of human memory access.
"""
from __future__ import annotations
import re

# Cross-language topic aliases.  These are retrieval hints, NEVER evidence.
TOPICS = {
    'frequency': (r'频率|多频繁|经常|多久|次数|每(天|周)|how often|frequency',
                  r'\boften\b|\bfrequently\b|\brarely\b|\bsometimes\b|\bnever\b|\bvery often\b|经常|频繁|偶尔|很少|每天|每周',
                  r'经常|频率'),
    'coding': (r'编程|写代码|代码|开发|功能|调试|排查|bug|coding|programming|feature',
               r'coding|programming|code|feature|debug|bug|copilot|development|代码|编程|开发|功能|调试',r'代码|开发'),
    'debug': (r'调试|排查|修复|bug|debug',r'bug|debug|fix|problem|排查|修复|调试',r'调试|修复'),
    'feature': (r'新功能|写功能|功能|feature',r'feature|functionality|implement|add|新功能|功能',r'功能'),
    'example': (r'例子|案例|经历|最近一次|具体说说|recent|example|experience',
                r'recent|example|instance|last time|had to|once|最近|那次|例子|案例',r'案例|最近'),
    'start': (r'最早|什么时候开始|何时开始|多久前|start|began|first',
              r'when did you start|started|months|years|began|first|第一次|最初|开始|大学期间|个月前',r'开始|时间'),
    'integration': (r'集成|开发环境|ide|vscode|vs code|integration',
                    r'integration|ide|visual studio|vs code|vscode|plugin|copilot|集成|插件',r'集成|开发环境'),
    'tools': (r'哪些工具|什么工具|工具|copilot|chatgpt|tool',
              r'copilot|chatgpt|claude|gemini|tools|ide|工具',r'工具'),
    'opinion': (r'优缺点|利弊|评价|怎么看|看法|pros and cons|opinion',
                r'优势|劣势|优点|缺点|利弊|认为|看法|好处|benefit|drawback|advantage|disadvantage|opinion',r'看法|利弊'),
    'recording': (r'录音|录制|隐私|同意录|recording|consent',
                  r'recording|recorded|audio|consent|privacy|录音|隐私|录制',r'录音|隐私'),
}

def _tokens(t):
    return set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}', t.lower()))

def _intent(question):
    q=question.lower()
    return {k for k,(query,_,_) in TOPICS.items() if re.search(query,q,re.I)}

def _score(question, record):
    q=question.lower()
    answer=str(record.get('text') or '')
    source_q=str(record.get('source_question') or '')
    a=(answer+' '+source_q).lower()
    intent=_intent(q)
    if not intent: return 0.0
    # Safety: no recording-consent evidence may be replaced by general AI facts.
    if 'opinion' in intent and not re.search(TOPICS['opinion'][1],a,re.I):
        return 0.0
    if 'recording' in intent and not re.search(TOPICS['recording'][1],a,re.I):
        return 0.0
    matched={name for name in intent if re.search(TOPICS[name][1],a,re.I)}
    # Require domain overlap. Frequency alone should not retrieve every 'often'.
    domain=intent.intersection({'coding','debug','feature','integration','tools','recording'})
    if domain and not matched.intersection(domain): return 0.0
    if 'frequency' in intent and 'frequency' not in matched: return 0.0
    if 'start' in intent and 'start' not in matched: return 0.0
    if 'example' in intent and 'example' not in matched: return 0.0
    if 'opinion' in intent and 'opinion' not in matched: return 0.0
    s=2.0*len(matched)
    if 'frequency' in intent and 'frequency' in matched: s+=3
    if 'start' in intent and re.search(r'\b\d+\s*[-–]\s*\d+\s*months?\b',a):s+=5
    if 'example' in intent and re.search(r'recent example|had to add|new feature',a):s+=4
    if 'debug' in intent and re.search(r'bug fixing|debugging',a):s+=3
    if 'integration' in intent and re.search(r'ide integration|with ide',a):s+=3
    # Weak lexical boost, but never enough to turn irrelevant records into matches.
    s+=min(1.5, .25*len(_tokens(q)&_tokens(a)))
    return s

def lookup(question, records, max_items=8):
    """Return cited cached source facts; no answer synthesis or memory bypass.

    Every returned item is an eligible source record with an original source ID.
    """
    ranked=[]
    for rec in records:
        if not isinstance(rec,dict) or not rec.get('id') or not rec.get('text'):continue
        score=_score(question,rec)
        if score>=2.0: ranked.append((score,rec))
    ranked.sort(key=lambda v:(-v[0],v[1]['id']))
    # Prevent weak same-topic answers from blocking a valid cache decision:
    # dimension-specific questions require a strong score relative to the
    # best answer, while ordinary questions retain the broader candidate set.
    intents=_intent(question)
    top=ranked[0][0] if ranked else 0
    floor=max(2.0, top*.70) if intents.intersection({'frequency','start','example'}) else 2.0
    out=[]
    for score,rec in ranked:
        if score<floor or len(out)>=max_items:continue
        out.append({'source_id':rec['id'],'text':rec['text'],
                    'source_question':rec.get('source_question',''),
                    'score':round(score,2),'cache_source':'verified_original_fact',
                    'review_pending':bool(rec.get('review_pending'))})
    return out
