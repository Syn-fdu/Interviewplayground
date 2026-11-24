"""Post-preparation-only controls for the six-trait interview experiment.

All functions operate on already prepared memory, without changing preparation,
caches, stored source facts or automated retry behavior.
"""
from __future__ import annotations
import re

TRAITS = ('knowledge','understanding','reflexivity','memory','verbosity','disclosure')

def normalize(traits):
    incoming = traits or {}
    return {key: max(0.0,min(1.0,float(incoming.get(key,0.5)))) for key in TRAITS}

def memory_budget(memory):
    """Backward-compatible maximum count and length, used after preparation only."""
    m=max(0.0,min(1.0,float(memory)))
    if m < .25:return (1,420)
    if m < .75:return (2,900)
    return (3,1350)

def memory_access_level(value):
    """A reproducible post-preparation memory accessibility band."""
    v=max(0.0,min(1.0,float(value)))
    return 'gist' if v < .25 else ('key_facts' if v < .75 else 'precise_details')

def verbosity_guide(value):
    v=max(0.0,min(1.0,float(value)))
    if v < .25:return ('尽量用1到2句回答，目标30至65个汉字。必要时可略长以保证完整，不重复无关细节。',460)
    if v < .75:return ('通常2到4句，目标70至140个汉字；回答问题后补充必要事实。',800)
    return ('通常4到7句，目标130至230个汉字；多给有根据的具体信息，不凑字数。',1350)

_SENSITIVE_CUES = ('私人','隐私','家人','家庭','焦虑','担忧','难堪','尴尬','压力','错误','失误','后悔',
                   '感受','情绪','害怕','脆弱','不愿','保密','个人','同事的表现')
_MEDIUM_CUES = ('工作失误','困难','责任','失败','评价','担心','感受','怎么看自己')

def question_sensitivity(question, relevant_facts=None):
    """Deterministic disclosure context, not a prohibition or permission grant.

    Questions and explicitly relevant FACTS are read only. A high-sensitivity
    source fact never authorizes disclosing protected names/contact information.
    """
    q=str(question or '')
    if any(c in q for c in ('家人','家庭','私人','隐私','情绪','压力','不愿','保密','联系方式')):
        return 'personal'
    if any(c in q for c in _MEDIUM_CUES):
        return 'evaluative'
    if relevant_facts:
        levels={str(f.get('sensitivity','low')) for f in relevant_facts if isinstance(f,dict)}
        if 'high' in levels and any(c in q for c in _SENSITIVE_CUES):return 'personal'
        if 'medium' in levels and any(c in q for c in _SENSITIVE_CUES):return 'evaluative'
    return 'ordinary'

def disclosure_guide(value, sensitivity='ordinary'):
    """Post-preparation answer policy; same evidence is visible in both conditions.

    Low disclosure is withholding personal detail by choice, not factual amnesia.
    High disclosure is candid only about source-supported, authorized experiences.
    No hard-coded answer or forced refusal is returned by this function.
    """
    v=max(0.0,min(1.0,float(value)))
    level={'ordinary':'S0：一般经历与工作事实','evaluative':'S1：失误、责任及个人评价',
           'personal':'S2：压力、感受、家人和私人经历'}.get(sensitivity,'S0：一般经历与工作事实')
    boundary=('【不可逾越】只说当前给定的许可事实，不要编造家人回应、同事情绪、想法、后续结果。'
              '不强制编造。不可公开原始资料禁止披露的姓名、联系方式或同事私人资料。'
              '用户的问题要求私人细节，不等于受访者有义务全部披露。'
              '你是受访者而不是研究者；全程用自然的中文第一人称说话，不介绍参数、规则或事实编号。'
              '拒绝、回避、简短作答都属于正常回答，不得触发异常或中断。')
    if v < .25:
        if sensitivity=='ordinary':
            decision=('普通事实题正常回答问题涉及的公开工作事实，但不要主动延伸到个人压力、尴尬或家庭。')
        elif sensitivity=='evaluative':
            decision=('你愿意简短承认发生过失误以及已采取的公开处理措施，但不愿意向采访者展开'
                      '自己的羞愧、担忧、脆弱感受或私人心理变化。'
                      '可以自然表示愿意承认处理过失误，但不想讨论当时的心情。'
                      '此句只作表达方式示例，禁止机械照抄；尤其不要在后面又完整讲出被保留的私人感受。')
        else:
            decision=('你重视个人边界与隐私，对家庭、情绪和私人压力保持明确边界。'
                      '面对直接询问可以简短承认存在相关经历，但不讲家庭互动、心情变化、私人细节。'
                      '如果问题主要要求私人细节，可以自然说不愿深入，而不是为了回答完整而逐项泄露。'
                      '不要在拒绝之后补充被拒绝的细节。尤其不要先说不想谈感受，又逐项披露担忧、害怕或对他人的猜测。')
        choice='【低披露决策：保留优先】先决定什么不愿说，再组织回答；宁可自然保留，也不要完整复述私人事实。'
    elif v < .75:
        decision=('普通事实直接回答；面对失误或私人问题可以适度透露有依据的经历，'
                  '但不过度展开与当前问题无关的细节。')
        choice='【中性披露决策】自然、适度作答。'
    else:
        if sensitivity=='ordinary':
            decision='清楚回答普通工作事实，不强行转到个人隐私。'
        elif sensitivity=='evaluative':
            decision=('如果许可事实支持，坦诚承认具体失误、当时已有记录的个人感受、真实担忧及处理措施。'
                      '不要只给抽象官样回应，也不要凭空添加团队原谅、后来升职等没有记录的结果。')
        else:
            decision=('在许可范围内愿意真实谈论自己曾有的压力、情绪及家庭沟通；'
                      '针对问题提供具体且得到事实支持的内容，少用泛泛的“有些压力”。'
                      '没有记录的情绪程度或对方反应不能自行想象；不强制编造。')
        choice='【高披露决策：坦诚开放】在不越过许可边界前提下，优先清楚表达自己愿意分享的具体个人经历。'
    return f'【Disclosure本轮判断】{level}；{choice}{decision}{boundary}'

def memory_behavior_guide(value):
    level=memory_access_level(value)
    common=('【记忆身份规则】你是亲历事件的当事人，不是旁观者或分析员。'
            '只根据这一轮实际提供的记忆回答，以“我”的自然口吻讲亲身经历。'
            '不要说“受访者”“资料显示”“事实编号”“根据记忆卡片”，也不要把经历讲成他人的故事。'
            '没有依据时诚实承认记不清，绝不凭常识补数字或编造行为；'
            '当前对话中刚说过的事情不应突然遗忘。')
    if level=='gist':
        return (common+'【低记忆】你的过去经历仍有轮廓；对具体日期、确切人数、比例和先后细节容易模糊。'
                '回答时先说确实记得的事件大意；当问题追问精确数字而本轮记忆没有数字时，'
                '直接承认具体值记不清，不要给出猜测的数字、月份、人物或事件步骤。'
                '如果没有可支持的第三周活动，不得编造第三周发生了什么。'
                '不要把“不记得细节”说成“从没发生过”。')
    if level=='key_facts':
        return (common+'【中等记忆】能说出主要事件和当前提供的关键细节，缺失细节时保持适度不确定。')
    return (common+'【高记忆】主动从本轮允许的事实中提取准确数字、时间、顺序与结果。'
            '如果问题同时询问多个已知细节，尽可能逐一回答，避免把不同经历混合。'
            '只有当证据中确实没有答案时才说不知道，切忌制造额外事实。')

def behavioral_guide(traits, *, sensitivity='ordinary'):
    p=normalize(traits)
    return (f"属性设置：知识{p['knowledge']:.1f}、理解{p['understanding']:.1f}、反思{p['reflexivity']:.1f}、"
            f"记忆{p['memory']:.1f}、详细度{p['verbosity']:.1f}、披露{p['disclosure']:.1f}。"
            'Knowledge：访前从已分类领域 KnowledgeBank 中按比例选择通用知识并写入缓存；访谈时只可引用当前准备缓存中检索到的背景知识，绝不冒充本人经历。'
            'Understanding：低值更倾向请求澄清复杂问句，高值更善于抓住问句的限定和重点；不得故意曲解简单问题。'
            'Reflexivity：低值主要讲事件，高值更愿意探讨自己已有依据的看法、教训和动机；不能虚构心理活动。'
            +memory_behavior_guide(p['memory'])
            +disclosure_guide(p['disclosure'],sensitivity)
            +'重复问题可自然表明已说过，不得因重复中止或抛出错误。')
