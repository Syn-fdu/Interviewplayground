"""Respondent interaction routing and human conversational behavior.

Current conversation is valid information about what was just agreed, but it
must never be promoted into historical autobiographical facts. No new LLM
planner is needed; the existing respondent model remains the sole generator
for free-form responses.
"""
from __future__ import annotations
import re
from question_intent import classify, RECORDING
from source_recall import _question_dimension

AFFIRM=re.compile(
    r"这样.{0,22}(?:可以|行|接受|同意|没问题)|"
    r"(?:这些|上述|刚才|之前).{0,24}(?:条件|安排|要求|说的|约定)|"
    r"(?:可以|愿意|能够|能否|是否).{0,20}(?:接受|继续|同意|确认)|"
    r"(?:do you|are you).{0,24}(?:agree|accept|consent|still comfortable)",
    re.I)
CLARIFY=re.compile(
    r"刚才.{0,20}(?:意思|说的|提到)|你(?:是)?(?:指|说)的|"
    r"您(?:是)?(?:指|说)的|能再解释|能否澄清|什么意思|"
    r"what did you mean|could you clarify",re.I)
SITUATIONAL=re.compile(
    r"听(?:得)?清|听明白|看得到|能看到|听见|"
    r"要不要(?:休息|暂停)|是否(?:需要|愿意|方便).*?(?:休息|暂停|继续)|"
    r"audio (?:is|okay)|can you (?:hear|see)|ready to proceed",
    re.I)
BOUNDARY=re.compile(r"不方便回答|不愿意回答|不想回答|太私人|敏感信息|"
                    r"refuse to answer|too personal",re.I)
REASON=re.compile(r"为什么|您怎么看|你怎么看|你如何看|您如何看|"
                  r"would you say|why do you think",re.I)

def _recent(history):
    return [x for x in (history or []) if isinstance(x,dict)][-4:]

AUTOBIOGRAPHICAL_REQUEST=re.compile(
    r"哪些.{0,18}(?:任务|工作|项目|工具|活动|类型|经历)|"
    r"(?:主要|平时|日常|通常|具体).{0,25}(?:做|用|使用|处理|协助|完成|工作)|"
    r"能(?:否|不能|不).{0,18}(?:讲|聊|说|回忆|描述)|"
    r"您.{0,25}(?:使用|利用|尝试|遇到|经历|做过|选择|开发|处理|参与)|"
    r"你.{0,25}(?:使用|利用|尝试|遇到|经历|做过|选择|开发|处理|参与)|"
    r"最早|第一次|具体项目|怎么用|怎么做|工作习惯|"
    r"how often|which (?:tasks|tools|projects)|what (?:tasks|projects) "
    r"(?:do|did) you|how (?:do|did) you use",
    re.I
)

def requires_personal_experience(question,history=None):
    """Identify factual follow-up *after* retrieval, not as a retrieval gate.

    When uncertain, prefer autobiographical handling to a canned generic
    clarification. This classifier cannot authorize unsourced facts.
    """
    q=str(question or "").strip()
    intent=classify(q)
    # Current consent / recording arrangements are live social acts; a
    # normal "您怎么看录音" is not a request for a past personal method.
    if intent["kind"]=="procedural" and not re.search(
        r"过去|之前.{0,10}(?:录音|经历)|上次.{0,10}录音",q):
        return False
    # A broad opinion ("您怎么看待 AI 行业的发展") contains the ordinary
    # Chinese word "怎么" but is not a request for the person's past method.
    if intent["kind"] in ("general_opinion","general_knowledge") and not re.search(
        r"亲身|自己的经历|曾经|过去|我为什么|您为什么|你为什么|"
        r"您.{0,14}(?:遇到|经历|做过|使用过)|"
        r"你.{0,14}(?:遇到|经历|做过|使用过)",q):
        return False
    if AUTOBIOGRAPHICAL_REQUEST.search(q):
        return True
    if _question_dimension(q) in (
        "frequency","onset_time","specific_episode","method","reason",
        "impact","comparison","challenges"
    ) and re.search(r"您|你|刚才|此前|之前|工作|项目|任务",q):
        return True
    if re.search(r"在.{0,10}(?:开发|工作)|(?:AI|工具).{0,16}(?:哪|什么|如何|怎么)",q,re.I):
        return True
    return False

def route(question,history=None):
    """Propose a conversational response mode, never block Memory.

    Actual Memory probing happens before this hint can decide final behavior.
    """
    intent=classify(question)
    q=str(question or "").strip()
    recent=_recent(history)
    last_answer=str(recent[-1].get("answer") or "") if recent else ""
    if requires_personal_experience(q,history):
        return {"mode":"recall","trigger_memory":True,
                "reason":"personal_question_not_just_clarification",
                "intent":intent}
    if SITUATIONAL.search(q):
        return {"mode":"conversational","trigger_memory":False,
                "reason":"live_interview_situation","intent":intent}
    if AFFIRM.search(q) and (recent or RECORDING.search(q)):
        if (RECORDING.search(q) or
            re.search(r"录音|保密|公开|妥善|可以|接受|同意|继续|研究|用途",last_answer)):
            return {"mode":"conversational","trigger_memory":False,
                    "reason":"confirm_existing_conversation","intent":intent}
    if intent["kind"]=="procedural":
        return {"mode":"conversational","trigger_memory":False,
                "reason":intent["reason"],"intent":intent}
    # A reference to what was said earlier is not necessarily a request
    # to clarify wording: "how often do you do each activity?" requires
    # autobiographical frequency recall even if it mentions "刚才".
    frequency_question=(_question_dimension(q)=="frequency")
    if CLARIFY.search(q) and recent and not frequency_question and not re.search(
        r"当时|具体项目|哪年|最早|第一次|发生了什么|那次经历",q):
        return {"mode":"clarification","trigger_memory":False,
                "reason":"clarify_current_conversation","intent":intent}
    if frequency_question:
        return {"mode":"recall","trigger_memory":True,
                "reason":"personal_frequency_requires_source","intent":intent}
    if (intent["kind"]=="general_knowledge" or
        (not requires_personal_experience(q,history)
         and re.search(r"(?:你|您)(?:了[解]|知道|懂|听说过).{1,35}(?:吗|么|？|\\?)",
                       q,re.I)
         and not re.search(r"过去|以前|第一次|当时|具体项目|亲身|你做过|您做过",q))):
        return {"mode":"reasoning","trigger_memory":False,
                "reason":"general_knowledge_after_recall","intent":intent}
    if intent["kind"]=="general_opinion" and not re.search(
        r"亲身|自己的经历|曾经|过去|我为什么|您为什么|你为什么",q):
        return {"mode":"reasoning","trigger_memory":False,
                "reason":"general_opinion","intent":intent}
    if BOUNDARY.search(q) and not re.search(r"经历|具体|事件|当时",q):
        return {"mode":"boundary","trigger_memory":False,
                "reason":"present_conversational_boundary","intent":intent}
    return {"mode":"recall","trigger_memory":True,
            "reason":"personal_fact_or_related_explanation","intent":intent}


def conversational_reply(question,history,language="zh"):
    """Answer confirmations from present conversational state; not memory."""
    from question_intent import procedural_reply
    q=str(question or "")
    recent=_recent(history)
    last=str(recent[-1].get("answer") or "") if recent else ""
    if AFFIRM.search(q) and (recent or RECORDING.search(q)):
        if language=="en":
            return "Yes, that arrangement works for me. We can continue."
        if re.search(r"保密|公开|用途|使用范围|用于研究|用于这次研究",last):
            return "可以，只要按刚才说好的用途处理录音就行。我们继续吧。"
        return "可以，这样我能接受，我们继续吧。"
    if CLARIFY.search(q) and last:
        # Pure conversational clarification is never a new biographical claim.
        if re.search(r"录音|保密|不公开|用途",last):
            return ("我主要是希望录音只用于这次研究，其他用途先跟我确认。"
                    if language!="en" else
                    "I mainly mean recording for this study, with other uses agreed first.")
    if SITUATIONAL.search(q):
        if re.search(r"听(?:得)?清|听明白|听见|hear",q,re.I):
            return "听得清，可以继续。" if language!="en" else "Yes, I can hear you. We can continue."
        if re.search(r"看得到|能看到|see",q,re.I):
            return "能看到，您继续。" if language!="en" else "Yes, I can see it."
    return procedural_reply(q,language)


def uncertainty_guidance(question,history,language="zh"):
    """Contextualized uncertainty, preserving missing-fact boundary.

    This is a grounding-safe fallback only. Prefer a verified partial-memory
    answer when some relevant facts are available.
    """
    dim=_question_dimension(question)
    if dim=="frequency":
        if language=="en":
            return "I can't put an accurate count on each activity right now."
        return "要说每种任务一天具体用几次，我一下子还真说不准。"
    if language=="en":
        return {
            "onset_time":"I can't place the start date accurately right now.",
            "specific_episode":"I can't pin down a particular instance well enough to describe it.",
            "reason":"I can't really say what drove that particular decision.",
            "method":"I don't remember the steps clearly enough to walk through them.",
        }.get(dim,"I can't recall enough to answer that precisely.")
    if dim=="onset_time":return "最早具体从什么时候开始，我一下子想不准。"
    if dim=="specific_episode":return "让我想想……我现在想不起一个能说清过程的具体例子。"
    if dim=="reason":return "当时具体为什么会那样想，我现在说不太准。"
    if dim=="method":return "具体步骤我一下子记不全，不想说错。"
    if dim=="impact":return "说到具体影响有多大，我现在不太能确定。"
    if dim=="feeling":return "当时到底是什么感受，我现在说不太准。"
    if dim=="comparison":return "要准确比较这两种情况，我现在还真说不准。"
    if dim=="challenges":return "具体遇到过哪些麻烦，我现在一时想不起来。"
    return "这个我一时说不准，您可以换个角度问问。"


def answer_focus_guidance(question, language="zh"):
    """A presentation hint only. Never select retrieval routes or facts here."""
    q=str(question or "")
    dimension=_question_dimension(q)
    if language=="en":
        core=("Answer the precise thing asked in the first sentence; lead with a supported "
              "fact, not a preamble. Give at most one or two useful follow-up details if "
              "the supplied evidence supports them. Do not repeat the question, pad for "
              "length, re-tell the whole previous answer, or invent motives or chronology. "
              "If one detail is unavailable, state only that limitation concisely. "
              "Follow the configured verbosity when evidence genuinely warrants detail.")
        hints={"frequency":"Say how often first, rather than just listing tools.",
               "onset_time":"Give the supported starting time or range first.",
               "specific_episode":"Describe an actual supported episode, or say you cannot recall one.",
               "method":"Describe the supported steps rather than generic benefits.",
               "reason":"Give a supported reason rather than inventing motivation."}
        return core+" "+hints.get(dimension, "")
    core=("【回答焦点与表达】第一句话就回答采访者这轮真正问的对象或维度；"
          "有直接证据就先说答案，不用‘这个问题我想了想’等铺垫。"
          "必要时补充一两项直接相关且有来源的细节，不要为了凑长度讲背景、"
          "推测动机、重复上一轮或把同一事实换几种说法。"
          "对于未获证实的部分，只简短说明那一部分不确定，已知部分仍直接回答。"
          "句数和字数是弹性建议，不要求填满；确有丰富证据时可按既定详略参数展开。"
          "只改变语言组织，不改变来源授权、记忆门控或检索结果。")
    hints={
        "frequency":"问频率：第一句说明经常、偶尔、很少或有证据的频次，不用工具名称代替频率。",
        "onset_time":"问起始时间：第一句说明有依据的时间或区间，不凭空补充开始的原因。",
        "specific_episode":"问具体案例：优先描述有依据的一次事件；只有一般习惯时坦言没有具体案例。",
        "method":"问如何操作：先说明有依据的动作或步骤，而非介绍工具的通用优点。",
        "reason":"问原因：只说有根据的原因，不替受访者虚构动机。",
    }
    if re.search(r"哪(?:一|个|款|种)|什么(?:编辑器|软件|工具|平台|IDE)|which (?:editor|ide|tool|software)",q,re.I):
        return core+"问具体名称：有获准名称就第一句直接说名称；没有就简短说不确定。"
    return core+hints.get(dimension,"")


def natural_style(mode,history,language="zh"):
    if language=="en":
        return ("Speak as a real interview participant: direct, concise, context-aware. "
                "Do not repeat boilerplate, enumerate points, or pretend to be an archive.")
    return (
        "你是在真实访谈现场说话的人，不是检索系统。先回应当前交流动作，"
        "可以简短确认、承接上一轮、澄清、适度迟疑或表示不愿多谈。"
        "不要机械地套用‘没有足够个人经历信息’、不要重复完整前文、"
        "不要写标题、列表或研究报告。事实只能来自获准的个人记忆；"
        "当前访谈刚刚达成的安排可根据对话历史承接，不能变成虚构的过去经历。"
        "提问涉及个人事实而记忆不足时，以问题角度自然说明没想起的具体部分。"
        "尤其注意问题问什么维度就先回答什么维度：问频率回答经常/偶尔/少用，"
        "问次数回答次数或说明没数过，问具体任务就说任务或承认记不清具体哪次。"
        "不要把‘使用什么工具’冒充‘使用多频繁’，也不要把‘通常怎么用’"
        "冒充‘上一次具体做了什么’。对于已有来源的粗略频率，可以自然给出"
        "高低判断；不得推导出处于来源之外的一天几次、每周几次。"
    )


def recording_answer_guidance(language="zh"):
    """Model instruction, not a canned response or a source memory.

    Recording consent is a present-tense conversational decision. Do not
    confuse it with recollection of autobiographical events.
    """
    if language == "en":
        return ("For recording/consent questions, answer as a respondent, not as "
                "a memory search system. A suitable standard reply is: 'I'm "
                "comfortable with recording for accurately documenting this "
                "interview, provided it is kept confidential and used for "
                "the agreed research purpose; please ask before any additional "
                "use.' If asked what would help you feel comfortable, mention "
                "confidentiality, purpose and asking before wider sharing. "
                "Do not claim a historical recording experience or imply that "
                "the interviewer's privacy promises have been verified.")
    return (
        "【录音与同意类问题的标准应答规范】这属于当前访谈现场的同意与感受，"
        "不是过去经历，不要因为没有个人记忆而拒绝回答，也不要谈记忆库或证据不足。"
        "用自然第一人称回答，可参考：‘如果录音只是为了准确记录这次访谈，"
        "并且妥善保管、不随意公开，我可以接受；如果要作其他用途，请先征求我的同意。’"
        "如果问‘怎样更自在’，说明清楚录音用途、保密措施及后续使用前征求同意即可。"
        "这是现场条件性意愿，不能当作本人过去的事实，不得虚构既往录音经历，"
        "也不得声称研究方已经采取了未得到确认的保护措施。"
    )


def _plain_evasion(answer):
    """Reject vacuous transitions without rejecting natural uncertainty."""
    a=re.sub(r"[\s。！？!?,，.；;～~]+", "", str(answer or "").lower())
    return a in {
        "可以", "好的", "嗯", "嗯嗯", "行", "好", "是的", "没问题",
        "可以我们继续聊吧", "我们继续聊吧", "好的我们继续", "请继续",
        "sure", "okay", "yes", "let'scontinue", "wecancontinue",
        "好的我们继续聊吧", "好我们继续聊吧", "嗯我们继续聊吧",
        "好的请继续提问", "你继续问吧", "请继续提问"
    }


def topical_response(question,answer):
    """Conservative focus check: do not mistake paraphrases for errors.

    The source-grounding guard, not lexical overlap, is responsible for
    rejecting invented personal facts. This guard only rejects clear evasion.
    """
    from memory_matcher import facets
    from source_recall import _question_dimension
    from question_intent import classify
    q=str(question or "").strip()
    a=str(answer or "").strip()
    if not a or len(a)>2000 or _plain_evasion(a):
        return False
    intent=classify(q)
    # Recording is a present-moment consent question. Natural privacy- or
    # comfort-oriented replies are valid without matching the question words.
    if intent.get("reason")=="recording_consent":
        return bool(re.search(
            r"录音|录制|录像|记录|研究|保密|隐私|公开|用途|使用|同意|接受|"
            r"自在|安心|放心|介意|舒适|顾虑|事先|征求|可以|不方便|"
            r"record|privacy|confidential|consent|comfortable|purpose|share|agree",
            a,re.I))
    dim=_question_dimension(q)
    if dim=="frequency":
        return bool(re.search(
            r"经常|频繁|偶尔|有时|很少|每天|每周|每月|几次|次数|规律|"
            r"没数|数不清|不太清楚|说不准|记不清|不确定|大约|不一定|"
            r"often|sometimes|rarely|daily|weekly|monthly|frequency|"
            r"not sure|don't count|haven.t counted|cannot recall",a,re.I))
    # Explicitly answered uncertainty/partial evidence is valid. It does not
    # assert an invented biography; grounding_guard handles actual claims.
    uncertainty=bool(re.search(
        r"记不清|想不起来|不太记得|说不准|无法确认|不能确定|不太确定|"
        r"不知道|没留意|没注意|没有把握|不能准确|不方便讲|不愿意|"
        r"can't remember|cannot recall|not sure|don't know|don't remember|"
        r"can't confirm|not comfortable sharing",a,re.I))
    if uncertainty:
        return True
    qfac=facets(q)-{"experience","coding","tools"}
    afac=facets(a)
    if qfac and not (qfac & afac):
        return False
    if re.search(r"哪些.{0,8}(?:任务|活动|工作)|what (?:tasks|activities)",q,re.I):
        return bool(re.search(r"编程|写代码|调试|测试|文档|解释|学习|任务|"
                              r"code|debug|test|document|task|work",a,re.I))
    return True
