"""Isolate conversational interaction, general concepts, and autobiographical recall.

No external classifier or hidden LLM call. Applied before probabilistic recall,
so a question about recording permission cannot turn into "I forgot".
"""
from __future__ import annotations
import re

RECORDING=re.compile(r"录音|录像|录制|录下来|录屏|record(?:ing|ed)?\\b|audio\\s*record|tape\\s*this",re.I)
SOCIAL=re.compile(r"^(?:您?好|你好|早上好|下午好|晚上好|谢谢|感谢|辛苦了|可以开始|我们开始|准备好了吗|"
                  r"hello|hi\\b|thank you|thanks|ready to begin)",re.I)
OPINION=re.compile(r"怎么看|如何看待|怎么看待|有什么看法|有什么顾虑|你认为|您认为|"
                   r"你的观点|您的观点|what do you think|how do you feel about",re.I)
PROCEDURAL=re.compile(r"可以继续|继续访谈|愿意参与|可以开始|能开始吗|需要休息|暂停一下|是否方便|"
                      r"能不能继续|可以问|能问|能否进行|是否同意",re.I)
CONCEPT=re.compile(r"什么是|是什么意思|如何理解|请解释|请定义|一般来说|通常来说|"
                   r"通用概念|基本原理|what is|define|explain the concept",re.I)
PERSONAL=re.compile(r"什么时候开始|最早|第一次|经历|做过|遇到过|参与过|"
                    r"你的经历|您.*经历|当时|那次|具体时间|何时|哪一年|"
                    r"我.*做|你.*做|您.*做|怎么使用|如何使用|"
                    r"when did you|have you ever|tell me about your",re.I)

def focus_question(question):
    """Keep the current question, not completed prefatory remarks.

    "录音确认好了。您什么时候开始使用 AI？" is a personal-recall
    question, not a request to authorize recording again.
    """
    q=str(question or "").strip()
    segments=[x.strip() for x in re.split(r"[。！？!?]+",q) if x.strip()]
    if len(segments)<=1:
        return q
    last=segments[-1]
    # "最早什么时候使用 AI？当时的具体情况呢？" is a linked question;
    # restore its previous topical segment. A completed recording preface
    # ("录音确认好了") is not linked to the following AI-tools question.
    if re.search(r"^(?:那|当时|这|您?能.*当时|能.*具体情况|请再说|能详细说|"
                 r"您?能.*(?:时间点|那件事|那次|刚才))",
                 last) and not re.search(r"录音确认好了|录音没问题",segments[-2]):
        return segments[-2]+"；"+last
    return last


def classify(question):
    original=str(question or "").strip()
    focus=focus_question(original)
    q=focus
    recording_in_focus=bool(RECORDING.search(q))
    # A recording-consent opener often ends with a conditional invitation
    # ("如果有具体顾虑，也欢迎随时告诉我"). This must not turn the
    # whole utterance into a request for unrelated autobiographical recall.
    # Only apply this to an ACTUAL recording-question clause, never a
    # completed recording-confirmation preface followed by an AI question.
    clauses=[c.strip() for c in re.split(r"[。！？!?；;]+",original) if c.strip()]
    recording_question_clause=any(
        RECORDING.search(c) and
        re.search(r"感受|顾虑|介意|同意|接受|愿意|可以|方便|是否|"
                  r"feel|consent|comfortable|concern|mind|agree",c,re.I)
        for c in clauses
    )
    personal_ai_question=bool(re.search(
        r"(?:什么时候|最早|第一次|何时).{0,40}(?:AI|人工智能|编程|代码)|"
        r"(?:AI|人工智能|编程|代码).{0,40}(?:什么时候|最早|第一次|何时)|"
        r"when did you (?:first |start ).*(?:AI|coding|programming)",
        original,re.I))
    recording_question_in_original=recording_question_clause and not personal_ai_question
    refers_to_previous_recording=(
        bool(RECORDING.search(original)) and
        bool(re.search(r"这种方式|这件事|这个安排|对此|这次录制",focus)) and
        not PERSONAL.search(focus)
    )
    if recording_in_focus or refers_to_previous_recording or recording_question_in_original:
        past_event=bool(re.search(r"第一次录音|以前.*录音|上次.*录音|"
                                  r"录音经历|被录音的经历|过去.*录音|"
                                  r"when were you recorded",focus,re.I))
        if not past_event:
            return {"kind":"procedural","reason":"recording_consent",
                    "focus_question":original if recording_question_in_original else focus}
    if (SOCIAL.search(q) or PROCEDURAL.search(q)) and not PERSONAL.search(q):
        return {"kind":"procedural","reason":"conversation_management",
                "focus_question":focus}
    if CONCEPT.search(q) and not PERSONAL.search(q):
        return {"kind":"general_knowledge","reason":"conceptual_question",
                "focus_question":focus}
    if OPINION.search(q) and not PERSONAL.search(q):
        return {"kind":"general_opinion","reason":"non_autobiographical_opinion",
                "focus_question":focus}
    return {"kind":"personal_recall","reason":"experience_or_context",
            "focus_question":focus}

def procedural_reply(question,language="zh"):
    q=str(question or "")
    en=language=="en"
    if RECORDING.search(q):
        return ("For this interview, recording is fine if it is only used to accurately "
                "document our conversation and handled confidentially. Please tell me "
                "before sharing it beyond the research." if en else
                "如果录音只是为了准确整理这次访谈，并且会妥善保管、不随意公开，我可以接受。"
                "如果之后要公开使用，希望提前跟我说明。")
    if re.search(r"休息|暂停|break|pause",q,re.I):
        return "Sure, we can pause for a moment." if en else "可以，咱们稍微暂停一下就好。"
    if re.search(r"谢谢|感谢|thank",q,re.I):
        return "You're welcome." if en else "不客气，希望刚才说的能帮上忙。"
    return "Sure, we can continue." if en else "可以，我们继续聊吧。"
