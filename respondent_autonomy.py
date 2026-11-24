"""Interview Playground v0.12.13 - optional respondent autonomy layer.

No Ollama calls are made here: the existing respondent call stays single-pass.
This module provides pre-answer context and a post-answer, nonblocking ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import json
import os
import re
import tempfile
import time

VALID_MODES = frozenset({"answer", "expand", "clarify", "already_answered", "decline", "exhausted", "uncertain"})

SYSTEM_POLICY_ZH = """你现在是正在接受真实研究访谈的受访者，不是客服，也不是负责管理访谈流程的程序。
在正式作答前，在心中比较当前问题与既往访谈问题、已讲过的信息和许可记忆；不要在最终答复中解释内部判定。
按下面原则自然作答：
1. 新问题且有可信依据：正面、具体、自然地回答，避免复述问题。
2. 与之前问题有关，但询问的是一个尚未解释的角度：仅补充真正新增的细节或澄清误解。不要重复大段经历。
3. 当前问题实质已回答且没有新增内容：直接用符合人物语气的一两句话说明刚才说过；不要机械复读旧答案。
4. 已经把能够讲的内容讲完：自然表示没有更多可扩展的部分，必要时建议换个方向。
5. 不愿披露：可以自然拒绝回答，不需要虚构借口。
6. 没有足够依据或记不清：如实说不知道/记不清；不要编造事实，也不要将不知道等同于不愿回答。
7. 同一事实可回答不同角度的问题，不可单纯因 fact_id 曾出现就拒绝；重点是新问题是否需要新的解释。
8. 不可由重复内容自行停止系统、抛异常、返回特殊 HTTP 状态或声明访谈已终止。只需像真实受访者那样作答，让采访者继续。
9. 保持当前人物的第一人称说话方式，遵守现有许可范围、角色约束和答复长度参数。
10. 允许自然变化表达；不要固定使用“这个问题我已经回答过”作为万能模板。
""".strip()

@dataclass
class TurnUsage:
    round: int
    question: str
    answer: str
    topic_id: Optional[str] = None
    used_fact_ids: list[str] = field(default_factory=list)
    answered_aspects: list[str] = field(default_factory=list)
    response_mode: str = "answer"
    topic_exhausted: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.response_mode not in VALID_MODES:
            raise ValueError("unknown response_mode")
        if self.round <= 0:
            raise ValueError("round must be positive")
        self.used_fact_ids = list(dict.fromkeys(map(str, self.used_fact_ids)))
        self.answered_aspects = list(dict.fromkeys(map(str, self.answered_aspects)))


def _short(text: Any, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def build_preanswer_context(question: str, history: Iterable[TurnUsage], *, topic_id: Optional[str] = None,
                            max_history_turns: int = 12, max_answer_chars: int = 550) -> str:
    """Append to the existing respondent system context before its ONE LLM call.

    Existing full integrated memories and consent boundaries must remain in scope.
    Do not treat this as a substitute for those inputs.
    """
    if not question or not question.strip():
        raise ValueError("question is required")
    turns = list(history)[-max_history_turns:]
    rows = []
    for t in turns:
        rows.append({
            "round": t.round, "topic_id": t.topic_id,
            "question": _short(t.question, 320),
            "answer": _short(t.answer, max_answer_chars),
            "used_fact_ids": t.used_fact_ids,
            "answered_aspects": t.answered_aspects,
            "response_mode": t.response_mode,
            "topic_exhausted": t.topic_exhausted,
        })
    state = {"current_topic_id": topic_id, "current_question": question,
             "prior_turns": rows, "note": "历史事实编号只作为来源追踪；同一来源可能支持新的回答角度。"}
    return SYSTEM_POLICY_ZH + "\n\n【当前问答与已表达信息】\n" + json.dumps(state, ensure_ascii=False)


def make_turn_usage(*, round: int, question: str, answer: str,
                    topic_id: Optional[str] = None, used_fact_ids: Optional[Iterable[str]] = None,
                    answered_aspects: Optional[Iterable[str]] = None, response_mode: str = "answer",
                    topic_exhausted: bool = False) -> TurnUsage:
    """Record ACTUALLY expressed information only; never assume all prepared facts were said.

    If fact/aspect identification is unavailable, keep both arrays empty rather
    than guessing. The model can still compare question/answer history directly.
    """
    return TurnUsage(round=round, question=question, answer=answer, topic_id=topic_id,
                     used_fact_ids=list(used_fact_ids or []), answered_aspects=list(answered_aspects or []),
                     response_mode=response_mode, topic_exhausted=topic_exhausted)


def add_completed_turn(history: list[TurnUsage], item: TurnUsage) -> None:
    """Only call after the existing respondent request completed successfully.

    A refusal / already-covered / exhausted response is a completed turn, not an
    error. Model/network failures still follow the project's strict error path.
    """
    if history and item.round != history[-1].round + 1:
        raise ValueError("turn order mismatch")
    if not history and item.round != 1:
        raise ValueError("first turn must be round 1")
    history.append(item)


def save_ledger_atomic(path: str | Path, turns: Iterable[TurnUsage]) -> None:
    """For use with the host project's existing per-session privacy controls.

    Do not save sensitive interview content without permission; prefer host's
    established protected session storage and cleanup policy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "turns": [asdict(t) for t in turns]}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".usage-", suffix=".tmp", delete=False) as f:
            tmp = f.name
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def load_ledger(path: str | Path) -> list[TurnUsage]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported usage ledger version")
    return [TurnUsage(**turn) for turn in data["turns"]]
