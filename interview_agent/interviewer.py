from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from .agents.base import BehaviorAgent
from .agents.behaviors import DisruptiveAgent, FollowUpAgent, QuestioningAgent, ResponseAgent
from .agents.planner import PlannerAgent
from .agents.reflection import ReflectionAgent
from .llm.base import LLMClient
from .outline import Outline
from .policy import InterviewPolicy
from .skills_loader import SkillLoader
from .state import DISRUPTIVE, FOLLOWUP, QUESTIONING, RESPONSE, InterviewState


class Interviewer:
    """Planner (top layer) + four behavior agents. One round = one planner decision."""

    def __init__(self, outline: Outline, llms: Dict[str, LLMClient], skills: Optional[SkillLoader] = None,
                 policy: Optional[InterviewPolicy] = None, max_turns: int = 60):
        """`llms` maps role -> client; roles: planner, reflection, questioning, followup, response, disruptive.
        A "default" entry is used for roles not listed (reflection falls back to the planner's client)."""
        get = lambda role: llms.get(role) or llms["default"]  # noqa: E731
        skills = skills or SkillLoader()
        self.state = InterviewState(outline, policy)
        self.planner = PlannerAgent(get("planner"))
        self.reflector = ReflectionAgent(llms.get("reflection") or get("planner"))
        self.agents: Dict[str, BehaviorAgent] = {
            QUESTIONING: QuestioningAgent(get("questioning"), skills),
            FOLLOWUP: FollowUpAgent(get("followup"), skills),
            RESPONSE: ResponseAgent(get("response"), skills),
            DISRUPTIVE: DisruptiveAgent(get("disruptive"), skills),
        }
        self.max_turns = max_turns

    def step(self, participant_reply: Optional[str] = None) -> str:
        """Run one round. Pass the participant's reply to the previous utterance (None on the first call)."""
        s = self.state
        if participant_reply is not None and s.last_turn():
            s.last_turn().participant_reply = participant_reply
            self.reflector.reflect(s)                    # analyse the reply; commits validated updates to the state
        decision = self.planner.decide(s)                # route the next round from the updated state
        utterance = self.agents[decision.action].run(s, decision)
        turn = s.start_turn(decision)
        turn.utterance = utterance
        if len(s.turns) >= self.max_turns:
            s.finish()
        return utterance

    def run(self, ask: Callable[[str], str], emit: Optional[Callable[[str, str], None]] = None) -> InterviewState:
        """Drive a full interview. `ask(utterance)` returns the participant's reply."""
        reply = None
        while True:
            utterance = self.step(reply)
            if emit:
                emit(self.state.last_turn().decision.action, utterance)
            if self.state.finished:
                break
            reply = ask(utterance)
        return self.state

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
