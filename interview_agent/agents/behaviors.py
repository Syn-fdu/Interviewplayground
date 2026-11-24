from __future__ import annotations

from ..state import DISRUPTIVE, FOLLOWUP, QUESTIONING, RESPONSE, Decision, InterviewState
from .base import BehaviorAgent


class QuestioningAgent(BehaviorAgent):
    name, role = QUESTIONING, "questioning"

    def task_context(self, state: InterviewState, decision: Decision) -> str:
        new_topic = ""
        q = state.outline.get(decision.question_id) if decision.question_id else None
        if q:
            t = state.outline.topic_of(q.id)
            first = t and t.questions[0].id == q.id
            new_topic = f"Topic: {t.title if t else ''}. " + ("This opens a NEW topic.\n" if first else "Same topic as before.\n")
        needs = "; ".join(s.description for s in q.required_slots) if q else ""
        return (f"Ask this main question now (adapt wording per the skill): {q.text if q else ''}\n{new_topic}"
                f"Information the question should establish (do not read it out as a checklist): {needs}\n")


class FollowUpAgent(BehaviorAgent):
    name, role = FOLLOWUP, "followup"

    def task_context(self, state: InterviewState, decision: Decision) -> str:
        asked = [t.utterance for t in state.turns if t.decision.action == FOLLOWUP
                 and t.decision.question_id == decision.question_id]
        q = state.outline.get(decision.question_id) if decision.question_id else None
        slot = q.slot(decision.target_slot) if q and decision.target_slot else None
        target = f"{slot.id}: {slot.description}" if slot else "(find one yourself)"
        return (f"Information this follow-up must obtain (ask about this ONLY): {target}\n"
                f"Goal: {decision.goal or '(n/a)'}\n"
                f"Already obtained for this question (do not ask again): {state.render_obtained(q.id) if q else ''}\n"
                f"Follow-ups already asked for this question (do not repeat): {asked or '(none)'}\n")


class ResponseAgent(BehaviorAgent):
    name, role = RESPONSE, "response"

    def task_context(self, state: InterviewState, decision: Decision) -> str:
        if state.finished:
            return "The interview is finished: perform the CLOSE move.\n"
        prev = [t.utterance for t in state.turns[-3:] if t.decision.action == RESPONSE]
        return f"Your previous reactions (vary your wording): {prev or '(none)'}\n"


class DisruptiveAgent(BehaviorAgent):
    name, role = DISRUPTIVE, "disruptive"

    def task_context(self, state: InterviewState, decision: Decision) -> str:
        d = state.disruption
        return (f"Detected disruption type: {d.type if d.active else 'other'} (severity {d.severity}); "
                f"triggered by: \"{d.evidence}\"; repair attempts so far: {d.repair_attempts}\n")
