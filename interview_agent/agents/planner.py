"""Top-level planner: reads the explicit InterviewState (and the latest reflection) and routes to one of four
behavior agents. Reflection lives in `agents/reflection.py`; follow-up budgets in `policy.py`."""
from __future__ import annotations

import json
from typing import Optional

from ..llm.base import LLMClient
from ..policy import FollowupAssessment, assess_followup
from ..skills_loader import SkillLoader
from ..state import ACTIONS, DISRUPTIVE, FOLLOWUP, QUESTIONING, RESPONSE, Decision, InterviewState

_DECIDE_SYSTEM = """You are the top-level planner of an AI interviewer. Choose the interviewer's NEXT behavior mode:
- "questioning": ask the next main guide question (or a branch target the guide allows)
- "followup": probe ONE slot of the CURRENT main question; only slots listed under ALLOWED FOLLOW-UPS may be chosen
- "response": react to the participant (acknowledge / encourage / summarize) to maintain flow, without a new question
- "disruptive": repair a disruptive event (misunderstanding, refusal, contradiction, off-topic drift, distress)

You are given the structured interview state (slot statuses, evidence, follow-up attempts, last action and its
verified outcome, disruption state) and the reflection on the last reply. Decide from the STATE, not from impressions.
1. An unresolved disruption is handled by "disruptive" (the system already forces this while repairs remain).
2. Required information still missing and a follow-up is allowed -> "followup" is usually best.
3. Core question covered: follow up only when an allowed deepening slot is clearly worth an extra turn;
   otherwise move on with "questioning".
4. No allowed follow-up, or the stop reasons say budget is exhausted / nothing new is being produced -> "questioning".
   Never ask the participant to guess something they said they do not know or refused to share.
5. "response" when the participant is emotional or hesitant, or to confirm understanding. Never twice in a row.
6. Vary strategy if the last action's verified outcome shows its goal was not achieved.

Return ONLY JSON: {"action": "questioning|followup|response|disruptive", "question_id": str,
"target_slot": str, "focus": str, "rationale": str}
"target_slot" is required for followup. "focus" tells the sub-agent what to work on."""


class PlannerAgent:
    def __init__(self, llm: LLMClient, skills: Optional[SkillLoader] = None):
        self.llm = llm

    # ---- routing ------------------------------------------------------------------------
    def decide(self, state: InterviewState) -> Decision:
        last = state.last_turn()
        if last is None:
            return self._advance(state, "Open the interview with the first main question.")
        qid = state.active_qid
        ds = state.disruption
        if ds.active and qid and ds.repair_attempts < state.policy.max_disruption_repairs:
            return self._finalize(state, Decision(DISRUPTIVE, qid, f"repair {ds.type}",
                                                  f"{ds.severity} {ds.type}: {ds.evidence}"))

        refl = last.reflection
        assess = assess_followup(state, refl.followup_candidates if refl else [])
        if not assess.options and state.next_pending() is None:
            return self._advance(state, "; ".join(assess.stop_reasons) or "all guide questions covered")
        return self._guard(state, self._llm_decide(state, assess), assess)

    def _llm_decide(self, state: InterviewState, assess: FollowupAssessment) -> Decision:
        qid = state.active_qid
        refl = state.last_turn().reflection
        q = state.outline.get(qid) if qid else None
        allowed = [f"- {o.slot_id} ({'required' if o.required else 'deepening, ' + o.priority}): {o.reason}"
                   + (f" | participant's words: {o.lead}" if o.lead else "") for o in assess.options]
        user = (
            f"Interview guide status ([x] done, [!] closed incomplete, [>] active, [ ] pending, [-] skipped):\n"
            f"{state.render_outline()}\n\n"
            f"Active question:\n{state.render_question(qid)}\n"
            + (f"Guide branch rules: {[r.condition or r.__dict__ for r in q.branch_rules]}\n" if q and q.branch_rules else "")
            + f"\nSlots:\n{state.render_slots(qid)}\n\n"
            f"Structured state: {json.dumps(state.snapshot(), ensure_ascii=False)}\n\n"
            f"Recent conversation:\n{state.render_transcript(6)}\n\n"
            + (f"Reflection: relevance={refl.answer_analysis.relevance} specificity={refl.answer_analysis.specificity} "
               f"novelty={refl.answer_analysis.novelty} core={refl.core_coverage.status} failed={refl.failed} "
               f"| {refl.rationale}\n" if refl else "")
            + f"ALLOWED FOLLOW-UPS:\n{chr(10).join(allowed) if allowed else '(none)'}\n"
            f"Follow-up stop reasons: {assess.stop_reasons or '(none)'}\n"
            f"Recent interviewer actions: {state.recent_actions(3)}\nNext pending question: {state.next_pending()}\n"
            f"Write the free-text fields \"focus\" and \"rationale\" in {state.policy.output_language}.\n"
        )
        try:
            d = self.llm.chat_json(
                [{"role": "system", "content": _DECIDE_SYSTEM}, {"role": "user", "content": user}], temperature=0.2)
            action = str(d.get("action", "")).lower().strip()
            if action not in ACTIONS:
                raise ValueError(f"bad action {action!r}")
            return Decision(action, str(d.get("question_id") or qid or "") or None, str(d.get("focus", "") or ""),
                            str(d.get("rationale", "") or ""), str(d.get("target_slot", "") or "") or None)
        except Exception:  # noqa: BLE001 - deterministic fallback
            if assess.options:
                return Decision(FOLLOWUP, qid, "", "fallback policy", assess.options[0].slot_id)
            return Decision(QUESTIONING, None, "", "fallback: advance")

    def _guard(self, state: InterviewState, d: Decision, assess: FollowupAssessment) -> Decision:
        """Enforce hard interview constraints regardless of what the model chose."""
        qid = state.active_qid
        repeated_response = d.action == RESPONSE and state.recent_actions(1) == [RESPONSE]
        if d.action == DISRUPTIVE or repeated_response:   # genuine disruptions were routed before the model
            d = Decision(FOLLOWUP if assess.options else QUESTIONING, qid, "", d.rationale, d.target_slot)
        if d.action == FOLLOWUP:
            if not assess.options or not qid:
                return self._advance(state, "; ".join(assess.stop_reasons) or "no follow-up allowed")
            opt = next((o for o in assess.options if o.slot_id == d.target_slot), assess.options[0])
            desc = state.qstate[qid].slots[opt.slot_id].description
            focus = f"{desc} (participant said: {opt.lead})" if opt.lead else desc
            return self._finalize(state, Decision(FOLLOWUP, qid, focus, d.rationale or opt.reason, opt.slot_id))
        if d.action == QUESTIONING:
            if d.question_id and d.question_id != qid and state.can_jump_to(d.question_id):
                return self._finalize(state, Decision(QUESTIONING, d.question_id, d.focus, d.rationale))
            return self._advance(state, d.rationale or "move on", d.focus)
        d.question_id = qid
        return self._finalize(state, d)

    def _advance(self, state: InterviewState, why: str, focus: str = "") -> Decision:
        nxt = state.next_pending()
        if nxt is None:
            state.finish()
            return self._finalize(state, Decision(RESPONSE, state.active_qid, "close the interview",
                                                  "all guide questions covered"))
        return self._finalize(state, Decision(QUESTIONING, nxt, focus, why))

    @staticmethod
    def _finalize(state: InterviewState, d: Decision) -> Decision:
        """Record what the action is meant to achieve, so the next reflection can judge it."""
        q = state.outline.get(d.question_id) if d.question_id else None
        slot = q.slot(d.target_slot) if q and d.target_slot else None
        if d.goal:
            return d
        if d.action == QUESTIONING and q:
            d.goal = f"获得 {q.id} 的必需信息:" + "；".join(f"{s.id}（{s.description}）" for s in q.required_slots)
        elif d.action == FOLLOWUP and slot:
            d.goal = f"补充信息槽 {slot.id}:{slot.description}"
        elif d.action == DISRUPTIVE:
            d.goal = f"处理互动异常（{state.disruption.type}）并恢复正常访谈"
        elif state.finished:
            d.goal = "结束访谈"
        else:
            d.goal = "维持交流、确认理解或自然回应受访者"
        return d
