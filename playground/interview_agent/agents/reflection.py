"""Reflection module: analyses the participant's latest reply and proposes structured, evidence-grounded
state updates. It does NOT choose the next behavior (that is the planner's job) and does not mutate state;
`InterviewState.apply_reflection` validates and commits its output."""
from __future__ import annotations

import json

from ..llm.base import LLMClient
from ..reflection_schema import ReflectionResult, parse_reflection
from ..state import InterviewState

_EVIDENCE_REPAIR_SYSTEM = """You repair evidence for proposed interview slot updates.
For each rejected proposal, independently decide whether the participant's actual words support the proposed
status for that slot. If they do, provide one contiguous verbatim excerpt from a participant reply as evidence.
The quoted analysis passages are suggestions, not proof that they support the proposed status.
Do not paraphrase, combine passages, change the slot or status, or use the interviewer's words.
Omit any proposal you cannot support. Return ONLY JSON:
{"slot_updates": [{"slot_id": str, "new_status": str, "evidence": str}]}"""

_REFLECT_SYSTEM = """You are the reflection module of an AI interviewer. After each participant reply you REFLECT on
the round that just ended and return a structured, evidence-grounded analysis. You never decide the next move.

Slots: every main question has REQUIRED slots (they decide whether the question is complete) and OPTIONAL
deepening slots (nice to have; never mandatory). Work slot by slot.
- Update a slot ONLY when the PARTICIPANT's own words support it. Copy the supporting words VERBATIM (an exact
  excerpt of a participant message) into "evidence". Never mark a slot because the answer "should" imply it.
- Statuses you may assign: answered (the information for that slot is given), partial (touched on but vague or
  incomplete), refused (participant explicitly declines), unknown (participant explicitly says they do not know or
  do not remember), not_applicable (participant says the slot does not apply to them), contradicted (conflicts
  with something they said earlier; quote the conflicting words). Do not emit "missing".
- Report only slots whose status CHANGES because of information not yet in the evidence bank. Information from
  earlier participant messages may satisfy a slot too (quote it), but never re-report what is recorded.
- Distinguish absence of information (leave the slot alone), refusal, not knowing and contradiction.
- Judge relevance / specificity / novelty from content, not from length or fluency: none|low|medium|high.
  specificity is high only for concrete episodes, actions or examples that you quote in answer_analysis.evidence.
- previous_action_effectiveness: judge the previous move ONLY against its own stated goal. A follow-up that got
  its target slot answered succeeded even if other slots remain open.
- disruption: only for genuinely abnormal interaction (misunderstanding | refusal | contradiction | off_topic |
  other = distress, hostility, ...), with a verbatim trigger quote. Missing information alone is NOT a disruption.
- followup_candidates: at most 3 still-unresolved slots of THIS question worth probing given what the participant
  just said (priority low|medium|high, optional "lead" = the participant's words to echo). An empty list is fine.
  Do not turn every gap into a candidate.
The system computes core coverage itself; do not report it.

Return ONLY JSON:
{"answer_analysis": {"relevance": str, "specificity": str, "novelty": str,
                     "evidence": [{"quote": str, "related_slots": [str]}]},
 "slot_updates": [{"slot_id": str, "new_status": str, "evidence": str}],
 "previous_action_effectiveness": {"goal_achieved": bool, "reason": str},
 "followup_candidates": [{"target_slot": str, "reason": str, "priority": str, "lead": str}],
 "disruption": {"type": "none|misunderstanding|refusal|contradiction|off_topic|other",
                "severity": "none|low|medium|high", "evidence": str},
 "rationale": str}"""


class ReflectionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reflect(self, state: InterviewState) -> ReflectionResult:
        """Analyse the last turn (which already holds the participant's reply) and commit it to the state."""
        turn = state.last_turn()
        qid = turn.decision.question_id or state.active_qid or ""
        q = state.outline.get(qid)
        if q is None or not turn.participant_reply.strip():
            r = ReflectionResult.neutral(qid, "" if q else "no active question")
            r.rationale = r.rationale or "empty participant reply"
            return state.apply_reflection(r)

        user = (
            f"Main question:\n{state.render_question(qid)}\nIntent: {q.intent or q.goal or '(n/a)'}\n\n"
            f"Slots of this question (current status):\n{state.render_slots(qid)}\n\n"
            f"Evidence already recorded:\n{state.render_obtained(qid) or '(none)'}\n\n"
            f"Previous interviewer move: [{turn.decision.action}] target_slot={turn.decision.target_slot or '-'}\n"
            f"Goal of that move: {turn.decision.goal or '(unspecified)'}\n\n"
            f"Conversation (latest participant reply is the last line):\n{state.render_transcript(10)}\n\n"
            "Reflect on the participant's latest reply. Quotes stay verbatim; write the free-text fields "
            f"(reason, rationale) in {state.policy.output_language}."
        )
        try:
            data = self.llm.chat_json(
                [{"role": "system", "content": _REFLECT_SYSTEM}, {"role": "user", "content": user}], temperature=0.0)
            r = parse_reflection(data, qid, [s.id for s in q.slots()], state.participant_replies(), turn.index)
        except Exception as e:  # noqa: BLE001 - degrade gracefully: learn nothing rather than invent state
            r = ReflectionResult.neutral(qid, f"{type(e).__name__}: {e}")
        if not r.failed:
            self._repair_evidence(state, r)
        return state.apply_reflection(r)

    def _repair_evidence(self, state: InterviewState, result: ReflectionResult) -> None:
        rejected = [u for u in result.rejected_updates if u["reason"] == "evidence is not a quote from the participant"]
        if not rejected:
            return
        for u in rejected:
            u["resolution"] = "unresolved"
            u["repair_reason"] = "repair returned no valid matching update"
        turn = state.last_turn()
        q = state.outline.get(result.question_id)
        if turn is None or q is None:
            return
        proposals = {(u["slot_id"], u["new_status"]) for u in rejected}
        user = (
            f"Question:\n{state.render_question(result.question_id)}\n\n"
            f"Slots and current status:\n{state.render_slots(result.question_id)}\n\n"
            f"Rejected proposals:\n{json.dumps(rejected, ensure_ascii=False)}\n\n"
            f"Verified analysis excerpts:\n{json.dumps([{'quote': e.quote, 'related_slots': e.related_slots} for e in result.answer_analysis.evidence], ensure_ascii=False)}\n\n"
            f"Participant replies (turn ID and text):\n{json.dumps(state.participant_replies(), ensure_ascii=False)}"
        )
        try:
            data = self.llm.chat_json(
                [{"role": "system", "content": _EVIDENCE_REPAIR_SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0)
            repaired = parse_reflection(data, result.question_id, [s.id for s in q.slots()],
                                        state.participant_replies(), turn.index)
        except Exception as e:  # noqa: BLE001 - retain the original result and expose the repair failure
            for u in rejected:
                u["repair_error"] = f"{type(e).__name__}: {e}"
            return
        for attempt in repaired.rejected_updates:
            for u in rejected:
                if (u["slot_id"], u["new_status"]) == (attempt["slot_id"], attempt["new_status"]):
                    u["repair_reason"] = attempt["reason"]
                    u["repair_attempt_evidence"] = attempt["evidence"]
        for update in repaired.slot_updates:
            key = (update.slot_id, update.new_status)
            if key not in proposals:
                continue
            proposals.remove(key)
            result.slot_updates.append(update)
            for u in rejected:
                if (u["slot_id"], u["new_status"]) == key:
                    u["resolution"] = "repaired"
                    u["repaired_evidence"] = update.evidence
                    u.pop("repair_reason", None)
                    u.pop("repair_attempt_evidence", None)
