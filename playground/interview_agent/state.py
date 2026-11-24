from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .outline import Outline
from .policy import InterviewPolicy
from .reflection_schema import DISRUPTIONS, CoreCoverage, ReflectionResult  # noqa: F401 - DISRUPTIONS re-exported
from .slots import (ANSWERED, CONTRADICTED, FINAL_UNRESOLVED, MISSING, NOT_APPLICABLE, OPEN, PARTIAL, SATISFIED,
                    EvidenceItem, SlotState, normalize, transition_allowed)

QUESTIONING, FOLLOWUP, RESPONSE, DISRUPTIVE = "questioning", "followup", "response", "disruptive"
ACTIONS = (QUESTIONING, FOLLOWUP, RESPONSE, DISRUPTIVE)


@dataclass
class Decision:
    action: str
    question_id: Optional[str] = None  # main question this round is anchored to
    focus: str = ""                    # what the sub-agent should concentrate on (lead, repair target, ...)
    rationale: str = ""
    target_slot: Optional[str] = None  # slot a follow-up is meant to fill
    goal: str = ""                     # what this action should achieve; judged by the next reflection


@dataclass
class Turn:
    index: int
    decision: Decision
    utterance: str = ""                # interviewer's output
    participant_reply: str = ""        # filled once the participant responds
    reflection: Optional[ReflectionResult] = None  # assessment of this round, made after the reply


@dataclass
class Disruption:
    type: str = "none"
    severity: str = "none"
    evidence: str = ""
    turn_id: int = 0
    repair_attempts: int = 0

    @property
    def active(self) -> bool:
        return self.type != "none"


@dataclass
class QuestionState:
    status: str = "pending"            # pending | active | done | incomplete | skipped
    core_status: str = "not_covered"   # covered | partial | not_covered (required slots only)
    followups: int = 0
    optional_followups: int = 0
    unproductive_followups: int = 0    # consecutive follow-ups that changed nothing
    slots: Dict[str, SlotState] = field(default_factory=dict)
    note: str = ""                     # e.g. why the question was skipped


class InterviewState:
    """Single source of truth for the interview. Only `start_turn`, `apply_reflection` and `finish` mutate it."""

    def __init__(self, outline: Outline, policy: Optional[InterviewPolicy] = None):
        self.outline = outline
        self.policy = policy or InterviewPolicy()
        self.qstate: Dict[str, QuestionState] = {}
        for q in outline.questions():
            self.qstate[q.id] = QuestionState(slots={
                s.id: SlotState(s.id, s.required, s.description, s.priority) for s in q.slots()})
        self.turns: List[Turn] = []
        self.evidence_bank: List[EvidenceItem] = []
        self.disruption = Disruption()
        self.active_qid: Optional[str] = None
        self.finished = False

    # ---- queries -------------------------------------------------------------------------
    def next_pending(self) -> Optional[str]:
        return next((q.id for q in self.outline.questions() if self.qstate[q.id].status == "pending"), None)

    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    def recent_actions(self, n: int = 3) -> List[str]:
        return [t.decision.action for t in self.turns[-n:]]

    def participant_replies(self) -> List[tuple]:
        return [(t.index, t.participant_reply) for t in self.turns if t.participant_reply.strip()]

    def can_jump_to(self, target: str) -> bool:
        """A guide branch rule on the active question explicitly allows jumping to `target`."""
        q = self.outline.get(self.active_qid) if self.active_qid else None
        return bool(q and target in self.qstate and self.qstate[target].status == "pending"
                    and any(r.goto == target for r in q.branch_rules))

    def last_action(self) -> Optional[Dict[str, Any]]:
        t = self.last_turn()
        if not t:
            return None
        eff = t.reflection.previous_action_effectiveness if t.reflection else None
        return {"mode": t.decision.action, "question_id": t.decision.question_id,
                "target_slot": t.decision.target_slot, "goal": t.decision.goal,
                "outcome": asdict(eff) if eff else None}

    def core_coverage(self, qid: str) -> CoreCoverage:
        req = [s for s in self.qstate[qid].slots.values() if s.required]
        if all(s.status in SATISFIED for s in req):
            return CoreCoverage("covered")
        status = "partial" if any(s.status in SATISFIED | {PARTIAL} | FINAL_UNRESOLVED for s in req) else "not_covered"
        return CoreCoverage(status, [s.slot_id for s in req if s.status in OPEN],
                            [s.slot_id for s in req if s.status in FINAL_UNRESOLVED])

    # ---- updates (the only mutation entry points) ----------------------------------------
    def apply_reflection(self, r: ReflectionResult) -> ReflectionResult:
        """Validate and commit a reflection. Slot changes need grounded evidence and a legal transition;
        anything else is moved to `r.rejected_updates`."""
        turn = self.last_turn()
        if turn is None:
            return r
        qid = turn.decision.question_id or self.active_qid
        qs = self.qstate.get(qid) if qid else None
        turn.reflection = r
        if qs is None:
            return r
        r.question_id = qid

        accepted = []
        for u in r.slot_updates:
            ss = qs.slots.get(u.slot_id)
            ok, why = (False, "unknown slot for this question") if ss is None else transition_allowed(ss.status, u.new_status)
            if ok and any(e.question_id == qid and e.slot_id == u.slot_id and e.status == u.new_status
                          and normalize(e.quote) == normalize(u.evidence) for e in self.evidence_bank):
                ok, why = False, "evidence already recorded"
            if not ok:
                r.rejected_updates.append({"slot_id": u.slot_id, "new_status": u.new_status, "reason": why,
                                           "evidence": u.evidence})
                continue
            u.old_status, ss.status = ss.status, u.new_status
            item = EvidenceItem(u.slot_id, qid, u.turn_id, u.evidence, u.new_status)
            ss.evidence.append(item)
            self.evidence_bank.append(item)
            accepted.append(u)
        r.slot_updates = accepted

        r.core_coverage = self.core_coverage(qid)
        qs.core_status = r.core_coverage.status
        self._apply_branch_rules(qid)
        self._verify_effectiveness(turn, r, accepted)
        fresh = [u for u in accepted if u.turn_id == turn.index]
        if not fresh:
            r.answer_analysis.novelty = "none"   # old information is never counted as new again
        elif r.answer_analysis.novelty in ("none", "unknown"):
            r.answer_analysis.novelty = "low"

        if turn.decision.action == FOLLOWUP:
            qs.unproductive_followups = 0 if accepted else qs.unproductive_followups + 1
        elif accepted:
            qs.unproductive_followups = 0

        if not r.failed:
            d = r.disruption
            if d.type == "none":
                self.disruption = Disruption()
            elif self.disruption.type == d.type:
                self.disruption.severity, self.disruption.evidence = d.severity, d.evidence
            else:
                self.disruption = Disruption(d.type, d.severity, d.evidence, turn.index)
        return r

    @staticmethod
    def _verify_effectiveness(turn: Turn, r: ReflectionResult, accepted: list) -> None:
        """Judge the previous action against ITS OWN goal using actual state changes where possible."""
        eff, act = r.previous_action_effectiveness, turn.decision
        by_slot = {u.slot_id: u for u in accepted}
        if act.action == FOLLOWUP and act.target_slot:
            u = by_slot.get(act.target_slot)
            eff.goal_achieved = bool(u and u.new_status == ANSWERED)
            check = (f"目标槽 {act.target_slot}: {u.old_status}->{u.new_status}" if u
                     else f"目标槽 {act.target_slot} 没有被接受的状态更新")
        elif act.action == QUESTIONING:
            got = [u for u in accepted if u.new_status in SATISFIED | {PARTIAL}]
            eff.goal_achieved = bool(got)
            check = f"有进展的槽: {[u.slot_id for u in got] or '无'}"
        elif act.action == DISRUPTIVE:
            eff.goal_achieved = r.disruption.type == "none"
            check = "异常已解决" if eff.goal_achieved else f"异常仍存在（{r.disruption.type}）"
        else:
            return  # response: no slot-level check possible; keep the model's (unverified) claim
        eff.verified = True
        eff.reason = f"{eff.reason} [状态核验: {check}]".strip()

    def start_turn(self, decision: Decision) -> Turn:
        qid = decision.question_id
        if decision.action == QUESTIONING and qid and qid != self.active_qid:
            self._leave_active()
            order = [q.id for q in self.outline.questions()]
            for other in order[:order.index(qid)]:       # jumping ahead skips what lies in between
                if self.qstate[other].status == "pending":
                    self.qstate[other].status, self.qstate[other].note = "skipped", f"jumped to {qid} by guide rule"
            self.qstate[qid].status = "active"
            self.active_qid = qid
        elif decision.action == FOLLOWUP and qid:
            qs = self.qstate[qid]
            qs.followups += 1
            ss = qs.slots.get(decision.target_slot or "")
            if ss:
                ss.attempts += 1
                qs.optional_followups += 0 if ss.required else 1
        elif decision.action == DISRUPTIVE and self.disruption.active:
            self.disruption.repair_attempts += 1
        turn = Turn(index=len(self.turns) + 1, decision=decision)
        self.turns.append(turn)
        return turn

    def finish(self) -> None:
        if not self.finished:
            self._leave_active()
        self.finished = True

    def _leave_active(self) -> None:
        qid = self.active_qid
        qs = self.qstate.get(qid) if qid else None
        if not qs or qs.status != "active":
            return
        qs.core_status = self.core_coverage(qid).status
        qs.status = "done" if qs.core_status == "covered" else "incomplete"

    def _apply_branch_rules(self, qid: str) -> None:
        """Skip questions as soon as a guide rule's slot condition holds, so routing never offers them."""
        order = [x.id for x in self.outline.questions()]
        for rule in self.outline.get(qid).branch_rules:
            if not rule.slot_id or self.qstate[qid].slots[rule.slot_id].status != rule.when_status:
                continue
            targets = list(rule.skip)
            if rule.goto:
                targets += order[order.index(qid) + 1:order.index(rule.goto)]
            for t in targets:
                if self.qstate[t].status == "pending":
                    self.qstate[t].status = "skipped"
                    self.qstate[t].note = f"{qid} rule: {rule.condition or rule.slot_id + '=' + rule.when_status}"

    # ---- rendering for prompts -----------------------------------------------------------
    def render_outline(self) -> str:
        marks = {"pending": " ", "active": ">", "done": "x", "incomplete": "!", "skipped": "-"}
        out = []
        for t in self.outline.topics:
            out.append(f"## {t.id} {t.title}")
            for q in t.questions:
                s = self.qstate[q.id]
                out.append(f"[{marks[s.status]}] {q.id}: {q.text}  (core: {s.core_status}, follow-ups used {s.followups})")
        return "\n".join(out)

    def render_question(self, qid: Optional[str]) -> str:
        q = self.outline.get(qid) if qid else None
        if not q:
            return "(none)"
        lines = [f"{q.id}: {q.text}", f"Goal: {q.goal or '(not specified)'}"]
        if q.probes:
            lines.append("Guide probes: " + "; ".join(q.probes))
        got = self.render_obtained(q.id)
        if got:
            lines.append("Information gathered so far:\n" + got)
        return "\n".join(lines)

    def render_obtained(self, qid: str) -> str:
        return "\n".join(f'- {e.slot_id} ({e.status}): "{e.quote}"' for e in self.evidence_bank if e.question_id == qid)

    def render_slots(self, qid: Optional[str]) -> str:
        qs = self.qstate.get(qid) if qid else None
        if not qs:
            return "(none)"
        lines = []
        for ss in qs.slots.values():
            lines.append(f"- {ss.slot_id} [{'required' if ss.required else 'optional/' + ss.priority}] "
                         f"status={ss.status} follow-ups={ss.attempts}: {ss.description}")
        return "\n".join(lines)

    def render_transcript(self, last_n: int = 6) -> str:
        lines = []
        for t in self.turns[-last_n:]:
            lines.append(f"Interviewer[{t.decision.action}]: {t.utterance}")
            if t.participant_reply:
                lines.append(f"Participant: {t.participant_reply}")
        return "\n".join(lines) or "(interview not started)"

    # ---- export ----------------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Compact view of the current structured state (what the planner reads each round)."""
        qid = self.active_qid
        qs = self.qstate.get(qid) if qid else None
        return {
            "current_question_id": qid,
            "question_status": "in_progress" if qs and qs.status == "active" else (qs.status if qs else None),
            "core_status": qs.core_status if qs else None,
            "slot_status": {k: s.status for k, s in qs.slots.items()} if qs else {},
            "evidence_bank": [asdict(e) for e in self.evidence_bank if e.question_id == qid],
            "followup_attempts": {k: s.attempts for k, s in qs.slots.items()} if qs else {},
            "last_action": self.last_action(),
            "disruption_state": asdict(self.disruption),
            "completed_questions": [k for k, v in self.qstate.items() if v.status in ("done", "incomplete", "skipped")],
            "next_pending": self.next_pending(),
        }

    def report(self) -> Dict[str, Any]:
        """Per-question outcome, including what is still missing and why (budget, refusal, ...)."""
        out = {}
        for qid, qs in self.qstate.items():
            out[qid] = {
                "status": qs.status, "core_status": qs.core_status, "note": qs.note,
                "answered": [k for k, s in qs.slots.items() if s.status == ANSWERED],
                "missing_required": [k for k, s in qs.slots.items() if s.required and s.status in OPEN],
                "refused_or_unknown": {k: s.status for k, s in qs.slots.items() if s.status in FINAL_UNRESOLVED},
                "not_applicable": [k for k, s in qs.slots.items() if s.status == NOT_APPLICABLE],
                "contradicted": [k for k, s in qs.slots.items() if s.status == CONTRADICTED],
                "unreached_optional": [k for k, s in qs.slots.items() if not s.required and s.status == MISSING],
            }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outline": self.outline.to_dict(),
            "policy": asdict(self.policy),
            "questions": {k: asdict(v) for k, v in self.qstate.items()},
            "evidence_bank": [asdict(e) for e in self.evidence_bank],
            "report": self.report(),
            "turns": [asdict(t) for t in self.turns],
        }
