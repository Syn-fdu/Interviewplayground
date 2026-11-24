"""Structured reflection result + tolerant, grounded parsing of the model's JSON.

The parser never trusts the model blindly: unknown slots, invalid statuses and quotes that do not appear
in the participant's own words are dropped (and reported in `rejected_updates`)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .outline.models import PRIORITIES
from .slots import MISSING, SLOT_STATUSES, find_quote

DISRUPTIONS = ("none", "misunderstanding", "refusal", "contradiction", "off_topic", "other")
LEVELS = ("none", "low", "medium", "high")
MAX_CANDIDATES = 3


@dataclass
class EvidenceQuote:
    quote: str
    related_slots: List[str] = field(default_factory=list)


@dataclass
class AnswerAnalysis:
    relevance: str = "unknown"
    specificity: str = "unknown"
    novelty: str = "unknown"             # overridden by the state: only newly accepted evidence is "new"
    evidence: List[EvidenceQuote] = field(default_factory=list)


@dataclass
class CoreCoverage:
    status: str = "not_covered"          # covered | partial | not_covered  (computed by the state, not the model)
    missing_required_slots: List[str] = field(default_factory=list)
    unresolved_required_slots: List[str] = field(default_factory=list)  # refused / unknown


@dataclass
class SlotUpdate:
    slot_id: str
    new_status: str
    evidence: str
    old_status: str = MISSING            # filled by the state from its own record
    turn_id: int = 0                     # turn whose participant reply contains the evidence


@dataclass
class ActionEffectiveness:
    goal_achieved: bool = False
    reason: str = ""
    verified: bool = False               # True when the state could check the claim against slot changes


@dataclass
class FollowupCandidate:
    target_slot: str
    reason: str = ""
    priority: str = "medium"
    lead: str = ""                       # optional pointer into the participant's words worth echoing


@dataclass
class DisruptionReport:
    type: str = "none"
    severity: str = "none"
    evidence: str = ""


@dataclass
class ReflectionResult:
    question_id: str = ""
    answer_analysis: AnswerAnalysis = field(default_factory=AnswerAnalysis)
    core_coverage: CoreCoverage = field(default_factory=CoreCoverage)
    slot_updates: List[SlotUpdate] = field(default_factory=list)
    rejected_updates: List[Dict[str, str]] = field(default_factory=list)
    previous_action_effectiveness: ActionEffectiveness = field(default_factory=ActionEffectiveness)
    followup_candidates: List[FollowupCandidate] = field(default_factory=list)
    disruption: DisruptionReport = field(default_factory=DisruptionReport)
    rationale: str = ""
    failed: bool = False                 # the model output was unusable; nothing was learned this round
    error: str = ""

    @classmethod
    def neutral(cls, question_id: str, error: str = "") -> "ReflectionResult":
        return cls(question_id=question_id, failed=bool(error), error=error,
                   rationale=error and f"reflection failed ({error}); no state changes")


def _level(v: Any, allowed: Sequence[str] = LEVELS, default: str = "unknown") -> str:
    v = str(v or "").strip().lower()
    return v if v in allowed else default


def _items(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def parse_reflection(data: Dict[str, Any], question_id: str, valid_slots: Sequence[str],
                     replies: Sequence[Tuple[int, str]], latest_turn: int) -> ReflectionResult:
    """`replies` are all (turn_id, participant_text) pairs so far; the latest one has id `latest_turn`."""
    if not isinstance(data, dict):
        raise ValueError("reflection is not a JSON object")
    valid = set(valid_slots)
    latest = [(t, x) for t, x in replies if t == latest_turn]
    res = ReflectionResult(question_id=question_id, rationale=str(data.get("rationale", "") or ""))

    aa = data.get("answer_analysis") if isinstance(data.get("answer_analysis"), dict) else {}
    for ev in _items(aa.get("evidence")):
        if not isinstance(ev, dict) or find_quote(str(ev.get("quote", "")), latest) is None:
            continue
        slots = [s for s in _items(ev.get("related_slots")) if s in valid]
        res.answer_analysis.evidence.append(EvidenceQuote(str(ev["quote"]).strip(), slots))
    res.answer_analysis.relevance = _level(aa.get("relevance"))
    res.answer_analysis.specificity = _level(aa.get("specificity"))
    res.answer_analysis.novelty = _level(aa.get("novelty"))
    if not res.answer_analysis.evidence and res.answer_analysis.specificity in ("medium", "high"):
        res.answer_analysis.specificity = "low"   # specificity claims need a quoted passage

    for u in _items(data.get("slot_updates")):
        if not isinstance(u, dict):
            continue
        sid, new = str(u.get("slot_id", "")), str(u.get("new_status", "")).strip().lower()
        quote = str(u.get("evidence", "") or "")

        def reject(reason: str) -> None:
            res.rejected_updates.append({"slot_id": sid, "new_status": new, "reason": reason, "evidence": quote})

        if sid not in valid:
            reject("unknown slot for this question")
        elif new not in SLOT_STATUSES or new == MISSING:
            reject("invalid target status")
        else:
            turn = find_quote(quote, replies)
            if turn is None:
                reject("evidence is not a quote from the participant")
            else:
                res.slot_updates.append(SlotUpdate(sid, new, quote.strip(), turn_id=turn))

    seen = set()
    for c in _items(data.get("followup_candidates")):
        if isinstance(c, dict) and c.get("target_slot") in valid and c["target_slot"] not in seen:
            seen.add(c["target_slot"])
            res.followup_candidates.append(FollowupCandidate(
                c["target_slot"], str(c.get("reason", "") or ""), _level(c.get("priority"), PRIORITIES, "medium"),
                str(c.get("lead", "") or "")))
    res.followup_candidates = res.followup_candidates[:MAX_CANDIDATES]

    pae = data.get("previous_action_effectiveness")
    if isinstance(pae, dict):
        res.previous_action_effectiveness = ActionEffectiveness(
            bool(pae.get("goal_achieved", False)), str(pae.get("reason", "") or ""))

    d = data.get("disruption")
    d = d if isinstance(d, dict) else {"type": d}
    dtype = _level(d.get("type"), DISRUPTIONS, "none" if not d.get("type") else "other")
    quote = str(d.get("evidence", "") or "")
    if dtype != "none" and find_quote(quote, latest) is not None:  # disruptions need a quoted trigger too
        res.disruption = DisruptionReport(dtype, _level(d.get("severity"), LEVELS, "medium"), quote.strip())
    return res
