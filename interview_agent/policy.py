"""Interview policy: configurable follow-up budgets and the deterministic follow-up gate."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .outline.models import PRIORITIES
from .slots import OPEN

if TYPE_CHECKING:  # pragma: no cover
    from .reflection_schema import FollowupCandidate
    from .state import InterviewState

CONFIG_ENV = "INTERVIEW_CONFIG"
DEFAULT_CONFIG_FILE = "interview_config.json"


@dataclass(frozen=True)
class InterviewPolicy:
    max_followups_per_slot: int = 2             # follow-ups spent on any single slot
    max_followups_per_question: int = 4         # all follow-ups on one main question
    max_optional_followups_per_question: int = 2  # deepening follow-ups after the core is covered
    optional_min_priority: str = "medium"       # lowest priority of a deepening slot worth an extra turn
    max_unproductive_followups: int = 2         # consecutive follow-ups that produced no state change
    max_disruption_repairs: int = 2             # repair attempts for one unresolved disruption
    output_language: str = "中文"               # language of free-text planner/reflection output (focus, reasons)

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "output_language":
                if not isinstance(v, str) or not v.strip():
                    raise ValueError("output_language must be a non-empty string")
            elif f.name == "optional_min_priority":
                if v not in PRIORITIES:
                    raise ValueError(f"{f.name} must be one of {PRIORITIES}, got {v!r}")
            elif isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{f.name} must be a non-negative integer, got {v!r}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterviewPolicy":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return cls(**d)


def load_policy(path: Optional[str | Path] = None, **overrides: Any) -> InterviewPolicy:
    """Policy from a JSON file (explicit path, $INTERVIEW_CONFIG, or ./interview_config.json if present),
    with non-None keyword overrides applied last."""
    path = path or os.getenv(CONFIG_ENV)
    if not path and Path(DEFAULT_CONFIG_FILE).is_file():
        path = DEFAULT_CONFIG_FILE
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(data, dict):
        raise ValueError("interview config must be a JSON object")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return InterviewPolicy.from_dict(data)


@dataclass
class FollowupOption:
    slot_id: str
    required: bool
    priority: str
    reason: str
    lead: str = ""


@dataclass
class FollowupAssessment:
    options: List[FollowupOption]
    stop_reasons: List[str]


_RANK = {p: i for i, p in enumerate(PRIORITIES)}


def assess_followup(state: "InterviewState", candidates: List["FollowupCandidate"]) -> FollowupAssessment:
    """Which follow-ups are still allowed for the active question, and why not (if none)?

    Checks: slot still open, not already obtained, not refused/unknown, per-slot / per-question budgets,
    deepening value, and whether recent follow-ups stopped producing anything."""
    p = state.policy
    qid = state.active_qid
    qs = state.qstate.get(qid) if qid else None
    if not qs:
        return FollowupAssessment([], ["no active question"])
    if qs.followups >= p.max_followups_per_question:
        return FollowupAssessment(
            [], [f"question follow-up budget exhausted ({qs.followups}/{p.max_followups_per_question})"])
    if qs.unproductive_followups >= p.max_unproductive_followups:
        return FollowupAssessment(
            [], [f"{qs.unproductive_followups} consecutive follow-ups produced no new information"])

    cand = {c.target_slot: c for c in candidates}
    options: List[FollowupOption] = []
    stops: List[str] = []

    def eligible(ss) -> bool:
        if ss.status not in OPEN:
            stops.append(f"{ss.slot_id}: already {ss.status}")
            return False
        if ss.attempts >= p.max_followups_per_slot:
            stops.append(f"{ss.slot_id}: follow-up budget exhausted ({ss.attempts}/{p.max_followups_per_slot}); "
                         f"status stays {ss.status}")
            return False
        return True

    for ss in qs.slots.values():
        if ss.required and eligible(ss):
            c = cand.get(ss.slot_id)
            options.append(FollowupOption(ss.slot_id, True, "high",
                                          c.reason if c and c.reason else f"required slot still {ss.status}",
                                          c.lead if c else ""))

    if qs.core_status == "covered":
        if qs.optional_followups >= p.max_optional_followups_per_question:
            stops.append(f"deepening budget exhausted ({qs.optional_followups}/{p.max_optional_followups_per_question})")
        else:
            deep = []
            for c in candidates:
                ss = qs.slots.get(c.target_slot)
                if not ss or ss.required or not eligible(ss):
                    continue
                prio = min(ss.priority, c.priority, key=lambda x: _RANK.get(x, 1))
                if _RANK[prio] < _RANK[p.optional_min_priority]:
                    stops.append(f"{ss.slot_id}: deepening value {prio} below threshold {p.optional_min_priority}")
                    continue
                deep.append(FollowupOption(ss.slot_id, False, prio, c.reason or "worth deepening", c.lead))
            options += sorted(deep, key=lambda o: -_RANK[o.priority])
    elif not options:
        stops.append("required information unresolved but no further follow-up is allowed")
    return FollowupAssessment(options, [] if options else stops or ["no worthwhile follow-up target"])
