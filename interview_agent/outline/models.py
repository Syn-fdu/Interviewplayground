from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PRIORITIES = ("low", "medium", "high")


@dataclass
class Slot:
    """One unit of information a question should elicit."""
    id: str                      # unique within its question, e.g. "tools_used"
    description: str
    required: bool = True        # required slots decide whether the question is complete
    priority: str = "medium"     # value of obtaining it (required slots default to high)
    source: str = "llm"          # llm | guide_probe | fallback  (traceability of auto-extracted slots)


@dataclass
class BranchRule:
    """Skip/jump rule from the guide. `condition` keeps the guide's wording; `slot_id` + `when_status`
    make it machine-evaluable (e.g. skip Q3 if `uses_ai` is not_applicable)."""
    condition: str = ""
    slot_id: str = ""
    when_status: str = ""
    skip: List[str] = field(default_factory=list)
    goto: str = ""


@dataclass
class Question:
    id: str                      # e.g. "T1Q2"
    text: str                    # main question as written in the guide (original text)
    topic_id: str
    goal: str = ""               # what information this question should elicit
    probes: List[str] = field(default_factory=list)  # optional probes suggested by the guide
    intent: str = ""
    section: str = ""            # title of the topic/section the question belongs to
    required_slots: List[Slot] = field(default_factory=list)
    optional_deepening_slots: List[Slot] = field(default_factory=list)
    branch_rules: List[BranchRule] = field(default_factory=list)

    def slots(self) -> List[Slot]:
        return self.required_slots + self.optional_deepening_slots

    def slot(self, slot_id: str) -> Optional[Slot]:
        return next((s for s in self.slots() if s.id == slot_id), None)


@dataclass
class Topic:
    id: str
    title: str
    questions: List[Question] = field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _parse_slots(raw: Any, required: bool, used: set, prefix: str, source: str) -> List[Slot]:
    slots: List[Slot] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or item.get("desc") or "").strip()
        if not desc:
            continue
        base = _slug(str(item.get("slot_id") or item.get("id") or "")) or f"{prefix}{len(used) + 1}"
        sid, n = base, 2
        while sid in used:
            sid, n = f"{base}_{n}", n + 1
        used.add(sid)
        prio = str(item.get("priority", "high" if required else "medium")).lower()
        slots.append(Slot(id=sid, description=desc, required=required,
                          priority=prio if prio in PRIORITIES else "medium",
                          source=str(item.get("source") or source)))
    return slots


def _parse_rules(raw: Any) -> List[BranchRule]:
    rules: List[BranchRule] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            item = {"condition": item}
        if not isinstance(item, dict):
            continue
        skip = item.get("skip") or []
        rule = BranchRule(
            condition=str(item.get("condition", "") or ""), slot_id=str(item.get("slot_id", "") or ""),
            when_status=str(item.get("when_status", "") or ""),
            skip=[str(s) for s in (skip if isinstance(skip, list) else [skip])],
            goto=str(item.get("goto", "") or ""),
        )
        if rule.condition or rule.skip or rule.goto:
            rules.append(rule)
    return rules


@dataclass
class Outline:
    title: str
    objective: str = ""
    topics: List[Topic] = field(default_factory=list)
    notes: str = ""              # participant info, ethics, tone guidance etc.

    def questions(self) -> List[Question]:
        return [q for t in self.topics for q in t.questions]

    def get(self, qid: str) -> Optional[Question]:
        return next((q for q in self.questions() if q.id == qid), None)

    def topic_of(self, qid: str) -> Optional[Topic]:
        q = self.get(qid)
        return next((t for t in self.topics if q and t.id == q.topic_id), None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Outline":
        topics: List[Topic] = []
        for ti, t in enumerate(d.get("topics", []), 1):
            tid = str(t.get("id") or f"T{ti}")
            title = str(t.get("title", tid))
            qs = []
            for qi, q in enumerate(t.get("questions", []), 1):
                if isinstance(q, str):
                    q = {"text": q}
                if not str(q.get("text", "")).strip():
                    continue
                qs.append(cls._question(q, str(q.get("id") or f"{tid}Q{qi}"), tid, title))
            if qs:
                topics.append(Topic(id=tid, title=title, questions=qs))
        outline = cls(title=str(d.get("title", "Interview")), objective=str(d.get("objective", "")),
                      topics=topics, notes=str(d.get("notes", "")))
        outline._validate_rules()
        return outline

    @staticmethod
    def _question(q: Dict[str, Any], qid: str, tid: str, section: str) -> Question:
        used: set = set()
        required = _parse_slots(q.get("required_slots"), True, used, "req", "llm")
        probes = [str(p) for p in q.get("probes", []) or []]
        optional = _parse_slots(q.get("optional_deepening_slots", q.get("optional_slots")), False, used, "opt", "llm")
        if not optional and probes:
            optional = _parse_slots(probes, False, used, "probe", "guide_probe")
        text, goal = str(q["text"]).strip(), str(q.get("goal", ""))
        if not required:  # never leave a question without a completion criterion
            required = _parse_slots([{"slot_id": "main_answer", "description": goal or text}], True, used,
                                    "req", "fallback")
        return Question(id=qid, text=text, topic_id=tid, goal=goal, probes=probes,
                        intent=str(q.get("intent", "") or goal), section=section, required_slots=required,
                        optional_deepening_slots=optional, branch_rules=_parse_rules(q.get("branch_rules")))

    def _validate_rules(self) -> None:
        """Drop references to unknown questions/slots; keep the rule's original condition text."""
        ids = {q.id for q in self.questions()}
        for q in self.questions():
            for r in q.branch_rules:
                r.skip = [s for s in r.skip if s in ids and s != q.id]
                r.goto = r.goto if r.goto in ids and r.goto != q.id else ""
                if r.slot_id and not q.slot(r.slot_id):
                    r.slot_id = r.when_status = ""
                if not r.slot_id:
                    r.when_status = ""
