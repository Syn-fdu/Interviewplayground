"""Information-slot statuses, legal transitions and evidence grounding (no LLM, no interview state)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

MISSING, PARTIAL, ANSWERED = "missing", "partial", "answered"
REFUSED, UNKNOWN, NOT_APPLICABLE, CONTRADICTED = "refused", "unknown", "not_applicable", "contradicted"
SLOT_STATUSES = (MISSING, PARTIAL, ANSWERED, REFUSED, UNKNOWN, NOT_APPLICABLE, CONTRADICTED)

OPEN = frozenset({MISSING, PARTIAL, CONTRADICTED})           # still worth asking about
SATISFIED = frozenset({ANSWERED, NOT_APPLICABLE})            # counts towards completing a question
FINAL_UNRESOLVED = frozenset({REFUSED, UNKNOWN})             # participant closed the topic; never re-ask

_ALLOWED = {
    MISSING: {PARTIAL, ANSWERED, REFUSED, UNKNOWN, NOT_APPLICABLE, CONTRADICTED},
    PARTIAL: {ANSWERED, REFUSED, UNKNOWN, NOT_APPLICABLE, CONTRADICTED},
    ANSWERED: {CONTRADICTED},
    REFUSED: {PARTIAL, ANSWERED},          # participant volunteers the information later
    UNKNOWN: {PARTIAL, ANSWERED},          # participant remembers later
    NOT_APPLICABLE: set(),
    CONTRADICTED: {PARTIAL, ANSWERED, REFUSED, UNKNOWN},
}


def transition_allowed(old: str, new: str) -> Tuple[bool, str]:
    if old not in _ALLOWED or new not in _ALLOWED:
        return False, f"unknown status {old!r}->{new!r}"
    if old == new:
        return False, "status unchanged"
    if new not in _ALLOWED[old]:
        return False, f"transition {old}->{new} not allowed"
    return True, ""


_NOISE = re.compile(r"[\s\u3000，。、；：！？,.;:!?\"'“”‘’（）()\[\]【】…\-—]+")


def normalize(text: str) -> str:
    return _NOISE.sub("", text or "").lower()


def find_quote(quote: str, replies: Sequence[Tuple[int, str]]) -> Optional[int]:
    """Return the turn id of the most recent participant reply containing `quote` (whitespace/punctuation
    insensitive), or None. A quote must be a real excerpt: this is what grounds every state update."""
    q = normalize(quote)
    if len(q) < 2:
        return None
    for turn_id, text in reversed(list(replies)):
        if q in normalize(text):
            return turn_id
    return None


@dataclass
class EvidenceItem:
    slot_id: str
    question_id: str
    turn_id: int
    quote: str
    status: str


@dataclass
class SlotState:
    slot_id: str
    required: bool
    description: str
    priority: str = "medium"
    status: str = MISSING
    attempts: int = 0                      # follow-ups already spent on this slot
    evidence: List[EvidenceItem] = field(default_factory=list)
