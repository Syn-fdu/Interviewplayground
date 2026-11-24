"""Respondent-only prompt defaults.

The interviewer's prompt and behavior are exclusively owned by the bundled
`interview_agent` package (Planner, Reflection and Behavior skill files).
This module must not define or override any interviewer instructions.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RESPONDENT=(
    "Researcher-selected respondent conversational guidance:\n"
    "Answer the specific question naturally with the allowed facts. Distinguish "
    "uncertainty from refusal and do not repeat a stock sentence. Follow the "
    "configured six traits and do not add biographical facts beyond the "
    "evidence supplied for this turn."
)
MAX_CHARS=16000

def read_templates():
    # Preserve a saved respondent-only instruction from existing projects without
    # loading or applying their obsolete interviewer prompt / model settings.
    try:
        obj=json.loads((ROOT/"resources"/"prompts"/"respondent_prompt_only.json").read_text(encoding="utf-8"))
        text=obj.get("respondent")
        if isinstance(text,str) and 0<len(text)<=MAX_CHARS and "\\x00" not in text:
            return {"respondent":text}
    except (OSError,ValueError,UnicodeError,TypeError):
        pass
    return {"respondent":DEFAULT_RESPONDENT}
