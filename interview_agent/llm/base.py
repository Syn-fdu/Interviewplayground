from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

Message = Dict[str, str]


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    """Minimal provider-agnostic chat interface. Implement `chat` to add a backend."""

    @abstractmethod
    def chat(self, messages: List[Message], temperature: Optional[float] = None,
             json_mode: bool = False) -> str:
        ...

    def chat_json(self, messages: List[Message], temperature: Optional[float] = None) -> Dict[str, Any]:
        """Chat and parse a JSON object, retrying once with a repair hint."""
        raw = self.chat(messages, temperature=temperature, json_mode=True)
        try:
            return parse_json(raw)
        except ValueError:
            retry = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your reply was not valid JSON. Reply again with ONLY one valid JSON object."},
            ]
            return parse_json(self.chat(retry, temperature=0.0, json_mode=True))


def parse_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from model output (handles fences / reasoning residue)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise ValueError("no JSON object found")
        depth, in_str, esc = 0, False, False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            raise ValueError("unterminated JSON object")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj
