from __future__ import annotations

from typing import Callable, List, Optional, Union

from .base import LLMClient, Message


class MockLLM(LLMClient):
    """Scripted LLM for tests/offline demos. `responses` is a list of strings or a callable
    (messages) -> str. Records all calls in `.calls`."""

    def __init__(self, responses: Union[List[str], Callable[[List[Message]], str]]):
        self.responses = responses
        self.calls: List[List[Message]] = []

    def chat(self, messages: List[Message], temperature: Optional[float] = None,
             json_mode: bool = False) -> str:
        self.calls.append(messages)
        if callable(self.responses):
            return self.responses(messages)
        if not self.responses:
            raise RuntimeError("MockLLM exhausted")
        return self.responses.pop(0)
