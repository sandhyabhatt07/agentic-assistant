from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from agentic_assistant.agent import Agent
from agentic_assistant.config import Settings
from agentic_assistant.llm import LLMError, Message
from agentic_assistant.tools import Tool
from agentic_assistant.tools.calculator import calculator_tool
from agentic_assistant.tools.datetime_tool import make_datetime_tool


class FakeLLM:
    """Scripted LLM: each call pops the next reply (a string, or an exception to raise)."""

    def __init__(
        self, replies: list[str | Exception] | None = None, responder: Callable[[list[Message]], str] | None = None
    ) -> None:
        self.replies = list(replies or [])
        self.responder = responder
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, max_tokens: int = 300, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if self.responder is not None:
            return self.responder(messages)
        if not self.replies:
            raise LLMError("no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


FIXED_NOW = datetime(2026, 10, 24, 15, 30)


def fake_wikipedia(query: str) -> str:
    return f"Summary of {query}."


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key=None, log_file=None)


@pytest.fixture
def offline_tools() -> list[Tool]:
    return [
        calculator_tool,
        Tool("wikipedia", "Searches Wikipedia.", fake_wikipedia),
        make_datetime_tool(clock=lambda: FIXED_NOW),
    ]


@pytest.fixture
def offline_agent(settings: Settings, offline_tools: list[Tool]) -> Agent:
    return Agent(settings, tools=offline_tools, use_llm=False)
