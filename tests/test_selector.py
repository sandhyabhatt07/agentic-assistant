from __future__ import annotations

import pytest

from agentic_assistant.llm import LLMError
from agentic_assistant.selector import ToolSelector
from agentic_assistant.tools import Tool

from .conftest import FakeLLM


@pytest.fixture
def tools(offline_tools: list[Tool]) -> dict[str, Tool]:
    return {t.name: t for t in offline_tools}


@pytest.mark.parametrize(
    ("task", "tool"),
    [
        ("what is 5 + 3", "calculator"),
        ("calculate 25 times 4", "calculator"),
        ("Explain GPT-4", "wikipedia"),  # the hyphen used to route this to the calculator
        ("what is the address of the White House", "wikipedia"),  # "add" inside "address"
        ("do you know Python", "wikipedia"),  # "now" inside "know"
        ("what time is it", "datetime"),
        ("what's the date tomorrow", "datetime"),
        ("tell me about Einstein", "wikipedia"),
    ],
)
def test_fallback_heuristics(tools: dict[str, Tool], task: str, tool: str) -> None:
    assert ToolSelector(tools).select(task).tool == tool


def test_fallback_extracts_wikipedia_topic(tools: dict[str, Tool]) -> None:
    selection = ToolSelector(tools).select("tell me about Albert Einstein?")
    assert (selection.tool, selection.input, selection.source) == ("wikipedia", "Albert Einstein", "heuristic")


def test_llm_selection_is_used(tools: dict[str, Tool]) -> None:
    llm = FakeLLM(['{"tool": "calculator", "input": "50 * 2", "reasoning": "follow-up"}'])
    selection = ToolSelector(tools, llm).select("multiply that by 2", context="previous result 50")
    assert (selection.tool, selection.input, selection.source) == ("calculator", "50 * 2", "llm")
    assert any("Context:" in m["content"] for m in llm.calls[0])


@pytest.mark.parametrize("reply", ["not json", '{"tool": "weather", "input": "x"}', LLMError("down")])
def test_invalid_llm_output_falls_back(tools: dict[str, Tool], reply: str | Exception) -> None:
    selection = ToolSelector(tools, FakeLLM([reply])).select("what is 2 + 2")
    assert (selection.tool, selection.source) == ("calculator", "heuristic")


def test_prompt_lists_registered_tools_and_examples(tools: dict[str, Tool]) -> None:
    prompt = ToolSelector(tools).build_prompt()
    for name in tools:
        assert f"- {name}:" in prompt
    assert "Examples:" in prompt
