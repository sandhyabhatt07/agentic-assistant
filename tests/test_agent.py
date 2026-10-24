from __future__ import annotations

import json

import pytest

from agentic_assistant.agent import Agent, is_multi_part_question, split_multi_part
from agentic_assistant.config import Settings
from agentic_assistant.tools import Tool

from .conftest import FakeLLM


def test_agent_starts_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: ToolSelector used to build OpenAI(api_key=None), which raises at startup
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = Agent(Settings(openai_api_key=None, log_file=None))
    assert agent.llm is None
    assert set(agent.tools) == {"calculator", "wikipedia", "datetime"}


def test_single_tool_answer(offline_agent: Agent) -> None:
    response = offline_agent.ask("what is 25 * 4?")
    assert response.answer == "100"
    assert response.mode == "single-tool"
    assert response.steps[0].tool == "calculator"
    assert offline_agent.session.get_session_stats()["successful_calls"] == 1


def test_offline_follow_up_uses_previous_result(offline_agent: Agent) -> None:
    assert offline_agent.run("what is 10 * 5?") == "50"
    assert offline_agent.run("multiply that by 2") == "100"


def test_errors_are_reported_not_raised(offline_agent: Agent) -> None:
    response = offline_agent.ask("divide 100 by 0")
    assert response.answer == "Error: division by zero"
    assert not response.success


def test_broken_tool_does_not_crash_agent(settings: Settings) -> None:
    def broken(_: str) -> str:
        raise RuntimeError("bug")

    agent = Agent(settings, tools=[Tool("calculator", "math", broken)], use_llm=False)
    response = agent.ask("1 + 1")
    assert not response.success
    assert "failed unexpectedly" in response.answer
    assert agent.session.get_session_stats()["total_errors"] == 1


def test_multi_tool_offline(offline_agent: Agent) -> None:
    response = offline_agent.ask("what is 5 + 3 and also tell me about Python")
    assert response.mode == "multi-tool"
    assert [s.tool for s in response.steps] == ["calculator", "wikipedia"]
    assert "8" in response.answer and "Summary of Python" in response.answer
    assert offline_agent.memory.history[-1].tool == "multi-tool"


def test_multi_tool_with_llm(settings: Settings, offline_tools: list[Tool]) -> None:
    def respond(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if system.startswith("Break the user's message"):
            return json.dumps(
                [
                    {"description": "calculate 5+3", "query": "5+3"},
                    {"description": "about Python", "query": "Python programming language"},
                ]
            )
        if system.startswith("You are a tool selector"):
            query = messages[-1]["content"]
            tool = "calculator" if "+" in query else "wikipedia"
            return json.dumps({"tool": tool, "input": query, "reasoning": "test"})
        if system.startswith("Combine the tool results"):
            return "5+3 is 8, and Python is a programming language."
        raise AssertionError(f"unexpected prompt: {system[:40]}")

    agent = Agent(settings, llm=FakeLLM(responder=respond), tools=offline_tools)
    response = agent.answer_multi_tool("what is 5+3 and tell me about Python")
    assert response.answer == "5+3 is 8, and Python is a programming language."
    assert [s.result for s in response.steps] == ["8", "Summary of Python programming language."]


def test_direct_answer_from_context(settings: Settings, offline_tools: list[Tool]) -> None:
    llm = FakeLLM(
        responder=lambda m: (
            '{"needs_tool": false, "response": "Yes, 5 + 3 is 8."}'
            if m[0]["content"].startswith("Decide whether")
            else '{"tool": "calculator", "input": "5 + 3"}'
        )
    )
    agent = Agent(settings, llm=llm, tools=offline_tools)
    agent.memory.add_interaction("what is 5 + 3", "calculator", "5 + 3", "8")
    response = agent.ask("are you sure")
    assert (response.mode, response.answer) == ("direct", "Yes, 5 + 3 is 8.")


def test_add_and_remove_tool(offline_agent: Agent) -> None:
    offline_agent.add_tool("weather", lambda city: f"Sunny in {city}", "Get the weather for a city")
    assert offline_agent.execute_tool("weather", "Pune").output == "Sunny in Pune"
    offline_agent.remove_tool("weather")
    assert not offline_agent.execute_tool("weather", "Pune").success


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("what is 5+3 and also tell me about Python", ["what is 5+3", "tell me about Python"]),
        ("what's the time and what is 100/5?", ["what's the time", "what is 100/5"]),
        ("what is 2+2? who is Einstein?", ["what is 2+2", "who is Einstein"]),
        ("calculate 25*4 and search for Einstein", ["calculate 25*4", "search for Einstein"]),
        ("tell me about salt and pepper", ["tell me about salt and pepper"]),
    ],
)
def test_split_multi_part(text: str, parts: list[str]) -> None:
    assert split_multi_part(text) == parts
    assert is_multi_part_question(text) == (len(parts) > 1)
