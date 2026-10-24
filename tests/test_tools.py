from __future__ import annotations

from datetime import datetime

import pytest

from agentic_assistant.llm import LLMError
from agentic_assistant.tools import Tool, ToolError, ToolResult
from agentic_assistant.tools.calculator import calculate, calculator_tool
from agentic_assistant.tools.datetime_tool import format_datetime, make_datetime_tool, parse_datetime_query

from .conftest import FIXED_NOW, FakeLLM


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("5 + 3", "8"),
        ("4 * 5", "20"),
        ("2 ** 10", "1024"),
        ("(4 - 1) / 2", "1.5"),
        ("10 / 2", "5"),
        ("7 // 2", "3"),
        ("-3 + 1", "-2"),
        ("sqrt(81)", "9"),
        ("round(pi, 2)", "3.14"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_calculator_evaluates(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "lambda: 1",
        "x + 1",
        "9 ** 9 ** 9",
        "",
        "1 +",
    ],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression: str) -> None:
    result = calculator_tool(expression)
    assert not result.success
    assert result.output.startswith("Error:")


def test_calculator_division_by_zero() -> None:
    assert calculator_tool("100 / 0") == ToolResult("Error: division by zero", success=False)


def test_tool_wraps_plain_strings_and_tool_errors() -> None:
    def flaky(text: str) -> str:
        if text == "bad":
            raise ToolError("nope")
        return text.upper()

    tool = Tool("echo", "Echo", flaky)
    assert tool("hi") == ToolResult("HI")
    assert tool("bad") == ToolResult("Error: nope", success=False)


@pytest.mark.parametrize(
    ("query", "offset", "include_time"),
    [
        ("what's the date tomorrow", 1, False),
        ("what is the date day after tomorrow", 2, False),  # used to return tomorrow
        ("day before yesterday", -2, False),
        ("yesterday", -1, False),
        ("3 days ago", -3, False),
        ("2 weeks from now", 14, False),
        ("next week", 7, False),
        ("what time is it", 0, True),
        ("today's date", 0, False),
    ],
)
def test_parse_datetime_query(query: str, offset: int, include_time: bool) -> None:
    assert parse_datetime_query(query, FIXED_NOW) == (offset, include_time)


def test_format_datetime_ordinals() -> None:
    assert format_datetime(datetime(2026, 10, 1)) == "Thursday, October 1st, 2026"
    assert format_datetime(datetime(2026, 10, 12)) == "Monday, October 12th, 2026"
    assert format_datetime(datetime(2026, 10, 23)) == "Friday, October 23rd, 2026"
    assert format_datetime(FIXED_NOW, include_time=True) == "Saturday, October 24th, 2026 at 03:30 PM"


def test_datetime_tool_uses_llm_offset() -> None:
    tool = make_datetime_tool(
        FakeLLM(['```json\n{"days_offset": 10, "include_time": false}\n```']), clock=lambda: FIXED_NOW
    )
    assert tool("a week and a half from now").output == "Tuesday, November 3rd, 2026"


def test_datetime_tool_falls_back_when_llm_fails() -> None:
    tool = make_datetime_tool(FakeLLM([LLMError("boom")]), clock=lambda: FIXED_NOW)
    assert tool("tomorrow").output == "Sunday, October 25th, 2026"


@pytest.mark.network
def test_wikipedia_live_lookup() -> None:
    from agentic_assistant.tools.wikipedia_search import make_wikipedia_tool

    result = make_wikipedia_tool()("Python programming language")
    assert result.success
    assert "Python" in result.output


class _FakeWikipedia:
    """Stands in for the `wikipedia` package so tests never touch the network."""

    def __init__(self, titles: list[str], summary: str | Exception = "A summary.") -> None:
        import wikipedia

        self.exceptions = wikipedia.exceptions
        self.wikipedia = wikipedia.wikipedia
        self._titles = titles
        self._summary = summary

    def set_lang(self, _: str) -> None: ...

    def set_user_agent(self, _: str) -> None: ...

    def search(self, query: str, results: int = 10) -> list[str]:
        return self._titles

    def summary(self, title: str, sentences: int = 2, auto_suggest: bool = True) -> str:
        assert auto_suggest is False
        if isinstance(self._summary, Exception):
            raise self._summary
        return self._summary


def _wiki_tool(monkeypatch: pytest.MonkeyPatch, fake: _FakeWikipedia) -> Tool:
    from agentic_assistant.tools import wikipedia_search

    monkeypatch.setattr(wikipedia_search, "wikipedia", fake)
    return wikipedia_search.make_wikipedia_tool()


def test_wikipedia_uses_https(monkeypatch: pytest.MonkeyPatch) -> None:
    import wikipedia

    from agentic_assistant.tools.wikipedia_search import make_wikipedia_tool

    monkeypatch.setattr(wikipedia.wikipedia, "API_URL", wikipedia.wikipedia.API_URL)
    make_wikipedia_tool(language="de")
    assert wikipedia.wikipedia.API_URL == "https://de.wikipedia.org/w/api.php"


def test_wikipedia_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _wiki_tool(monkeypatch, _FakeWikipedia(["Python (programming language)"]))
    assert tool("python") == ToolResult("A summary.")


def test_wikipedia_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import wikipedia

    assert not _wiki_tool(monkeypatch, _FakeWikipedia([]))("zzzz").success
    disambiguation = wikipedia.exceptions.DisambiguationError("Mercury", ["Mercury (planet)", "Mercury (element)"])
    result = _wiki_tool(monkeypatch, _FakeWikipedia(["Mercury"], disambiguation))("Mercury")
    assert not result.success and "Mercury (planet)" in result.output
    offline = _wiki_tool(monkeypatch, _FakeWikipedia(["X"], ConnectionError("no network")))("X")
    assert not offline.success and "ConnectionError" in offline.output
