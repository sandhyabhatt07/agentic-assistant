"""Built-in tools."""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient
from .base import Tool, ToolError, ToolResult
from .calculator import calculator_tool
from .datetime_tool import make_datetime_tool
from .wikipedia_search import make_wikipedia_tool

__all__ = ["Tool", "ToolError", "ToolResult", "default_tools"]


def default_tools(settings: Settings, llm: LLMClient | None = None) -> list[Tool]:
    """The calculator, Wikipedia and datetime tools, configured from ``settings``."""
    return [
        calculator_tool,
        make_wikipedia_tool(settings.wikipedia_sentences, settings.wikipedia_language),
        make_datetime_tool(llm),
    ]
