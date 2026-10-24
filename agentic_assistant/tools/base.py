"""Tool abstraction shared by every tool the agent can call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    output: str
    success: bool = True

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(output=f"Error: {message}", success=False)


class ToolError(Exception):
    """Raise inside a tool to report a user-facing failure (it becomes a failed ``ToolResult``)."""


@dataclass(frozen=True)
class Tool:
    """A named capability with a description the LLM uses to decide when to call it."""

    name: str
    description: str
    func: Callable[[str], str | ToolResult]

    def __call__(self, tool_input: str) -> ToolResult:
        try:
            result = self.func(tool_input)
        except ToolError as exc:
            return ToolResult.error(str(exc))
        if isinstance(result, ToolResult):
            return result
        return ToolResult(str(result))
