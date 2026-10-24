"""Agentic Assistant: a multi-tool LLM agent with task decomposition and offline fallbacks."""

from __future__ import annotations

__version__ = "1.0.0"

from .agent import Agent, AgentResponse, Step
from .config import Settings
from .tools import Tool, ToolError, ToolResult

__all__ = ["Agent", "AgentResponse", "Settings", "Step", "Tool", "ToolError", "ToolResult", "__version__"]
