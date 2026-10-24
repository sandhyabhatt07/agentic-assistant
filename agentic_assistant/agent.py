"""The agent: routes a question to tools (one or several) and turns the results into an answer."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .llm import LLMClient, LLMError, create_llm
from .memory import ConversationMemory
from .parsing import preprocess_math_expression, safe_parse_json, safe_parse_json_list
from .selector import ToolSelector
from .telemetry import SessionLog
from .tools import Tool, ToolResult, default_tools

logger = logging.getLogger(__name__)

# Phrases that usually join two independent questions ("what is 5+3 and also tell me about Python")
_MULTI_PART_SPLIT = re.compile(
    r"\s*(?:\?\s*(?:and|also)?\s+|,?\s+and also\s+|,?\s+and then\s+|,?\s+also\s+|,\s+and\s+"
    r"|\s+and\s+(?=(?:what|what's|who|tell|calculate|compute|search|how|when|explain|describe)\b))",
    re.IGNORECASE,
)
_FOLLOW_UP_REFERENCE = re.compile(r"\b(that|it|the result|the answer|previous result)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Step:
    """One tool call made while answering a question."""

    description: str
    tool: str
    tool_input: str
    result: str
    success: bool


@dataclass
class AgentResponse:
    answer: str
    mode: str  # "direct", "single-tool" or "multi-tool"
    steps: list[Step] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.mode == "direct" or any(step.success for step in self.steps)


def is_multi_part_question(user_input: str) -> bool:
    """Heuristic: does the input contain more than one question?"""
    if user_input.count("?") > 1:
        return True
    return len(split_multi_part(user_input)) > 1


def split_multi_part(user_input: str) -> list[str]:
    """Split a compound question on joining phrases (offline fallback for decomposition)."""
    parts = [p.strip(" ,.?") for p in _MULTI_PART_SPLIT.split(user_input)]
    return [p for p in parts if p]


class Agent:
    """Orchestrates tool selection, execution, memory and answer synthesis."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        tools: Iterable[Tool] | None = None,
        memory: ConversationMemory | None = None,
        session: SessionLog | None = None,
        use_llm: bool = True,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm if llm is not None else (create_llm(self.settings) if use_llm else None)
        tool_list = list(tools) if tools is not None else default_tools(self.settings, self.llm)
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list}
        self.memory = memory or ConversationMemory(self.settings.max_memory_size)
        self.session = session or SessionLog()
        self.selector = ToolSelector(self.tools, self.llm)
        logger.info("Agent initialised with tools: %s (LLM: %s)", ", ".join(self.tools), bool(self.llm))

    # ------------------------------------------------------------------ public API

    def ask(self, task: str) -> AgentResponse:
        """Answer ``task``, using multi-tool mode automatically for compound questions."""
        if is_multi_part_question(task):
            return self.answer_multi_tool(task)
        return self.answer(task)

    def run(self, task: str, use_memory: bool = True) -> str:
        """Answer a single question and return the answer text."""
        return self.answer(task, use_memory=use_memory).answer

    def run_multi_tool(self, task: str) -> str:
        """Decompose a compound question, answer each part with a tool, and combine the results."""
        return self.answer_multi_tool(task).answer

    def answer(self, task: str, use_memory: bool = True) -> AgentResponse:
        task = task.strip()
        if not task:
            return AgentResponse("Please ask a question.", mode="direct")
        start = time.perf_counter()
        context = self._context() if use_memory else ""

        direct = self._direct_answer(task, context)
        if direct is not None:
            if use_memory:
                self.memory.add_interaction(task, "none", "", direct)
            return AgentResponse(direct, "direct", duration_ms=self._elapsed_ms(start))

        step = self._run_step(task, task, context)
        if use_memory:
            self.memory.add_interaction(task, step.tool, step.tool_input, step.result)
        return AgentResponse(step.result, "single-tool", [step], self._elapsed_ms(start))

    def answer_multi_tool(self, task: str) -> AgentResponse:
        task = task.strip()
        start = time.perf_counter()
        sub_tasks = self.decompose_question(task)
        if len(sub_tasks) <= 1:
            return self.answer(task)

        context = self._context()
        steps = [self._run_step(sub["description"], sub["query"], context) for sub in sub_tasks]
        final_answer = self._synthesize(task, steps)
        self.memory.add_interaction(task, "multi-tool", "; ".join(s.description for s in steps), final_answer)
        return AgentResponse(final_answer, "multi-tool", steps, self._elapsed_ms(start))

    def decompose_question(self, task: str) -> list[dict[str, str]]:
        """Split ``task`` into ``{"description", "query"}`` sub-tasks (LLM first, heuristic fallback)."""
        if self.llm is not None:
            prompt = (
                "Break the user's message into independent sub-questions. Each sub-question must be "
                f"answerable by exactly one of these tools: {', '.join(self.tools)}.\n"
                "Respond ONLY with a JSON array, for example:\n"
                '[{"description": "calculate 5+3", "query": "5+3"}, '
                '{"description": "get info about Python", "query": "Python programming language"}]\n'
                "If it is a single question, return an array with one item."
            )
            try:
                text = self.llm.complete(
                    [{"role": "system", "content": prompt}, {"role": "user", "content": task}], max_tokens=300
                )
                items = safe_parse_json_list(text) or []
                sub_tasks = [
                    {"description": str(i.get("description") or i["query"]), "query": str(i["query"])}
                    for i in items
                    if isinstance(i, dict) and str(i.get("query", "")).strip()
                ]
                if sub_tasks:
                    return sub_tasks
                logger.warning("Could not parse decomposition %r, using heuristic split", text)
            except LLMError as exc:
                logger.warning("Decomposition LLM call failed, using heuristic split: %s", exc)
        return [{"description": part, "query": part} for part in split_multi_part(task)] or [
            {"description": task, "query": task}
        ]

    def add_tool(self, name: str, func: Callable[[str], str | ToolResult], description: str | None = None) -> Tool:
        """Register a new tool at runtime; it is picked up by selection and decomposition immediately."""
        tool = Tool(name=name, description=description or (func.__doc__ or name).strip(), func=func)
        self.tools[name] = tool
        logger.info("Added tool: %s", name)
        return tool

    def remove_tool(self, name: str) -> None:
        if self.tools.pop(name, None) is not None:
            logger.info("Removed tool: %s", name)

    def get_stats(self) -> dict[str, Any]:
        return {
            "available_tools": list(self.tools),
            "llm_enabled": self.llm is not None,
            "model": self.settings.openai_model if self.llm is not None else None,
            "memory": self.memory.get_summary(),
            "session": self.session.get_session_stats(),
        }

    def reset(self) -> None:
        self.memory.clear()
        logger.info("Agent memory cleared")

    # ------------------------------------------------------------------ internals

    def _context(self) -> str:
        return self.memory.get_recent_context(self.settings.context_window)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    def _direct_answer(self, task: str, context: str) -> str | None:
        """Ask the LLM whether the task can be answered from context alone (e.g. 'are you sure?')."""
        if self.llm is None or not context:
            return None
        prompt = (
            "Decide whether the user's message needs a tool or can be answered directly from the "
            f"conversation history. Tools: {', '.join(self.tools)}.\n"
            "Answer directly ONLY for confirmations, clarifications or small talk about earlier results. "
            "Anything needing calculation, facts or dates needs a tool.\n"
            'Respond ONLY with JSON: {"needs_tool": true} or {"needs_tool": false, "response": "<answer>"}'
        )
        try:
            text = self.llm.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "system", "content": f"Conversation history:\n{context}"},
                    {"role": "user", "content": task},
                ],
                max_tokens=150,
            )
        except LLMError as exc:
            logger.warning("Direct-answer check failed, defaulting to tool use: %s", exc)
            return None
        parsed = safe_parse_json(text)
        if parsed.get("needs_tool") is False and str(parsed.get("response", "")).strip():
            return str(parsed["response"]).strip()
        return None

    def _run_step(self, description: str, query: str, context: str) -> Step:
        selection = self.selector.select(query, context)
        tool_input = self._prepare_input(selection.tool, selection.input)
        result = self.execute_tool(selection.tool, tool_input)
        return Step(description, selection.tool, tool_input, result.output, result.success)

    def _prepare_input(self, tool_name: str, tool_input: str) -> str:
        if tool_name != "calculator":
            return tool_input
        # Offline follow-ups like "multiply that by 2": substitute the last numeric answer
        previous = self.memory.last_numeric_result()
        if previous is not None and _FOLLOW_UP_REFERENCE.search(tool_input):
            tool_input = _FOLLOW_UP_REFERENCE.sub(previous, tool_input, count=1)
        return preprocess_math_expression(tool_input)

    def execute_tool(self, tool_name: str, tool_input: str) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"unknown tool '{tool_name}'. Available: {', '.join(self.tools)}")

        start = time.perf_counter()
        try:
            result = tool(tool_input)
        except Exception as exc:  # a buggy tool must not take the whole agent down
            logger.exception("Tool %s raised", tool_name)
            self.session.log_error("TOOL_EXCEPTION", str(exc), {"tool": tool_name, "input": tool_input})
            result = ToolResult.error(f"tool '{tool_name}' failed unexpectedly ({type(exc).__name__})")
        self.session.log_tool_call(tool_name, tool_input, result.output, time.perf_counter() - start, result.success)
        return result

    def _synthesize(self, original_task: str, steps: list[Step]) -> str:
        successful = [s for s in steps if s.success]
        if self.llm is None or not successful:
            return self._combine_simple(steps)

        results_text = "\n".join(f"- {s.description}: {s.result}" for s in steps)
        try:
            return self.llm.complete(
                [
                    {
                        "role": "system",
                        "content": "Combine the tool results into one concise, natural answer to the user's "
                        "question. Use only the facts in the results; if a result is an error, say that "
                        "part could not be answered.",
                    },
                    {"role": "user", "content": f"Question: {original_task}\n\nResults:\n{results_text}"},
                ],
                max_tokens=400,
                temperature=0.3,
            )
        except LLMError as exc:
            logger.warning("Synthesis failed, returning combined results: %s", exc)
            return self._combine_simple(steps)

    @staticmethod
    def _combine_simple(steps: list[Step]) -> str:
        return "\n".join(f"- {s.description}: {s.result}" for s in steps)
