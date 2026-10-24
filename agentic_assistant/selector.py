"""Chooses which tool should answer a task: LLM with few-shot prompting, rule-based fallback."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .llm import LLMClient, LLMError
from .parsing import safe_parse_json
from .tools import Tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    tool: str
    input: str
    reasoning: str = ""
    source: str = "llm"  # "llm" or "heuristic"


@dataclass(frozen=True)
class FewShotExample:
    task: str
    tool: str
    input: str
    reasoning: str


DEFAULT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample("what is 12 times 7?", "calculator", "12 * 7", "Arithmetic question"),
    FewShotExample("who was Marie Curie?", "wikipedia", "Marie Curie", "Factual question about a person"),
    FewShotExample("what's the date tomorrow?", "datetime", "tomorrow", "Relative date question"),
    FewShotExample(
        "multiply that by 2",
        "calculator",
        "50 * 2",
        "Follow-up arithmetic; 'that' is the previous result (50) from the context",
    ),
)

# An arithmetic expression such as "5 + 3", "2**10" or "(4 - 1) / 2"
_MATH_EXPRESSION = re.compile(r"\d\s*(\*\*|[-+*/×÷^%x])\s*[\d(]")
_MATH_WORDS = re.compile(
    r"\b(plus|minus|multiply|multiplied|divide|divided|add|subtract|times|calculate|compute|"
    r"sum|difference|product|power|squared|cubed|sqrt|square root|modulo)\b"
)
_DATETIME_WORDS = re.compile(r"\b(time|date|day|today|now|tomorrow|yesterday|datetime|clock|week|weekday|when is)\b")
_WIKI_PREFIXES = re.compile(
    r"^\s*(please\s+)?(search\s+(wikipedia\s+)?for|search|look up|tell me about|who is|who was|what is|"
    r"what are|what was|explain|define|describe|information about|info on|wiki)\s+",
    re.IGNORECASE,
)


class ToolSelector:
    """Select a tool for a task. Uses the LLM when one is configured, otherwise keyword heuristics."""

    def __init__(
        self,
        tools: Mapping[str, Tool],
        llm: LLMClient | None = None,
        examples: Sequence[FewShotExample] = DEFAULT_EXAMPLES,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.examples = examples

    def build_prompt(self) -> str:
        """System prompt listing the registered tools, the response format and few-shot examples."""
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        prompt = (
            "You are a tool selector. Choose the best tool for the user's task.\n\n"
            f"Available tools:\n{tool_list}\n\n"
            "Respond ONLY with JSON in this exact format:\n"
            '{"tool": "<tool_name>", "input": "<input_text>", "reasoning": "<short reason>"}\n\n'
            "Rules:\n"
            "1. Choose exactly ONE tool from the list above.\n"
            "2. 'input' is what the tool needs: a plain math expression for calculator, "
            "the topic only for wikipedia, the date phrase for datetime.\n"
            "3. Use the conversation context to resolve references like 'that' or 'it'.\n"
        )
        examples = [ex for ex in self.examples if ex.tool in self.tools]
        if examples:
            prompt += "\nExamples:\n"
            for ex in examples:
                answer = json.dumps({"tool": ex.tool, "input": ex.input, "reasoning": ex.reasoning})
                prompt += f"User: {ex.task}\nResponse: {answer}\n\n"
        return prompt

    def select(self, task: str, context: str = "") -> Selection:
        if self.llm is None:
            return self.fallback_heuristics(task)

        messages = [{"role": "system", "content": self.build_prompt()}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": task})

        try:
            text = self.llm.complete(messages, max_tokens=200)
        except LLMError as exc:
            logger.warning("Tool selection LLM call failed, using heuristics: %s", exc)
            return self.fallback_heuristics(task)

        parsed = safe_parse_json(text)
        tool = parsed.get("tool")
        if not isinstance(tool, str) or tool not in self.tools:
            logger.warning("LLM returned an invalid tool selection %r, using heuristics", text)
            return self.fallback_heuristics(task)

        tool_input = str(parsed.get("input") or "").strip() or task
        selection = Selection(tool, tool_input, str(parsed.get("reasoning", "")), source="llm")
        logger.info("Selected %s via LLM (%s)", selection.tool, selection.reasoning or "no reasoning")
        return selection

    def fallback_heuristics(self, task: str) -> Selection:
        """Rule-based selection used offline or when the LLM response is unusable."""
        t = task.lower()

        if "calculator" in self.tools and (_MATH_EXPRESSION.search(t) or _MATH_WORDS.search(t)):
            return Selection("calculator", task, "Detected a math expression or keyword", "heuristic")

        if "datetime" in self.tools and _DATETIME_WORDS.search(t):
            return Selection("datetime", task, "Detected date/time keywords", "heuristic")

        if "wikipedia" in self.tools:
            topic = _WIKI_PREFIXES.sub("", task).strip(" ?.!") or task
            return Selection("wikipedia", topic, "General knowledge question", "heuristic")

        if not self.tools:
            raise ValueError("No tools are registered")
        first = next(iter(self.tools))
        return Selection(first, task, "Default fallback", "heuristic")
