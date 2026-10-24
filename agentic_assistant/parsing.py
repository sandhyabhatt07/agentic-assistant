"""Helpers for parsing LLM output and normalising natural-language math."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", text or "")


def safe_parse_json(text: str) -> dict[str, Any]:
    """Try to parse a JSON object from text, fallback to empty dict on error."""
    try:
        text = _strip_code_fences(text)
        start = text.index("{")
        end = text.rindex("}") + 1
        result = json.loads(text[start:end])
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def safe_parse_json_list(text: str) -> list[Any] | None:
    """Try to parse a JSON array from text, returning None on error."""
    try:
        text = _strip_code_fences(text)
        start = text.index("[")
        end = text.rindex("]") + 1
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else None
    except (ValueError, TypeError):
        return None


def preprocess_math_expression(expression: str) -> str:
    """
    Convert natural language math expressions into valid Python expressions.
    E.g., "what is 3 minus 2?" -> "3 - 2", "5 mul by 6" -> "5 * 6"
    """
    expr = expression.lower().strip()

    # Remove common question words and punctuation
    expr = re.sub(r"\b(what is|what's|calculate|compute|solve|find|tell me)\b", "", expr)
    expr = expr.replace("?", "").replace("!", "").replace("=", "").strip()

    # Remove filler words that connect operations
    expr = re.sub(r"\b(and|then|also|the|value of|result of)\b", "", expr)

    expr = re.sub(r"\bsquare\s+root\s+of\s+(\S+)", r"sqrt(\1)", expr)

    # Verb-first phrasings: "divide 100 by 4", "multiply 5 by 3", "add 2 to 3", "subtract 2 from 10"
    expr = re.sub(r"\bdivide\s+(\S+)\s+by\s+(\S+)", r"\1 / \2", expr)
    expr = re.sub(r"\bmultiply\s+(\S+)\s+by\s+(\S+)", r"\1 * \2", expr)
    expr = re.sub(r"\badd\s+(\S+)\s+to\s+(\S+)", r"\1 + \2", expr)
    expr = re.sub(r"\bsubtract\s+(\S+)\s+from\s+(\S+)", r"\2 - \1", expr)

    # Replace word operators with symbols (order matters: longer phrases first)
    replacements = [
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bmul(?:tiply)?\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bdivide\s+by\b", "/"),
        (r"\bto\s+the\s+power\s+of\b", "**"),
        (r"\bpower\s+of\b", "**"),
        (r"\bplus\b", "+"),
        (r"\badd\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bsubtract\b", "-"),
        (r"\bmultiply\b", "*"),
        (r"\btimes\b", "*"),
        (r"\bover\b", "/"),
        (r"\bmod(?:ulo)?\b", "%"),
        (r"\bsquared\b", "** 2"),
        (r"\bcubed\b", "** 3"),
        (r"÷", "/"),
        (r"\^", "**"),
    ]
    for pattern, symbol in replacements:
        expr = re.sub(pattern, symbol, expr)

    # "x" / "×" is only an operator between two numbers ("3 x 4")
    expr = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", " * ", expr)

    # Remove extra whitespace
    return " ".join(expr.split())
