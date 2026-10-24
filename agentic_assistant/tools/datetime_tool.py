"""Current and relative date/time lookup (LLM-parsed when available, keyword fallback otherwise)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta

from ..llm import LLMClient, LLMError
from ..parsing import safe_parse_json
from .base import Tool

logger = logging.getLogger(__name__)

_RELATIVE = re.compile(r"(\d+)\s*(day|week)s?\s*(ago|from now|later|from today)")


def format_datetime(dt: datetime, include_time: bool = False) -> str:
    """Format a datetime like 'Saturday, October 24th, 2026' (optionally with the time)."""
    day = dt.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    date_str = f"{dt.strftime('%A')}, {dt.strftime('%B')} {day}{suffix}, {dt.year}"
    if include_time:
        return f"{date_str} at {dt.strftime('%I:%M %p')}"
    return date_str


def parse_datetime_query(query: str, now: datetime) -> tuple[int, bool]:
    """Keyword parser: return ``(days_offset, include_time)`` for a natural-language query."""
    q = query.lower()
    match = _RELATIVE.search(q)
    if match:
        amount = int(match.group(1)) * (7 if match.group(2) == "week" else 1)
        return (-amount if match.group(3) == "ago" else amount), False
    # Longer phrases first: "day after tomorrow" also contains "tomorrow"
    for phrase, offset in (
        ("day after tomorrow", 2),
        ("overmorrow", 2),
        ("day before yesterday", -2),
        ("tomorrow", 1),
        ("yesterday", -1),
        ("next week", 7),
        ("last week", -7),
    ):
        if phrase in q:
            return offset, False
    if re.search(r"\btime\b", q):
        return 0, True
    if re.search(r"\b(date|day|today)\b", q):
        return 0, False
    return 0, True


_SYSTEM_PROMPT = """You are a date/time interpreter. Current date and time: {now}

Parse the user's query and determine:
1. How many days to add/subtract from today (0 for today, negative for the past)
2. Whether they want just the date or date + time

Respond with JSON only:
{{"days_offset": <integer>, "include_time": true/false}}

Examples:
- "tomorrow" -> {{"days_offset": 1, "include_time": false}}
- "day after tomorrow" -> {{"days_offset": 2, "include_time": false}}
- "3 days ago" -> {{"days_offset": -3, "include_time": false}}
- "next week" -> {{"days_offset": 7, "include_time": false}}
- "what time is it" -> {{"days_offset": 0, "include_time": true}}
"""


def make_datetime_tool(llm: LLMClient | None = None, clock: Callable[[], datetime] = datetime.now) -> Tool:
    def lookup(query: str) -> str:
        now = clock()
        offset, include_time = parse_datetime_query(query, now)
        if llm is not None:
            try:
                text = llm.complete(
                    [
                        {
                            "role": "system",
                            "content": _SYSTEM_PROMPT.format(now=now.strftime("%A, %B %d, %Y %I:%M %p")),
                        },
                        {"role": "user", "content": query},
                    ],
                    max_tokens=60,
                )
                parsed = safe_parse_json(text)
                if "days_offset" in parsed:
                    offset = int(parsed["days_offset"])
                    include_time = bool(parsed.get("include_time", False))
            except (LLMError, ValueError, TypeError) as exc:
                logger.warning("LLM date parsing failed, using keyword fallback: %s", exc)
        return format_datetime(now + timedelta(days=offset), include_time)

    return Tool(
        name="datetime",
        description=(
            "Returns the current date/time or a relative date from natural language "
            "(tomorrow, yesterday, next week, day after tomorrow, 3 days from now, ...)."
        ),
        func=lookup,
    )
