"""Bounded conversation history used to give the LLM (and offline fallbacks) context."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Interaction:
    user_input: str
    tool: str
    tool_input: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ConversationMemory:
    """Stores the most recent interactions (oldest are dropped first)."""

    def __init__(self, max_history: int = 50) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_history = max_history
        self._history: deque[Interaction] = deque(maxlen=max_history)

    @property
    def history(self) -> list[Interaction]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def add_interaction(self, user_input: str, tool_used: str, tool_input: str, result: str) -> None:
        self._history.append(Interaction(user_input, tool_used, tool_input, result))

    def get_recent_context(self, n: int = 3) -> str:
        """Formatted recent context for an LLM prompt (empty string when there is none)."""
        if not self._history or n <= 0:
            return ""
        lines = ["Previous interactions:"]
        for i, item in enumerate(self.history[-n:], 1):
            lines.append(f"{i}. User asked: '{item.user_input}' -> Used {item.tool} -> Result: {item.result[:200]}")
        return "\n".join(lines)

    def last_numeric_result(self) -> str | None:
        """Most recent result that is a plain number, used to resolve 'that' / 'it' in follow-ups."""
        for item in reversed(self._history):
            if _NUMBER.match(item.result.strip()):
                return item.result.strip()
        return None

    def clear(self) -> None:
        self._history.clear()

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_interactions": len(self._history),
            "max_history": self.max_history,
            "recent_tools_used": [h.tool for h in self.history[-5:]],
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(h) for h in self._history]
