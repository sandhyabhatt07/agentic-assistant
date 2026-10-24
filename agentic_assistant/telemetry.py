"""Logging setup and per-session metrics (tool calls, errors, latency) with JSON export."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(settings: Settings) -> None:
    """Configure the package logger: full detail to the log file, only warnings to the console."""
    root = logging.getLogger("agentic_assistant")
    root.setLevel(settings.log_level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel(settings.console_log_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setLevel(settings.log_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    input: str
    output: str
    duration_ms: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass(frozen=True)
class ErrorEvent:
    type: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class SessionLog:
    """Collects metrics for one agent session."""

    def __init__(self) -> None:
        self.session_start = datetime.now()
        self._started = time.monotonic()
        self.tool_calls: list[ToolCall] = []
        self.errors: list[ErrorEvent] = []

    def log_tool_call(self, tool_name: str, input_data: str, output: str, duration: float, success: bool) -> None:
        call = ToolCall(
            tool=tool_name,
            input=input_data[:200],
            output=output[:500],
            duration_ms=round(duration * 1000, 2),
            success=success,
        )
        self.tool_calls.append(call)
        if success:
            logger.info("Tool %r succeeded in %sms", tool_name, call.duration_ms)
        else:
            logger.info("Tool %r failed: %s", tool_name, output)  # the user already sees the error

    def log_error(self, error_type: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.errors.append(ErrorEvent(error_type, message, context or {}))
        logger.error("%s: %s", error_type, message)

    def get_session_stats(self) -> dict[str, Any]:
        successful = sum(1 for call in self.tool_calls if call.success)
        durations = [call.duration_ms for call in self.tool_calls]
        return {
            "session_duration_s": round(time.monotonic() - self._started, 2),
            "total_tool_calls": len(self.tool_calls),
            "successful_calls": successful,
            "failed_calls": len(self.tool_calls) - successful,
            "avg_tool_latency_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "total_errors": len(self.errors),
        }

    def export(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        data = {
            "session_start": self.session_start.isoformat(timespec="seconds"),
            "session_end": datetime.now().isoformat(timespec="seconds"),
            "stats": self.get_session_stats(),
            "tool_calls": [asdict(c) for c in self.tool_calls],
            "errors": [asdict(e) for e in self.errors],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Session exported to %s", path)
        return path
