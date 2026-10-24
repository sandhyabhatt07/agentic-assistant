from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_assistant.config import Settings
from agentic_assistant.memory import ConversationMemory
from agentic_assistant.telemetry import SessionLog


def test_memory_is_bounded() -> None:
    memory = ConversationMemory(max_history=2)
    for i in range(3):
        memory.add_interaction(f"q{i}", "calculator", "", str(i))
    assert [h.user_input for h in memory.history] == ["q1", "q2"]
    assert memory.last_numeric_result() == "2"
    assert "q2" in memory.get_recent_context(1) and "q1" not in memory.get_recent_context(1)


def test_memory_clear() -> None:
    memory = ConversationMemory()
    memory.add_interaction("q", "none", "", "a")
    memory.clear()
    assert len(memory) == 0 and memory.get_recent_context() == ""


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    monkeypatch.setenv("MAX_MEMORY_SIZE", "7")
    monkeypatch.setenv("LOG_FILE", "")
    settings = Settings.from_env(dotenv=False)
    assert settings.llm_enabled and settings.openai_model == "my-model"
    assert settings.max_memory_size == 7 and settings.log_file is None
    assert "sk-test" not in repr(settings)  # secrets never end up in logs


def test_settings_rejects_bad_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MEMORY_SIZE", "lots")
    with pytest.raises(ValueError, match="MAX_MEMORY_SIZE"):
        Settings.from_env(dotenv=False)


def test_session_export(tmp_path: Path) -> None:
    session = SessionLog()
    session.log_tool_call("calculator", "1+1", "2", 0.01, True)
    session.log_error("TEST", "oops")
    data = json.loads(session.export(tmp_path / "s.json").read_text())
    assert data["stats"]["total_tool_calls"] == 1 and data["stats"]["total_errors"] == 1
