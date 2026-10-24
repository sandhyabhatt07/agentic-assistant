from __future__ import annotations

import pytest

from agentic_assistant import cli


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOG_FILE", "")
    monkeypatch.setenv("SESSION_EXPORT_FILE", str(tmp_path / "session.json"))
    monkeypatch.chdir(tmp_path)  # keep any .env in the repo from being loaded


def test_one_shot_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--offline", "what", "is", "6", "*", "7"]) == 0
    assert capsys.readouterr().out.strip() == "42"


def test_one_shot_failure_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--offline", "1 / 0"]) == 1
    assert "division by zero" in capsys.readouterr().out


def test_repl_session(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path) -> None:  # type: ignore[no-untyped-def]
    inputs = iter(["what is 10 * 5", "multiply that by 2", "stats", "memory", "clear", "help", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "Agent: 50" in out and "Agent: 100" in out
    assert "[STATS]" in out and "[MEMORY] Memory cleared." in out
    assert (tmp_path / "session.json").exists()


def test_repl_handles_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli.main(["--offline"]) == 0
