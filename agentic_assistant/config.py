"""Runtime settings, loaded from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """All tunable settings. Every field can be overridden with the matching upper-case env var."""

    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    request_timeout: int = 30
    max_retries: int = 3

    max_memory_size: int = 50
    context_window: int = 3

    log_level: str = "INFO"
    log_file: str | None = "agent.log"
    console_log_level: str = "WARNING"
    session_export_file: str = "agent_session.json"

    wikipedia_sentences: int = 2
    wikipedia_language: str = "en"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @classmethod
    def from_env(cls, dotenv: bool = True) -> Settings:
        if dotenv:
            load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=_env_str("OPENAI_MODEL", cls.openai_model),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            request_timeout=_env_int("REQUEST_TIMEOUT", cls.request_timeout, minimum=1),
            max_retries=_env_int("MAX_RETRIES", cls.max_retries),
            max_memory_size=_env_int("MAX_MEMORY_SIZE", cls.max_memory_size, minimum=1),
            context_window=_env_int("CONTEXT_WINDOW", cls.context_window),
            log_level=_env_str("LOG_LEVEL", cls.log_level).upper(),
            # LOG_FILE="" disables file logging
            log_file=os.getenv("LOG_FILE", "agent.log").strip() or None,
            console_log_level=_env_str("CONSOLE_LOG_LEVEL", cls.console_log_level).upper(),
            session_export_file=_env_str("SESSION_EXPORT_FILE", cls.session_export_file),
            wikipedia_sentences=_env_int("WIKIPEDIA_SENTENCES", cls.wikipedia_sentences, minimum=1),
            wikipedia_language=_env_str("WIKIPEDIA_LANGUAGE", cls.wikipedia_language),
        )
