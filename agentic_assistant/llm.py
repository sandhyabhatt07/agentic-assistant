"""Thin LLM abstraction so the agent can run against OpenAI, a compatible server, or a test double."""

from __future__ import annotations

import logging
from typing import Protocol

from .config import Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the LLM call fails after the client's own retries."""


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, max_tokens: int = 300, temperature: float = 0.0) -> str:
        """Return the assistant message text for a chat completion."""
        ...


class OpenAIChatClient:
    """`LLMClient` backed by the OpenAI Chat Completions API (or any compatible endpoint)."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # imported lazily so offline mode never needs credentials

        self._model = settings.openai_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    def complete(self, messages: list[Message], *, max_tokens: int = 300, temperature: float = 0.0) -> str:
        from openai import OpenAIError

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return (response.choices[0].message.content or "").strip()


def create_llm(settings: Settings) -> LLMClient | None:
    """Return an LLM client, or ``None`` when no API key is configured (offline heuristics mode)."""
    if not settings.llm_enabled:
        logger.warning("OPENAI_API_KEY is not set; running in offline mode with rule-based fallbacks")
        return None
    return OpenAIChatClient(settings)
