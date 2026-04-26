"""Абстракция LLM-бэкенда: позволяет fork'у проекта подключить OpenAI /
Anthropic / Mistral вместо локального llama.cpp без правки пайплайна.

Сейчас единственная реализация — ``LlamaCppProvider`` (тонкая обёртка над
существующими ``llm.chat`` / ``llm.chat_stream``). Когда понадобится
cloud-fallback, добавьте ``OpenAIProvider`` / ``AnthropicProvider``,
зарегистрируйте через ``get_llm_provider()`` и переключайте через
``settings.LLM_PROVIDER``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Минимальный интерфейс LLM-бэкенда, который ждёт qa_pipeline."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Однократный chat-completion. Должен возвращать чистый ответ
        (без think-блоков)."""
        ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """Стриминг по delta.content; yields только непустые куски."""
        ...

    def ping(self) -> bool:
        """Health-check: бэкенд жив и принимает запросы."""
        ...


class LlamaCppProvider:
    """Локальный llama.cpp через OpenAI-совместимый HTTP. Default-провайдер."""

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        from src.rag.llm import chat as _chat

        return _chat(messages, **kwargs)

    def chat_stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        from src.rag.llm import chat_stream as _stream

        return _stream(messages, **kwargs)

    def ping(self) -> bool:
        from src.rag.llm import ping as _ping

        return _ping()


_DEFAULT_PROVIDER: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Возвращает singleton-провайдер согласно ``settings.LLM_PROVIDER``.

    Сейчас единственный валидный — ``openai_compat`` (=llama.cpp). Расширения:
    добавить ветку, импортировать класс лениво (чтобы SDK не тянулся при
    импорте), вернуть инстанс.
    """
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = LlamaCppProvider()
    return _DEFAULT_PROVIDER


def reset_provider() -> None:
    """Только для тестов."""
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = None
