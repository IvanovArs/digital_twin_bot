"""Тесты LLMProvider Protocol + default LlamaCpp реализации."""

from __future__ import annotations

import pytest

from src.rag.llm_provider import (
    LlamaCppProvider,
    LLMProvider,
    get_llm_provider,
    reset_provider,
)


def test_llamacpp_provider_satisfies_protocol() -> None:
    provider = LlamaCppProvider()
    assert isinstance(provider, LLMProvider)


def test_get_llm_provider_returns_singleton() -> None:
    reset_provider()
    a = get_llm_provider()
    b = get_llm_provider()
    assert a is b
    reset_provider()


def test_provider_chat_delegates_to_llm_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(messages, **kwargs):  # type: ignore[no-untyped-def]
        captured["msgs"] = messages
        captured["kwargs"] = kwargs
        return "OK"

    import src.rag.llm as llm_mod

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    provider = LlamaCppProvider()
    out = provider.chat([{"role": "user", "content": "hi"}], temperature=0.2)
    assert out == "OK"
    assert captured["kwargs"] == {"temperature": 0.2}


def test_provider_ping_delegates(monkeypatch) -> None:
    import src.rag.llm as llm_mod

    monkeypatch.setattr(llm_mod, "ping", lambda: True)
    assert LlamaCppProvider().ping() is True
    monkeypatch.setattr(llm_mod, "ping", lambda: False)
    assert LlamaCppProvider().ping() is False


@pytest.mark.asyncio
async def test_provider_chat_stream_delegates(monkeypatch) -> None:
    async def fake_stream(messages, **kwargs):  # type: ignore[no-untyped-def]
        for piece in ["a", "b", "c"]:
            yield piece

    import src.rag.llm as llm_mod

    monkeypatch.setattr(llm_mod, "chat_stream", fake_stream)
    provider = LlamaCppProvider()
    chunks: list[str] = []
    async for piece in provider.chat_stream([{"role": "user", "content": "q"}]):
        chunks.append(piece)
    assert chunks == ["a", "b", "c"]
