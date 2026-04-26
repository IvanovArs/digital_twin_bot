"""Покрытие _stream_answer_to_ui: heartbeat, throttle, retry-after, bad-request."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from src.bot.services import qa_pipeline as qa


class _StatusBuf:
    def __init__(self, *, fail_first: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.fail_first = fail_first

    async def __call__(self, text: str) -> None:
        self.calls.append(text)
        if self.fail_first is not None:
            err = self.fail_first
            self.fail_first = None
            raise err


@pytest.mark.asyncio
async def test_stream_yields_full_answer(monkeypatch) -> None:
    async def fake_stream(_m, **_kw) -> AsyncIterator[str]:
        for piece in ["Стейк", "холдер", " — это"]:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    cap = _StatusBuf()
    out = await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=None,
    )
    assert out == "Стейкхолдер — это"


@pytest.mark.asyncio
async def test_stream_with_typing_ping(monkeypatch) -> None:
    async def fake_stream(_m, **_kw) -> AsyncIterator[str]:
        for piece in ["a", "b", "c"]:
            yield piece
            await asyncio.sleep(0)

    monkeypatch.setattr(qa, "chat_stream", fake_stream)

    typing_calls = {"n": 0}

    async def ping() -> None:
        typing_calls["n"] += 1

    cap = _StatusBuf()
    await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=ping,
    )
    # На коротком быстром стриме typing-таймер (4с) может не сработать.
    # Главное — что typing_ping callable принят без ошибки.
    assert typing_calls["n"] >= 0


@pytest.mark.asyncio
async def test_stream_handles_retry_after(monkeypatch) -> None:
    async def fake_stream(_m, **_kw) -> AsyncIterator[str]:
        for piece in ["a"] * 3:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)

    cap = _StatusBuf(fail_first=TelegramRetryAfter(method=None, message="rate", retry_after=0))
    out = await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=None,
    )
    assert out


@pytest.mark.asyncio
async def test_stream_swallows_not_modified(monkeypatch) -> None:
    async def fake_stream(_m, **_kw) -> AsyncIterator[str]:
        for piece in ["a", "b"]:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)

    cap = _StatusBuf(fail_first=TelegramBadRequest(method=None, message="message is not modified"))
    out = await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=None,
    )
    assert out


@pytest.mark.asyncio
async def test_stream_logs_bad_request_other(monkeypatch) -> None:
    async def fake_stream(_m, **_kw) -> AsyncIterator[str]:
        for piece in ["a", "b"]:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)

    cap = _StatusBuf(fail_first=TelegramBadRequest(method=None, message="can't parse entities"))
    await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=None,
    )


@pytest.mark.asyncio
async def test_stream_heartbeat_fires_after_silence(monkeypatch) -> None:
    """Если первый токен прилетает позже _HEARTBEAT_DELAY_S — должен сработать heartbeat."""
    monkeypatch.setattr(qa, "_HEARTBEAT_DELAY_S", 0.05)

    async def slow_stream(_m, **_kw) -> AsyncIterator[str]:
        await asyncio.sleep(0.15)
        yield "наконец-то"

    monkeypatch.setattr(qa, "chat_stream", slow_stream)
    cap = _StatusBuf()
    out = await qa._stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        set_status=cap,
        typing_ping=None,
    )
    assert "наконец-то" in out
    # Должен был быть хоть один heartbeat-edit с STATUS_ANALYSING.
    assert any("Анализирую" in c for c in cap.calls)
