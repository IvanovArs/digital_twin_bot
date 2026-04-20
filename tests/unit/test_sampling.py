"""Tests that the factual-answer stream uses tight sampling params."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from src.bot.services.qa_pipeline import _stream_answer_to_ui


@pytest.mark.asyncio
async def test_stream_overrides_temperature_and_top_p(monkeypatch) -> None:
    """The final-answer stream must request deterministic sampling — the
    default 0.7/0.8 caused "same question, different fabricated author"
    drift that the pre-P11 runs produced."""
    captured: dict[str, object] = {}

    async def fake_chat_stream(_messages, **kwargs) -> AsyncIterator[str]:
        captured.update(kwargs)
        yield "ok"

    from src.bot.services import qa_pipeline as qa

    monkeypatch.setattr(qa, "chat_stream", fake_chat_stream)

    async def _noop_status(_text: str) -> None: ...

    await _stream_answer_to_ui(
        messages=[{"role": "user", "content": "q"}],
        lang="ru",
        subject_title=None,
        set_status=_noop_status,
        typing_ping=None,
    )
    assert captured.get("temperature") == 0.1
    assert captured.get("top_p") == 0.9
    assert captured.get("top_k") == 40
    assert captured.get("repeat_penalty") == 1.1
