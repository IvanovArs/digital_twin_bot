"""Тесты глобального error-handler'а."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.handlers.errors import on_error


def _err_event(*, exc: Exception, update) -> SimpleNamespace:
    return SimpleNamespace(exception=exc, update=update)


@pytest.mark.asyncio
async def test_on_error_handles_message_update() -> None:
    msg = MagicMock()
    msg.answer = AsyncMock()
    msg.from_user = SimpleNamespace(language_code="ru")
    upd = SimpleNamespace(update_id=42, message=msg, callback_query=None)
    handled = await on_error(_err_event(exc=RuntimeError("boom"), update=upd))
    assert handled is True
    msg.answer.assert_awaited()
    body = msg.answer.call_args.args[0]
    assert "пошло не так" in body or "wrong" in body


@pytest.mark.asyncio
async def test_on_error_acks_callback_then_messages() -> None:
    cbm = MagicMock()
    cbm.answer = AsyncMock()
    cbm.from_user = SimpleNamespace(language_code="en")
    cbq = MagicMock()
    cbq.answer = AsyncMock()
    cbq.message = cbm
    upd = SimpleNamespace(update_id=7, message=None, callback_query=cbq)
    await on_error(_err_event(exc=ValueError("x"), update=upd))
    # Spinner снят.
    cbq.answer.assert_awaited()
    # Сообщение отправлено.
    cbm.answer.assert_awaited()


@pytest.mark.asyncio
async def test_on_error_swallows_send_failures() -> None:
    msg = MagicMock()
    msg.answer = AsyncMock(side_effect=Exception("network down"))
    msg.from_user = None
    upd = SimpleNamespace(update_id=1, message=msg, callback_query=None)
    # Не должен пробрасывать — это last-line-of-defence.
    handled = await on_error(_err_event(exc=KeyError("k"), update=upd))
    assert handled is True
