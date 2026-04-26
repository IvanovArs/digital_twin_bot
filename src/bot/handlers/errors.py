"""Глобальный обработчик ошибок — последняя линия обороны."""

from __future__ import annotations

import contextlib

import structlog
from aiogram import Router
from aiogram.types import ErrorEvent, Message

from src.bot import texts

log = structlog.get_logger(__name__)
router = Router(name="errors")


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    log.exception(
        "handler_exception",
        exc_type=type(event.exception).__name__,
        update_id=getattr(event.update, "update_id", None),
    )
    upd = event.update
    # Сначала всегда снимаем callback-spinner у клиента — иначе он крутится
    # ~30 с в ожидании answerCallbackQuery, который не придёт.
    if upd.callback_query is not None:
        with contextlib.suppress(Exception):
            await upd.callback_query.answer()

    message: Message | None = None
    if upd.message is not None:
        message = upd.message
    elif upd.callback_query is not None and upd.callback_query.message is not None:
        message = upd.callback_query.message  # type: ignore[assignment]

    if message is not None:
        lang = texts.normalize_lang(message.from_user.language_code if message.from_user else None)
        with contextlib.suppress(Exception):
            await message.answer(texts.tr(lang, texts.INTERNAL_ERROR))
    return True  # mark as handled
