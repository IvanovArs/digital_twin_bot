"""Global error handler — the last line of defence."""

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
    # Always clear the client-side callback spinner first — otherwise it spins
    # for ~30 s waiting for an answerCallbackQuery that never comes.
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
