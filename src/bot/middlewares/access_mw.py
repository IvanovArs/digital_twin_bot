"""Allowlist middleware: if ALLOWED_TELEGRAM_IDS is set, drop everyone else.

Used for closed testing. Empty allowlist disables the check.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, InlineQuery, Message, TelegramObject
from aiogram.types import User as TgUser

from src.bot import texts

log = structlog.get_logger(__name__)


class AccessMiddleware(BaseMiddleware):
    def __init__(self, allowed_ids: set[int]):
        self._allowed = allowed_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not self._allowed:
            return await handler(event, data)

        tg_user: TgUser | None = data.get("event_from_user")
        if tg_user is not None and tg_user.id in self._allowed:
            return await handler(event, data)

        lang = data.get("lang") or texts.normalize_lang(tg_user.language_code if tg_user else None)
        msg = texts.tr(lang, texts.ACCESS_DENIED)

        log.info("access_denied", telegram_id=getattr(tg_user, "id", None))
        if isinstance(event, Message):
            await event.answer(msg)
        elif isinstance(event, CallbackQuery):
            await event.answer(msg, show_alert=False)
        elif isinstance(event, InlineQuery):
            await event.answer(results=[], cache_time=1, is_personal=True)
        return None
