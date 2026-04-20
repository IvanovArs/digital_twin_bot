"""Inject the user's UI language (``lang``) into handler kwargs.

Uses Telegram's ``language_code`` on the incoming user. Supported: ru, en.
All other codes fall back to English except ex-USSR locales → Russian
(see :func:`src.bot.texts.normalize_lang`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from aiogram.types import User as TgUser

from src.bot.texts import normalize_lang


class LangMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user: TgUser | None = data.get("event_from_user")
        data["lang"] = normalize_lang(tg_user.language_code if tg_user else None)
        return await handler(event, data)
