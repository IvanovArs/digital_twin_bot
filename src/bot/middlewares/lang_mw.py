"""Инжектит UI-язык юзера (``lang``) в kwargs хендлера.

Берёт ``language_code`` из Telegram-юзера. Поддержаны: ru, en. Остальные
коды падают в EN, кроме экс-СССР-локалей → RU (см. :func:`src.bot.texts.normalize_lang`).
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
