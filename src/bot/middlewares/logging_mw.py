"""Структурный logging-context для каждого update'а."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject, Update
from aiogram.types import User as TgUser

log = structlog.get_logger(__name__)


def _describe(event: TelegramObject) -> dict[str, Any]:
    """Компактный dict с описанием прилетевшего update'а."""
    info: dict[str, Any] = {"event_type": type(event).__name__}
    if isinstance(event, Update):
        inner = (
            event.message
            or event.edited_message
            or event.channel_post
            or event.callback_query
            or event.inline_query
            or event.chosen_inline_result
        )
        info["inner_type"] = type(inner).__name__ if inner is not None else "None"
        if isinstance(inner, Message):
            info["text_prefix"] = (inner.text or "")[:40]
            info["via_bot_id"] = inner.via_bot.id if inner.via_bot else None
            info["chat_type"] = inner.chat.type if inner.chat else None
    return info


class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        request_id = uuid.uuid4().hex[:12]
        tg_user: TgUser | None = data.get("event_from_user")
        user_id = tg_user.id if tg_user else None

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, telegram_id=user_id)

        log.info("update_received", **_describe(event))
        try:
            return await handler(event, data)
        except Exception:
            log.exception("update_failed")
            raise
