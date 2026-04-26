"""Per-user throttling — burst одного пользователя не должен DOS-ить LLM.

Бот отвечает с одного pinned-CPU/GPU llama-server'а (``--parallel 1``).
Один разрешённый юзер с 10 вопросами подряд сериализуется в минуты
wall-clock и блокирует всех остальных студентов.

Два слоя защиты:

1. **Min-interval gate** — тихо дропаем свежий heavy-update, если тот же
   юзер отправил предыдущий менее чем ``_MIN_INTERVAL_S`` назад.
2. **Per-user lock** — если предыдущий heavy-update этого юзера ещё
   крутится в ``run_qa_pipeline``, дропаем новый.

Оба слоя in-process (single-instance bot). Для multi-instance — переезд
на Redis-backed throttling.

Heavy = всё, что запускает ``run_qa_pipeline``:
  * свободный текст в PM
  * ``chosen_inline_result`` (auto-fire пайплайна, когда /setinlinefeedback
    включён в BotFather)

НЕ heavy (всегда пропускаем):
  * ``inline_query`` — только возвращает список результатов в Telegram, без
    LLM. Раньше тоже throttle'ился, но это съедало собственный
    ``chosen_inline_result`` юзера, прилетающий в 1.5 с от последнего
    keystroke, и ломало auto-answer-flow.
  * ``callback_query`` — feedback 👍/👎, «что сейчас делается?», меню.
  * Slash-команды — /ask, /help, /admin_*.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject, Update

log = structlog.get_logger(__name__)

# Два heavy-update'а одного юзера в этом окне → второй дропаем.
# 1.5 с ловит самый частый «double-tap send» и не ощущается лагом.
_MIN_INTERVAL_S = 1.5

# Записи в ``_last_ts`` старше этого вытесняются на write. Без GC dict
# растёт безгранично с каждым новым юзером (на долго-живущем инстансе
# 100k юзеров × 72 байта = 7 МБ на dict, два dict'а, без освобождения).
# 24 ч idle — щедрый порог для реального human-spacing'а, без false-positive'ов.
_GC_IDLE_SECONDS = 24 * 3600


def _gc_stale(now: float) -> None:
    stale = [uid for uid, ts in _last_ts.items() if now - ts > _GC_IDLE_SECONDS]
    for uid in stale:
        _last_ts.pop(uid, None)
        lock = _locks.get(uid)
        if lock is not None and not lock.locked():
            _locks.pop(uid, None)


_last_ts: dict[int, float] = {}
_locks: dict[int, asyncio.Lock] = {}


def _is_heavy_update(update: Update) -> bool:
    """Heavy-update запускает retrieval + LLM. Стоит троттлить.

    ``inline_query`` намеренно НЕ heavy — он только возвращает список
    результатов, иначе бы съел следующий ``chosen_inline_result`` того же
    юзера через 1.5-секундный min-interval.
    """
    if update.chosen_inline_result is not None:
        return True
    msg = update.message
    if msg is None or not isinstance(msg, Message):
        return False
    # Сообщение, отправленное через inline-режим самого бота (`via_bot`),
    # Telegram доставляет обратно в чат как Message — это НЕ свежий запрос
    # юзера, а наш собственный inline-плейсхолдер. Если считать heavy,
    # оно съедает throttle-слот, и следующий ``chosen_inline_result``
    # того же юзера дропается как burst (gap_ms=0). Auto-fire ломается.
    if getattr(msg, "via_bot", None) is not None:
        return False
    # В группах/супергруппах не считаем heavy любой текст без `/` —
    # хендлер `on_question` всё равно отбросит сообщения, не адресованные
    # боту (без reply / @mention). Если оставить heavy, мы зря заберём
    # throttle-слот у того же юзера и дропнем его следующий настоящий
    # запрос как burst.
    chat = getattr(msg, "chat", None)
    if (
        chat is not None
        and getattr(chat, "type", None) in ("group", "supergroup")
        and not _is_message_addressed_to_bot(msg)
    ):
        return False
    text = (msg.text or msg.caption or "").lstrip()
    if text.startswith("/"):
        return False  # /ask, /help, /admin_* — не heavy
    return bool(text)


def _is_message_addressed_to_bot(msg: Message) -> bool:
    """Лёгкая sync-проверка @mention / reply на бота. Username/ID берём из
    кеша aiogram (`bot._me`); если ещё не прогрет — возвращаем True, чтобы
    не молча дропать настоящие вопросы при первом hit'е после рестарта."""
    bot = msg.bot
    if bot is None:
        return True
    me = getattr(bot, "_me", None)
    if me is None:
        return True
    bot_id = me.id
    bot_username = (me.username or "").lower() or None
    if (
        msg.reply_to_message
        and msg.reply_to_message.from_user
        and msg.reply_to_message.from_user.id == bot_id
    ):
        return True
    text = msg.text or msg.caption or ""
    for ent in msg.entities or msg.caption_entities or []:
        et = getattr(ent, "type", "")
        if et == "mention" and bot_username:
            mention = text[ent.offset : ent.offset + ent.length].lstrip("@").lower()
            if mention == bot_username:
                return True
        elif et == "text_mention" and ent.user and ent.user.id == bot_id:
            return True
    return False


class ThrottleMiddleware(BaseMiddleware):
    """Outer-middleware на ``dp.update``. Дропает burst'ы и concurrent-run'ы."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Update) or not _is_heavy_update(event):
            return await handler(event, data)

        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)
        uid = int(user.id)

        now = time.monotonic()
        last = _last_ts.get(uid, 0.0)
        if now - last < _MIN_INTERVAL_S:
            log.info("throttle_drop_burst", telegram_id=uid, gap_ms=int((now - last) * 1000))
            return None
        _last_ts[uid] = now
        # Амортизируем GC по обычному трафику (≈1 раз на 200 проходов) —
        # без фонового таймера и без обхода всего dict'а на каждое сообщение.
        if len(_last_ts) % 200 == 0:
            _gc_stale(now)

        lock = _locks.setdefault(uid, asyncio.Lock())
        if lock.locked():
            log.info("throttle_drop_in_flight", telegram_id=uid)
            return None

        async with lock:
            return await handler(event, data)
