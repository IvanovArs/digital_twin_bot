"""Per-user throttling — keep one user's burst from DOSing the LLM.

The bot serves answers from a single CPU/GPU pinned llama-server
(``--parallel 1``). One allowed user firing 10 questions back-to-back
serialises into ~minutes of wall-clock and starves every other student.

Two layers of defence:

1. **Min-interval gate** — silently drop a fresh heavy update if the same
   user sent another less than ``_MIN_INTERVAL_S`` ago.
2. **Per-user lock** — if a previous heavy update from the same user is
   still running through ``run_qa_pipeline``, drop the new one.

Both layers are in-process (single-instance bot). For multi-instance
deploys move to Redis-backed throttling.

Heavy = anything that triggers ``run_qa_pipeline``:
  * free-text PM messages
  * ``chosen_inline_result`` (auto-fires the pipeline when /setinlinefeedback
    is enabled in BotFather)

NOT heavy (always pass through):
  * ``inline_query`` — only returns the result list to Telegram, no LLM. Used
    to be throttled too, but that ate the user's own ``chosen_inline_result``
    that arrives within 1.5 s of the last keystroke, breaking the auto-answer
    flow and leaving the «🔍 Получить ответ» button unswapped.
  * ``callback_query`` — feedback 👍/👎, «что сейчас делается?», menu taps.
  * Slash-command messages — /ask, /help, /admin_*.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, Update

log = structlog.get_logger(__name__)

# Two heavy updates from the same user within this window → drop the second.
# 1.5 s catches the most common "double-tap send" without feeling laggy.
_MIN_INTERVAL_S = 1.5

# Entries in ``_last_ts`` older than this are evicted on write. Without
# GC the dict grows unboundedly with every user who ever hit the bot
# (problematic on a long-running instance: 100k users × 72 bytes = 7 MB
# per dict, two dicts, no reclamation). A 24 h idle threshold is
# generous for real human spacing and doesn't cause false positives.
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
    """A heavy update kicks off retrieval + LLM. Worth throttling.

    ``inline_query`` is intentionally NOT heavy — it just returns the result
    list and would otherwise eat the very next ``chosen_inline_result`` from
    the same user via the 1.5 s min-interval gate.
    """
    if update.chosen_inline_result is not None:
        return True
    msg = update.message
    if msg is None or not isinstance(msg, Message):
        return False
    text = (msg.text or msg.caption or "").lstrip()
    if text.startswith("/"):
        return False  # /ask, /help, /admin_* — not heavy
    return bool(text)


class ThrottleMiddleware(BaseMiddleware):
    """Outer-middleware on ``dp.update``. Drops bursts and concurrent runs."""

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
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
        # Amortise GC over normal traffic (≈1 in 200 passes) so we don't
        # need a background timer and don't walk the whole dict on every
        # message.
        if len(_last_ts) % 200 == 0:
            _gc_stale(now)

        lock = _locks.setdefault(uid, asyncio.Lock())
        if lock.locked():
            log.info("throttle_drop_in_flight", telegram_id=uid)
            return None

        async with lock:
            return await handler(event, data)
