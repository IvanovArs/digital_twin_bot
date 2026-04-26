"""Реестр in-flight-задач для graceful shutdown.

Трекает долгие RAG/LLM-корутины — SIGTERM ждёт их завершения (или дедлайна),
а не отменяет mid-stream, оставляя half-written DB-строки.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Coroutine, Iterator
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_TASKS: set[asyncio.Task[Any]] = set()


def register_current() -> asyncio.Task[Any] | None:
    """Добавить текущую задачу в реестр (caller должен звать ``unregister``).

    Использовать в начале долгой корутины, которая должна дофлушиться на
    shutdown'е. No-op вне task-контекста (например, в unit-тесте).
    """
    task = asyncio.current_task()
    if task is not None:
        _TASKS.add(task)
    return task


def unregister(task: asyncio.Task[Any] | None) -> None:
    if task is not None:
        _TASKS.discard(task)


@contextlib.contextmanager
def tracked() -> Iterator[None]:
    """Context-manager: регистрирует текущую задачу на enter, снимает на exit
    (включая early-return — благодаря with-протоколу).
    """
    task = register_current()
    try:
        yield
    finally:
        unregister(task)


def track(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Стартует ``coro`` как трекаемую Task — сама себя удалит на completion."""
    task = asyncio.create_task(coro)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


async def run_tracked(awaitable: Awaitable[Any]) -> Any:
    """Await-нуть ``awaitable``, оставаясь видимым в реестре.

    Когда shutdown должен ждать работу, но caller'у нужен return-value
    inline (в отличие от ``track``, который fire-and-forget).
    """
    task = asyncio.ensure_future(awaitable)
    _TASKS.add(task)
    try:
        return await task
    finally:
        _TASKS.discard(task)


async def drain(timeout: float = 25.0) -> None:
    """Ждать завершения трекаемых задач, не дольше ``timeout`` секунд.

    Всё, что осталось после дедлайна, cancel'им, чтобы процесс мог
    выйти. ``stop_grace_period`` в compose должен быть на пару секунд
    длиннее этого таймаута, чтобы у нас был запас.
    """
    if not _TASKS:
        return
    pending = list(_TASKS)
    log.info("shutdown_drain_start", in_flight=len(pending), timeout=timeout)
    done, still = await asyncio.wait(pending, timeout=timeout)
    if still:
        log.warning("shutdown_drain_cancel", remaining=len(still))
        for t in still:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(*still, return_exceptions=True)
    log.info("shutdown_drain_done", finished=len(done), cancelled=len(still))
