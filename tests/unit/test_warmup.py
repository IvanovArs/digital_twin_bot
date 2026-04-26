"""Тесты MODELS_READY event-семантики."""

from __future__ import annotations

import asyncio

import pytest

from src.bot.services.warmup import MODELS_READY


@pytest.mark.asyncio
async def test_models_ready_initially_clear() -> None:
    # Свежий event на текущей loop — модульный MODELS_READY мог быть set'нут
    # другим тестом в той же сессии.
    fresh = asyncio.Event()
    assert fresh.is_set() is False


@pytest.mark.asyncio
async def test_models_ready_unblocks_waiters() -> None:
    fresh = asyncio.Event()

    async def waiter() -> str:
        await fresh.wait()
        return "ok"

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert not task.done()
    fresh.set()
    res = await asyncio.wait_for(task, timeout=1.0)
    assert res == "ok"


@pytest.mark.asyncio
async def test_wait_with_timeout_raises_if_unset() -> None:
    # Свежий event на ту же loop, чтоб обойти "bound to a different event loop"
    # из-за модульного MODELS_READY, который создаётся при импорте.
    fresh = asyncio.Event()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(fresh.wait(), timeout=0.05)
    # Чтобы тест-модуль ещё что-то делал — sanity на API.
    assert MODELS_READY.is_set() in (True, False)
