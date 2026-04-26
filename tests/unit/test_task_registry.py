"""Тесты graceful-shutdown реестра задач."""

from __future__ import annotations

import asyncio

import pytest

from src.bot.services import task_registry


@pytest.mark.asyncio
async def test_tracked_registers_and_releases() -> None:
    seen_during_call = False

    async def work() -> None:
        nonlocal seen_during_call
        with task_registry.tracked():
            seen_during_call = bool(task_registry._TASKS)
            await asyncio.sleep(0)

    await work()
    assert seen_during_call is True
    assert not task_registry._TASKS  # после exit реестр пуст


@pytest.mark.asyncio
async def test_drain_waits_until_done() -> None:
    finished = False

    async def slow() -> None:
        nonlocal finished
        with task_registry.tracked():
            await asyncio.sleep(0.05)
            finished = True

    bg = asyncio.create_task(slow())
    await asyncio.sleep(0)  # даём slow зарегистрироваться
    await task_registry.drain(timeout=1.0)
    assert finished is True
    assert bg.done()


@pytest.mark.asyncio
async def test_drain_cancels_overrun_tasks() -> None:
    cancelled = False

    async def stuck() -> None:
        nonlocal cancelled
        with task_registry.tracked():
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled = True
                raise

    bg = asyncio.create_task(stuck())
    await asyncio.sleep(0)
    await task_registry.drain(timeout=0.1)
    assert cancelled is True
    assert bg.done()


@pytest.mark.asyncio
async def test_drain_noop_when_empty() -> None:
    # Без зарегистрированных задач — мгновенный возврат, без исключений.
    await task_registry.drain(timeout=0.5)


@pytest.mark.asyncio
async def test_track_creates_tracked_task() -> None:
    done = False

    async def quick() -> int:
        nonlocal done
        await asyncio.sleep(0)
        done = True
        return 42

    task = task_registry.track(quick())
    assert task in task_registry._TASKS
    res = await task
    assert res == 42
    # done_callback убирает задачу из реестра.
    await asyncio.sleep(0)
    assert task not in task_registry._TASKS
    assert done is True
