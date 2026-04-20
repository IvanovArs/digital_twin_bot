"""Tests for the throttle middleware's idle-user GC."""

from __future__ import annotations

from src.bot.middlewares import throttle_mw


def _reset() -> None:
    throttle_mw._last_ts.clear()
    throttle_mw._locks.clear()


def test_gc_removes_stale_entries() -> None:
    _reset()
    now = throttle_mw._GC_IDLE_SECONDS + 2_000.0
    throttle_mw._last_ts[1] = 0.0  # very stale
    throttle_mw._last_ts[2] = 0.0  # very stale
    throttle_mw._last_ts[3] = now - 100.0  # fresh
    throttle_mw._gc_stale(now=now)
    assert 1 not in throttle_mw._last_ts
    assert 2 not in throttle_mw._last_ts
    assert 3 in throttle_mw._last_ts


def test_gc_keeps_locked_locks() -> None:
    """A user whose pipeline is still running must not lose their lock
    even if their last-ts is stale — mid-call eviction would let a
    concurrent second call through."""
    import asyncio

    _reset()
    throttle_mw._last_ts[7] = 0.0  # stale timestamp
    lock = asyncio.Lock()

    async def _acquire() -> None:
        await lock.acquire()

    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(_acquire())
        throttle_mw._locks[7] = lock
        throttle_mw._gc_stale(now=throttle_mw._GC_IDLE_SECONDS + 10.0)
        # Timestamp entry evicted, but the locked lock survives.
        assert 7 not in throttle_mw._last_ts
        assert 7 in throttle_mw._locks
    finally:
        lock.release()
        loop.close()


def test_gc_keeps_recent_users() -> None:
    _reset()
    now = 5_000.0
    throttle_mw._last_ts[42] = now - 10.0
    throttle_mw._gc_stale(now=now)
    assert 42 in throttle_mw._last_ts


def test_gc_is_safe_on_empty_state() -> None:
    _reset()
    throttle_mw._gc_stale(now=0.0)  # must not raise
