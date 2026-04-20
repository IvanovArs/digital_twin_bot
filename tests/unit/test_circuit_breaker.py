"""Tests for the LLM circuit breaker."""

from __future__ import annotations

import time

import pytest

from src.rag.circuit_breaker import CircuitBreaker, CircuitOpenError


def test_fresh_breaker_is_closed() -> None:
    cb = CircuitBreaker(fail_threshold=3, cooldown_s=30)
    cb.guard()  # must not raise
    assert not cb.is_open()


def test_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker(fail_threshold=3, cooldown_s=30)
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open()  # 2 < 3, still closed
    cb.record_failure()
    assert cb.is_open()  # 3 == threshold → open


def test_guard_raises_when_open() -> None:
    cb = CircuitBreaker(fail_threshold=1, cooldown_s=30)
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.guard()


def test_success_resets_failure_counter() -> None:
    cb = CircuitBreaker(fail_threshold=3, cooldown_s=30)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # Back under threshold — should still be closed.
    assert not cb.is_open()


def test_cooldown_moves_to_half_open(monkeypatch) -> None:
    """After the cooldown elapses, a single probe is allowed through."""
    cb = CircuitBreaker(fail_threshold=1, cooldown_s=10.0)
    base = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: base)
    cb.record_failure()
    assert cb.is_open()
    # Advance past cooldown.
    monkeypatch.setattr(time, "monotonic", lambda: base + 11.0)
    assert not cb.is_open()
    cb.guard()  # probe call succeeds (doesn't raise)


def test_probe_success_closes_breaker(monkeypatch) -> None:
    cb = CircuitBreaker(fail_threshold=2, cooldown_s=5.0)
    base = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: base)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    # Simulate cooldown elapsing.
    monkeypatch.setattr(time, "monotonic", lambda: base + 6.0)
    cb.guard()
    cb.record_success()
    # Counter is reset; one further failure shouldn't re-open (still
    # below threshold=2).
    cb.record_failure()
    assert not cb.is_open()


def test_probe_failure_reopens_breaker(monkeypatch) -> None:
    cb = CircuitBreaker(fail_threshold=2, cooldown_s=5.0)
    base = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: base)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    monkeypatch.setattr(time, "monotonic", lambda: base + 6.0)
    # is_open() has the side effect of moving to half-open; probe now
    # permitted. If the probe also fails, the breaker re-trips after
    # ``fail_threshold`` more failures — we verify the half-open state
    # actually resets the counter.
    assert not cb.is_open()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()


def test_circuit_open_error_message_is_informative() -> None:
    cb = CircuitBreaker(fail_threshold=1, cooldown_s=30)
    cb.record_failure()
    with pytest.raises(CircuitOpenError) as exc:
        cb.guard()
    assert "circuit open" in str(exc.value).lower()
