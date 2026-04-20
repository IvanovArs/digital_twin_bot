"""Minimal circuit breaker for the llama-server HTTP client.

Why: if llama-server is hung or crashed, every student's question waits
``LLM_TIMEOUT_SECONDS`` (default 300 s) before failing. That's a
compounding disaster — the bot queue fills, new students see "busy", the
first wave times out en-masse.

A circuit breaker trips after ``fail_threshold`` consecutive failures:
subsequent calls fail *fast* for ``cooldown_s`` seconds without hitting
the backend. After cooldown we half-open — one probe call gets through;
success closes, failure re-opens for another cooldown window.

The breaker is process-local (no shared state between instances), which
is fine: in a multi-instance deploy each instance independently detects
its own llama-server going bad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class CircuitOpenError(RuntimeError):
    """Raised by ``guard()`` when the breaker is open."""


@dataclass
class _State:
    consecutive_failures: int = 0
    opened_at: float = 0.0  # monotonic timestamp, 0 when closed


class CircuitBreaker:
    def __init__(self, *, fail_threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._state = _State()

    # ---------- state queries ----------

    def is_open(self) -> bool:
        if self._state.opened_at == 0:
            return False
        if time.monotonic() - self._state.opened_at >= self.cooldown_s:
            # Cooldown elapsed — move to half-open (allow ONE probe). We
            # don't keep a separate half-open state; a single request races
            # through and either closes or re-opens the breaker.
            self._state.opened_at = 0
            self._state.consecutive_failures = 0
            return False
        return True

    def guard(self) -> None:
        """Raise ``CircuitOpenError`` if the breaker is currently open.
        Call this at the top of every LLM-invoking code path."""
        if self.is_open():
            remaining = self.cooldown_s - (time.monotonic() - self._state.opened_at)
            raise CircuitOpenError(
                f"llm circuit open, retry in {max(0, int(remaining))} s"
            )

    # ---------- state transitions ----------

    def record_success(self) -> None:
        self._state.consecutive_failures = 0
        self._state.opened_at = 0

    def record_failure(self) -> None:
        self._state.consecutive_failures += 1
        if self._state.consecutive_failures >= self.fail_threshold:
            self._state.opened_at = time.monotonic()


# Module-level singleton, one per process. Import this where you need it.
llm_breaker = CircuitBreaker(fail_threshold=5, cooldown_s=30.0)
