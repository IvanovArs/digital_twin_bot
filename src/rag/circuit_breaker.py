"""Process-local circuit-breaker для HTTP-клиента llama-server.

Срабатывает после N подряд-failures, fail-fast'ит на cooldown-окно,
half-open'ится после, close'ится на success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class CircuitOpenError(RuntimeError):
    """Бросается ``guard()``, когда breaker открыт."""


@dataclass
class _State:
    consecutive_failures: int = 0
    opened_at: float = 0.0  # monotonic-таймштамп, 0 когда closed


class CircuitBreaker:
    def __init__(self, *, fail_threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._state = _State()

    # ---------- запросы состояния ----------

    def is_open(self) -> bool:
        if self._state.opened_at == 0:
            return False
        if time.monotonic() - self._state.opened_at >= self.cooldown_s:
            # Cooldown истёк — half-open (пропускаем ОДИН probe). Отдельного
            # half-open-состояния не держим; один запрос пробегает и либо
            # close'ит, либо снова open'ит breaker.
            self._state.opened_at = 0
            self._state.consecutive_failures = 0
            return False
        return True

    def guard(self) -> None:
        """Бросить ``CircuitOpenError``, если breaker сейчас открыт.
        Вызывай в начале каждого code-path'а, дёргающего LLM."""
        if self.is_open():
            remaining = self.cooldown_s - (time.monotonic() - self._state.opened_at)
            raise CircuitOpenError(
                f"llm circuit open, retry in {max(0, int(remaining))} s"
            )

    # ---------- переходы состояния ----------

    def record_success(self) -> None:
        self._state.consecutive_failures = 0
        self._state.opened_at = 0

    def record_failure(self) -> None:
        self._state.consecutive_failures += 1
        if self._state.consecutive_failures >= self.fail_threshold:
            self._state.opened_at = time.monotonic()


# Module-level singleton, один на процесс. Импортируй где нужен.
llm_breaker = CircuitBreaker(fail_threshold=5, cooldown_s=30.0)
