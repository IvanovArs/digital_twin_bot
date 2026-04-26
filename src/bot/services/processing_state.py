"""In-process-карта ``{request_id: current_status_text}``, общая для PM и
inline-флоу — кнопка «👀 Что сейчас делается?» отвечает реальным актуальным
лейблом стадии.

Каждый ``run_qa_pipeline`` получает свежий ``rid``; обёртка над
``set_status`` пишет сюда на каждом edit'е Telegram. ``clear(rid)``
вызывается в финальном ``set_final`` — поздний тап по кэшированной
client-кнопке проваливается через TTL-гейт и показывает дружелюбный alert.

TTL короткий специально: после доставки ответа rid бессмысленен, и мы не
хотим течь памятью от заброшенных диалогов.
"""

from __future__ import annotations

import time
import uuid

_TTL_S = 60.0
_STATE: dict[str, tuple[str, float]] = {}


def _gc() -> None:
    now = time.time()
    stale = [k for k, (_, ts) in _STATE.items() if now - ts > _TTL_S]
    for k in stale:
        _STATE.pop(k, None)


def new_rid() -> str:
    """12 hex-символов → "wh:<rid>" хорошо помещается в 64-байтный лимит callback_data."""
    return uuid.uuid4().hex[:12]


def set_status(rid: str, text: str) -> None:
    _gc()
    _STATE[rid] = (text, time.time())


def get_status(rid: str) -> str | None:
    _gc()
    hit = _STATE.get(rid)
    return hit[0] if hit else None


def clear(rid: str) -> None:
    _STATE.pop(rid, None)


__all__ = ["clear", "get_status", "new_rid", "set_status"]
