"""In-process map ``{request_id: current_status_text}`` shared between the PM
and inline flows so the «👀 Что сейчас делается?» button can answer with a
real, up-to-the-second stage label.

Each ``run_qa_pipeline`` invocation gets a fresh ``rid``; the wrapper around
``set_status`` writes here every time Telegram is edited. ``clear(rid)`` is
called on the terminal ``set_final`` so a late tap against a cached client
button falls through the TTL gate and surfaces the friendly fallback alert.

TTL is short on purpose — once an answer is delivered the rid is meaningless,
and we don't want to leak memory from abandoned dialogs.
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
    """12-hex chars → "wh:<rid>" stays well under the 64-byte callback_data cap."""
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
