"""Cross-module signal for "RAG models are loaded and ready to serve".

The bot warms up bge-m3 and the unified index in the background at startup.
On first boot that involves downloading ~3 GB from HuggingFace, which can
take 30-60 s on a fast link and *minutes* on a slow one.

If a student sends a question before the warm-up finishes, the retrieval
step blocks on ``SentenceTransformer(...)`` inside a worker thread, and the
UX looks like "bot hangs on placeholder". This event lets the QA pipeline
show an honest status ("прогреваюсь…") instead, and lets the warm-up task
signal completion exactly once.

The Event is created lazily on first access so it always binds to the
**currently running** asyncio loop. Historically we created it at module
import; that worked in production (one loop ever) but blew up in tests
that use pytest-asyncio's per-test loops with "got Future attached to a
different loop".
"""

from __future__ import annotations

import asyncio

_EVENT: asyncio.Event | None = None


def models_ready_event() -> asyncio.Event:
    """Return the singleton readiness Event, creating it on first call.

    Lazy creation defers binding to the running loop until someone actually
    needs the Event — safer in tests and for any future "graceful restart"
    code path that recreates the event loop.
    """
    global _EVENT
    if _EVENT is None:
        _EVENT = asyncio.Event()
    return _EVENT


# Backwards-compat alias for existing call sites that read/write directly.
class _LazyEventProxy:
    """Forward set/wait/is_set to the lazily-created Event."""

    def set(self) -> None:
        models_ready_event().set()

    def clear(self) -> None:
        models_ready_event().clear()

    def is_set(self) -> bool:
        return models_ready_event().is_set()

    async def wait(self) -> bool:
        return await models_ready_event().wait()


MODELS_READY: _LazyEventProxy = _LazyEventProxy()
