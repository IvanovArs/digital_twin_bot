"""Кросс-модульный сигнал «RAG-модели загружены и готовы отвечать».

Бот прогревает bge-m3 и unified-индекс в фоне на старте. При первом запуске
это означает скачать ~3 ГБ с HuggingFace — 30–60 с на быстром линке и
*минуты* на медленном.

Если студент пишет вопрос до прогрева, retrieval блокируется на
``SentenceTransformer(...)`` в worker-thread'е, и UX выглядит как «бот
завис на плейсхолдере». Этот event позволяет QA-пайплайну показать
честный статус («прогреваюсь…»), а warm-up task'у — однократно
просигналить о завершении.

Event создаётся лениво на первый доступ — всегда биндится к **текущему**
asyncio-loop'у. Раньше создавали на module-import; в проде работало
(один loop на жизнь), но в тестах с pytest-asyncio per-test-loop'ами
вылетало с «got Future attached to a different loop».
"""

from __future__ import annotations

import asyncio

_EVENT: asyncio.Event | None = None


def models_ready_event() -> asyncio.Event:
    """Singleton-Event готовности, создаётся при первом вызове.

    Ленивое создание откладывает binding к running-loop'у до момента, когда
    Event реально нужен — безопаснее в тестах и в любом будущем
    «graceful-restart»-сценарии, пересоздающем event-loop.
    """
    global _EVENT
    if _EVENT is None:
        _EVENT = asyncio.Event()
    return _EVENT


# Backwards-compat alias для старых call-site'ов, читающих/пишущих напрямую.
class _LazyEventProxy:
    """Прокси set/wait/is_set к лениво-создаваемому Event'у."""

    def set(self) -> None:
        models_ready_event().set()

    def clear(self) -> None:
        models_ready_event().clear()

    def is_set(self) -> bool:
        return models_ready_event().is_set()

    async def wait(self) -> bool:
        return await models_ready_event().wait()


MODELS_READY: _LazyEventProxy = _LazyEventProxy()
