"""Централизованная настройка structlog + stdlib-logging.

Зови ``configure_logging()`` один раз на старте процесса.
JSON-вывод включается через LOG_JSON=true; иначе — human-friendly console.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog

from src.config import settings


def _force_utf8_stdio() -> None:
    """Windows-консоль в RU-локалях по умолчанию cp1251 — emoji и кириллица
    в log-строках («📊», «стейкхолдер») валят весь процесс UnicodeEncodeError'ом
    до того, как сработает фильтр логов. Форсим UTF-8 на stdout/stderr один
    раз на старте. На POSIX — no-op."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr,unused-ignore]


def configure_logging() -> None:
    _force_utf8_stdio()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # ``format_exc_info`` плющит exc_info в строку — хорошо для JSON-вывода,
    # но ConsoleRenderer делает своё (более красивое) форматирование исключений
    # и warning'ует, если они уже flattened. Поэтому добавляем только в JSON-mode.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.LOG_JSON:
        shared_processors.append(structlog.processors.format_exc_info)
        renderer: Any = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Также роутим stdlib-логгеры (aiogram, sqlalchemy, httpx) в тот же stream
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # primp (HTTP-клиент внутри duckduckgo_search) каждый запуск пишет
    # «Impersonate 'safari_16.5' does not exist, using 'random'» — это
    # хардкод устаревшего профиля в нашей версии duckduckgo_search; primp
    # сам корректно фолбэчится на random. Шум, не баг → ERROR-уровень.
    logging.getLogger("primp").setLevel(logging.ERROR)
    logging.getLogger("primp.impersonate").setLevel(logging.ERROR)
