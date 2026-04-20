"""Centralized structlog + stdlib logging setup.

Call ``configure_logging()`` once at process startup.
JSON output is enabled by LOG_JSON=true; otherwise human-friendly console.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog

from src.config import settings


def _force_utf8_stdio() -> None:
    """Windows console defaults to cp1251 in RU locales — emoji and
    Cyrillic in log lines (e.g. «📊», «стейкхолдер») crash the whole
    process with UnicodeEncodeError before any log filter runs. Force
    utf-8 on stdout/stderr once at startup. No-op on POSIX."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def configure_logging() -> None:
    _force_utf8_stdio()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # ``format_exc_info`` flattens exc_info → a string; good for JSON output,
    # but ConsoleRenderer does its own (prettier) exception formatting and
    # warns if exceptions are pre-flattened. So we only add it in JSON mode.
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

    # Also route stdlib loggers (aiogram, sqlalchemy, httpx) to the same stream
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
