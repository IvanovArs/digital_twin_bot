"""Async SQLAlchemy engine + session-factory, лениво создаётся один раз на процесс."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    kwargs: dict[str, Any] = {"echo": False}

    # Pool-опции имеют смысл только для серверных драйверов; SQLite использует
    # NullPool и pool_size/max_overflow полностью отбивает.
    # Размер под маленький VPS (4 vCPU) — Telegram-боту не нужно много
    # concurrent-сессий. Recycle idle-соединений каждые 30 мин, чтобы Postgres
    # не дропал их за нашей спиной.
    if not settings.DB_URL.startswith("sqlite"):
        kwargs.update(
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=2,
            pool_recycle=1800,
            pool_timeout=10,
        )

    return create_async_engine(settings.DB_URL, **kwargs)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )
