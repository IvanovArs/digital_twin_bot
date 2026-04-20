"""Async SQLAlchemy engine + session factory, lazily constructed once per process."""

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

    # Pool options only make sense for server-based drivers; SQLite uses NullPool
    # and rejects pool_size / max_overflow entirely.
    # Sized for a small VPS (4 vCPU) — a Telegram bot doesn't need many
    # concurrent sessions. Recycle idle connections every 30 min so Postgres
    # doesn't drop them behind our back.
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
