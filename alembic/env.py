"""Alembic environment for digital_twin_bot.

Sync migrations (sqlalchemy.url set by src.config.settings.DB_URL). Async DB
access at runtime still uses asyncpg; migrations themselves run via psycopg
for simplicity — we translate the asyncpg URL to a sync one here.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from src.config import settings
from src.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _sync_url(async_url: str) -> str:
    """Translate async SQLAlchemy URL → sync one for Alembic.

    asyncpg → psycopg (v3), aiosqlite → plain sqlite. Other drivers pass through.
    """
    if "+asyncpg" in async_url:
        return async_url.replace("+asyncpg", "+psycopg")
    if "+aiosqlite" in async_url:
        return async_url.replace("+aiosqlite", "")
    return async_url


url = _sync_url(settings.DB_URL)
config.set_main_option("sqlalchemy.url", url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
