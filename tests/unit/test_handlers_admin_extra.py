"""Дополнительные тесты handlers/admin.py: teacher_stats, teacher_gaps, etc."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin import (
    on_admin_reindex,
    on_teacher_gaps,
    on_teacher_review,
    on_teacher_stats,
)
from src.db.models import Base, Subject, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _u(role: UserRole = UserRole.admin) -> User:
    u = User(telegram_id=1, full_name="A")
    u.role = role
    return u


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


# ---------- /teacher_stats ----------


@pytest.mark.asyncio
async def test_teacher_stats_no_arg_shows_usage(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_teacher_stats(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_teacher_stats_unknown_slug(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="ghost")
    await on_teacher_stats(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "Неизвестный" in body


@pytest.mark.asyncio
async def test_teacher_stats_invalid_days(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems abc")
    await on_teacher_stats(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "числом" in body


@pytest.mark.asyncio
async def test_teacher_stats_known_subject(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems")
    await on_teacher_stats(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "Теория систем" in body or "📊" in body


@pytest.mark.asyncio
async def test_teacher_stats_denies_student(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="ghost")
    await on_teacher_stats(m, cmd, session, _u(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


# ---------- /teacher_gaps ----------


@pytest.mark.asyncio
async def test_teacher_gaps_empty_celebrates(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_teacher_gaps(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "🎉" in body or "учебник" in body


@pytest.mark.asyncio
async def test_teacher_gaps_invalid_days(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="abc")
    await on_teacher_gaps(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "числами" in body or "числом" in body


@pytest.mark.asyncio
async def test_teacher_gaps_denies_student(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_teacher_gaps(m, cmd, session, _u(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


# ---------- /teacher_review ----------


@pytest.mark.asyncio
async def test_teacher_review_empty(session) -> None:
    m = _msg()
    await on_teacher_review(m, session, _u())
    body = m.answer.call_args.args[0]
    # Нет 👎-feedback в БД → дружелюбное сообщение.
    assert body  # не упало


@pytest.mark.asyncio
async def test_teacher_review_denies_student(session) -> None:
    m = _msg()
    await on_teacher_review(m, session, _u(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


# ---------- /admin_reindex ----------


@pytest.mark.asyncio
async def test_admin_reindex_unknown_slug(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="ghost")
    await on_admin_reindex(m, cmd, session, _u())
    body = m.answer.call_args.args[0]
    assert "Неизвестный" in body or "ghost" in body


@pytest.mark.asyncio
async def test_admin_reindex_denies_student(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_admin_reindex(m, cmd, session, _u(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body
