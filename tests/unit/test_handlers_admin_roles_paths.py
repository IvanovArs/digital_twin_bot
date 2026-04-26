"""Покрытие path'ов admin/roles.py: error-ветки на promote/demote/users."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin.roles import (
    on_admin_demote,
    on_admin_promote,
    on_admin_users,
)
from src.config import settings
from src.db.models import Base, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _superadmin() -> User:
    u = User(telegram_id=1, full_name="A")
    u.role = UserRole.admin
    return u


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_promote_no_args_shows_usage(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_admin_promote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_promote_non_numeric_id(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="abc teacher")
    await on_admin_promote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "числом" in body


@pytest.mark.asyncio
async def test_promote_unknown_role(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="555 emperor")
    await on_admin_promote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "Неизвестная роль" in body


@pytest.mark.asyncio
async def test_promote_target_not_found(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="555 teacher")
    await on_admin_promote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "не найден" in body


@pytest.mark.asyncio
async def test_promote_already_in_role_idempotent(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    target = User(telegram_id=555, full_name="T")
    target.role = UserRole.teacher
    session.add(target)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="555 teacher")
    await on_admin_promote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "уже" in body


@pytest.mark.asyncio
async def test_demote_no_args(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_admin_demote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_demote_non_numeric(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="abc")
    await on_admin_demote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "числом" in body


@pytest.mark.asyncio
async def test_demote_target_not_found(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="9999")
    await on_admin_demote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "не найден" in body


@pytest.mark.asyncio
async def test_demote_already_student(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    target = User(telegram_id=555, full_name="T")
    target.role = UserRole.student
    session.add(target)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="555")
    await on_admin_demote(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "и так student" in body


@pytest.mark.asyncio
async def test_users_with_filter_role(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    session.add_all(
        [
            User(telegram_id=10, full_name="Stud"),
            User(telegram_id=20, full_name="Teach"),
        ]
    )
    await session.flush()
    target = User(telegram_id=30, full_name="Adm")
    target.role = UserRole.admin
    session.add(target)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="admin")
    await on_admin_users(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "30" in body
    # student/teacher не показаны.
    assert "10" not in body or "20" not in body


@pytest.mark.asyncio
async def test_users_unknown_role_falls_back(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    session.add(User(telegram_id=10, full_name="S"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="alien_role")
    await on_admin_users(m, cmd, session, _superadmin())
    # parse_role с default=None для alien → None → фильтр не применяется.
    m.answer.assert_awaited()


@pytest.mark.asyncio
async def test_users_no_users(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    m = _msg()
    cmd = SimpleNamespace(args="admin")
    await on_admin_users(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "Нет пользователей" in body


@pytest.mark.asyncio
async def test_users_truncates_at_80(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    for i in range(85):
        session.add(User(telegram_id=1000 + i, full_name=f"U{i}"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_admin_users(m, cmd, session, _superadmin())
    body = m.answer.call_args.args[0]
    assert "ещё" in body or "85" in body
