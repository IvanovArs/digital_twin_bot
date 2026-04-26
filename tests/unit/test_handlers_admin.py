"""Тесты handlers/admin.py — pure helpers + командные хендлеры (моки aiogram)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin import (
    _deny,
    _is_admin,
    _is_superadmin,
    _parse_role,
    on_admin_demote,
    on_admin_promote,
    on_admin_stats,
    on_admin_subjects,
    on_admin_upload,
    on_admin_upload_cancel,
    on_admin_users,
    on_whoami,
)
from src.config import settings
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


def _user(tg: int = 1, role: UserRole = UserRole.admin) -> User:
    u = User(telegram_id=tg, full_name="Adm")
    u.role = role
    return u


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


# ---------- pure helpers ----------


def test_is_admin_for_admin_role() -> None:
    assert _is_admin(_user(role=UserRole.admin)) is True


def test_is_admin_for_teacher_role() -> None:
    assert _is_admin(_user(role=UserRole.teacher)) is True


def test_is_admin_false_for_student() -> None:
    assert _is_admin(_user(role=UserRole.student)) is False


def test_is_superadmin_only_env_listed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "999")
    assert _is_superadmin(_user(tg=999)) is True
    assert _is_superadmin(_user(tg=1)) is False


def test_parse_role_value() -> None:
    assert _parse_role("teacher") is UserRole.teacher
    assert _parse_role("student") is UserRole.student
    assert _parse_role("admin") is UserRole.admin


def test_parse_role_default_when_empty() -> None:
    assert _parse_role(None) is UserRole.teacher
    assert _parse_role("") is UserRole.teacher


def test_parse_role_unknown_returns_none() -> None:
    assert _parse_role("president") is None


# ---------- access guards ----------


@pytest.mark.asyncio
async def test_deny_sends_visible_message() -> None:
    m = _msg()
    await _deny(m)
    body = m.answer.call_args.args[0]
    assert "⛔" in body or "Команда" in body


@pytest.mark.asyncio
async def test_admin_subjects_denies_student(session) -> None:
    m = _msg()
    await on_admin_subjects(m, session, _user(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


@pytest.mark.asyncio
async def test_admin_upload_denies_student(session) -> None:
    m = _msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    cmd = SimpleNamespace(args="theory_of_systems")
    await on_admin_upload(m, cmd, state, session, _user(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


# ---------- admin_subjects ----------


@pytest.mark.asyncio
async def test_admin_subjects_empty(session) -> None:
    m = _msg()
    await on_admin_subjects(m, session, _user())
    body = m.answer.call_args.args[0]
    assert "Нет предметов" in body


@pytest.mark.asyncio
async def test_admin_subjects_lists_all(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"))
    await session.flush()
    m = _msg()
    await on_admin_subjects(m, session, _user())
    body = m.answer.call_args.args[0]
    assert "theory_of_systems" in body
    assert "🟢" in body or "⚪" in body


# ---------- admin_upload ----------


@pytest.mark.asyncio
async def test_admin_upload_no_arg_shows_usage(session) -> None:
    m = _msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    cmd = SimpleNamespace(args=None)
    await on_admin_upload(m, cmd, state, session, _user())
    body = m.answer.call_args.args[0]
    assert "Использование" in body
    state.set_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_upload_unknown_slug(session) -> None:
    m = _msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    cmd = SimpleNamespace(args="ghost")
    await on_admin_upload(m, cmd, state, session, _user())
    body = m.answer.call_args.args[0]
    assert "Неизвестный" in body


@pytest.mark.asyncio
async def test_admin_upload_known_subject_sets_state(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    state.update_data = AsyncMock()
    cmd = SimpleNamespace(args="theory_of_systems")
    await on_admin_upload(m, cmd, state, session, _user())
    state.set_state.assert_awaited()
    state.update_data.assert_awaited_with(subject_slug="theory_of_systems")


@pytest.mark.asyncio
async def test_admin_upload_cancel_clears_state() -> None:
    m = _msg()
    state = MagicMock()
    state.clear = AsyncMock()
    await on_admin_upload_cancel(m, state)
    state.clear.assert_awaited()


# ---------- admin_stats ----------


@pytest.mark.asyncio
async def test_admin_stats_denies_student(session) -> None:
    m = _msg()
    await on_admin_stats(m, session, _user(role=UserRole.student))
    body = m.answer.call_args.args[0]
    assert "⛔" in body


@pytest.mark.asyncio
async def test_admin_stats_returns_zeros_on_empty(session) -> None:
    m = _msg()
    await on_admin_stats(m, session, _user())
    body = m.answer.call_args.args[0]
    # На пустой БД — 0 диалогов, без падений.
    assert "0" in body


# ---------- whoami ----------


@pytest.mark.asyncio
async def test_whoami_shows_role_and_id(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    m = _msg()
    user = _user(tg=12345, role=UserRole.teacher)
    await on_whoami(m, user)
    body = m.answer.call_args.args[0]
    assert "12345" in body
    assert "teacher" in body.lower()


# ---------- promote / demote / users ----------


@pytest.mark.asyncio
async def test_promote_denies_non_superadmin(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    m = _msg()
    cmd = SimpleNamespace(args="555 teacher")
    await on_admin_promote(m, cmd, session, _user(tg=1, role=UserRole.admin))
    body = m.answer.call_args.args[0]
    assert "только" in body.lower() or "super" in body.lower() or "⛔" in body


@pytest.mark.asyncio
async def test_promote_changes_role(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    target = User(telegram_id=555, full_name="Target")
    target.role = UserRole.student
    session.add(target)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="555 teacher")
    await on_admin_promote(m, cmd, session, _user(tg=1, role=UserRole.admin))
    await session.refresh(target)
    assert target.role is UserRole.teacher


@pytest.mark.asyncio
async def test_demote_resets_to_student(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    target = User(telegram_id=555, full_name="Target")
    target.role = UserRole.teacher
    session.add(target)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="555")
    await on_admin_demote(m, cmd, session, _user(tg=1, role=UserRole.admin))
    await session.refresh(target)
    assert target.role is UserRole.student


@pytest.mark.asyncio
async def test_users_lists_all(session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "1")
    session.add(User(telegram_id=100, full_name="A"))
    session.add(User(telegram_id=200, full_name="B"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_admin_users(m, cmd, session, _user(tg=1, role=UserRole.admin))
    body = m.answer.call_args.args[0]
    assert "100" in body or "200" in body
