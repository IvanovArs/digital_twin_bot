"""Тесты обработчиков common.py через лёгкие моки aiogram-объектов."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.common import (
    _send_help,
    on_glossary,
    on_help,
    on_mode,
    on_start,
    on_subjects,
)
from src.db.models import AnswerMode, Base, Subject, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _user(tg: int = 1, role: UserRole = UserRole.student) -> User:
    u = User(telegram_id=tg, full_name="Stud Ent")
    u.role = role
    u.answer_mode = AnswerMode.verbose
    return u


def _msg(text: str = "") -> MagicMock:
    m = MagicMock()
    m.text = text
    m.answer = AsyncMock()
    m.answer_animation = AsyncMock(side_effect=Exception("video missing"))
    me = SimpleNamespace(username="testbot")
    m.bot = MagicMock()
    m.bot.get_me = AsyncMock(return_value=me)
    return m


@pytest.mark.asyncio
async def test_on_start_sends_greeting() -> None:
    m = _msg()
    await on_start(m, _user(), "ru")
    m.answer.assert_awaited()
    body = m.answer.call_args.args[0]
    assert "Stud Ent" in body or "друг" in body


@pytest.mark.asyncio
async def test_on_help_includes_admin_block_for_admin_role() -> None:
    m = _msg()
    await on_help(m, _user(role=UserRole.admin), "ru")
    body = m.answer.call_args.args[0]
    assert "/admin" in body or "Преподаватель" in body or "Teacher" in body


@pytest.mark.asyncio
async def test_on_help_no_admin_block_for_student() -> None:
    m = _msg()
    await on_help(m, _user(role=UserRole.student), "ru")
    body = m.answer.call_args.args[0]
    assert "/admin_promote" not in body


@pytest.mark.asyncio
async def test_on_mode_no_arg_shows_current(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    user = _user()
    await on_mode(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "verbose" in body or "развёрнутый" in body


@pytest.mark.asyncio
async def test_on_mode_set_brief_persists(session) -> None:
    m = _msg()
    user = _user()
    session.add(user)
    await session.flush()
    cmd = SimpleNamespace(args="brief")
    await on_mode(m, cmd, session, user, "ru")
    assert user.answer_mode is AnswerMode.brief


@pytest.mark.asyncio
async def test_on_mode_invalid_arg_shows_help(session) -> None:
    m = _msg()
    user = _user()
    session.add(user)
    await session.flush()
    cmd = SimpleNamespace(args="absurd")
    await on_mode(m, cmd, session, user, "ru")
    # Mode не поменялся.
    assert user.answer_mode is AnswerMode.verbose


@pytest.mark.asyncio
async def test_on_subjects_empty_returns_friendly_message(session) -> None:
    m = _msg()
    await on_subjects(m, session, "ru")
    body = m.answer.call_args.args[0]
    assert "Пока нет предметов" in body or "No subjects" in body


@pytest.mark.asyncio
async def test_on_subjects_lists_active_subjects(session) -> None:
    session.add_all(
        [
            Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"),
            Subject(
                slug="systems_engineering",
                title_en="SE",
                title_ru="Инженерия",
                is_active=False,
            ),
        ]
    )
    await session.flush()
    m = _msg()
    await on_subjects(m, session, "ru")
    body = m.answer.call_args.args[0]
    assert "Теория систем" in body
    # Неактивный — не показан.
    assert "systems_engineering" not in body


@pytest.mark.asyncio
async def test_on_glossary_no_query_lists_terms(session) -> None:
    from src.db.models import GlossaryTerm

    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    session.add(subj)
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj.id, term="X", definition="Y"))
    await session.flush()

    m = _msg()
    user = _user()
    user.current_subject_slug = "theory_of_systems"
    cmd = SimpleNamespace(args=None)
    await on_glossary(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "X" in body and "Y" in body


@pytest.mark.asyncio
async def test_on_glossary_with_query_finds_match(session) -> None:
    from src.db.models import GlossaryTerm

    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    session.add(subj)
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj.id, term="Stakeholder", definition="A person."))
    await session.flush()

    m = _msg()
    user = _user()
    user.current_subject_slug = "theory_of_systems"
    cmd = SimpleNamespace(args="stake")
    await on_glossary(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Stakeholder" in body


@pytest.mark.asyncio
async def test_on_glossary_query_misses(session) -> None:
    m = _msg()
    user = _user()
    cmd = SimpleNamespace(args="несуществующее")
    await on_glossary(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "ничего не нашёл" in body or "Nothing found" in body


@pytest.mark.asyncio
async def test_send_help_uses_bot_username(session) -> None:
    m = _msg()
    await _send_help(m, _user(), "ru")
    body = m.answer.call_args.args[0]
    assert "testbot" in body
