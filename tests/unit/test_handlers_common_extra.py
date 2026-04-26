"""Покрытие оставшихся веток common.py: deep_link/start payload, menu callbacks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.common import (
    on_menu_ask,
    on_menu_help,
    on_menu_subjects,
    on_start_with_payload,
)
from src.db.models import AnswerMode, Base, GlossaryTerm, Subject, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _user() -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    m.answer_animation = AsyncMock(side_effect=Exception("no video"))
    me = SimpleNamespace(username="testbot")
    m.bot = MagicMock()
    m.bot.get_me = AsyncMock(return_value=me)
    return m


@pytest.mark.asyncio
async def test_start_payload_help_sends_help() -> None:
    m = _msg()
    cmd = SimpleNamespace(args="help")
    await on_start_with_payload(m, cmd, _user(), "ru")
    body = m.answer.call_args.args[0]
    assert "🎯" in body or "/ask" in body or "What I can do" in body


@pytest.mark.asyncio
async def test_start_payload_unknown_falls_through_to_greeting() -> None:
    m = _msg()
    cmd = SimpleNamespace(args="unknown_payload")
    await on_start_with_payload(m, cmd, _user(), "ru")
    body = m.answer.call_args.args[0]
    assert "Привет" in body or "Hi" in body


@pytest.mark.asyncio
async def test_start_payload_empty_falls_through() -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_start_with_payload(m, cmd, _user(), "ru")
    m.answer.assert_awaited()


# ---------- menu callbacks ----------


def _cb_no_msg() -> MagicMock:
    cb = MagicMock()
    cb.answer = AsyncMock()
    cb.message = None
    return cb


def _cb_with_msg() -> MagicMock:
    cb = MagicMock()
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.answer = AsyncMock()
    me = SimpleNamespace(username="testbot")
    cb.message.bot = MagicMock()
    cb.message.bot.get_me = AsyncMock(return_value=me)
    return cb


@pytest.mark.asyncio
async def test_menu_ask_no_message_sends_alert() -> None:
    cb = _cb_no_msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    await on_menu_ask(cb, state, "ru")
    cb.answer.assert_awaited()
    args = cb.answer.call_args
    assert args.kwargs.get("show_alert") is True


@pytest.mark.asyncio
async def test_menu_ask_with_message_sets_state() -> None:
    cb = _cb_with_msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    await on_menu_ask(cb, state, "ru")
    state.set_state.assert_awaited()
    cb.message.answer.assert_awaited()


@pytest.mark.asyncio
async def test_menu_help_with_message_sends_help() -> None:
    cb = _cb_with_msg()
    await on_menu_help(cb, _user(), "ru")
    cb.message.answer.assert_awaited()


@pytest.mark.asyncio
async def test_menu_help_no_message_just_acks() -> None:
    cb = _cb_no_msg()
    await on_menu_help(cb, _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_menu_subjects_no_message_just_acks(session) -> None:
    cb = _cb_no_msg()
    await on_menu_subjects(cb, session, "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_menu_subjects_with_message_sends_list(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория"))
    await session.flush()
    cb = _cb_with_msg()
    await on_menu_subjects(cb, session, "ru")
    cb.message.answer.assert_awaited()


@pytest.mark.asyncio
async def test_glossary_with_known_subject_header(session) -> None:
    """on_glossary без query, но с current_subject_slug — заголовок про subject."""
    from src.bot.handlers.common import on_glossary

    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")
    session.add(subj)
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj.id, term="X", definition="Y"))
    await session.flush()

    user = _user()
    user.current_subject_slug = "theory_of_systems"
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_glossary(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    # subject-header должен включать имя предмета.
    assert "Теория систем" in body or "theory_of_systems" in body


@pytest.mark.asyncio
async def test_glossary_truncated_at_20(session) -> None:
    """При >20 терминах должно прийти GLOSSARY_TRUNCATED."""
    from src.bot.handlers.common import on_glossary

    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    session.add(subj)
    await session.flush()
    for i in range(25):
        session.add(GlossaryTerm(subject_id=subj.id, term=f"термин{i:02d}", definition=f"def{i}"))
    await session.flush()

    user = _user()
    user.current_subject_slug = "theory_of_systems"
    m = _msg()
    cmd = SimpleNamespace(args=None)
    await on_glossary(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "первые 20" in body or "first 20" in body
