"""Дополнительные тесты handlers/student.py: history/favourites/term/ref/find."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student import (
    on_favourites,
    on_find,
    on_history,
    on_ref,
    on_term,
)
from src.db.models import (
    AnswerMode,
    Base,
    Dialog,
    GlossaryTerm,
    Subject,
    User,
    UserRole,
)


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


async def _u(session: AsyncSession, tg: int = 1) -> User:
    u = User(telegram_id=tg, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    session.add(u)
    await session.flush()
    return u


# ---------- /history ----------


@pytest.mark.asyncio
async def test_history_empty_says_no_questions(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args=None)
    m = _msg()
    await on_history(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Пока нет" in body or "No questions" in body


@pytest.mark.asyncio
async def test_history_lists_dialogs(session) -> None:
    user = await _u(session)
    d = Dialog(user_id=user.id)
    d.question = "вопрос про систему"
    d.answer = "ответ"
    d.created_at = datetime.now(UTC)
    session.add(d)
    await session.flush()
    cmd = SimpleNamespace(args=None)
    m = _msg()
    await on_history(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "вопрос про систему" in body


@pytest.mark.asyncio
async def test_history_invalid_page_arg_defaults_to_zero(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args="abc")
    m = _msg()
    await on_history(m, cmd, session, user, "ru")
    m.answer.assert_awaited()  # не упало


# ---------- /favourites ----------


@pytest.mark.asyncio
async def test_favourites_empty_special_message(session) -> None:
    user = await _u(session)
    m = _msg()
    await on_favourites(m, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "избранных" in body or "favourites" in body


@pytest.mark.asyncio
async def test_favourites_lists_starred(session) -> None:
    user = await _u(session)
    d = Dialog(user_id=user.id)
    d.question = "это избранное"
    d.answer = "ответ"
    d.is_favourite = True
    session.add(d)
    await session.flush()
    m = _msg()
    await on_favourites(m, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "это избранное" in body


# ---------- /term ----------


@pytest.mark.asyncio
async def test_term_no_arg_shows_usage(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args=None)
    m = _msg()
    await on_term(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_term_unknown_glossary_friendly_message(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args="ghost")
    m = _msg()
    await on_term(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "В глоссарии нет" in body or "ghost" in body


@pytest.mark.asyncio
async def test_term_returns_glossary_body(session) -> None:
    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    session.add(subj)
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj.id, term="Стейкхолдер", definition="Лицо."))
    await session.flush()
    user = await _u(session)
    cmd = SimpleNamespace(args="стейкхолдер")
    m = _msg()
    await on_term(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Стейкхолдер" in body or "Лицо" in body


# ---------- /ref ----------


@pytest.mark.asyncio
async def test_ref_no_arg_shows_usage(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args=None)
    m = _msg()
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_ref_non_numeric_id_rejected(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args="abc")
    m = _msg()
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "числом" in body


@pytest.mark.asyncio
async def test_ref_unknown_id_returns_message(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args="999")
    m = _msg()
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    # Текст про «не нашёл» / «not found».
    assert "999" in body or "Не " in body or "not " in body.lower()


# ---------- /find ----------


@pytest.mark.asyncio
async def test_find_no_arg_shows_usage(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args=None)
    m = _msg()
    await on_find(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Использование" in body or "/find" in body


@pytest.mark.asyncio
async def test_find_no_match(session) -> None:
    user = await _u(session)
    cmd = SimpleNamespace(args="ничего такого")
    m = _msg()
    await on_find(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "ничего такого" in body
