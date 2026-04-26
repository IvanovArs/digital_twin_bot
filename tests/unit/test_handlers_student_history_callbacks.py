"""Тесты на history-callback'и: on_history_page, on_star_toggle, on_export."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student.history import (
    on_export,
    on_history_page,
    on_star_toggle,
)
from src.db.models import AnswerMode, Base, Dialog, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _user(tg: int = 1) -> User:
    u = User(telegram_id=tg, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _cb(data: str) -> MagicMock:
    cb = MagicMock()
    cb.data = data
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.edit_text = AsyncMock()
    return cb


@pytest.mark.asyncio
async def test_history_page_invalid_data() -> None:
    cb = _cb("hst:")
    await on_history_page(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_history_page_non_numeric() -> None:
    cb = _cb("hst:abc:0")
    await on_history_page(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_history_page_valid(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "вопрос"
    d.answer = "ответ"
    d.created_at = datetime.now(UTC)
    session.add(d)
    await session.flush()
    cb = _cb("hst:0:0")
    await on_history_page(cb, session, user, "ru")
    cb.answer.assert_awaited()
    cb.message.edit_text.assert_awaited()


@pytest.mark.asyncio
async def test_star_toggle_invalid_data() -> None:
    cb = _cb("star:")
    await on_star_toggle(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_star_toggle_non_numeric_dialog() -> None:
    cb = _cb("star:abc:1:0:0")
    await on_star_toggle(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_star_toggle_marks_favourite(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    cb = _cb(f"star:{d.id}:1:0:0")
    await on_star_toggle(cb, session, user, "ru")
    await session.refresh(d)
    assert d.is_favourite is True


@pytest.mark.asyncio
async def test_star_toggle_rejects_other_user(session) -> None:
    owner = _user(tg=1)
    intruder = _user(tg=2)
    session.add_all([owner, intruder])
    await session.flush()
    d = Dialog(user_id=owner.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    cb = _cb(f"star:{d.id}:1:0:0")
    await on_star_toggle(cb, session, intruder, "ru")
    await session.refresh(d)
    assert d.is_favourite is False


@pytest.mark.asyncio
async def test_export_empty_history(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    m.answer_document = AsyncMock()
    await on_export(m, session, user, "ru")
    m.answer.assert_awaited()
    m.answer_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_with_dialogs_sends_document(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    for i in range(3):
        d = Dialog(user_id=user.id)
        d.question = f"вопрос {i}"
        d.answer = f"ответ {i}"
        d.created_at = datetime.now(UTC)
        session.add(d)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    m.answer_document = AsyncMock()
    await on_export(m, session, user, "ru")
    m.answer_document.assert_awaited()
    args = m.answer_document.call_args
    caption = args.kwargs.get("caption", "")
    assert "3" in caption
