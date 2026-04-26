"""Тесты handlers/student/followup.py: 👍/👎, expand, simplify, whats_happening."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student.followup import (
    on_expand_answer,
    on_feedback,
    on_followup,
    on_whats_happening,
)
from src.bot.services import processing_state
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


def _cb(data: str, *, with_message: bool = True, inline_msg_id: str | None = None) -> MagicMock:
    cb = MagicMock()
    cb.data = data
    cb.answer = AsyncMock()
    cb.inline_message_id = inline_msg_id
    if with_message:
        cb.message = MagicMock()
        cb.message.answer = AsyncMock()
        cb.message.edit_text = AsyncMock()
        cb.message.edit_reply_markup = AsyncMock()
        cb.message.text = "old"
        cb.message.bot = MagicMock()
        cb.message.chat = MagicMock()
        cb.message.chat.id = 100
        cb.message.message_id = 5
    else:
        cb.message = None
    cb.bot = MagicMock()
    cb.bot.set_message_reaction = AsyncMock()
    cb.bot.edit_message_reply_markup = AsyncMock()
    return cb


# ---------- on_whats_happening ----------


@pytest.mark.asyncio
async def test_whats_happening_unknown_rid_shows_default() -> None:
    cb = _cb("wh:unknown-rid")
    await on_whats_happening(cb, "ru")
    cb.answer.assert_awaited()
    args = cb.answer.call_args
    assert args.kwargs.get("show_alert") is True


@pytest.mark.asyncio
async def test_whats_happening_returns_stored_status() -> None:
    rid = processing_state.new_rid()
    processing_state.set_status(rid, "Тестовый статус")
    try:
        cb = _cb(f"wh:{rid}")
        await on_whats_happening(cb, "ru")
        args = cb.answer.call_args
        assert "Тестовый статус" in args.kwargs.get("text", "")
    finally:
        processing_state.clear(rid)


@pytest.mark.asyncio
async def test_whats_happening_truncates_long_status() -> None:
    rid = processing_state.new_rid()
    long_text = "X" * 500
    processing_state.set_status(rid, long_text)
    try:
        cb = _cb(f"wh:{rid}")
        await on_whats_happening(cb, "ru")
        args = cb.answer.call_args
        assert len(args.kwargs.get("text", "")) <= 200
    finally:
        processing_state.clear(rid)


@pytest.mark.asyncio
async def test_whats_happening_strips_html_tags_and_cursor() -> None:
    rid = processing_state.new_rid()
    processing_state.set_status(rid, "<b>Stake</b> ▍")
    try:
        cb = _cb(f"wh:{rid}")
        await on_whats_happening(cb, "ru")
        body = cb.answer.call_args.kwargs.get("text", "")
        assert "<b>" not in body
    finally:
        processing_state.clear(rid)


# ---------- on_expand_answer ----------


@pytest.mark.asyncio
async def test_expand_invalid_callback_data() -> None:
    cb = _cb("expand:")
    await on_expand_answer(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_expand_non_numeric_dialog_id() -> None:
    cb = _cb("expand:abc")
    await on_expand_answer(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_expand_dialog_not_found(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    cb = _cb("expand:9999")
    await on_expand_answer(cb, session, user, "ru")
    cb.answer.assert_awaited_with("Диалог не найден.", show_alert=True)


# ---------- on_followup ----------


@pytest.mark.asyncio
async def test_followup_invalid_data_format() -> None:
    cb = _cb("fu:bad")
    await on_followup(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_followup_unknown_modifier() -> None:
    cb = _cb("fu:1:invalid_mod")
    await on_followup(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_followup_no_message_returns_silently() -> None:
    cb = _cb("fu:1:simplify", with_message=False)
    await on_followup(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_followup_non_numeric_dialog() -> None:
    cb = _cb("fu:abc:simplify")
    await on_followup(cb, MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


# ---------- on_feedback ----------


@pytest.mark.asyncio
async def test_feedback_records_rating(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    cb = _cb(f"fb:{d.id}:5")
    await on_feedback(cb, session, user, "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_feedback_inline_path_swaps_keyboard(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    cb = _cb(f"fb:{d.id}:5", with_message=False, inline_msg_id="inline-xyz")
    await on_feedback(cb, session, user, "ru")
    cb.bot.edit_message_reply_markup.assert_awaited()


@pytest.mark.asyncio
async def test_feedback_thumbs_down_no_reaction(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    cb = _cb(f"fb:{d.id}:1")
    await on_feedback(cb, session, user, "ru")
    cb.bot.set_message_reaction.assert_not_awaited()
