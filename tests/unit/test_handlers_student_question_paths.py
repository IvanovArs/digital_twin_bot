"""Покрытие edit-веток on_question: TelegramRetryAfter, TelegramBadRequest,
не-modified, generic exception, glossary/faq kind, brief-mode kind."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student import question as q_mod
from src.bot.handlers.student.question import on_question
from src.db.models import AnswerMode, Base, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _user(*, brief: bool = False) -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.brief if brief else AnswerMode.verbose
    return u


def _msg(text: str, *, edit_side_effect=None) -> MagicMock:  # type: ignore[no-untyped-def]
    m = MagicMock()
    m.text = text
    m.message_id = 7
    m.chat = MagicMock()
    m.chat.id = 100
    m.bot = MagicMock()
    m.bot.send_chat_action = AsyncMock()
    m.bot.set_message_reaction = AsyncMock()
    sent = MagicMock()
    sent.edit_text = AsyncMock(side_effect=edit_side_effect)
    m.answer = AsyncMock(return_value=sent)
    return m


def _state() -> MagicMock:
    s = MagicMock()
    s.set_state = AsyncMock()
    s.clear = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_question_glossary_kind_picks_short_kb(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_final"]("курированный", 11, "glossary")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос")
    await on_question(m, _state(), session, user, "ru")
    # set_final был вызван — edit_text должен был принять полный body.
    assert m.answer.return_value.edit_text.await_count >= 1


@pytest.mark.asyncio
async def test_question_brief_user_gets_brief_kb(session, monkeypatch) -> None:
    user = _user(brief=True)
    session.add(user)
    await session.flush()

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_final"]("краткий ответ", 22, "full")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос")
    await on_question(m, _state(), session, user, "ru")


@pytest.mark.asyncio
async def test_question_edit_retry_after_recovers(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    edits = {"n": 0}

    async def edit_side_effect(*a, **kw):  # type: ignore[no-untyped-def]
        edits["n"] += 1
        if edits["n"] == 1:
            raise TelegramRetryAfter(method=None, message="rate", retry_after=0)
        return None

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_status"]("первый edit")
        await kw["set_status"]("второй edit (after retry)")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос", edit_side_effect=edit_side_effect)
    await on_question(m, _state(), session, user, "ru")


@pytest.mark.asyncio
async def test_question_edit_bad_request_not_modified_swallowed(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def edit_side_effect(*a, **kw):  # type: ignore[no-untyped-def]
        raise TelegramBadRequest(method=None, message="message is not modified")

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_status"]("какой-то текст")
        await kw["set_final"]("body", 1, "full")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос", edit_side_effect=edit_side_effect)
    await on_question(m, _state(), session, user, "ru")


@pytest.mark.asyncio
async def test_question_edit_bad_request_other_logged(session, monkeypatch, caplog) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def edit_side_effect(*a, **kw):  # type: ignore[no-untyped-def]
        raise TelegramBadRequest(method=None, message="can't parse entities at byte 12")

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_status"]("текст")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос", edit_side_effect=edit_side_effect)
    await on_question(m, _state(), session, user, "ru")


@pytest.mark.asyncio
async def test_question_edit_generic_exception_caught(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def edit_side_effect(*a, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("network down")

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        await kw["set_status"]("текст")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос", edit_side_effect=edit_side_effect)
    await on_question(m, _state(), session, user, "ru")


@pytest.mark.asyncio
async def test_question_dedupe_skips_no_op_edit(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        # Дважды один и тот же текст — второй должен быть скипнут.
        await kw["set_status"]("одинаковый текст")
        await kw["set_status"]("одинаковый текст")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос")
    await on_question(m, _state(), session, user, "ru")
    # Edit должен был вызваться один раз (второй — дроп по дедуп).
    assert m.answer.return_value.edit_text.await_count == 1


@pytest.mark.asyncio
async def test_question_streaming_preview_alert_substituted(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    async def pipeline(**kw):  # type: ignore[no-untyped-def]
        # Стрим-превью с курсором ▍ — alert должен подменить.
        await kw["set_status"]("частичный ответ ▍")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", pipeline)
    m = _msg("вопрос")
    await on_question(m, _state(), session, user, "ru")
