"""Тесты handlers/student/question.py: on_question (главный entry)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
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


def _user() -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _msg(text: str) -> MagicMock:
    m = MagicMock()
    m.text = text
    m.message_id = 7
    m.chat = MagicMock()
    m.chat.id = 100
    m.bot = MagicMock()
    m.bot.send_chat_action = AsyncMock()
    m.bot.set_message_reaction = AsyncMock()
    sent = MagicMock()
    sent.edit_text = AsyncMock()
    m.answer = AsyncMock(return_value=sent)
    return m


@pytest.fixture(autouse=True)
def _mock_pipeline(monkeypatch):
    """Подменяем тяжёлый run_qa_pipeline — нас интересует только UI-обвязка."""
    captured: dict = {}

    async def noop_pipeline(**kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        # Симулируем один edit-цикл, чтобы зацепить set_status / set_final.
        await kw["set_status"]("в процессе")
        await kw["set_final"]("готовый ответ", 42, "full")

    monkeypatch.setattr(q_mod, "run_qa_pipeline", noop_pipeline)
    return captured


@pytest.mark.asyncio
async def test_on_question_empty_text_prompts(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    state = MagicMock()
    state.set_state = AsyncMock()
    state.clear = AsyncMock()
    m = _msg("   ")
    await on_question(m, state, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Напиши" in body or "Type" in body


@pytest.mark.asyncio
async def test_on_question_runs_pipeline_and_sets_final(session, _mock_pipeline) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    state = MagicMock()
    state.set_state = AsyncMock()
    state.clear = AsyncMock()
    m = _msg("что такое стейкхолдер")
    await on_question(m, state, session, user, "ru")
    # FSM должна очиститься в finally.
    state.clear.assert_awaited()
    # Передали оригинальный вопрос внутрь pipeline'а.
    assert _mock_pipeline.get("question") == "что такое стейкхолдер"


@pytest.mark.asyncio
async def test_on_question_brief_mode_keyboard(session, _mock_pipeline) -> None:
    user = _user()
    user.answer_mode = AnswerMode.brief
    session.add(user)
    await session.flush()
    state = MagicMock()
    state.set_state = AsyncMock()
    state.clear = AsyncMock()
    m = _msg("вопрос")
    await on_question(m, state, session, user, "ru")
    state.clear.assert_awaited()
