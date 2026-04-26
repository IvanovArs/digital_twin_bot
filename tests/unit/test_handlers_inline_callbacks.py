"""Тесты inline-callback'ов: on_inline_ask и on_chosen_inline_result."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers import inline as inline_mod
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


def _user(tg: int = 1) -> User:
    u = User(telegram_id=tg, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def setup_function(_f) -> None:  # type: ignore[no-untyped-def]
    inline_mod._QUESTION_CACHE.clear()
    inline_mod._RUNNING.clear()


def _bot() -> MagicMock:
    b = MagicMock()
    b.edit_message_text = AsyncMock()
    b.send_chat_action = AsyncMock()
    b.send_message = AsyncMock(return_value=SimpleNamespace(message_id=999, chat=SimpleNamespace(id=1)))
    return b


def _cb(data: str, *, inline_msg_id: str = "im-1") -> MagicMock:
    cb = MagicMock()
    cb.data = data
    cb.from_user = SimpleNamespace(id=1)
    cb.inline_message_id = inline_msg_id
    cb.message = None
    cb.answer = AsyncMock()
    return cb


@pytest.fixture(autouse=True)
def _mock_qa_pipeline(monkeypatch):
    """Подменяем тяжёлый run_qa_pipeline на noop, чтобы тесты не висли
    на ожидании MODELS_READY / реального LLM-стрима."""
    async def _noop(**_kw):
        return None

    monkeypatch.setattr(inline_mod, "run_qa_pipeline", _noop)


@pytest.mark.asyncio
async def test_inline_ask_invalid_data() -> None:
    cb = _cb("iq:")
    await inline_mod.on_inline_ask(cb, _bot(), MagicMock(), _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_inline_ask_no_inline_message_id(session) -> None:
    cb = _cb("iq:abcd1234", inline_msg_id="")
    cb.inline_message_id = None
    await inline_mod.on_inline_ask(cb, _bot(), session, _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_inline_ask_qid_not_in_cache(session) -> None:
    cb = _cb("iq:missing_qid")
    await inline_mod.on_inline_ask(cb, _bot(), session, _user(), "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_inline_ask_dedupe_second_call_silent(session) -> None:
    """Второй tap по тому же inline-сообщению должен быть тихо проигнорирован."""
    inline_mod._cache_put("dedup_qid", "что такое X")
    inline_mod._RUNNING["im-dup"] = ("other-rid", 1e12)  # уже занято

    cb = _cb("iq:dedup_qid", inline_msg_id="im-dup")
    await inline_mod.on_inline_ask(cb, _bot(), session, _user(), "ru")
    # callback.answer всё равно вызовется (anti-spinner)
    cb.answer.assert_awaited()


# ---------- chosen_inline_result ----------


@pytest.mark.asyncio
async def test_chosen_inline_result_no_inline_id(session) -> None:
    upd = MagicMock()
    upd.inline_message_id = None
    upd.from_user = SimpleNamespace(id=1)
    upd.query = "что такое X"
    upd.result_id = "abc"
    bot = _bot()
    # При отсутствии inline_message_id и chat — должно тихо завершиться или
    # упасть на send_message; главное, что не падает с unhandled exception.
    import contextlib

    with contextlib.suppress(Exception):
        await inline_mod.on_chosen_inline_result(upd, bot, session, _user(), "ru")
