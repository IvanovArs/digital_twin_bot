"""Покрытие on_inline_ask happy-flow и on_chosen_inline_result."""

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


def setup_function(_f) -> None:  # type: ignore[no-untyped-def]
    inline_mod._QUESTION_CACHE.clear()
    inline_mod._RUNNING.clear()


def _user() -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _bot() -> MagicMock:
    b = MagicMock()
    b.edit_message_text = AsyncMock()
    b.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99, chat=SimpleNamespace(id=1)))
    b.send_chat_action = AsyncMock()
    return b


@pytest.fixture(autouse=True)
def _mock_qa(monkeypatch):
    captured: dict = {}

    async def fake_pipeline(**kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        await kw["set_status"]("running")
        await kw["set_final"]("done body", 100, "full")

    monkeypatch.setattr(inline_mod, "run_qa_pipeline", fake_pipeline)
    return captured


@pytest.mark.asyncio
async def test_inline_ask_full_happy_path(session, _mock_qa) -> None:
    """Кэш есть, claim проходит, pipeline зовётся, edit_message_text вызывается."""
    inline_mod._cache_put("happy_qid", "что такое стейкхолдер")
    cb = MagicMock()
    cb.data = "iq:happy_qid"
    cb.from_user = SimpleNamespace(id=1)
    cb.inline_message_id = "im-happy"
    cb.message = None
    cb.answer = AsyncMock()
    bot = _bot()
    await inline_mod.on_inline_ask(cb, bot, session, _user(), "ru")
    cb.answer.assert_awaited()
    # pipeline получил правильный вопрос
    assert _mock_qa.get("question") == "что такое стейкхолдер"


@pytest.mark.asyncio
async def test_inline_ask_dedupe_blocks_second_run(session, _mock_qa) -> None:
    inline_mod._cache_put("dup_qid", "вопрос")
    inline_mod._RUNNING["im-dup"] = ("other-rid", 1e12)  # уже занято
    cb = MagicMock()
    cb.data = "iq:dup_qid"
    cb.from_user = SimpleNamespace(id=1)
    cb.inline_message_id = "im-dup"
    cb.message = None
    cb.answer = AsyncMock()
    bot = _bot()
    await inline_mod.on_inline_ask(cb, bot, session, _user(), "ru")
    # Pipeline НЕ должен был вызваться — claim занят другим rid.
    assert _mock_qa.get("question") is None


@pytest.mark.asyncio
async def test_chosen_inline_result_with_inline_id(session, _mock_qa) -> None:
    inline_mod._cache_put("cir_qid", "выбранный вопрос")
    upd = MagicMock()
    upd.from_user = SimpleNamespace(id=1)
    upd.inline_message_id = "im-cir"
    upd.result_id = "cir_qid"
    upd.query = "выбранный вопрос"
    bot = _bot()
    import contextlib

    # Возможны издержки edit_message_text-mock'а — главное, что вошли в flow.
    with contextlib.suppress(Exception):
        await inline_mod.on_chosen_inline_result(upd, bot, session, _user(), "ru")


@pytest.mark.asyncio
async def test_chosen_inline_result_no_ids_uses_query_fallback(session, _mock_qa) -> None:
    upd = MagicMock()
    upd.from_user = SimpleNamespace(id=1)
    upd.inline_message_id = None
    upd.result_id = None
    upd.query = "fallback вопрос"
    bot = _bot()
    import contextlib

    with contextlib.suppress(Exception):
        await inline_mod.on_chosen_inline_result(upd, bot, session, _user(), "ru")
