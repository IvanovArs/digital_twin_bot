"""Покрытие оставшихся happy-path: on_ref, on_followup, on_expand_answer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student import entry as entry_mod
from src.bot.handlers.student import followup as fu_mod
from src.bot.handlers.student.followup import on_expand_answer, on_followup
from src.bot.handlers.student.quick_lookup import on_ref
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


# ---------- on_ref happy paths ----------


@pytest.mark.asyncio
async def test_ref_no_sources(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    d.sources = None
    session.add(d)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    from types import SimpleNamespace

    cmd = SimpleNamespace(args=str(d.id))
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "глоссария" in body or "FAQ" in body


@pytest.mark.asyncio
async def test_ref_textbook_sources_render(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "ответ"
    d.sources = [
        {"book": "a.pdf", "page": 12, "score": 0.85},
        {"book": "b.pdf", "page": 5, "score": 0.72},
    ]
    session.add(d)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    from types import SimpleNamespace

    cmd = SimpleNamespace(args=str(d.id))
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "a.pdf" in body and "b.pdf" in body
    assert "стр. 12" in body


@pytest.mark.asyncio
async def test_ref_web_sources_render(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    d.sources = [{"type": "web", "title": "Wikipedia", "url": "https://en.wikipedia.org/x"}]
    session.add(d)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    from types import SimpleNamespace

    cmd = SimpleNamespace(args=str(d.id))
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Wikipedia" in body
    assert "wikipedia.org" in body


@pytest.mark.asyncio
async def test_ref_ignores_garbage_source_rows(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    d.sources = ["plain string", 42, {"book": "ok.pdf", "page": 1}]
    session.add(d)
    await session.flush()
    m = MagicMock()
    m.answer = AsyncMock()
    from types import SimpleNamespace

    cmd = SimpleNamespace(args=str(d.id))
    await on_ref(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "ok.pdf" in body


# ---------- on_followup happy path ----------


@pytest.mark.asyncio
async def test_followup_invokes_pipeline(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()

    cb = MagicMock()
    cb.data = f"fu:{d.id}:simplify"
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.bot = MagicMock()
    cb.message.chat = MagicMock()
    cb.message.chat.id = 1
    placeholder = MagicMock()
    placeholder.text = "..."
    placeholder.edit_text = AsyncMock()
    cb.message.answer = AsyncMock(return_value=placeholder)

    captured: dict = {}

    async def fake_followup(**kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        await kw["set_status"]("работаю")
        await kw["set_final"]("готовый ответ", 99, "full")
        return True

    monkeypatch.setattr(fu_mod, "run_followup_pipeline", fake_followup)
    await on_followup(cb, session, user, "ru")
    assert captured.get("modifier") == "simplify"
    assert captured.get("dialog_id") == d.id


@pytest.mark.asyncio
async def test_followup_returns_expired_message(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()

    cb = MagicMock()
    cb.data = f"fu:{d.id}:example"
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.bot = MagicMock()
    cb.message.chat = MagicMock()
    cb.message.chat.id = 1
    cb.message.text = "старый ответ бота"
    cb.message.edit_text = AsyncMock()

    async def fake_followup(**kw):  # type: ignore[no-untyped-def]
        return False  # context expired

    monkeypatch.setattr(fu_mod, "run_followup_pipeline", fake_followup)
    await on_followup(cb, session, user, "ru")
    # Теперь follow-up edit-in-place — пишем на ту же cb.message, не на отдельный
    # placeholder.
    cb.message.edit_text.assert_awaited()


# ---------- on_expand_answer happy path ----------


@pytest.mark.asyncio
async def test_expand_runs_full_pipeline(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "что такое X"
    d.answer = "..."
    session.add(d)
    await session.flush()

    cb = MagicMock()
    cb.data = f"expand:{d.id}"
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.bot = MagicMock()
    cb.message.text = "old"
    cb.message.edit_text = AsyncMock()

    captured: dict = {}

    async def fake_pipeline(**kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        await kw["set_status"]("ищу")
        await kw["set_final"]("полный ответ", 100, "full")

    monkeypatch.setattr(fu_mod, "run_qa_pipeline", fake_pipeline)
    await on_expand_answer(cb, session, user, "ru")
    assert captured.get("question") == "что такое X"
    assert captured.get("skip_short_circuit") is True


@pytest.mark.asyncio
async def test_expand_empty_question_returns_silently(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "   "
    d.answer = "."
    session.add(d)
    await session.flush()

    cb = MagicMock()
    cb.data = f"expand:{d.id}"
    cb.answer = AsyncMock()
    cb.message = MagicMock()

    called = {"hit": False}

    async def fake_pipeline(**kw):  # type: ignore[no-untyped-def]
        called["hit"] = True

    monkeypatch.setattr(fu_mod, "run_qa_pipeline", fake_pipeline)
    await on_expand_answer(cb, session, user, "ru")
    assert called["hit"] is False


# ---------- on_ask_sample ----------


@pytest.mark.asyncio
async def test_ask_sample_invalid_idx(session, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    cb = MagicMock()
    cb.data = "ask_sample:999"
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.answer = AsyncMock()
    cb.from_user = MagicMock()

    state = MagicMock()
    state.set_state = AsyncMock()
    await entry_mod.on_ask_sample(cb, state, session, user, "ru")
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_ask_sample_no_message_returns(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    cb = MagicMock()
    cb.data = "ask_sample:0"
    cb.answer = AsyncMock()
    cb.message = None
    state = MagicMock()
    await entry_mod.on_ask_sample(cb, state, session, user, "ru")
    cb.answer.assert_awaited()
