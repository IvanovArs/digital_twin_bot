"""Тесты middlewares: access, lang, logging, session."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.bot.middlewares.access_mw import AccessMiddleware
from src.bot.middlewares.lang_mw import LangMiddleware
from src.bot.middlewares.logging_mw import LoggingMiddleware, _describe
from src.bot.middlewares.session_mw import SessionMiddleware
from src.db.models import Base

# ---------- LangMiddleware ----------


@pytest.mark.asyncio
async def test_lang_middleware_normalizes_ru() -> None:
    mw = LangMiddleware()
    captured: dict = {}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        captured.update(data)
        return "ok"

    tg = SimpleNamespace(language_code="ru-RU")
    res = await mw(handler, MagicMock(), {"event_from_user": tg})
    assert res == "ok"
    assert captured["lang"] == "ru"


@pytest.mark.asyncio
async def test_lang_middleware_no_user_falls_back() -> None:
    mw = LangMiddleware()
    captured: dict = {}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        captured.update(data)
        return None

    await mw(handler, MagicMock(), {})
    assert captured["lang"] == "ru"  # дефолт


# ---------- AccessMiddleware ----------


@pytest.mark.asyncio
async def test_access_middleware_open_when_allowlist_empty() -> None:
    mw = AccessMiddleware(set())

    async def handler(event, data):  # type: ignore[no-untyped-def]
        return "called"

    res = await mw(handler, MagicMock(), {"event_from_user": SimpleNamespace(id=1)})
    assert res == "called"


@pytest.mark.asyncio
async def test_access_middleware_passes_listed_user() -> None:
    mw = AccessMiddleware({42})

    async def handler(event, data):  # type: ignore[no-untyped-def]
        return "ok"

    res = await mw(handler, MagicMock(), {"event_from_user": SimpleNamespace(id=42, language_code="ru")})
    assert res == "ok"


@pytest.mark.asyncio
async def test_access_middleware_blocks_outsider_message() -> None:
    from aiogram.types import Message

    mw = AccessMiddleware({42})

    async def handler(event, data):  # type: ignore[no-untyped-def]
        return "should not be called"

    msg = MagicMock(spec=Message)
    msg.answer = AsyncMock()
    res = await mw(
        handler, msg, {"event_from_user": SimpleNamespace(id=99, language_code="ru")}
    )
    assert res is None
    msg.answer.assert_awaited()


@pytest.mark.asyncio
async def test_access_middleware_blocks_callback_query() -> None:
    from aiogram.types import CallbackQuery

    mw = AccessMiddleware({42})

    async def handler(event, data):  # type: ignore[no-untyped-def]
        return None

    cb = MagicMock(spec=CallbackQuery)
    cb.answer = AsyncMock()
    await mw(
        handler, cb, {"event_from_user": SimpleNamespace(id=99, language_code="en")}
    )
    cb.answer.assert_awaited()


@pytest.mark.asyncio
async def test_access_middleware_blocks_inline_query() -> None:
    from aiogram.types import InlineQuery

    mw = AccessMiddleware({42})

    async def handler(event, data):  # type: ignore[no-untyped-def]
        return None

    iq = MagicMock(spec=InlineQuery)
    iq.answer = AsyncMock()
    await mw(handler, iq, {"event_from_user": SimpleNamespace(id=99, language_code="ru")})
    iq.answer.assert_awaited_with(results=[], cache_time=1, is_personal=True)


# ---------- LoggingMiddleware ----------


@pytest.mark.asyncio
async def test_logging_middleware_passes_through_and_logs() -> None:
    mw = LoggingMiddleware()
    called = {"hit": False}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        called["hit"] = True
        return "x"

    res = await mw(handler, MagicMock(), {"event_from_user": SimpleNamespace(id=1)})
    assert called["hit"] is True
    assert res == "x"


@pytest.mark.asyncio
async def test_logging_middleware_propagates_exception() -> None:
    mw = LoggingMiddleware()

    async def handler(event, data):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await mw(handler, MagicMock(), {})


def test_describe_unknown_event_type() -> None:
    info = _describe(MagicMock())
    assert "event_type" in info


def test_describe_update_with_message() -> None:
    from aiogram.types import Chat, Message, Update

    msg = MagicMock(spec=Message)
    msg.text = "что такое X"
    msg.via_bot = None
    msg.chat = MagicMock(spec=Chat)
    msg.chat.type = "private"
    upd = MagicMock(spec=Update)
    upd.message = msg
    upd.edited_message = None
    upd.channel_post = None
    upd.callback_query = None
    upd.inline_query = None
    upd.chosen_inline_result = None
    info = _describe(upd)
    assert info["inner_type"] == "Message" or "MagicMock" in info["inner_type"]


# ---------- SessionMiddleware ----------


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.mark.asyncio
async def test_session_middleware_injects_session(sessionmaker, monkeypatch) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    mw = SessionMiddleware(sessionmaker)
    captured: dict = {}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        captured.update(data)
        return "ok"

    tg = SimpleNamespace(id=42, full_name="Test", language_code="ru")
    res = await mw(handler, MagicMock(), {"event_from_user": tg})
    assert res == "ok"
    assert "session" in captured
    assert "user" in captured
    assert captured["user"].telegram_id == 42


@pytest.mark.asyncio
async def test_session_middleware_rolls_back_on_exception(sessionmaker, monkeypatch) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    mw = SessionMiddleware(sessionmaker)

    async def handler(event, data):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    tg = SimpleNamespace(id=43, full_name="X", language_code="ru")
    with pytest.raises(RuntimeError):
        await mw(handler, MagicMock(), {"event_from_user": tg})
