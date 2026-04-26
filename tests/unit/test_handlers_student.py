"""Тесты handlers/student.py — pure helpers + командные хендлеры."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.student import (
    _history_keyboard,
    _shorten,
    on_ask,
    on_subject_command,
)
from src.db.models import AnswerMode, Base, Dialog, Subject, User, UserRole


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


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


# ---------- _shorten ----------


def test_shorten_keeps_short_text() -> None:
    assert _shorten("привет", 100) == "привет"


def test_shorten_truncates_long_with_ellipsis() -> None:
    out = _shorten("a" * 50, 10)
    assert out.endswith("…")
    assert len(out) == 10


def test_shorten_replaces_newlines_with_spaces() -> None:
    assert _shorten("line1\nline2", 100) == "line1 line2"


# ---------- on_ask ----------


@pytest.mark.asyncio
async def test_on_ask_sets_state_and_shows_samples() -> None:
    m = _msg()
    state = MagicMock()
    state.set_state = AsyncMock()
    await on_ask(m, state, "ru")
    state.set_state.assert_awaited()
    m.answer.assert_awaited()


# ---------- on_subject_command ----------


@pytest.mark.asyncio
async def test_subject_command_no_arg_no_pinned_says_cleared(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    user = _user()
    session.add(user)
    await session.flush()
    await on_subject_command(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "снят" in body or "active again" in body or "unpinned" in body


@pytest.mark.asyncio
async def test_subject_command_clear_resets(session) -> None:
    m = _msg()
    user = _user()
    user.current_subject_slug = "theory_of_systems"
    session.add(user)
    await session.flush()
    cmd = SimpleNamespace(args="clear")
    await on_subject_command(m, cmd, session, user, "ru")
    assert user.current_subject_slug is None


@pytest.mark.asyncio
async def test_subject_command_unknown_slug(session) -> None:
    m = _msg()
    user = _user()
    session.add(user)
    await session.flush()
    cmd = SimpleNamespace(args="ghost")
    await on_subject_command(m, cmd, session, user, "ru")
    body = m.answer.call_args.args[0]
    assert "Неизвестный" in body or "Unknown" in body


@pytest.mark.asyncio
async def test_subject_command_locks_subject(session) -> None:
    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")
    session.add(subj)
    user = _user()
    session.add(user)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems")
    await on_subject_command(m, cmd, session, user, "ru")
    assert user.current_subject_slug == "theory_of_systems"


# ---------- _history_keyboard ----------


def _dialog(uid: int, dialog_id: int, fav: bool = False) -> Dialog:
    d = Dialog(user_id=uid)
    d.id = dialog_id
    d.question = "?"
    d.answer = "."
    d.is_favourite = fav
    return d


def test_history_keyboard_pagination_buttons() -> None:
    dialogs = [_dialog(1, i) for i in range(5)]
    kb = _history_keyboard(dialogs, page=1, total=20, favourites_only=False, search=None)
    cbs = {b.callback_data for row in kb.inline_keyboard for b in row if b.callback_data}
    # Должна быть «следующая» (есть ещё страницы) и «предыдущая».
    assert any(c.startswith("hst:") for c in cbs)


def test_history_keyboard_no_prev_on_first_page() -> None:
    dialogs = [_dialog(1, i) for i in range(2)]
    kb = _history_keyboard(dialogs, page=0, total=2, favourites_only=False, search=None)
    cbs = {b.callback_data for row in kb.inline_keyboard for b in row if b.callback_data}
    # Первый page=0 — нет hst:prev.
    assert not any(c == "hst:0:0" for c in cbs)


def test_history_keyboard_star_toggles() -> None:
    dialogs = [_dialog(1, 42, fav=False), _dialog(1, 43, fav=True)]
    kb = _history_keyboard(dialogs, page=0, total=2, favourites_only=False, search=None)
    cbs = {b.callback_data for row in kb.inline_keyboard for b in row if b.callback_data}
    # callback_data: star:<dialog_id>:<target>:<page>:<fav_flag>
    assert any(c.startswith("star:42:1") for c in cbs)
    assert any(c.startswith("star:43:0") for c in cbs)
