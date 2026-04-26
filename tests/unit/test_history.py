"""Tests for the paginated history + ⭐ favourite flag."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.dialog_service import set_favourite, user_dialogs_page
from src.db.models import Base, Dialog, Subject, User, UserRole


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
    u = User(telegram_id=tg, full_name="U")
    u.role = UserRole.student
    return u


def _dialog(user_id: int, q: str, *, at: datetime | None = None) -> Dialog:
    d = Dialog(user_id=user_id)
    d.question = q
    d.answer = "a"
    if at is not None:
        d.created_at = at
    return d


# ---------- user_dialogs_page ----------


@pytest.mark.asyncio
async def test_pagination_respects_offset_and_limit(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    base = datetime.now(UTC)
    for i in range(12):
        session.add(_dialog(user.id, f"q{i}", at=base - timedelta(minutes=i)))
    await session.commit()

    first, total = await user_dialogs_page(session, user_id=user.id, offset=0, limit=5)
    second, _ = await user_dialogs_page(session, user_id=user.id, offset=5, limit=5)
    third, _ = await user_dialogs_page(session, user_id=user.id, offset=10, limit=5)

    assert total == 12
    assert [d.question for d in first] == [f"q{i}" for i in range(0, 5)]
    assert [d.question for d in second] == [f"q{i}" for i in range(5, 10)]
    assert [d.question for d in third] == ["q10", "q11"]


@pytest.mark.asyncio
async def test_favourites_only_filters(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    regular = _dialog(user.id, "обычный")
    fav = _dialog(user.id, "любимый")
    fav.is_favourite = True
    session.add_all([regular, fav])
    await session.commit()

    rows, total = await user_dialogs_page(session, user_id=user.id, favourites_only=True)
    assert total == 1
    assert rows[0].question == "любимый"


@pytest.mark.asyncio
async def test_search_is_case_insensitive_substring(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    session.add(_dialog(user.id, "что такое Стейкхолдер"))
    session.add(_dialog(user.id, "что такое эмерджентность"))
    session.add(_dialog(user.id, "подсистема это?"))
    await session.commit()

    rows, total = await user_dialogs_page(session, user_id=user.id, search="стейкхолдер")
    assert total == 1
    assert rows[0].question == "что такое Стейкхолдер"


@pytest.mark.asyncio
async def test_search_combined_with_favourites(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d1 = _dialog(user.id, "что такое система")
    d1.is_favourite = True
    d2 = _dialog(user.id, "что такое система")  # not favourite
    d3 = _dialog(user.id, "про онтологию")
    d3.is_favourite = True
    session.add_all([d1, d2, d3])
    await session.commit()

    rows, total = await user_dialogs_page(
        session, user_id=user.id, favourites_only=True, search="система"
    )
    assert total == 1
    assert rows[0] is d1 or rows[0].id == d1.id


@pytest.mark.asyncio
async def test_pagination_orders_newest_first(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    now = datetime.now(UTC)
    session.add(_dialog(user.id, "old", at=now - timedelta(hours=10)))
    session.add(_dialog(user.id, "new", at=now))
    await session.commit()
    rows, _ = await user_dialogs_page(session, user_id=user.id, limit=5)
    assert [d.question for d in rows] == ["new", "old"]


# ---------- set_favourite ----------


@pytest.mark.asyncio
async def test_set_favourite_toggles_flag(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = _dialog(user.id, "q")
    session.add(d)
    await session.flush()

    ok = await set_favourite(session, dialog_id=d.id, user_id=user.id, is_favourite=True)
    assert ok is True
    assert d.is_favourite is True

    ok = await set_favourite(session, dialog_id=d.id, user_id=user.id, is_favourite=False)
    assert ok is True
    assert d.is_favourite is False


@pytest.mark.asyncio
async def test_set_favourite_refuses_foreign_dialog(session: AsyncSession) -> None:
    me = _user(tg=1)
    you = _user(tg=2)
    session.add_all([me, you])
    await session.flush()
    their_dialog = _dialog(you.id, "theirs")
    session.add(their_dialog)
    await session.flush()

    ok = await set_favourite(session, dialog_id=their_dialog.id, user_id=me.id, is_favourite=True)
    assert ok is False
    # Flag must not have been flipped.
    assert their_dialog.is_favourite is False


@pytest.mark.asyncio
async def test_pagination_attaches_subject_eagerly(session: AsyncSession) -> None:
    """``joinedload`` should populate ``Dialog.subject`` in one query so the
    handler can render titles without triggering lazy-load in sync context."""
    user = _user()
    subj = Subject(slug="tos", title_en="Theory of Systems", title_ru="Теория систем")
    session.add_all([user, subj])
    await session.flush()
    d = _dialog(user.id, "q")
    d.subject_id = subj.id
    session.add(d)
    await session.commit()

    rows, _ = await user_dialogs_page(session, user_id=user.id)
    assert rows[0].subject is not None
    assert rows[0].subject.slug == "tos"
