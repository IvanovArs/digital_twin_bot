"""Tests for the /term, /ref, /export quick-reference commands."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.dialog_service import user_dialogs_page
from src.bot.services.glossary_upload import (
    format_glossary_body,
    lookup_term,
)
from src.db.models import Base, Dialog, GlossaryTerm, Subject, User


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ---------- /term: relies on lookup_term (covered elsewhere) + formatting ----------


@pytest.mark.asyncio
async def test_term_lookup_and_format(session: AsyncSession) -> None:
    subj = Subject(slug="tos", title_en="Theory of Systems", title_ru="Теория систем")
    session.add(subj)
    await session.flush()
    session.add(
        GlossaryTerm(
            subject_id=subj.id,
            term="Подсистема",
            definition="Часть системы, рассматриваемая как целое.",
        )
    )
    await session.commit()

    hit = await lookup_term(session, question="Что такое подсистема?", subject_id=subj.id)
    assert hit is not None
    body = format_glossary_body(hit, "ru")
    assert "Подсистема" in body
    assert "Часть системы" in body


# ---------- /ref: stored sources round-trip through Dialog.sources ----------


@pytest.mark.asyncio
async def test_dialog_sources_roundtrip_textbook(session: AsyncSession) -> None:
    user = User(telegram_id=1, full_name="U")
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "q"
    d.answer = "a"
    d.sources = [
        {"book": "a.pdf", "page": 12, "subject": "tos", "score": 0.87},
        {"book": "b.pdf", "page": 3, "subject": "tos", "score": 0.71},
    ]
    session.add(d)
    await session.commit()

    from sqlalchemy import select

    from src.db.models import Dialog as DRow

    got = (
        await session.execute(select(DRow).where(DRow.id == d.id))
    ).scalar_one()
    assert got.sources is not None
    assert len(got.sources) == 2
    first = got.sources[0]
    assert first["book"] == "a.pdf"
    assert first["page"] == 12
    assert abs(first["score"] - 0.87) < 1e-9


@pytest.mark.asyncio
async def test_dialog_sources_roundtrip_web(session: AsyncSession) -> None:
    user = User(telegram_id=1, full_name="U")
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "q"
    d.answer = "a"
    d.sources = [
        {"type": "web", "title": "Wikipedia", "url": "https://wiki/foo", "snippet": "s"}
    ]
    session.add(d)
    await session.commit()
    # Retrieve and verify structure the handler depends on.
    from sqlalchemy import select

    from src.db.models import Dialog as DRow

    got = (
        await session.execute(select(DRow).where(DRow.id == d.id))
    ).scalar_one()
    assert got.sources is not None
    row = got.sources[0]
    assert row["type"] == "web"
    assert row["title"] == "Wikipedia"
    assert row["url"].startswith("https://")


# ---------- /export: user_dialogs_page covers lots; here we confirm large-limit path ----------


@pytest.mark.asyncio
async def test_export_fetches_all_when_limit_high(session: AsyncSession) -> None:
    """Export passes limit=10_000 so the user sees their full history in
    one file. Make sure that actually returns every row, not a silent cap."""
    user = User(telegram_id=1, full_name="U")
    session.add(user)
    await session.flush()
    for i in range(25):
        d = Dialog(user_id=user.id)
        d.question = f"q{i}"
        d.answer = "a"
        session.add(d)
    await session.commit()

    rows, total = await user_dialogs_page(
        session, user_id=user.id, offset=0, limit=10_000
    )
    assert len(rows) == 25
    assert total == 25


@pytest.mark.asyncio
async def test_export_respects_ownership(session: AsyncSession) -> None:
    """A call with user_id=me should never return another user's dialogs."""
    me = User(telegram_id=1, full_name="me")
    you = User(telegram_id=2, full_name="you")
    session.add_all([me, you])
    await session.flush()
    mine = Dialog(user_id=me.id)
    mine.question = "mine"
    mine.answer = "a"
    yours = Dialog(user_id=you.id)
    yours.question = "yours"
    yours.answer = "a"
    session.add_all([mine, yours])
    await session.commit()

    rows, total = await user_dialogs_page(session, user_id=me.id, limit=100)
    assert total == 1
    assert rows[0].question == "mine"
