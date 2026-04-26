"""Тесты синхронизации courses.yaml ↔ subjects-таблица."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.subjects_sync import sync_subjects
from src.db.models import Base
from src.db.models import Subject as SubjectRow
from src.subjects import Catalog
from src.subjects.schema import Subject as SubjectCfg


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _catalog(*subjects: SubjectCfg) -> Catalog:
    return Catalog(list(subjects))


@pytest.mark.asyncio
async def test_sync_inserts_new_subjects(session: AsyncSession) -> None:
    cat = _catalog(
        SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"),
        SubjectCfg(slug="systems_engineering", title_en="SE", title_ru="Системная инженерия"),
    )
    await sync_subjects(session, cat)
    rows = (await session.execute(select(SubjectRow))).scalars().all()
    assert {r.slug for r in rows} == {"theory_of_systems", "systems_engineering"}


@pytest.mark.asyncio
async def test_sync_updates_titles(session: AsyncSession) -> None:
    session.add(SubjectRow(slug="theory_of_systems", title_en="OLD", title_ru="OLD"))
    await session.flush()
    cat = _catalog(SubjectCfg(slug="theory_of_systems", title_en="NEW EN", title_ru="NEW RU"))
    await sync_subjects(session, cat)
    row = (
        await session.execute(select(SubjectRow).where(SubjectRow.slug == "theory_of_systems"))
    ).scalar_one()
    assert row.title_en == "NEW EN"
    assert row.title_ru == "NEW RU"


@pytest.mark.asyncio
async def test_sync_deactivates_missing(session: AsyncSession) -> None:
    session.add(SubjectRow(slug="legacy_course", title_en="L", title_ru="L", is_active=True))
    await session.flush()
    cat = _catalog(SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await sync_subjects(session, cat)
    row = (
        await session.execute(select(SubjectRow).where(SubjectRow.slug == "legacy_course"))
    ).scalar_one()
    assert row.is_active is False
    # FK-исторические записи сохранились — строка не удалена.
    assert row.id is not None
