"""Тесты sync_glossary_from_yaml + list_terms / find_term."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.glossary_service import (
    find_term,
    list_terms,
    sync_glossary_from_yaml,
)
from src.db.models import Base, Subject


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_sync_inserts_terms_for_known_subject(
    session: AsyncSession, tmp_path: Path
) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"))
    await session.flush()

    (tmp_path / "theory_of_systems.yaml").write_text(
        "terms:\n"
        "  - term: Стейкхолдер\n"
        "    definition: Заинтересованное лицо\n"
        "  - term: Эмерджентность\n"
        "    definition: Свойство целого, которого нет у частей\n",
        encoding="utf-8",
    )
    await sync_glossary_from_yaml(session, tmp_path)
    terms = await list_terms(session, subject_slug="theory_of_systems")
    assert {t.term for t in terms} == {"Стейкхолдер", "Эмерджентность"}


@pytest.mark.asyncio
async def test_sync_skips_unknown_subject(session: AsyncSession, tmp_path: Path) -> None:
    (tmp_path / "ghost.yaml").write_text(
        "terms:\n  - term: X\n    definition: Y\n", encoding="utf-8"
    )
    await sync_glossary_from_yaml(session, tmp_path)
    assert await list_terms(session) == []


@pytest.mark.asyncio
async def test_sync_updates_existing_term_definition(
    session: AsyncSession, tmp_path: Path
) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"))
    await session.flush()
    (tmp_path / "theory_of_systems.yaml").write_text(
        "terms:\n  - term: Стейкхолдер\n    definition: V1\n", encoding="utf-8"
    )
    await sync_glossary_from_yaml(session, tmp_path)
    (tmp_path / "theory_of_systems.yaml").write_text(
        "terms:\n  - term: Стейкхолдер\n    definition: V2\n", encoding="utf-8"
    )
    await sync_glossary_from_yaml(session, tmp_path)
    terms = await list_terms(session, subject_slug="theory_of_systems")
    assert len(terms) == 1
    assert terms[0].definition == "V2"


@pytest.mark.asyncio
async def test_find_term_substring_case_insensitive(
    session: AsyncSession, tmp_path: Path
) -> None:
    """SQLite ILIKE — ASCII-only, поэтому проверяем латиницей. Postgres
    в проде корректно работает с Unicode."""
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем"))
    await session.flush()
    (tmp_path / "theory_of_systems.yaml").write_text(
        "terms:\n"
        "  - term: Stakeholder\n    definition: X\n"
        "  - term: Emergence\n    definition: Y\n",
        encoding="utf-8",
    )
    await sync_glossary_from_yaml(session, tmp_path)
    out = await find_term(session, "stakeholder", subject_slug="theory_of_systems")
    assert len(out) == 1
    assert out[0].term == "Stakeholder"


@pytest.mark.asyncio
async def test_sync_handles_invalid_yaml(session: AsyncSession, tmp_path: Path) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    (tmp_path / "theory_of_systems.yaml").write_text("not: yaml: ::: {[", encoding="utf-8")
    # Не падать — только лог.
    await sync_glossary_from_yaml(session, tmp_path)
    assert await list_terms(session) == []
