"""Tests for teacher-uploaded glossary parsing + DB replace + term lookup."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.glossary_upload import (
    format_glossary_body,
    lookup_term,
    parse_glossary_payload,
    replace_glossary,
)
from src.db.models import Base, GlossaryTerm, Subject


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _subject(slug: str = "tos") -> Subject:
    return Subject(slug=slug, title_en="Theory of Systems", title_ru="Теория систем")


# ---------- parse_glossary_payload ----------


def test_parse_csv_with_header() -> None:
    payload = (
        "term,definition\n"
        "Стейкхолдер,Заинтересованное лицо\n"
        "Эмерджентность,Свойство системы\n"
    ).encode()
    rows = parse_glossary_payload(payload, "g.csv")
    assert rows == [
        ("Стейкхолдер", "Заинтересованное лицо"),
        ("Эмерджентность", "Свойство системы"),
    ]


def test_parse_csv_no_header() -> None:
    payload = ("Система,Совокупность элементов\n" "Подсистема,Часть системы\n").encode()
    rows = parse_glossary_payload(payload, "g.csv")
    assert rows == [
        ("Система", "Совокупность элементов"),
        ("Подсистема", "Часть системы"),
    ]


def test_parse_csv_semicolon_delimiter() -> None:
    payload = "Система;Совокупность элементов\n".encode()
    rows = parse_glossary_payload(payload, "g.csv")
    assert rows == [("Система", "Совокупность элементов")]


def test_parse_yaml() -> None:
    yaml_src = (
        "terms:\n"
        "  - term: Стейкхолдер\n"
        "    definition: Заинтересованное лицо\n"
        "  - term: Эмерджентность\n"
        "    definition: Свойство системы\n"
    )
    rows = parse_glossary_payload(yaml_src.encode("utf-8"), "g.yaml")
    assert len(rows) == 2
    assert rows[0] == ("Стейкхолдер", "Заинтересованное лицо")


def test_parse_yaml_without_terms_key_raises() -> None:
    with pytest.raises(ValueError, match="terms"):
        parse_glossary_payload(b"hello: world\n", "g.yaml")


def test_parse_unsupported_extension_raises() -> None:
    with pytest.raises(ValueError, match="Поддерживаются"):
        parse_glossary_payload(b"foo,bar\n", "g.pdf")


def test_parse_yaml_bad_syntax_raises() -> None:
    with pytest.raises(ValueError, match="YAML"):
        parse_glossary_payload(b"terms: [unbalanced\n", "g.yaml")


def test_parse_csv_handles_utf8_bom() -> None:
    # A BOM at the start crashes naive utf-8 decoders on Windows-exported
    # CSVs. utf-8-sig should swallow it transparently.
    payload = ("\ufefftermin,определение\nСистема,это\n").encode()
    rows = parse_glossary_payload(payload, "g.csv")
    # Header auto-detection with Russian column names ("термин"/"определение"
    # in lowercase) would need to match; here we use «termin» which isn't
    # a recognised header, so it becomes a data row.
    assert ("termin", "определение") in rows


# ---------- replace_glossary ----------


@pytest.mark.asyncio
async def test_replace_glossary_inserts_entries(session: AsyncSession) -> None:
    subj = _subject()
    session.add(subj)
    await session.flush()

    result = await replace_glossary(
        session,
        subject=subj,
        entries=[
            ("Стейкхолдер", "Заинтересованное лицо"),
            ("Система", "Совокупность"),
        ],
    )
    await session.commit()
    assert result.inserted == 2
    assert result.replaced_previous == 0

    rows = list(
        (
            await session.execute(select(GlossaryTerm).where(GlossaryTerm.subject_id == subj.id))
        ).scalars()
    )
    assert {r.term for r in rows} == {"Стейкхолдер", "Система"}


@pytest.mark.asyncio
async def test_replace_glossary_wipes_previous(session: AsyncSession) -> None:
    subj = _subject()
    session.add(subj)
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj.id, term="Старьё", definition="Ст."))
    await session.commit()

    result = await replace_glossary(
        session,
        subject=subj,
        entries=[("Новый", "Новое определение")],
    )
    await session.commit()
    assert result.replaced_previous == 1
    assert result.inserted == 1

    rows = list(
        (
            await session.execute(select(GlossaryTerm).where(GlossaryTerm.subject_id == subj.id))
        ).scalars()
    )
    assert [r.term for r in rows] == ["Новый"]


@pytest.mark.asyncio
async def test_replace_glossary_skips_empty_and_duplicates(session: AsyncSession) -> None:
    subj = _subject()
    session.add(subj)
    await session.flush()

    result = await replace_glossary(
        session,
        subject=subj,
        entries=[
            ("Term", "Def"),
            ("", "Пустой term"),
            ("Space term", ""),
            ("TERM", "дубль"),  # same as first (case-insensitive)
        ],
    )
    assert result.inserted == 1
    assert result.skipped_empty == 3


# ---------- lookup_term ----------


@pytest.mark.asyncio
async def test_lookup_matches_normalised_question(session: AsyncSession) -> None:
    subj = _subject()
    session.add(subj)
    await session.flush()
    session.add(
        GlossaryTerm(
            subject_id=subj.id,
            term="Стейкхолдер",
            definition="Заинтересованное лицо",
        )
    )
    await session.commit()

    hit = await lookup_term(session, question="Что такое стейкхолдер?", subject_id=subj.id)
    assert hit is not None
    assert hit.term == "Стейкхолдер"


@pytest.mark.asyncio
async def test_lookup_respects_subject_filter(session: AsyncSession) -> None:
    subj_a = _subject("a")
    subj_b = _subject("b")
    session.add_all([subj_a, subj_b])
    await session.flush()
    session.add(GlossaryTerm(subject_id=subj_a.id, term="система", definition="def A"))
    await session.commit()

    hit_a = await lookup_term(session, question="система", subject_id=subj_a.id)
    hit_b = await lookup_term(session, question="система", subject_id=subj_b.id)
    assert hit_a is not None and hit_a.definition == "def A"
    assert hit_b is None


@pytest.mark.asyncio
async def test_lookup_miss_returns_none(session: AsyncSession) -> None:
    subj = _subject()
    session.add(subj)
    await session.flush()
    hit = await lookup_term(session, question="неизвестное", subject_id=subj.id)
    assert hit is None


def test_format_glossary_body_contains_term_and_definition() -> None:
    entry = GlossaryTerm(subject_id=1, term="Система", definition="Совокупность элементов")
    ru = format_glossary_body(entry, "ru")
    en = format_glossary_body(entry, "en")
    assert "Система" in ru and "Совокупность элементов" in ru
    assert "Система" in en
