"""Tests for the teacher-curated FAQ short-circuit layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.faq_service import (
    format_faq_body,
    lookup_faq,
    pending_reviews,
    save_faq_from_dialog,
)
from src.db.models import Base, Dialog, FAQEntry, Feedback, Subject, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _user(tg: int = 1, role: UserRole = UserRole.student) -> User:
    u = User(telegram_id=tg, full_name="U")
    u.role = role
    return u


def _subject(slug: str = "tos") -> Subject:
    return Subject(slug=slug, title_en="Theory of Systems", title_ru="Теория систем")


def _dialog(user_id: int, subject_id: int | None, question: str, answer: str = "a") -> Dialog:
    d = Dialog(user_id=user_id, subject_id=subject_id)
    d.question = question
    d.answer = answer
    d.latency_ms = 100
    return d


# ---------- pending_reviews ----------


@pytest.mark.asyncio
async def test_pending_reviews_lists_downvoted(session: AsyncSession) -> None:
    user = _user()
    subj = _subject()
    session.add_all([user, subj])
    await session.flush()

    good = _dialog(user.id, subj.id, "хороший вопрос")
    bad = _dialog(user.id, subj.id, "плохой вопрос")
    session.add_all([good, bad])
    await session.flush()
    session.add(Feedback(dialog_id=good.id, rating=5))
    session.add(Feedback(dialog_id=bad.id, rating=1))
    await session.commit()

    result = await pending_reviews(session)
    assert len(result) == 1
    assert result[0].dialog_id == bad.id
    assert result[0].rating == 1
    assert result[0].subject_slug == "tos"


@pytest.mark.asyncio
async def test_pending_reviews_excludes_already_fixed(session: AsyncSession) -> None:
    user = _user()
    teacher = _user(tg=999, role=UserRole.teacher)
    subj = _subject()
    session.add_all([user, teacher, subj])
    await session.flush()

    d = _dialog(user.id, subj.id, "что такое эмерджентность")
    session.add(d)
    await session.flush()
    session.add(Feedback(dialog_id=d.id, rating=1))
    # Pre-populate a FAQ under the same normalised question.
    session.add(
        FAQEntry(
            subject_id=subj.id,
            question_normalised="эмерджентность",
            question_original="что такое эмерджентность",
            answer="teacher answer",
            parent_dialog_id=d.id,
            created_by_user_id=teacher.id,
        )
    )
    await session.commit()

    result = await pending_reviews(session)
    assert result == []


@pytest.mark.asyncio
async def test_pending_reviews_ordered_oldest_first(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()

    d1 = _dialog(user.id, None, "q1")
    d2 = _dialog(user.id, None, "q2")
    session.add_all([d1, d2])
    await session.flush()
    now = datetime.now(UTC)
    f1 = Feedback(dialog_id=d1.id, rating=1)
    f1.created_at = now - timedelta(hours=5)
    f2 = Feedback(dialog_id=d2.id, rating=2)
    f2.created_at = now - timedelta(hours=1)
    session.add_all([f1, f2])
    await session.commit()

    result = await pending_reviews(session)
    assert [r.dialog_id for r in result] == [d1.id, d2.id]


# ---------- save_faq_from_dialog ----------


@pytest.mark.asyncio
async def test_save_faq_creates_entry_from_dialog(session: AsyncSession) -> None:
    user = _user()
    teacher = _user(tg=999, role=UserRole.teacher)
    subj = _subject()
    session.add_all([user, teacher, subj])
    await session.flush()

    d = _dialog(user.id, subj.id, "что такое эмерджентность")
    session.add(d)
    await session.flush()

    entry = await save_faq_from_dialog(
        session, dialog_id=d.id, answer="Это свойство системы.", teacher=teacher
    )
    assert entry is not None
    assert entry.question_normalised == "эмерджентность"
    assert entry.answer == "Это свойство системы."
    assert entry.parent_dialog_id == d.id
    assert entry.created_by_user_id == teacher.id
    assert entry.subject_id == subj.id


@pytest.mark.asyncio
async def test_save_faq_upserts_on_repeat(session: AsyncSession) -> None:
    user = _user()
    teacher = _user(tg=999, role=UserRole.teacher)
    subj = _subject()
    session.add_all([user, teacher, subj])
    await session.flush()

    d = _dialog(user.id, subj.id, "что такое система")
    session.add(d)
    await session.flush()

    e1 = await save_faq_from_dialog(
        session, dialog_id=d.id, answer="первая", teacher=teacher
    )
    e2 = await save_faq_from_dialog(
        session, dialog_id=d.id, answer="вторая", teacher=teacher
    )
    assert e1 is not None and e2 is not None
    assert e1.id == e2.id  # same row
    assert e2.answer == "вторая"


@pytest.mark.asyncio
async def test_save_faq_returns_none_for_missing_dialog(session: AsyncSession) -> None:
    teacher = _user(tg=999, role=UserRole.teacher)
    session.add(teacher)
    await session.flush()
    entry = await save_faq_from_dialog(
        session, dialog_id=9999, answer="x", teacher=teacher
    )
    assert entry is None


# ---------- lookup_faq ----------


@pytest.mark.asyncio
async def test_lookup_finds_subject_scoped_faq(session: AsyncSession) -> None:
    teacher = _user(tg=999, role=UserRole.teacher)
    subj = _subject()
    session.add_all([teacher, subj])
    await session.flush()

    session.add(
        FAQEntry(
            subject_id=subj.id,
            question_normalised="эмерджентность",
            question_original="что такое эмерджентность",
            answer="teacher's RU answer",
            created_by_user_id=teacher.id,
        )
    )
    await session.commit()

    hit = await lookup_faq(
        session, question="Что такое эмерджентность?", subject_id=subj.id
    )
    assert hit is not None
    assert hit.answer == "teacher's RU answer"


@pytest.mark.asyncio
async def test_lookup_miss_returns_none(session: AsyncSession) -> None:
    hit = await lookup_faq(session, question="нечто неизвестное", subject_id=None)
    assert hit is None


@pytest.mark.asyncio
async def test_lookup_subject_scoped_wins_over_global(session: AsyncSession) -> None:
    teacher = _user(tg=999, role=UserRole.teacher)
    subj = _subject()
    session.add_all([teacher, subj])
    await session.flush()

    session.add(
        FAQEntry(
            subject_id=None,
            question_normalised="эмерджентность",
            question_original="что такое эмерджентность",
            answer="GLOBAL",
            created_by_user_id=teacher.id,
        )
    )
    session.add(
        FAQEntry(
            subject_id=subj.id,
            question_normalised="эмерджентность",
            question_original="что такое эмерджентность",
            answer="SUBJECT",
            created_by_user_id=teacher.id,
        )
    )
    await session.commit()

    hit = await lookup_faq(
        session, question="что такое эмерджентность", subject_id=subj.id
    )
    assert hit is not None and hit.answer == "SUBJECT"


# ---------- format_faq_body ----------


def test_format_faq_body_prefixes_ru_and_en() -> None:
    entry = FAQEntry()
    entry.answer = "Текст ответа"
    ru = format_faq_body(entry, "ru")
    en = format_faq_body(entry, "en")
    assert "Ответ преподавателя" in ru
    assert "Teacher's answer" in en
    assert "Текст ответа" in ru and "Текст ответа" in en
