"""Tests for the teacher-stats + coverage-gaps service.

We spin up an in-memory SQLite async engine, create the schema via the
existing declarative metadata, insert hand-crafted Dialog / Feedback rows,
and check the aggregation is correct. This is an integration test by
pytest's definition, but it's bounded (no network, no LLM) and fast
(<1 s), so it lives next to the unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.teacher_stats import (
    _normalise_question,
    coverage_gaps,
    subject_stats,
)
from src.db.models import Base, Dialog, Feedback, Subject, User, UserRole

# ---------- fixtures ----------


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _fresh_user(telegram_id: int = 1, current_subject: str | None = None) -> User:
    u = User(telegram_id=telegram_id, full_name="Stud Ent")
    u.current_subject_slug = current_subject
    u.role = UserRole.student
    return u


def _fresh_subject(slug: str = "tos", title: str = "Теория систем") -> Subject:
    s = Subject(slug=slug, title_en="Theory of Systems", title_ru=title)
    return s


def _fresh_dialog(
    user_id: int,
    subject_id: int | None,
    question: str,
    *,
    latency_ms: int | None = 1000,
    answer: str = "ответ",
    created_at: datetime | None = None,
) -> Dialog:
    d = Dialog(user_id=user_id, subject_id=subject_id)
    d.question = question
    d.answer = answer
    d.latency_ms = latency_ms
    if created_at is not None:
        d.created_at = created_at
    return d


# ---------- _normalise_question ----------


def test_normalise_strips_followup_tag() -> None:
    assert _normalise_question("[simplify] что такое система") == "система"
    assert _normalise_question("[EXAMPLE] система") == "система"


def test_normalise_collapses_leading_phrases() -> None:
    assert _normalise_question("Что такое система?") == "система"
    assert _normalise_question("расскажи про онтологию") == "онтологию"


def test_normalise_keeps_middle_word_variants_together() -> None:
    a = _normalise_question("что такое эмерджентность?")
    b = _normalise_question("Расскажи про эмерджентность.")
    c = _normalise_question("ЭМЕРДЖЕНТНОСТЬ")
    assert a == b == c == "эмерджентность"


# ---------- subject_stats ----------


@pytest.mark.asyncio
async def test_subject_stats_returns_none_for_unknown_slug(session: AsyncSession) -> None:
    assert await subject_stats(session, subject_slug="nope") is None


@pytest.mark.asyncio
async def test_subject_stats_counts_dialogs_and_feedback(session: AsyncSession) -> None:
    user = _fresh_user()
    subj = _fresh_subject()
    session.add_all([user, subj])
    await session.flush()

    # 3 dialogs inside window: two rated up (5, 4), one rated down (2).
    d1 = _fresh_dialog(user.id, subj.id, "что такое система", latency_ms=500)
    d2 = _fresh_dialog(user.id, subj.id, "что есть система?", latency_ms=1200)
    d3 = _fresh_dialog(user.id, subj.id, "что такое онтология", latency_ms=3000)
    session.add_all([d1, d2, d3])
    await session.flush()
    session.add(Feedback(dialog_id=d1.id, rating=5))
    session.add(Feedback(dialog_id=d2.id, rating=4))
    session.add(Feedback(dialog_id=d3.id, rating=2))
    await session.commit()

    s = await subject_stats(session, subject_slug="tos", days=7)
    assert s is not None
    assert s.subject_slug == "tos"
    assert s.dialogs_window == 3
    assert s.dialogs_total == 3
    assert s.up_count == 2
    assert s.down_count == 1
    assert s.avg_rating is not None
    assert abs(s.avg_rating - 11 / 3) < 1e-6
    # Top questions: d1 and d2 normalise to the same key → count=2.
    assert s.top_questions, "expected at least one repeat"
    assert s.top_questions[0][1] == 2
    # Latency p50 ≈ 1200 (middle of [500, 1200, 3000]).
    assert s.p50_latency_ms == 1200
    assert s.p95_latency_ms == 3000


@pytest.mark.asyncio
async def test_subject_stats_window_filters_old_dialogs(session: AsyncSession) -> None:
    user = _fresh_user()
    subj = _fresh_subject()
    session.add_all([user, subj])
    await session.flush()

    recent = _fresh_dialog(user.id, subj.id, "q1")
    old = _fresh_dialog(
        user.id,
        subj.id,
        "q2",
        created_at=datetime.now(UTC) - timedelta(days=30),
    )
    session.add_all([recent, old])
    await session.commit()

    s = await subject_stats(session, subject_slug="tos", days=7)
    assert s is not None
    assert s.dialogs_window == 1
    assert s.dialogs_total == 2


@pytest.mark.asyncio
async def test_subject_stats_web_fallback_ratio(session: AsyncSession) -> None:
    user = _fresh_user(current_subject="tos")
    subj = _fresh_subject()
    session.add_all([user, subj])
    await session.flush()

    # 2 textbook answers + 1 web-fallback under the same user whose
    # current_subject_slug="tos" at ask time.
    d1 = _fresh_dialog(user.id, subj.id, "система")
    d2 = _fresh_dialog(user.id, subj.id, "онтология")
    d3 = _fresh_dialog(user.id, None, "что-то другое")  # web-fallback
    session.add_all([d1, d2, d3])
    await session.commit()

    s = await subject_stats(session, subject_slug="tos", days=7)
    assert s is not None
    assert s.web_fallback_count == 1
    assert abs(s.web_fallback_ratio - 1 / 3) < 1e-6


# ---------- coverage_gaps ----------


@pytest.mark.asyncio
async def test_coverage_gaps_lists_web_fallback_questions(session: AsyncSession) -> None:
    user = _fresh_user()
    session.add(user)
    await session.flush()

    # 3 web-fallback dialogs with 2 distinct normalised questions.
    for q in [
        "что такое логарифм",
        "что есть логарифм?",
        "сколько лет коту васе",
    ]:
        session.add(_fresh_dialog(user.id, None, q))
    # One textbook dialog must NOT show up in gaps.
    session.add(_fresh_dialog(user.id, 1, "не должен попасть"))
    await session.commit()

    gaps = await coverage_gaps(session, days=30)
    assert len(gaps) == 2
    # Most-frequent first: the логарифм duplicate should top the list.
    assert gaps[0].count == 2
    assert "логарифм" in gaps[0].question.lower()
    assert gaps[1].count == 1


@pytest.mark.asyncio
async def test_coverage_gaps_respects_window(session: AsyncSession) -> None:
    user = _fresh_user()
    session.add(user)
    await session.flush()
    session.add(
        _fresh_dialog(
            user.id,
            None,
            "давно спрашивали",
            created_at=datetime.now(UTC) - timedelta(days=90),
        )
    )
    session.add(_fresh_dialog(user.id, None, "недавно спрашивали"))
    await session.commit()

    gaps = await coverage_gaps(session, days=30)
    assert len(gaps) == 1
    assert "недавно" in gaps[0].question.lower()


@pytest.mark.asyncio
async def test_coverage_gaps_empty_when_only_textbook_answers(session: AsyncSession) -> None:
    user = _fresh_user()
    subj = _fresh_subject()
    session.add_all([user, subj])
    await session.flush()
    session.add(_fresh_dialog(user.id, subj.id, "q"))
    await session.commit()
    assert await coverage_gaps(session, days=30) == []


@pytest.mark.asyncio
async def test_coverage_gaps_strips_followup_tag(session: AsyncSession) -> None:
    user = _fresh_user()
    session.add(user)
    await session.flush()
    session.add(_fresh_dialog(user.id, None, "[simplify] что такое фаза"))
    session.add(_fresh_dialog(user.id, None, "что такое фаза"))
    await session.commit()
    gaps = await coverage_gaps(session, days=30)
    # Both rows normalise to the same key → single entry of count 2.
    assert len(gaps) == 1
    assert gaps[0].count == 2
