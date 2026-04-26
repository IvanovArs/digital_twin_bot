"""Тесты dialog_service: save_dialog, record_feedback (атомарность), set_favourite."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.dialog_service import (
    record_feedback,
    save_dialog,
    set_favourite,
)
from src.db.models import Base, Feedback, Subject, User, UserRole
from src.rag.pipeline import AskResult
from src.rag.retriever import Hit
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


async def _user(session: AsyncSession, tg: int = 1) -> User:
    u = User(telegram_id=tg, full_name="U")
    u.role = UserRole.student
    session.add(u)
    await session.flush()
    return u


@pytest.mark.asyncio
async def test_save_dialog_persists_sources_with_subject(session: AsyncSession) -> None:
    user = await _user(session)
    subj_row = Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")
    session.add(subj_row)
    await session.flush()

    subj_cfg = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")
    hit = Hit(text="x", subject_slug="theory_of_systems", book="a.pdf", page=1, score=0.9)
    result = AskResult(answer="ответ", subject=subj_cfg, hits=[hit], route=None)  # type: ignore[arg-type]

    dialog = await save_dialog(session, user=user, question="Q", result=result, latency_ms=100)
    assert dialog.id is not None
    assert dialog.subject_id == subj_row.id
    assert dialog.sources == [
        {"book": "a.pdf", "page": 1, "subject": "theory_of_systems", "score": 0.9}
    ]


@pytest.mark.asyncio
async def test_record_feedback_first_call_inserts(session: AsyncSession) -> None:
    user = await _user(session)
    subj_cfg = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    res = AskResult(answer="x", subject=subj_cfg, hits=[], route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(session, user=user, question="?", result=res)

    ok = await record_feedback(session, dialog_id=dialog.id, user_id=user.id, rating=5)
    assert ok is True
    fb = (await session.execute(select(Feedback))).scalars().first()
    assert fb is not None and fb.rating == 5


@pytest.mark.asyncio
async def test_record_feedback_double_tap_returns_false(session: AsyncSession) -> None:
    """SQLite-путь через try/IntegrityError: два таппа → второй вернёт False."""
    user = await _user(session)
    subj_cfg = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    res = AskResult(answer="x", subject=subj_cfg, hits=[], route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(session, user=user, question="?", result=res)

    first = await record_feedback(session, dialog_id=dialog.id, user_id=user.id, rating=5)
    second = await record_feedback(session, dialog_id=dialog.id, user_id=user.id, rating=1)
    assert first is True
    assert second is False
    rows = (await session.execute(select(Feedback))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_record_feedback_rejects_other_users_dialog(session: AsyncSession) -> None:
    owner = await _user(session, tg=1)
    intruder = await _user(session, tg=2)
    subj_cfg = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    res = AskResult(answer="x", subject=subj_cfg, hits=[], route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(session, user=owner, question="?", result=res)

    ok = await record_feedback(session, dialog_id=dialog.id, user_id=intruder.id, rating=5)
    assert ok is False


@pytest.mark.asyncio
async def test_set_favourite_idor_protected(session: AsyncSession) -> None:
    owner = await _user(session, tg=1)
    intruder = await _user(session, tg=2)
    subj_cfg = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    res = AskResult(answer="x", subject=subj_cfg, hits=[], route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(session, user=owner, question="?", result=res)

    ok_owner = await set_favourite(
        session, dialog_id=dialog.id, user_id=owner.id, is_favourite=True
    )
    ok_intruder = await set_favourite(
        session, dialog_id=dialog.id, user_id=intruder.id, is_favourite=False
    )
    assert ok_owner is True
    assert ok_intruder is False
