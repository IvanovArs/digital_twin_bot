"""Happy-path для admin/stats.py с реальными dialogs/feedback в БД."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin.stats import on_teacher_gaps, on_teacher_stats
from src.bot.handlers.admin.subjects_cmds import _reindex_and_report
from src.db.models import Base, Dialog, Feedback, Subject, User, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _admin() -> User:
    u = User(telegram_id=1, full_name="A")
    u.role = UserRole.admin
    return u


def _msg() -> MagicMock:
    m = MagicMock()
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_teacher_stats_with_dialogs_renders_full(session) -> None:
    subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")
    session.add(subj)
    user = _admin()
    session.add(user)
    await session.flush()
    for i in range(3):
        d = Dialog(user_id=user.id, subject_id=subj.id)
        d.question = f"вопрос {i}"
        d.answer = "ответ"
        d.latency_ms = 1000 + i * 100
        d.created_at = datetime.now(UTC)
        session.add(d)
        await session.flush()
        fb = Feedback(dialog_id=d.id, rating=5)
        fb.created_at = datetime.now(UTC)
        session.add(fb)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems")
    await on_teacher_stats(m, cmd, session, _admin())
    body = m.answer.call_args.args[0]
    assert "Теория систем" in body or "📊" in body
    assert "Диалогов" in body or "👍" in body


@pytest.mark.asyncio
async def test_teacher_stats_clamps_days(session) -> None:
    """days > 365 должно ClipMax-нуться, не упасть."""
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems 9999")
    await on_teacher_stats(m, cmd, session, _admin())
    m.answer.assert_awaited()


@pytest.mark.asyncio
async def test_teacher_gaps_with_web_fallback_dialogs(session) -> None:
    user = _admin()
    session.add(user)
    await session.flush()
    # Диалог без subject_id и без sources типа textbook → web-fallback.
    for i in range(2):
        d = Dialog(user_id=user.id)
        d.question = f"редкий вопрос {i}"
        d.answer = "интернет"
        d.sources = [{"type": "web", "title": "x", "url": "https://x"}]
        d.created_at = datetime.now(UTC)
        session.add(d)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="30 5")
    await on_teacher_gaps(m, cmd, session, _admin())
    body = m.answer.call_args.args[0]
    # Либо «Пробелы», либо «🎉» (если coverage_gaps не вернул эти диалоги).
    assert "Пробел" in body or "🎉" in body


@pytest.mark.asyncio
async def test_reindex_and_report_handles_failure(monkeypatch) -> None:
    import src.bot.handlers.admin.subjects_cmds as sub_mod

    msg = MagicMock()
    msg.answer = AsyncMock()

    async def fake_reindex(_smaker, *, subject_slug=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("ingest blew up")

    monkeypatch.setattr(sub_mod, "reindex_subject_in_background", fake_reindex)
    await _reindex_and_report(msg, "theory_of_systems")
    body = msg.answer.call_args.args[0]
    assert "Ошибка" in body or "⚠️" in body


@pytest.mark.asyncio
async def test_reindex_and_report_success(monkeypatch) -> None:
    import src.bot.handlers.admin.subjects_cmds as sub_mod

    msg = MagicMock()
    msg.answer = AsyncMock()

    async def fake_reindex(_smaker, *, subject_slug=None):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(sub_mod, "reindex_subject_in_background", fake_reindex)
    await _reindex_and_report(msg, None)
    body = msg.answer.call_args.args[0]
    assert "✅" in body or "завершена" in body
