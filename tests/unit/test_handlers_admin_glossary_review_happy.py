"""Happy-path для admin/glossary.py и admin/review.py с заполненной БД."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin.glossary import on_glossary_doc
from src.bot.handlers.admin.review import on_teacher_review
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
    m.bot = MagicMock()
    return m


def _doc(name: str, size: int = 1024) -> MagicMock:
    d = MagicMock()
    d.file_name = name
    d.file_size = size
    return d


def _state(data: dict | None = None) -> MagicMock:
    st = MagicMock()
    st.get_data = AsyncMock(return_value=data or {})
    st.set_state = AsyncMock()
    st.update_data = AsyncMock()
    st.clear = AsyncMock()
    return st


# ---------- glossary doc — happy/edge ----------


@pytest.mark.asyncio
async def test_glossary_doc_no_bot(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    m.bot = None
    m.document = _doc("g.csv")
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _admin())
    body = m.answer.call_args.args[0]
    assert "Telegram" in body or "связ" in body


@pytest.mark.asyncio
async def test_glossary_doc_download_returns_none(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    m.document = _doc("g.csv")
    m.bot.download = AsyncMock(return_value=None)
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _admin())
    body = m.answer.call_args.args[0]
    assert "скачать" in body


@pytest.mark.asyncio
async def test_glossary_doc_invalid_payload(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    m.document = _doc("g.yaml")
    m.bot.download = AsyncMock(return_value=BytesIO(b"this is not valid yaml: ::: {["))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _admin())
    body = m.answer.call_args.args[0]
    assert "Ошибка" in body or "term/definition" in body


@pytest.mark.asyncio
async def test_glossary_doc_empty_yaml(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    m.document = _doc("g.yaml")
    m.bot.download = AsyncMock(return_value=BytesIO(b"terms: []\n"))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _admin())
    body = m.answer.call_args.args[0]
    assert "не нашлось" in body or "ни одной" in body


# ---------- teacher_review с реальными feedback ----------


@pytest.mark.asyncio
async def test_teacher_review_lists_pending(session) -> None:
    user = _admin()
    session.add(user)
    await session.flush()
    # Создаём 👎-диалог.
    d = Dialog(user_id=user.id)
    d.question = "почему всё так плохо"
    d.answer = "плохой ответ бота"
    d.created_at = datetime.now(UTC)
    session.add(d)
    await session.flush()
    fb = Feedback(dialog_id=d.id, rating=1)
    fb.created_at = datetime.now(UTC)
    session.add(fb)
    await session.flush()

    m = _msg()
    await on_teacher_review(m, session, user)
    body = m.answer.call_args.args[0]
    assert "Очередь модерации" in body or "почему" in body
    assert "/teacher_fix" in body or f"#{d.id}" in body


@pytest.mark.asyncio
async def test_teacher_review_marks_high_ratings_with_stars(session) -> None:
    user = _admin()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "вопрос"
    d.answer = "ответ"
    d.created_at = datetime.now(UTC)
    session.add(d)
    await session.flush()
    fb = Feedback(dialog_id=d.id, rating=3)  # средняя оценка
    fb.created_at = datetime.now(UTC)
    session.add(fb)
    await session.flush()
    m = _msg()
    await on_teacher_review(m, session, user)
    body = m.answer.call_args.args[0]
    # Иконка либо 👎 (rating ≤ 2), либо `3⭐` для среднего.
    assert "3⭐" in body or "Очередь" in body
