"""Тесты qa_pipeline._web_fallback и run_followup_pipeline."""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services import answer_cache, followup_cache
from src.bot.services import qa_pipeline as qa
from src.bot.services.warmup import MODELS_READY
from src.db.models import AnswerMode, Base, Dialog, User, UserRole
from src.rag.retriever import Hit
from src.rag.web_search import WebHit
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


def _user(tg: int = 1) -> User:
    u = User(telegram_id=tg, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _hit(text: str = "x") -> Hit:
    return Hit(text=text, subject_slug="theory_of_systems", book="a.pdf", page=1, score=0.9)


def _web(title: str = "Wiki") -> WebHit:
    return WebHit(title=title, url="https://example.org/x", snippet="A long snippet with more than 8 words about a thing.")


class _Capture:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: tuple = ()

    async def set_status(self, text: str) -> None:
        self.statuses.append(text)

    async def set_final(self, body: str, dialog_id: int, kind: str = "full") -> None:
        self.final = (body, dialog_id, kind)


async def _async_none(*_a: Any, **_kw: Any) -> Any:
    return None


@pytest.mark.asyncio
async def test_web_fallback_no_results_says_no_hits(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    cap = _Capture()

    async def empty_search(*a, **kw):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(qa.asyncio, "to_thread", empty_search)
    await qa._web_fallback(
        session=session,
        user=user,
        question="редкий вопрос",
        q_html="редкий вопрос",
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
        typing_ping=None,
        t0=0.0,
    )
    assert cap.final == ()  # set_final не вызвался
    assert any("материалах" in s or "Ask" in s for s in cap.statuses)


@pytest.mark.asyncio
async def test_web_fallback_with_results_streams_answer(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    cap = _Capture()

    web_hits = [_web()]

    async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        return web_hits

    async def fake_stream(_messages, **_kw):  # type: ignore[no-untyped-def]
        for piece in ["ответ ", "из ", "интернета"]:
            yield piece

    monkeypatch.setattr(qa.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroValidation()))

    await qa._web_fallback(
        session=session,
        user=user,
        question="что такое X",
        q_html="что такое X",
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
        typing_ping=None,
        t0=0.0,
    )
    assert cap.final != ()
    body, _, kind = cap.final
    assert kind == "web"
    assert "ответ" in body or "интернета" in body


@pytest.mark.asyncio
async def test_web_fallback_timeout(session: AsyncSession, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    cap = _Capture()

    async def hang(*a, **kw):  # type: ignore[no-untyped-def]
        raise TimeoutError("ddg hang")

    monkeypatch.setattr(qa.asyncio, "wait_for", hang)
    await qa._web_fallback(
        session=session,
        user=user,
        question="q",
        q_html="q",
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
        typing_ping=None,
        t0=0.0,
    )
    # Тайм-аут → web_hits=[] → NO_HITS, set_final не вызывается.
    assert cap.final == ()


@pytest.mark.asyncio
async def test_followup_returns_false_when_dialog_missing(
    session: AsyncSession,
) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    cap = _Capture()
    followup_cache.clear()
    MODELS_READY.set()

    ok = await qa.run_followup_pipeline(
        dialog_id=99999,
        modifier="simplify",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_followup_uses_cached_context(session: AsyncSession, monkeypatch) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "что такое X"
    d.answer = "..."
    session.add(d)
    await session.flush()

    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    followup_cache.store(
        d.id,
        question="что такое X",
        lang="ru",
        hits=[_hit()],
        subject=subj,
    )

    cap = _Capture()
    MODELS_READY.set()

    async def fake_stream(_messages, **_kw):  # type: ignore[no-untyped-def]
        for piece in ["проще: ", "это X"]:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroValidation()))

    ok = await qa.run_followup_pipeline(
        dialog_id=d.id,
        modifier="simplify",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert ok is True
    assert cap.final != ()


class _ZeroValidation:
    total = 0
    attributions_stripped = 0
    etymologies_stripped = 0
    foreign_scripts_stripped = 0
