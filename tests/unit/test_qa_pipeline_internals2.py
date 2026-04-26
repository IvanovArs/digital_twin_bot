"""Дополнительное покрытие qa_pipeline: web-fallback edge-cases, sources persisted."""

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


def _user() -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _hit() -> Hit:
    return Hit(text="x", subject_slug="theory_of_systems", book="a.pdf", page=1, score=0.9)


def _web() -> WebHit:
    return WebHit(
        title="W",
        url="https://example.org/x",
        snippet="long snippet with more than eight word tokens about a thing.",
    )


class _Cap:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: tuple = ()

    async def set_status(self, t: str) -> None:
        self.statuses.append(t)

    async def set_final(self, body: str, dialog_id: int, kind: str = "full") -> None:
        self.final = (body, dialog_id, kind)


async def _none(*_a: Any, **_kw: Any) -> Any:
    return None


class _ZeroV:
    total = 0
    attributions_stripped = 0
    etymologies_stripped = 0
    foreign_scripts_stripped = 0


@pytest.mark.asyncio
async def test_web_brief_mode_compact_body(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    user.answer_mode = AnswerMode.brief
    session.add(user)
    await session.flush()

    web_hits = [_web()]

    async def fake_to_thread(fn, *a, **kw):  # type: ignore[no-untyped-def]
        return web_hits

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        for piece in ["краткий", " ответ"]:
            yield piece

    monkeypatch.setattr(qa.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
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
    body, _, kind = cap.final
    assert kind == "web"
    assert "example.org" in body or "краткий" in body


@pytest.mark.asyncio
async def test_web_persists_sources_to_dialog(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    web_hits = [_web()]

    async def fake_to_thread(fn, *a, **kw):  # type: ignore[no-untyped-def]
        return web_hits

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        yield "ответ"

    monkeypatch.setattr(qa.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
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
    # В БД должен появиться диалог с sources типа web.
    rows = (await session.execute(__import__("sqlalchemy").select(Dialog))).scalars().all()
    assert any(
        isinstance(d.sources, list) and any(s.get("type") == "web" for s in d.sources or [])
        for d in rows
    )


@pytest.mark.asyncio
async def test_followup_reload_from_dialog_when_cache_missed(session, monkeypatch) -> None:
    """Cache promaхнулся → fallback на retrieval из Dialog.question."""
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "что такое X"
    d.answer = "..."
    session.add(d)
    await session.flush()

    followup_cache.clear()  # форсим cache miss
    MODELS_READY.set()

    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    monkeypatch.setattr(qa, "resolve_subject", lambda q, s: (subj, [_hit()], None))
    monkeypatch.setattr(qa, "detect_comparison", lambda q: None)

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        yield "проще"

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
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


@pytest.mark.asyncio
async def test_followup_reload_strips_modifier_prefix_from_question(session, monkeypatch) -> None:
    """Если question = '[simplify] Q', при reload берём только Q."""
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "[simplify] исходный вопрос"
    d.answer = "..."
    session.add(d)
    await session.flush()
    followup_cache.clear()
    MODELS_READY.set()

    captured_q: dict = {}

    def fake_resolve(q, s):  # type: ignore[no-untyped-def]
        captured_q["q"] = q
        subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
        return (subj, [_hit()], None)

    monkeypatch.setattr(qa, "resolve_subject", fake_resolve)
    monkeypatch.setattr(qa, "detect_comparison", lambda q: None)

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        yield "ok"

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
    await qa.run_followup_pipeline(
        dialog_id=d.id,
        modifier="example",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert captured_q.get("q") == "исходный вопрос"
