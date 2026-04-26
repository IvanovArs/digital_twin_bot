"""Интеграционные тесты qa_pipeline (моки на retrieval + LLM-стрим)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services import answer_cache
from src.bot.services import qa_pipeline as qa
from src.bot.services.warmup import MODELS_READY
from src.db.models import Base, User, UserRole
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
    u = User(telegram_id=tg, full_name="Stud Ent")
    u.role = UserRole.student
    session.add(u)
    await session.flush()
    return u


def _subject() -> SubjectCfg:
    return SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="Теория систем")


def _hit(text: str = "стейкхолдер — лицо") -> Hit:
    return Hit(text=text, subject_slug="theory_of_systems", book="a.pdf", page=12, score=0.85)


class _Capture:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: tuple[str, int, str] | None = None

    async def set_status(self, text: str) -> None:
        self.statuses.append(text)

    async def set_final(self, body: str, dialog_id: int, kind: str = "full") -> None:
        self.final = (body, dialog_id, kind)


def _patch_pipeline(monkeypatch, *, hits: list[Hit], answer: str) -> None:
    """Подменяет тяжёлые зависимости — модели готовы, retrieval мгновенный,
    LLM-стрим возвращает фиксированный ответ."""
    MODELS_READY.set()
    monkeypatch.setattr(
        qa, "resolve_subject", lambda q, slug: (_subject(), hits, None)
    )
    monkeypatch.setattr(qa, "lookup_faq", _async_none)
    monkeypatch.setattr(qa, "lookup_term", _async_none)

    async def fake_stream(_messages, **_kwargs) -> AsyncIterator[str]:
        for piece in answer.split(" "):
            yield piece + " "

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _texts: (a, _ZeroValidation()))


async def _async_none(*_a: Any, **_kw: Any) -> None:
    return None


class _ZeroValidation:
    total = 0
    attributions_stripped = 0
    etymologies_stripped = 0
    foreign_scripts_stripped = 0


@pytest.mark.asyncio
async def test_textbook_answer_path_persists_dialog_and_calls_set_final(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = await _user(session)
    cap = _Capture()
    _patch_pipeline(monkeypatch, hits=[_hit()], answer="<b>Стейкхолдер</b> — это лицо.")

    await qa.run_qa_pipeline(
        question="что такое стейкхолдер?",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert cap.final is not None
    body, dialog_id, kind = cap.final
    assert kind == "full"
    assert dialog_id > 0
    assert "Стейкхолдер" in body


@pytest.mark.asyncio
async def test_cache_hit_skips_retrieval(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = await _user(session)
    answer_cache.store(
        user_id=user.id,
        question="что это?",
        lang="ru",
        brief=False,
        body="cached body",
        kind="full",
        dialog_id=999,
    )

    called = {"resolve": False}

    def boom(*a, **kw):  # type: ignore[no-untyped-def]
        called["resolve"] = True
        raise RuntimeError("retrieval should be skipped on cache hit")

    monkeypatch.setattr(qa, "resolve_subject", boom)
    monkeypatch.setattr(qa, "lookup_faq", _async_none)
    monkeypatch.setattr(qa, "lookup_term", _async_none)
    cap = _Capture()
    await qa.run_qa_pipeline(
        question="что  это  ?",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert called["resolve"] is False
    assert cap.final == ("cached body", 999, "full")


@pytest.mark.asyncio
async def test_low_score_triggers_web_fallback(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = await _user(session)
    cap = _Capture()
    # top_score=0.1 — заведомо ниже MIN_TOP_SCORE.
    weak = Hit(text="x", subject_slug="theory_of_systems", book="a.pdf", page=1, score=0.1)
    _patch_pipeline(monkeypatch, hits=[weak], answer="x")

    web_called = {"hit": False}

    async def fake_web(**kwargs):  # type: ignore[no-untyped-def]
        web_called["hit"] = True

    monkeypatch.setattr(qa, "_web_fallback", fake_web)
    await qa.run_qa_pipeline(
        question="вопрос вне курса",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert web_called["hit"] is True


@pytest.mark.asyncio
async def test_llm_refusal_triggers_web_fallback(
    session: AsyncSession, monkeypatch
) -> None:
    answer_cache.invalidate_all()
    user = await _user(session)
    cap = _Capture()
    _patch_pipeline(
        monkeypatch,
        hits=[_hit()],
        answer="В материалах курса этого прямо не нашлось.",
    )

    web_called = {"hit": False}

    async def fake_web(**kwargs):  # type: ignore[no-untyped-def]
        web_called["hit"] = True

    monkeypatch.setattr(qa, "_web_fallback", fake_web)
    await qa.run_qa_pipeline(
        question="редкий нюанс",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert web_called["hit"] is True
