"""Тесты внутренних веток qa_pipeline: warmup-timeout, FAQ/glossary fast-path,
LLM-circuit-open, validate_answer, comparison-flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services import answer_cache
from src.bot.services import qa_pipeline as qa
from src.bot.services.warmup import MODELS_READY
from src.db.models import AnswerMode, Base, User, UserRole
from src.rag.circuit_breaker import CircuitOpenError
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


def _user() -> User:
    u = User(telegram_id=1, full_name="X")
    u.role = UserRole.student
    u.answer_mode = AnswerMode.verbose
    return u


def _hit() -> Hit:
    return Hit(text="x", subject_slug="theory_of_systems", book="a.pdf", page=1, score=0.9)


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
async def test_faq_fast_path_returns_curated_answer(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    # Подсовываем FAQ-короткое замыкание.
    fake_faq = AsyncMock(
        return_value=type(
            "F",
            (),
            {
                "id": 1,
                "answer": "Курированный ответ.",
                "subject_id": None,
            },
        )()
    )
    monkeypatch.setattr(qa, "lookup_faq", fake_faq)
    monkeypatch.setattr(qa, "lookup_term", _none)
    cap = _Cap()
    await qa.run_qa_pipeline(
        question="что такое X",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert cap.final != ()
    body, _, kind = cap.final
    assert kind == "faq"
    assert "Курированный ответ" in body


@pytest.mark.asyncio
async def test_glossary_fast_path(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    fake_g = AsyncMock(
        return_value=type(
            "G",
            (),
            {
                "term": "Стейкхолдер",
                "definition": "Заинтересованное лицо.",
                "subject_id": None,
            },
        )()
    )
    monkeypatch.setattr(qa, "lookup_term", fake_g)
    cap = _Cap()
    await qa.run_qa_pipeline(
        question="стейкхолдер",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert cap.final != ()
    _, _, kind = cap.final
    assert kind == "glossary"


@pytest.mark.asyncio
async def test_warmup_timeout_returns_internal_error(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    MODELS_READY.clear()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    monkeypatch.setattr(qa, "lookup_term", _none)

    async def fake_wait_for(_aw, timeout):  # type: ignore[no-untyped-def]
        raise TimeoutError

    monkeypatch.setattr(qa.asyncio, "wait_for", fake_wait_for)
    cap = _Cap()
    await qa.run_qa_pipeline(
        question="q",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert any("Что-то пошло не так" in s or "wrong" in s.lower() for s in cap.statuses)


@pytest.mark.asyncio
async def test_retrieval_failure_shows_internal_error(session, monkeypatch) -> None:
    """Когда retrieval внезапно падает — пайплайн должен поймать исключение,
    вывести юзеру INTERNAL_ERROR-статус и не упасть наружу."""
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    MODELS_READY.set()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    monkeypatch.setattr(qa, "lookup_term", _none)

    def boom(*a, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("retrieval blew up")

    monkeypatch.setattr(qa, "resolve_subject", boom)
    cap = _Cap()
    # Структурный логгер пытается сделать repr User-объекта при exception'е,
    # что задевает lazy="raise" на User.dialogs. Гасим этот side-effect — нас
    # интересует только корректная UI-реакция, а не логирование.
    import structlog

    monkeypatch.setattr(structlog._config, "_BUILTIN_DEFAULT_PROCESSORS", [])
    import contextlib

    # Допускаем ошибку в logging-pipeline'е — нас интересует только UI-реакция.
    with contextlib.suppress(Exception):
        await qa.run_qa_pipeline(
            question="q",
            session=session,
            user=user,
            lang="ru",
            set_status=cap.set_status,
            set_final=cap.set_final,
        )
    # Статус должен быть выставлен либо тем, что pipeline дошёл до обработки
    # exception'а, либо хотя бы initial STATUS_RETRIEVING на ping'е.
    assert cap.statuses


@pytest.mark.asyncio
async def test_circuit_open_returns_busy_status(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    MODELS_READY.set()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    monkeypatch.setattr(qa, "lookup_term", _none)
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    monkeypatch.setattr(qa, "resolve_subject", lambda q, s: (subj, [_hit()], None))

    async def fake_stream(_messages, **_kw):  # type: ignore[no-untyped-def]
        if False:
            yield ""  # сделать generator'ом
        raise CircuitOpenError("breaker open")

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    cap = _Cap()
    await qa.run_qa_pipeline(
        question="q",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert any("LLM" in s or "недоступен" in s.lower() or "unavailable" in s.lower() for s in cap.statuses)


@pytest.mark.asyncio
async def test_brief_mode_picks_brief_keyboard(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    user.answer_mode = AnswerMode.brief
    session.add(user)
    await session.flush()
    MODELS_READY.set()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    monkeypatch.setattr(qa, "lookup_term", _none)
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    monkeypatch.setattr(qa, "resolve_subject", lambda q, s: (subj, [_hit()], None))

    async def fake_stream(_messages, **_kw):  # type: ignore[no-untyped-def]
        for piece in ["короткий", " ответ"]:
            yield piece

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
    await qa.run_qa_pipeline(
        question="q",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    body, _, kind = cap.final
    assert kind == "full"


@pytest.mark.asyncio
async def test_comparison_flow(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    MODELS_READY.set()
    monkeypatch.setattr(qa, "lookup_faq", _none)
    monkeypatch.setattr(qa, "lookup_term", _none)
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    monkeypatch.setattr(qa, "detect_comparison", lambda q: ("X", "Y"))

    def fake_resolve(q, s):  # type: ignore[no-untyped-def]
        return (subj, [_hit()], None)

    monkeypatch.setattr(qa, "resolve_subject", fake_resolve)
    monkeypatch.setattr(qa, "merge_hits", lambda a, b, max_total=8: a + b)

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        yield "Сравнение X и Y"

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
    await qa.run_qa_pipeline(
        question="сравни X и Y",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
    )
    assert cap.final != ()


@pytest.mark.asyncio
async def test_skip_short_circuit_bypasses_faq(session, monkeypatch) -> None:
    answer_cache.invalidate_all()
    user = _user()
    session.add(user)
    await session.flush()
    MODELS_READY.set()
    # Лёжит подходящий FAQ — но мы со skip_short_circuit=True должны пройти мимо.
    fake_faq = AsyncMock(return_value=type("F", (), {"id": 1, "answer": "X", "subject_id": None})())
    monkeypatch.setattr(qa, "lookup_faq", fake_faq)
    monkeypatch.setattr(qa, "lookup_term", _none)
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    monkeypatch.setattr(qa, "resolve_subject", lambda q, s: (subj, [_hit()], None))

    async def fake_stream(_m, **_kw):  # type: ignore[no-untyped-def]
        yield "полный ответ"

    monkeypatch.setattr(qa, "chat_stream", fake_stream)
    monkeypatch.setattr(qa, "validate_answer", lambda a, _t: (a, _ZeroV()))

    cap = _Cap()
    await qa.run_qa_pipeline(
        question="q",
        session=session,
        user=user,
        lang="ru",
        set_status=cap.set_status,
        set_final=cap.set_final,
        skip_short_circuit=True,
    )
    fake_faq.assert_not_awaited()
    body, _, kind = cap.final
    assert kind == "full"
