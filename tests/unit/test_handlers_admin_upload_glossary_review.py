"""Тесты admin/upload, glossary, review хендлеров с mock'ом bot.download."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.handlers.admin.glossary import (
    on_glossary_cancel,
    on_glossary_doc,
    on_teacher_glossary_upload,
)
from src.bot.handlers.admin.review import (
    on_fix_cancel,
    on_fix_receive,
    on_teacher_fix,
)
from src.bot.handlers.admin.upload import on_admin_upload_doc
from src.db.models import Base, Dialog, Subject, User, UserRole


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
    u = User(telegram_id=1, full_name="A")
    u.role = UserRole.admin
    return u


def _msg(*, text: str = "", doc=None) -> MagicMock:  # type: ignore[no-untyped-def]
    m = MagicMock()
    m.answer = AsyncMock()
    m.text = text
    m.document = doc
    m.bot = MagicMock()
    m.bot.download = AsyncMock()
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


# ---------- admin upload doc ----------


@pytest.mark.asyncio
async def test_upload_doc_subject_disappeared(session) -> None:
    m = _msg(doc=_doc("foo.pdf"))
    st = _state({"subject_slug": "ghost"})
    await on_admin_upload_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "пропал" in body or "Начни заново" in body
    st.clear.assert_awaited()


@pytest.mark.asyncio
async def test_upload_doc_invalid_filename(session) -> None:
    """`../../etc/passwd` нормализуется в `passwd` (без расширения),
    что валится на проверке поддерживаемых расширений — это и есть защита."""
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("../../etc/passwd"))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_admin_upload_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    # Любая защитная реакция: либо «Недопустимое имя файла», либо
    # «Поддерживаются: .pdf …» — обе означают, что атака отбита.
    assert "Недопустимое" in body or "имя файла" in body or ".pdf" in body


@pytest.mark.asyncio
async def test_upload_doc_unsupported_extension(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("foo.exe"))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_admin_upload_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert ".pdf" in body and ".docx" in body


@pytest.mark.asyncio
async def test_upload_doc_oversize(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("huge.pdf", size=999_999_999))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_admin_upload_doc(m, st, session, _user())
    m.answer.assert_awaited()


@pytest.mark.asyncio
async def test_upload_doc_no_bot(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("foo.pdf"))
    m.bot = None
    st = _state({"subject_slug": "theory_of_systems"})
    await on_admin_upload_doc(m, st, session, _user())
    m.answer.assert_awaited()


@pytest.mark.asyncio
async def test_upload_doc_download_returns_none(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("foo.pdf"))
    m.bot.download = AsyncMock(return_value=None)
    st = _state({"subject_slug": "theory_of_systems"})
    await on_admin_upload_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "скачать" in body


@pytest.mark.asyncio
async def test_upload_cancel_clears_state() -> None:
    from src.bot.handlers.admin.upload import on_admin_upload_cancel

    m = _msg()
    st = _state()
    await on_admin_upload_cancel(m, st)
    st.clear.assert_awaited()


# ---------- teacher_glossary_upload ----------


@pytest.mark.asyncio
async def test_glossary_upload_no_arg(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    st = _state()
    await on_teacher_glossary_upload(m, cmd, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_glossary_upload_unknown_slug(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="ghost")
    st = _state()
    await on_teacher_glossary_upload(m, cmd, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "Неизвестный" in body


@pytest.mark.asyncio
async def test_glossary_upload_known_subject_sets_state(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args="theory_of_systems")
    st = _state()
    await on_teacher_glossary_upload(m, cmd, st, session, _user())
    st.set_state.assert_awaited()


@pytest.mark.asyncio
async def test_glossary_cancel_clears_state() -> None:
    m = _msg()
    st = _state()
    await on_glossary_cancel(m, st)
    st.clear.assert_awaited()


@pytest.mark.asyncio
async def test_glossary_doc_subject_gone(session) -> None:
    m = _msg(doc=_doc("g.csv"))
    st = _state({"subject_slug": "ghost"})
    await on_glossary_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "пропал" in body or "Начни" in body


@pytest.mark.asyncio
async def test_glossary_doc_unsupported_extension(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg(doc=_doc("g.txt"))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert ".csv" in body or ".yaml" in body


@pytest.mark.asyncio
async def test_glossary_doc_no_document(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    m = _msg()  # без документа
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _user())
    # Тихо вышел без crash'а.


@pytest.mark.asyncio
async def test_glossary_doc_full_flow_success(session) -> None:
    session.add(Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС"))
    await session.flush()
    payload = b"term,definition\nstakeholder,a person interested in the project\n"
    m = _msg(doc=_doc("g.csv", size=len(payload)))
    m.bot.download = AsyncMock(return_value=BytesIO(payload))
    st = _state({"subject_slug": "theory_of_systems"})
    await on_glossary_doc(m, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "✅" in body or "обновлён" in body


# ---------- teacher_fix / fix_receive / fix_cancel ----------


@pytest.mark.asyncio
async def test_teacher_fix_no_arg(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args=None)
    st = _state()
    await on_teacher_fix(m, cmd, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "Использование" in body


@pytest.mark.asyncio
async def test_teacher_fix_non_numeric(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="abc")
    st = _state()
    await on_teacher_fix(m, cmd, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "числом" in body


@pytest.mark.asyncio
async def test_teacher_fix_unknown_dialog(session) -> None:
    m = _msg()
    cmd = SimpleNamespace(args="9999")
    st = _state()
    await on_teacher_fix(m, cmd, st, session, _user())
    body = m.answer.call_args.args[0]
    assert "не найден" in body


@pytest.mark.asyncio
async def test_teacher_fix_known_dialog(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "что такое X"
    d.answer = "..."
    session.add(d)
    await session.flush()
    m = _msg()
    cmd = SimpleNamespace(args=str(d.id))
    st = _state()
    await on_teacher_fix(m, cmd, st, session, user)
    st.set_state.assert_awaited()
    st.update_data.assert_awaited_with(dialog_id=d.id)


@pytest.mark.asyncio
async def test_fix_cancel_clears_state() -> None:
    m = _msg()
    st = _state()
    await on_fix_cancel(m, st)
    st.clear.assert_awaited()


@pytest.mark.asyncio
async def test_fix_receive_empty_text(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    m = _msg(text="   ")
    st = _state({"dialog_id": d.id})
    await on_fix_receive(m, st, session, user)
    body = m.answer.call_args.args[0]
    assert "Пустой" in body


@pytest.mark.asyncio
async def test_fix_receive_too_long(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "?"
    d.answer = "."
    session.add(d)
    await session.flush()
    m = _msg(text="X" * 4000)
    st = _state({"dialog_id": d.id})
    await on_fix_receive(m, st, session, user)
    body = m.answer.call_args.args[0]
    assert "длинный" in body or "лимит" in body


@pytest.mark.asyncio
async def test_fix_receive_success_flow(session) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    d = Dialog(user_id=user.id)
    d.question = "что такое X"
    d.answer = "..."
    session.add(d)
    await session.flush()
    m = _msg(text="<b>X</b> — это правильный ответ")
    st = _state({"dialog_id": d.id})
    await on_fix_receive(m, st, session, user)
    body = m.answer.call_args.args[0]
    assert "✅" in body or "сохранён" in body
