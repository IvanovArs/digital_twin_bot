"""Покрытие throttle_mw + admin_service.save_material/reindex_subject_in_background."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.bot.middlewares.throttle_mw import (
    ThrottleMiddleware,
    _gc_stale,
    _is_heavy_update,
    _last_ts,
    _locks,
)
from src.bot.services.admin_service import (
    _invalidate_retrieval_caches,
    reindex_subject_in_background,
    save_material,
)
from src.db.models import Base, Subject, User, UserRole


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


# ---------- throttle ----------


def test_is_heavy_update_message_with_text() -> None:
    from aiogram.types import Message

    upd = MagicMock()
    upd.chosen_inline_result = None
    msg = MagicMock(spec=Message)
    msg.text = "обычный вопрос"
    msg.caption = None
    upd.message = msg
    assert _is_heavy_update(upd) is True


def test_is_heavy_update_slash_command_not_heavy() -> None:
    from aiogram.types import Message

    upd = MagicMock()
    upd.chosen_inline_result = None
    msg = MagicMock(spec=Message)
    msg.text = "/help"
    msg.caption = None
    upd.message = msg
    assert _is_heavy_update(upd) is False


def test_is_heavy_update_no_message() -> None:
    upd = MagicMock()
    upd.chosen_inline_result = None
    upd.message = None
    assert _is_heavy_update(upd) is False


def test_is_heavy_update_chosen_inline_result_is_heavy() -> None:
    upd = MagicMock()
    upd.chosen_inline_result = MagicMock()
    assert _is_heavy_update(upd) is True


@pytest.mark.asyncio
async def test_throttle_passes_non_heavy() -> None:
    mw = ThrottleMiddleware()
    called = {"hit": False}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        called["hit"] = True
        return "ok"

    upd = MagicMock()
    upd.chosen_inline_result = None
    upd.message = None
    res = await mw(handler, upd, {})
    assert called["hit"] is True
    assert res == "ok"


@pytest.mark.asyncio
async def test_throttle_drops_burst() -> None:
    from aiogram.types import Message, Update

    _last_ts.clear()
    _locks.clear()
    mw = ThrottleMiddleware()
    called = {"n": 0}

    async def handler(event, data):  # type: ignore[no-untyped-def]
        called["n"] += 1
        return "ok"

    msg = MagicMock(spec=Message)
    msg.text = "вопрос"
    msg.caption = None
    upd = MagicMock(spec=Update)
    upd.chosen_inline_result = None
    upd.message = msg

    await mw(handler, upd, {"event_from_user": SimpleNamespace(id=42)})
    # Сразу второй — должен быть дропнут.
    res2 = await mw(handler, upd, {"event_from_user": SimpleNamespace(id=42)})
    assert called["n"] == 1
    assert res2 is None


def test_gc_stale_evicts_old() -> None:
    import time

    _last_ts.clear()
    _last_ts[1] = time.monotonic() - 999_999  # очень старая запись
    _last_ts[2] = time.monotonic()  # свежая
    _gc_stale(time.monotonic())
    assert 1 not in _last_ts
    assert 2 in _last_ts


# ---------- admin_service.save_material ----------


@pytest.mark.asyncio
async def test_save_material_creates_file_and_row(sessionmaker, tmp_path: Path) -> None:
    async with sessionmaker() as session:
        subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
        session.add(subj)
        user = User(telegram_id=1, full_name="A")
        user.role = UserRole.admin
        session.add(user)
        await session.flush()

        row = await save_material(
            session,
            subject=subj,
            filename="x.pdf",
            payload=b"%PDF-1.4 hello",
            uploader=user,
            target_dir=tmp_path,
        )
        await session.commit()
        assert row.filename == "x.pdf"
        assert (tmp_path / "theory_of_systems" / "x.pdf").exists()


@pytest.mark.asyncio
async def test_save_material_overwrites_existing(sessionmaker, tmp_path: Path) -> None:
    async with sessionmaker() as session:
        subj = Subject(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
        session.add(subj)
        user = User(telegram_id=1, full_name="A")
        user.role = UserRole.admin
        session.add(user)
        await session.flush()
        await save_material(
            session,
            subject=subj,
            filename="x.pdf",
            payload=b"v1",
            uploader=user,
            target_dir=tmp_path,
        )
        await session.commit()
        await save_material(
            session,
            subject=subj,
            filename="x.pdf",
            payload=b"v2",
            uploader=user,
            target_dir=tmp_path,
        )
        await session.commit()
        # Та же строка, не дубликат.
        from sqlalchemy import func, select

        from src.db.models import SubjectMaterial

        cnt = (
            await session.execute(select(func.count(SubjectMaterial.id)))
        ).scalar_one()
        assert cnt == 1
        assert (tmp_path / "theory_of_systems" / "x.pdf").read_bytes() == b"v2"


@pytest.mark.asyncio
async def test_reindex_in_background_handles_failure(sessionmaker, monkeypatch) -> None:
    import src.bot.services.admin_service as adm

    def boom(_slug):  # type: ignore[no-untyped-def]
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(adm, "build_index", boom)
    # Не должна пробрасывать.
    await reindex_subject_in_background(sessionmaker, subject_slug="theory_of_systems")


def test_invalidate_retrieval_caches_runs() -> None:
    # Должен пройти без исключения, даже если кеши пустые.
    _invalidate_retrieval_caches()
