"""Тесты get_or_create_user — auto-promotion в admin, обновление имени/роли."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.services.user_service import get_or_create_user
from src.config import settings
from src.db.models import Base, UserRole


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _tg(uid: int, name: str = "Stud Ent") -> SimpleNamespace:
    return SimpleNamespace(id=uid, full_name=name)


@pytest.mark.asyncio
async def test_creates_new_user_as_student(session: AsyncSession, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    user = await get_or_create_user(session, _tg(101))
    assert user.id is not None
    assert user.role is UserRole.student
    assert user.full_name == "Stud Ent"


@pytest.mark.asyncio
async def test_admin_id_auto_promoted(session: AsyncSession, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "555")
    user = await get_or_create_user(session, _tg(555, "Admin Boss"))
    assert user.role is UserRole.admin


@pytest.mark.asyncio
async def test_existing_user_name_updates(session: AsyncSession, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    u1 = await get_or_create_user(session, _tg(1, "Old Name"))
    u2 = await get_or_create_user(session, _tg(1, "New Name"))
    assert u1.id == u2.id
    assert u2.full_name == "New Name"


@pytest.mark.asyncio
async def test_role_demote_on_admin_id_removal(session: AsyncSession, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "777")
    u = await get_or_create_user(session, _tg(777, "X"))
    assert u.role is UserRole.admin
    monkeypatch.setattr(settings, "ADMIN_TELEGRAM_IDS", "")
    u2 = await get_or_create_user(session, _tg(777, "X"))
    assert u2.role is UserRole.student
