"""User bootstrap/update: map Telegram user → row in `users`."""

from __future__ import annotations

from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import User, UserRole


async def get_or_create_user(session: AsyncSession, tg_user: TgUser) -> User:
    """Return a persistent User for the given Telegram user, creating if missing.

    Users whose telegram_id is listed in ADMIN_TELEGRAM_IDS are auto-promoted
    to the 'admin' role on first contact.
    """
    stmt = select(User).where(User.telegram_id == tg_user.id)
    user = (await session.execute(stmt)).scalar_one_or_none()

    role = UserRole.admin if tg_user.id in settings.admin_ids else UserRole.student

    if user is None:
        user = User(
            telegram_id=tg_user.id,
            full_name=tg_user.full_name,
            role=role,
        )
        session.add(user)
        await session.flush()
    else:
        # keep name / role fresh in case config changed
        changed = False
        if user.full_name != tg_user.full_name:
            user.full_name = tg_user.full_name
            changed = True
        if user.role != role:
            user.role = role
            changed = True
        if changed:
            await session.flush()

    return user
