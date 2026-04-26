"""User bootstrap/update: маппит Telegram-юзера → строку в таблице `users`."""

from __future__ import annotations

from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import User, UserRole


async def get_or_create_user(session: AsyncSession, tg_user: TgUser) -> User:
    """Вернуть сохранённого User'а для данного Telegram-юзера, создав при необходимости.

    Пользователи, чей telegram_id перечислен в ADMIN_TELEGRAM_IDS, при первом
    контакте автоматически промоутятся в роль 'admin'.
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
        # обновляем имя/роль на случай изменений в конфиге
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
