"""Общие хелперы для admin/teacher хендлеров."""

from __future__ import annotations

from aiogram.types import Message

from src.config import settings
from src.db.models import User, UserRole


def is_admin(user: User) -> bool:
    """Преподаватели и админы — оба имеют доступ к admin-командам."""
    return user.role in (UserRole.admin, UserRole.teacher)


def is_superadmin(user: User) -> bool:
    """Только те, кто в ADMIN_TELEGRAM_IDS env, могут менять чужие роли.
    DB-уровневый admin не может — иначе скомпрометированный DB-admin
    смог бы повышать сообщников мимо env-gate владельца сервера.
    """
    return user.telegram_id in settings.admin_ids


def parse_role(arg: str | None, default: UserRole | None = UserRole.teacher) -> UserRole | None:
    if not arg:
        return default
    name = arg.strip().lower()
    for role in UserRole:
        if role.value == name or role.name == name:
            return role
    return None


async def deny(message: Message) -> None:
    """Видимое сообщение об отказе. Тихий return на /команде делал вид, что
    бот сломан — пользователи трижды подряд писали admin-команду. Лучше
    одна строчка с явным отказом."""
    await message.answer("⛔ Команда доступна только преподавателям и админам.")
