"""Управление ролями (только суперадмин): /whoami, /admin_promote, /admin_demote, /admin_users."""

from __future__ import annotations

import structlog
from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import is_superadmin, parse_role
from src.db.models import User, UserRole

log = structlog.get_logger(__name__)
router = Router(name="admin_roles")


@router.message(Command("whoami"))
async def on_whoami(message: Message, user: User) -> None:
    """Любой может узнать свой telegram_id и текущую роль — нужно, чтобы
    кандидат в преподаватели мог отправить свой id суперадмину."""
    tag = "superadmin" if is_superadmin(user) else user.role.value
    await message.answer(
        f"<b>Вы:</b> {user.full_name or '—'}\n"
        f"• Telegram ID: <code>{user.telegram_id}</code>\n"
        f"• Роль: <b>{tag}</b>",
        parse_mode="HTML",
    )


@router.message(Command("admin_promote"))
async def on_admin_promote(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_promote <telegram_id> [teacher|admin|student]

    Дефолт — teacher (самый частый случай). Целевой пользователь должен хотя бы
    раз написать боту, чтобы его User-строка существовала в БД."""
    if not is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/admin_promote &lt;telegram_id&gt; [teacher|admin|student]</code>\n"
            "Пользователь должен хотя бы раз нажать /start у бота.",
            parse_mode="HTML",
        )
        return
    try:
        target_tg = int(args[0])
    except ValueError:
        await message.answer("Telegram ID должен быть числом.")
        return
    role = parse_role(args[1] if len(args) > 1 else None)
    if role is None:
        await message.answer("Неизвестная роль. Допустимо: teacher, admin, student.")
        return
    target = (
        await session.execute(select(User).where(User.telegram_id == target_tg))
    ).scalar_one_or_none()
    if target is None:
        await message.answer(
            f"Пользователь <code>{target_tg}</code> не найден в базе. "
            "Попроси его написать боту /start и повтори.",
            parse_mode="HTML",
        )
        return
    if target.role is role:
        await message.answer(
            f"У <code>{target_tg}</code> уже роль <b>{role.value}</b>.", parse_mode="HTML"
        )
        return
    previous = target.role
    target.role = role
    await session.commit()
    log.info(
        "role_promoted",
        by_user=user.telegram_id,
        target=target_tg,
        from_role=previous.value,
        to_role=role.value,
    )
    await message.answer(
        f"✅ <code>{target_tg}</code> ({target.full_name or '—'}) "
        f"теперь <b>{role.value}</b> (было: {previous.value}).",
        parse_mode="HTML",
    )


@router.message(Command("admin_demote"))
async def on_admin_demote(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_demote <telegram_id> — сбросить роль до student."""
    if not is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/admin_demote &lt;telegram_id&gt;</code>", parse_mode="HTML"
        )
        return
    try:
        target_tg = int(args[0])
    except ValueError:
        await message.answer("Telegram ID должен быть числом.")
        return
    target = (
        await session.execute(select(User).where(User.telegram_id == target_tg))
    ).scalar_one_or_none()
    if target is None:
        await message.answer(f"Пользователь <code>{target_tg}</code> не найден.", parse_mode="HTML")
        return
    if target.role is UserRole.student:
        await message.answer(f"<code>{target_tg}</code> и так student.", parse_mode="HTML")
        return
    previous = target.role
    target.role = UserRole.student
    await session.commit()
    log.info("role_demoted", by_user=user.telegram_id, target=target_tg, from_role=previous.value)
    await message.answer(
        f"✅ <code>{target_tg}</code> ({target.full_name or '—'}) "
        f"теперь <b>student</b> (было: {previous.value}).",
        parse_mode="HTML",
    )


@router.message(Command("admin_users"))
async def on_admin_users(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_users [teacher|admin|student] — список пользователей,
    опционально отфильтрованный по роли. Только суперадмин — чтобы личные
    данные не утекали DB-админу-teacher."""
    if not is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    filter_role = parse_role((command.args or "").strip() or None, default=None)
    stmt = select(User).order_by(User.role, User.telegram_id)
    if filter_role is not None:
        stmt = stmt.where(User.role == filter_role)
    users = list((await session.execute(stmt)).scalars())
    if not users:
        await message.answer("Нет пользователей под этот фильтр.")
        return
    # Обрезаем — иначе на большой БД пробьём 4096-cap Telegram'а.
    lines = [f"<b>Пользователи</b> ({len(users)}):"]
    for u in users[:80]:
        lines.append(
            f"• <code>{u.telegram_id}</code> — {u.full_name or '—'} ({u.role.value})"
        )
    if len(users) > 80:
        lines.append(f"… и ещё {len(users) - 80}")
    await message.answer("\n".join(lines), parse_mode="HTML")
