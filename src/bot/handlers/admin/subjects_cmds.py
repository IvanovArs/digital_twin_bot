"""Просмотр предметов и переиндексация."""

from __future__ import annotations

import asyncio

import structlog
from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import deny, is_admin
from src.bot.services.admin_service import reindex_subject_in_background
from src.db.models import Subject, SubjectMaterial, User
from src.db.session import get_sessionmaker

log = structlog.get_logger(__name__)
router = Router(name="admin_subjects")


@router.message(Command("admin_subjects"))
async def on_admin_subjects(message: Message, session: AsyncSession, user: User) -> None:
    if not is_admin(user):
        await deny(message)
        return

    subjects = list((await session.execute(select(Subject).order_by(Subject.title_ru))).scalars())
    if not subjects:
        await message.answer("Нет предметов в courses.yaml.")
        return

    lines = ["<b>Предметы:</b>"]
    for s in subjects:
        mats = list(
            (
                await session.execute(
                    select(SubjectMaterial).where(SubjectMaterial.subject_id == s.id)
                )
            ).scalars()
        )
        indexed = sum(1 for m in mats if m.indexed_at is not None)
        active = "🟢" if s.is_active else "⚪"
        lines.append(
            f"{active} <code>{s.slug}</code> — {s.title_ru} "
            f"(материалов: {len(mats)}, проиндексировано: {indexed})"
        )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("admin_reindex"))
async def on_admin_reindex(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    if not is_admin(user):
        await deny(message)
        return

    slug_arg = (command.args or "").strip() or None
    if slug_arg is not None:
        exists = (
            await session.execute(select(Subject).where(Subject.slug == slug_arg))
        ).scalar_one_or_none()
        if exists is None:
            await message.answer(
                f"Неизвестный предмет: <code>{slug_arg}</code>", parse_mode="HTML"
            )
            return

    scope = slug_arg or "все предметы"
    await message.answer(f"🔄 Запускаю переиндексацию ({scope})… Это может занять пару минут.")

    asyncio.create_task(_reindex_and_report(message, slug_arg))  # noqa: RUF006


async def _reindex_and_report(message: Message, slug: str | None) -> None:
    try:
        await reindex_subject_in_background(get_sessionmaker(), subject_slug=slug)
        await message.answer("✅ Переиндексация завершена.")
    except Exception:
        # Не светим текст исключения в чат — пути и трейс полезнее атакующему,
        # чем админу. Полный stack trace — в structlog.
        log.exception("reindex_task_failed")
        try:
            await message.answer("⚠️ Ошибка переиндексации. Подробности — в логах.")
        except Exception:
            log.exception("reindex_task_report_failed")
