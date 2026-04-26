"""Загрузка преподавательского глоссария (CSV/YAML)."""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import deny, is_admin
from src.bot.services.glossary_upload import parse_glossary_payload, replace_glossary
from src.bot.services.upload_guard import check_magic, check_size
from src.bot.states import AdminFlow
from src.db.models import Subject, User

log = structlog.get_logger(__name__)
router = Router(name="admin_glossary")


@router.message(Command("teacher_glossary_upload"))
async def on_teacher_glossary_upload(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_glossary_upload <slug> — принять CSV/YAML с парами term,definition
    и заменить глоссарий предмета загруженным набором."""
    if not is_admin(user):
        await deny(message)
        return
    slug = (command.args or "").strip()
    if not slug:
        await message.answer(
            "Использование: <code>/teacher_glossary_upload &lt;slug&gt;</code>", parse_mode="HTML"
        )
        return
    subj = (await session.execute(select(Subject).where(Subject.slug == slug))).scalar_one_or_none()
    if subj is None:
        await message.answer(f"Неизвестный slug: <code>{slug}</code>", parse_mode="HTML")
        return
    await state.set_state(AdminFlow.uploading_glossary)
    await state.update_data(subject_slug=slug)
    await message.answer(
        f"Пришли CSV или YAML с глоссарием для <b>{subj.title_ru}</b>. "
        "CSV: <code>term,definition</code>. "
        "YAML: <code>terms: [{term: …, definition: …}]</code>. "
        "Файл <b>заменит</b> существующий глоссарий предмета. Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.uploading_glossary)
async def on_glossary_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.")


@router.message(AdminFlow.uploading_glossary, F.document)
async def on_glossary_doc(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    try:
        data = await state.get_data()
        slug = str(data.get("subject_slug") or "")
        subj = (await session.execute(select(Subject).where(Subject.slug == slug))).scalar_one_or_none()
        if subj is None:
            await message.answer("Предмет пропал. Начни заново: /teacher_glossary_upload")
            return

        doc = message.document
        if doc is None:
            return
        fname = (doc.file_name or "glossary").lower()
        if not fname.endswith((".csv", ".yaml", ".yml")):
            await message.answer("Поддерживаются только .csv / .yaml / .yml.")
            return
        size_err = check_size(doc.file_size)
        if size_err is not None:
            await message.answer(size_err)
            return
        if message.bot is None:
            await message.answer("Не удалось связаться с Telegram API.")
            return
        file = await message.bot.download(doc)
        if file is None:
            await message.answer("Не удалось скачать файл.")
            return
        payload = file.read()
        magic_err = check_magic(fname, payload)
        if magic_err is not None:
            await message.answer(magic_err)
            return
        try:
            entries = parse_glossary_payload(payload, fname)
        except ValueError as exc:
            await message.answer(f"Ошибка разбора: {exc}")
            return
        if not entries:
            await message.answer("В файле не нашлось ни одной пары term/definition.")
            return

        try:
            result = await replace_glossary(session, subject=subj, entries=entries)
            await session.commit()
        except Exception:
            log.exception("glossary_save_failed", subject=subj.slug)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить глоссарий — попробуй позже.")
            return
        log.info(
            "glossary_uploaded",
            subject=subj.slug,
            inserted=result.inserted,
            replaced=result.replaced_previous,
            skipped=result.skipped_empty,
            by_user=user.telegram_id,
        )
        await message.answer(
            f"✅ Глоссарий обновлён для <b>{subj.title_ru}</b>:\n"
            f"• добавлено: <b>{result.inserted}</b>\n"
            f"• заменено предыдущих: {result.replaced_previous}\n"
            f"• пропущено пустых/дубликатов: {result.skipped_empty}",
            parse_mode="HTML",
        )
    finally:
        await state.clear()
