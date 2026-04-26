"""Загрузка PDF/DOCX/MD/TXT учебников через Telegram."""

from __future__ import annotations

from pathlib import PurePosixPath

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import Document, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import deny, is_admin
from src.bot.services.admin_service import save_material
from src.bot.services.upload_guard import check_magic, check_size
from src.bot.states import AdminFlow
from src.config import settings
from src.db.models import Subject, User

log = structlog.get_logger(__name__)
router = Router(name="admin_upload")


@router.message(Command("admin_upload"))
async def on_admin_upload(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    if not is_admin(user):
        await deny(message)
        return

    slug = (command.args or "").strip()
    if not slug:
        await message.answer(
            "Использование: <code>/admin_upload &lt;slug&gt;</code>", parse_mode="HTML"
        )
        return

    subj = (await session.execute(select(Subject).where(Subject.slug == slug))).scalar_one_or_none()
    if subj is None:
        await message.answer(f"Неизвестный предмет: <code>{slug}</code>", parse_mode="HTML")
        return

    await state.set_state(AdminFlow.uploading_material)
    await state.update_data(subject_slug=slug)
    await message.answer(
        f"Пришли файл для предмета <b>{subj.title_ru}</b>.\n"
        f"Поддерживаются: <code>.pdf .docx .md .txt</code>. Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.uploading_material)
async def on_admin_upload_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Загрузка отменена.")


@router.message(AdminFlow.uploading_material, F.document)
async def on_admin_upload_doc(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    # try/finally: любое исключение в загрузке/сохранении оставляло препода
    # в AdminFlow.uploading_material, и каждое следующее сообщение (включая
    # /cancel) трактовалось как новый документ.
    try:
        data = await state.get_data()
        slug = str(data.get("subject_slug") or "")
        subj = (
            await session.execute(select(Subject).where(Subject.slug == slug))
        ).scalar_one_or_none()
        if subj is None:
            await message.answer(
                "Предмет пропал. Начни заново: /admin_upload &lt;slug&gt;", parse_mode="HTML"
            )
            return

        doc: Document = message.document  # type: ignore[assignment]
        # ``doc.file_name`` контролируется клиентом. Срезаем path-компоненты,
        # чтобы клиент не мог писать вне BOOKS_DIR через "../../etc/foo.pdf"
        # или абсолютный POSIX-путь "/etc/passwd".
        raw_name = doc.file_name or "unnamed"
        safe_name = PurePosixPath(raw_name).name
        if not safe_name or safe_name.startswith(".") or "/" in safe_name or "\\" in safe_name:
            await message.answer("Недопустимое имя файла.")
            return
        fname_lower = safe_name.lower()
        if not fname_lower.endswith((".pdf", ".docx", ".md", ".txt")):
            await message.answer("Поддерживаются: .pdf .docx .md .txt")
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
        magic_err = check_magic(safe_name, payload)
        if magic_err is not None:
            await message.answer(magic_err)
            return

        try:
            await save_material(
                session,
                subject=subj,
                filename=safe_name,
                payload=payload,
                uploader=user,
                target_dir=settings.BOOKS_DIR,
            )
        except Exception:
            log.exception("save_material_failed", subject=subj.slug, filename=safe_name)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить файл — попробуй позже.")
            return
        await message.answer(
            f"✅ Файл сохранён. Запусти /admin_reindex <code>{subj.slug}</code>, "
            "чтобы включить его в индекс.",
            parse_mode="HTML",
        )
    finally:
        await state.clear()
