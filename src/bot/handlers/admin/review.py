"""Модерация фидбека: /teacher_review, /teacher_fix."""

from __future__ import annotations

from html import escape

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import deny, is_admin
from src.bot.services.faq_service import pending_reviews, save_faq_from_dialog
from src.bot.states import AdminFlow
from src.db.models import Dialog, User

log = structlog.get_logger(__name__)
router = Router(name="admin_review")


@router.message(Command("teacher_review"))
async def on_teacher_review(message: Message, session: AsyncSession, user: User) -> None:
    """/teacher_review — до 20 диалогов с 👎 без преподавательского FAQ.
    Каждый показывает вопрос, ответ бота (обрезанный) и dialog_id для /teacher_fix."""
    if not is_admin(user):
        await deny(message)
        return
    pending = await pending_reviews(session, limit=20)
    if not pending:
        await message.answer(
            "Очередь модерации пуста — негативных оценок без ответа препода нет. 🎉"
        )
        return

    lines = [f"<b>🧑‍🏫 Очередь модерации</b> ({len(pending)}):"]
    for pr in pending:
        q_preview = pr.question.replace("\n", " ")
        if len(q_preview) > 120:
            q_preview = q_preview[:119] + "…"
        a_preview = (pr.answer or "").replace("\n", " ")
        if len(a_preview) > 160:
            a_preview = a_preview[:159] + "…"
        subj = f" [<code>{escape(pr.subject_slug or 'web')}</code>]"
        # Иконка рейтинга — преподаватели просили контекст «почему студент недоволен».
        rating_icon = "👎" if pr.rating <= 2 else f"{pr.rating}⭐"
        lines.append(
            f"\n• <code>#{pr.dialog_id}</code>{subj} {rating_icon}\n"
            f"<b>В:</b> {escape(q_preview)}\n"
            f"<i>Бот ответил:</i> {escape(a_preview)}"
        )
    lines.append("\nПравь: <code>/teacher_fix &lt;id&gt;</code> — дальше бот спросит правильный ответ.")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("teacher_fix"))
async def on_teacher_fix(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_fix <dialog_id> — войти в FSM; следующее сообщение препода
    станет FAQ-ответом для нормализованного вопроса этого диалога."""
    if not is_admin(user):
        await deny(message)
        return
    arg = (command.args or "").strip()
    if not arg:
        await message.answer(
            "Использование: <code>/teacher_fix &lt;dialog_id&gt;</code>", parse_mode="HTML"
        )
        return
    try:
        dialog_id = int(arg)
    except ValueError:
        await message.answer("dialog_id должен быть числом.")
        return
    dialog = (
        await session.execute(select(Dialog).where(Dialog.id == dialog_id))
    ).scalar_one_or_none()
    if dialog is None:
        await message.answer(f"Диалог <code>#{dialog_id}</code> не найден.", parse_mode="HTML")
        return

    await state.set_state(AdminFlow.fixing_answer)
    await state.update_data(dialog_id=dialog_id)
    q_preview = dialog.question.replace("\n", " ")
    if len(q_preview) > 200:
        q_preview = q_preview[:199] + "…"
    await message.answer(
        f"<b>Вопрос студента:</b>\n{escape(q_preview)}\n\n"
        "Напиши правильный ответ одним сообщением. "
        "Можно HTML: <code>&lt;b&gt;…&lt;/b&gt;</code>, маркер «• ». "
        "Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.fixing_answer)
async def on_fix_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.")


@router.message(AdminFlow.fixing_answer, F.text)
async def on_fix_receive(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    # try/finally гарантирует очистку FSM даже при сбое DB или send-roundtrip'е —
    # без этого транзитный сбой оставлял препода в AdminFlow.fixing_answer навечно,
    # каждая следующая команда трактовалась как новый FAQ-body.
    try:
        data = await state.get_data()
        dialog_id = int(data.get("dialog_id") or 0)
        answer = (message.text or "").strip()
        if not answer:
            await message.answer("Пустой ответ — напиши текст или /cancel.")
            return
        if len(answer) > 3500:
            await message.answer("Слишком длинный ответ (лимит 3500 символов).")
            return
        try:
            entry = await save_faq_from_dialog(
                session, dialog_id=dialog_id, answer=answer, teacher=user
            )
            await session.commit()
        except Exception:
            log.exception("faq_save_failed", dialog_id=dialog_id)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить — попробуй позже.")
            return
        if entry is None:
            await message.answer("Не удалось сохранить — диалог пропал или пуст.")
            return
        log.info("faq_saved", faq_id=entry.id, parent_dialog=dialog_id, by_user=user.telegram_id)
        await message.answer(
            f"✅ Ответ сохранён как FAQ (<code>#{entry.id}</code>). "
            "Следующие студенты с таким же вопросом получат его без LLM.",
            parse_mode="HTML",
        )
    finally:
        await state.clear()
