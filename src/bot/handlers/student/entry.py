"""Команды-входы: /ask, /subject и кнопки-сэмплы."""

from __future__ import annotations

import html

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import ask_samples_inline
from src.bot.states import StudentFlow
from src.db.models import Subject, User

router = Router(name="student_entry")


@router.message(Command("ask"))
async def on_ask(message: Message, state: FSMContext, lang: str) -> None:
    await state.set_state(StudentFlow.awaiting_question)
    await message.answer(
        texts.tr(lang, texts.PROMPT_QUESTION_HINT),
        reply_markup=ask_samples_inline(lang),
    )


@router.callback_query(F.data.startswith("ask_sample:"))
async def on_ask_sample(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Тап по сэмпл-вопросу → запускаем тот же пайплайн, что и для типнутого
    вопроса. Синтезируем Message с текстом сэмпла через Pydantic ``model_copy``,
    чтобы не дублировать on_question. ``from_user`` подменяем на реального
    студента — иначе логирование/throttling увидели бы бота, не его."""
    from src.bot.handlers.student.question import on_question

    await callback.answer()
    if callback.message is None:
        return
    try:
        idx = int((callback.data or "ask_sample:0").split(":", 1)[1])
        pair = texts.ASK_SAMPLES[idx]
    except (ValueError, IndexError):
        return
    sample = pair[0] if lang == "ru" else pair[1]

    await callback.message.answer(f"🗣 <i>{html.escape(sample)}</i>", parse_mode="HTML")

    synthetic = callback.message.model_copy(
        update={"text": sample, "from_user": callback.from_user}
    )
    await state.set_state(StudentFlow.awaiting_question)
    await on_question(synthetic, state, session, user, lang)


@router.message(Command("subject"))
async def on_subject_command(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    arg = (command.args or "").strip()
    if not arg:
        current = user.current_subject_slug
        if current:
            subj = (
                await session.execute(select(Subject).where(Subject.slug == current))
            ).scalar_one_or_none()
            title = texts.subject_title(subj, lang) if subj else current
            await message.answer(
                texts.tr(lang, texts.SUBJECT_LOCKED).format(title=title), parse_mode="HTML"
            )
        else:
            await message.answer(texts.tr(lang, texts.SUBJECT_CLEARED))
        return

    if arg.lower() == "clear":
        user.current_subject_slug = None
        await message.answer(texts.tr(lang, texts.SUBJECT_CLEARED))
        return

    subj = (
        await session.execute(
            select(Subject).where(Subject.slug == arg, Subject.is_active.is_(True))
        )
    ).scalar_one_or_none()
    if subj is None:
        await message.answer(
            texts.tr(lang, texts.SUBJECT_UNKNOWN).format(slug=arg), parse_mode="HTML"
        )
        return

    user.current_subject_slug = subj.slug
    await message.answer(
        texts.tr(lang, texts.SUBJECT_LOCKED).format(title=texts.subject_title(subj, lang)),
        parse_mode="HTML",
    )
