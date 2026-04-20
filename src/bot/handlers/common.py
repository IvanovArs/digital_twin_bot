"""Common commands: /start, /help, /subjects, /glossary + main-menu callbacks."""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import main_inline
from src.bot.services.glossary_service import find_term, list_terms
from src.bot.states import StudentFlow
from src.config import ROOT
from src.db.models import AnswerMode, Subject, User, UserRole

log = structlog.get_logger(__name__)
router = Router(name="common")

START_VIDEO = ROOT / "data" / "bot" / "assets" / "start.mp4"


# ---------- /start ----------


@router.message(CommandStart(deep_link=True))
async def on_start_with_payload(
    message: Message,
    command: CommandObject,
    user: User,
    lang: str,
) -> None:
    """Deep-link /start payload handler. Inline mode's «📖 Инструкция» sends
    the user to ``t.me/<bot>?start=help`` — we resolve that to the help text.
    Unknown payloads fall through to the normal greeting.
    """
    payload = (command.args or "").strip().lower()
    if payload == "help":
        await _send_help(message, user, lang)
        return
    await _send_greeting(message, user, lang)


@router.message(CommandStart())
async def on_start(message: Message, user: User, lang: str) -> None:
    await _send_greeting(message, user, lang)


async def _send_greeting(message: Message, user: User, lang: str) -> None:
    caption = texts.tr(lang, texts.HELLO).format(name=user.full_name or "друг")

    if START_VIDEO.exists():
        try:
            await message.answer_animation(
                animation=FSInputFile(str(START_VIDEO)),
                caption=caption,
                parse_mode="HTML",
                reply_markup=main_inline(lang),
            )
            return
        except Exception:
            log.exception("start_video_failed", path=str(START_VIDEO))

    await message.answer(caption, parse_mode="HTML", reply_markup=main_inline(lang))


# ---------- /help ----------


async def _send_help(message: Message, user: User, lang: str) -> None:
    me = await message.bot.get_me()  # type: ignore[union-attr]
    body = texts.tr(lang, texts.HELP_STUDENT).format(bot_username=me.username)
    if user.role in (UserRole.admin, UserRole.teacher):
        body += texts.tr(lang, texts.HELP_ADMIN_EXTRA)
    # Env-var superadmin gets an additional role-management block. Compare
    # by telegram_id (not DB role) — a superadmin with role=student in the
    # DB is still the server-owner and needs the full block.
    from src.config import settings as _settings

    if user.telegram_id in _settings.admin_ids:
        body += texts.tr(lang, texts.HELP_SUPERADMIN_EXTRA)
    await message.answer(body, parse_mode="HTML", reply_markup=main_inline(lang))


@router.message(Command("help"))
async def on_help(message: Message, user: User, lang: str) -> None:
    await _send_help(message, user, lang)


# ---------- /mode ----------


@router.message(Command("mode"))
async def on_mode(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/mode [brief|verbose] — set or view answer style.

    Without an argument: show current mode + cheat-sheet. With an
    argument: set it, persist immediately (committed outside the handler
    by the middleware, but we flush here to be safe on SQLAlchemy 2.0
    autoflush-off sessions).
    """
    arg = (command.args or "").strip().lower()
    if not arg:
        label_key = (
            texts.MODE_BRIEF_LABEL
            if user.answer_mode is AnswerMode.brief
            else texts.MODE_VERBOSE_LABEL
        )
        current = texts.tr(lang, label_key)
        await message.answer(
            texts.tr(lang, texts.MODE_SET).format(mode=current)
            + "\n\n"
            + texts.tr(lang, texts.MODE_HELP),
            parse_mode="HTML",
        )
        return
    # Accept English labels only (brief/verbose) — they match the enum.
    if arg not in {m.value for m in AnswerMode}:
        await message.answer(texts.tr(lang, texts.MODE_HELP), parse_mode="HTML")
        return
    user.answer_mode = AnswerMode(arg)
    await session.flush()
    await session.commit()
    label_key = (
        texts.MODE_BRIEF_LABEL
        if user.answer_mode is AnswerMode.brief
        else texts.MODE_VERBOSE_LABEL
    )
    current = texts.tr(lang, label_key)
    await message.answer(
        texts.tr(lang, texts.MODE_SET).format(mode=current), parse_mode="HTML"
    )


# ---------- /subjects ----------


async def _send_subjects(message: Message, session: AsyncSession, lang: str) -> None:
    subjects = (
        (
            await session.execute(
                select(Subject).where(Subject.is_active.is_(True)).order_by(Subject.title_ru)
            )
        )
        .scalars()
        .all()
    )

    if not subjects:
        await message.answer(
            texts.tr(lang, texts.SUBJECTS_EMPTY),
            reply_markup=main_inline(lang),
        )
        return

    body = texts.tr(lang, texts.SUBJECTS_LIST_HEADER) + "\n".join(
        texts.tr(lang, texts.SUBJECTS_LIST_ITEM).format(
            title=texts.subject_title(s, lang), slug=s.slug
        )
        for s in subjects
    )
    await message.answer(body, parse_mode="HTML", reply_markup=main_inline(lang))


@router.message(Command("subjects"))
async def on_subjects(message: Message, session: AsyncSession, lang: str) -> None:
    await _send_subjects(message, session, lang)


# ---------- /glossary ----------


@router.message(Command("glossary"))
async def on_glossary(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    query = (command.args or "").strip()
    subject_slug = user.current_subject_slug

    if query:
        rows = await find_term(session, query, subject_slug=subject_slug)
        if not rows:
            await message.answer(
                texts.tr(lang, texts.GLOSSARY_NOT_FOUND).format(query=query),
                parse_mode="HTML",
                reply_markup=main_inline(lang),
            )
            return
        header = texts.tr(lang, texts.GLOSSARY_HEADER_SEARCH).format(query=query, count=len(rows))
    else:
        rows = await list_terms(session, subject_slug=subject_slug)
        if not rows:
            await message.answer(
                texts.tr(lang, texts.GLOSSARY_EMPTY), reply_markup=main_inline(lang)
            )
            return
        if subject_slug is not None:
            subj = (
                await session.execute(select(Subject).where(Subject.slug == subject_slug))
            ).scalar_one_or_none()
            subj_title = texts.subject_title(subj, lang) if subj else subject_slug
            header = texts.tr(lang, texts.GLOSSARY_HEADER_SUBJECT).format(
                subject=subj_title, count=len(rows)
            )
        else:
            header = texts.tr(lang, texts.GLOSSARY_HEADER_ALL).format(count=len(rows))

    items = [
        texts.tr(lang, texts.GLOSSARY_ITEM).format(term=r.term, definition=r.definition)
        for r in rows[:20]
    ]
    body = header + "\n".join(items)
    if len(rows) > 20:
        body += texts.tr(lang, texts.GLOSSARY_TRUNCATED)
    await message.answer(body, parse_mode="HTML", reply_markup=main_inline(lang))


# ---------- main-menu inline callbacks ----------


@router.callback_query(F.data == "menu:ask")
async def on_menu_ask(
    callback: CallbackQuery,
    state: FSMContext,
    lang: str,
) -> None:
    # If this callback came from an inline-sent message in some other chat
    # (not our bot's PM), the FSM flow would need a chat we don't have.
    if callback.message is None:
        await callback.answer(texts.tr(lang, texts.PROMPT_QUESTION), show_alert=True)
        return
    await state.set_state(StudentFlow.awaiting_question)
    await callback.answer()
    await callback.message.answer(texts.tr(lang, texts.PROMPT_QUESTION))


@router.callback_query(F.data == "menu:help")
async def on_menu_help(callback: CallbackQuery, user: User, lang: str) -> None:
    await callback.answer()
    if callback.message is not None:
        await _send_help(callback.message, user, lang)  # type: ignore[arg-type]


@router.callback_query(F.data == "menu:subjects")
async def on_menu_subjects(
    callback: CallbackQuery,
    session: AsyncSession,
    lang: str,
) -> None:
    await callback.answer()
    if callback.message is not None:
        await _send_subjects(callback.message, session, lang)  # type: ignore[arg-type]
