"""Обработка свободного вопроса от студента — главный entry в RAG-пайплайн."""

from __future__ import annotations

import asyncio
import contextlib
import html

import structlog
from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.enums.message_entity_type import MessageEntityType
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command  # noqa: F401  (для обратной совместимости импорта)
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReactionTypeEmoji
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import (
    disambig_keyboard,
    feedback_brief,
    feedback_inline,
    feedback_short_answer,
    processing_keyboard,
)
from src.bot.services import processing_state
from src.bot.services.qa_pipeline import run_qa_pipeline
from src.bot.states import StudentFlow
from src.db.models import User

log = structlog.get_logger(__name__)
router = Router(name="student_question")


_GROUP_CHAT_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}


async def _is_addressed_to_bot(message: Message) -> bool:
    """В группах принимаем вопрос только если бота явно позвали:

    - reply на сообщение бота, либо
    - в тексте есть @mention/text_mention бота.

    Privacy mode в BotFather у Telegram уже фильтрует это на стороне
    сервера, но строгая проверка нужна на случай выключенного privacy /
    админ-бота — иначе бот реагировал бы на каждое сообщение в чате.
    """
    if message.bot is None:
        return False
    me = await message.bot.me()  # aiogram кеширует — без сетевого hit
    bot_id = me.id
    bot_username = (me.username or "").lower() or None

    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    ):
        return True

    text = message.text or ""
    for ent in message.entities or []:
        if ent.type == MessageEntityType.MENTION and bot_username:
            mention = text[ent.offset : ent.offset + ent.length].lstrip("@").lower()
            if mention == bot_username:
                return True
        elif ent.type == MessageEntityType.TEXT_MENTION and ent.user and ent.user.id == bot_id:
            return True
    return False


@router.message(StudentFlow.awaiting_question, F.text)
@router.message(F.text & ~F.text.startswith("/"))
async def on_question(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Тонкий хендлер — вся RAG-логика в `qa_pipeline.run_qa_pipeline`."""
    # В группах/супергруппах отвечаем только если бот адресован (reply или
    # @mention). Без этого бот реагировал на любое сообщение в чате —
    # «ты че ебанутый?» в группе становился вопросом к учебнику.
    if (
        message.chat
        and message.chat.type in _GROUP_CHAT_TYPES
        and not await _is_addressed_to_bot(message)
    ):
        return

    question = (message.text or "").strip()
    if not question:
        await message.answer(texts.tr(lang, texts.PROMPT_QUESTION))
        return

    q_html = html.escape(question)
    q_with_status = texts.tr(lang, texts.Q_WITH_STATUS)

    initial_status_text = q_with_status.format(
        q=q_html, status=texts.tr(lang, texts.STATUS_RETRIEVING)
    )
    rid = processing_state.new_rid()
    processing_state.set_status(rid, texts.tr(lang, texts.STATUS_RETRIEVING))
    status_msg = await message.answer(
        initial_status_text,
        parse_mode="HTML",
        reply_markup=processing_keyboard(rid, lang),
    )
    bot = message.bot
    chat_id = message.chat.id

    # Запоминаем последний отправленный текст, чтобы пропускать no-op edits.
    # Иначе Telegram возвращает «Bad Request: message is not modified» и
    # наши edits незаметно ломаются — это раньше выглядело как «статус не меняется».
    last_sent = {"text": initial_status_text}

    async def _edit(new_text: str, reply_markup: object = None) -> None:
        # Дедупликация только по тексту — set_status каждый раз цепляет ту же
        # processing_keyboard, set_final меняет и текст и markup сразу.
        if new_text == last_sent["text"]:
            return  # дубль — спровоцировал бы «message is not modified»
        try:
            await status_msg.edit_text(new_text, parse_mode="HTML", reply_markup=reply_markup)
            last_sent["text"] = new_text
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.1)
            try:
                await status_msg.edit_text(
                    new_text, parse_mode="HTML", reply_markup=reply_markup
                )
                last_sent["text"] = new_text
            except Exception as retry_exc:
                log.warning("status_edit_retry_failed", exc_type=type(retry_exc).__name__)
        except TelegramBadRequest as exc:
            # «message is not modified» — ожидаемая идемпотентность; всё
            # остальное (parse entities, message too long, blocked) — реальное.
            if "not modified" in str(exc).lower():
                last_sent["text"] = new_text
                return
            log.warning(
                "status_edit_bad_request",
                exc=str(exc),
                text_prefix=new_text[:80].replace("\n", " "),
            )
        except Exception as exc:
            log.warning(
                "status_edit_failed",
                exc_type=type(exc).__name__,
                exc=str(exc),
                text_prefix=new_text[:80].replace("\n", " "),
            )

    async def set_status(status_text: str) -> None:
        # Стрим-превью оканчивается курсором «▍». В alert-сообщении кнопки
        # «Что сейчас делается?» это смотрелось бы как кусок ответа — для
        # alert-стейта подменяем на дружелюбный лейбл, а в Telegram-edit
        # передаём настоящий preview.
        alert_text = (
            texts.tr(lang, texts.STATUS_STREAMING)
            if status_text.rstrip().endswith("▍")
            else status_text
        )
        processing_state.set_status(rid, alert_text)
        # Перецепляем processing_keyboard на каждом edit'е. Telegram editMessageText
        # сбрасывает reply_markup, если его не передать — без этого кнопка
        # «Что сейчас делается?» исчезает после первого edit'а.
        await _edit(
            q_with_status.format(q=q_html, status=status_text),
            reply_markup=processing_keyboard(rid, lang),
        )

    async def _react(emoji: str) -> None:
        # Визуальный heartbeat на сообщение пользователя — переживает edit'ы
        # status-сообщения, так что студент видит «принял → ответил» даже
        # после прокрутки чата.
        if bot is None:
            return
        with contextlib.suppress(Exception):
            await bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message.message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )

    async def set_final(body: str, dialog_id: int, kind: str = "full") -> None:
        processing_state.clear(rid)  # снять wh-кнопку mapping
        # Клавиатура зависит от типа ответа:
        #   "disambig"          → одна кнопка «🔄 Да, про "Y"» (fuzzy-suggest).
        #   "glossary"/"faq"    → курированный однопараграфный + «📖 Развёрнутый».
        #   "full"/"web" brief  → только 👍/👎.
        #   "full"/"web" verbose → follow-ups + «задать ещё вопрос».
        if kind == "disambig":
            # body уже содержит фразу с предложенным термином в формате «… <b>«Y»</b> …»;
            # достаём его обратно для callback_data — самый дешёвый путь.
            import re as _re
            m = _re.search(r"<b>«([^»]+)»</b>\.\s*Возможно", body)
            if m is None:
                m = _re.search(r"<b>«([^»]+)»</b>\.\s*Did", body)
            term = (m.group(1) if m else "")[:50]
            kb = disambig_keyboard(term, lang) if term else feedback_brief(dialog_id, lang)
        elif kind in ("glossary", "faq"):
            kb = feedback_short_answer(dialog_id, lang)
        else:
            is_brief = (
                getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
            )
            kb = feedback_brief(dialog_id, lang) if is_brief else feedback_inline(dialog_id, lang)
        await _edit(body, reply_markup=kb)
        await _react("👍")

    async def typing_ping() -> None:
        if bot is not None:
            with contextlib.suppress(Exception):
                await bot.send_chat_action(chat_id, ChatAction.TYPING)

    await _react("👀")

    try:
        await run_qa_pipeline(
            question=question,
            session=session,
            user=user,
            lang=lang,
            set_status=set_status,
            set_final=set_final,
            typing_ping=typing_ping,
        )
    finally:
        processing_state.clear(rid)
        await state.clear()
