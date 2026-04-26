"""Callback'и поверх ответа: 👍/👎, «Проще/Пример/Подробнее», expand, «что сейчас?»."""

from __future__ import annotations

import asyncio
import contextlib
import html
import re

import structlog
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReactionTypeEmoji,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import ask_only_inline, feedback_inline, feedback_short_answer
from src.bot.services import processing_state
from src.bot.services.dialog_service import record_feedback
from src.bot.services.qa_pipeline import run_followup_pipeline, run_qa_pipeline
from src.db.models import Dialog, User

log = structlog.get_logger(__name__)
router = Router(name="student_followup")


@router.callback_query(F.data.startswith("wh:"))
async def on_whats_happening(callback: CallbackQuery, lang: str) -> None:
    rid = (callback.data or "wh:").split(":", 1)[1]
    stage = processing_state.get_status(rid) or texts.tr(lang, texts.ALERT_STAGE_UNKNOWN)
    # Telegram капает alert-текст на 200 chars (UTF-16 code units). Стрим-превью
    # легко это перебивает — обрезаем жёстко, чтобы не упасть с MESSAGE_TOO_LONG.
    # HTML-теги тоже срезаем: alert — plain text, и остатки <b> из preview уродуют.
    plain = re.sub(r"<[^>]+>", "", stage).replace("&lt;", "<").replace("&gt;", ">")
    if len(plain) > 190:
        plain = plain[:189] + "…"
    await callback.answer(text=plain, show_alert=True)


@router.callback_query(F.data.startswith("expand:"))
async def on_expand_answer(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Студент нажал «📖 Развёрнутый ответ» на короткий glossary/FAQ-ответ.
    Перезапускаем полный RAG на том же вопросе из Dialog'а и **редактируем
    то же сообщение** — короткое определение превращается в полный ответ
    без дублирующей карточки. Statbus в формате `<blockquote>question</blockquote>
    ⌛ …` чтобы студент не терял из виду, что ищется."""
    data = (callback.data or "").split(":", 1)
    if len(data) != 2:
        await callback.answer()
        return
    try:
        dialog_id = int(data[1])
    except ValueError:
        await callback.answer()
        return
    dialog = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user.id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        await callback.answer("Диалог не найден.", show_alert=True)
        return
    question = (dialog.question or "").strip()
    if not question:
        await callback.answer()
        return
    await callback.answer()

    msg = callback.message
    if msg is None or msg.bot is None:
        return

    q_html = html.escape(question)
    q_with_status = texts.tr(lang, texts.Q_WITH_STATUS)
    last_sent: dict[str, str] = {"text": msg.text or ""}

    async def _edit(text: str, reply_markup=None) -> None:  # type: ignore[no-untyped-def]
        if text == last_sent["text"]:
            return
        try:
            await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
            last_sent["text"] = text
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.1)
            with contextlib.suppress(Exception):
                await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
                last_sent["text"] = text
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc).lower():
                log.warning("expand_edit_bad_request", exc=str(exc))
        except Exception as exc:
            log.warning("expand_edit_failed", exc_type=type(exc).__name__, exc=str(exc))

    async def set_status(text: str) -> None:
        # Сохраняем вопрос в blockquote сверху — единый формат с обычным /ask.
        await _edit(q_with_status.format(q=q_html, status=text))

    async def set_final(body: str, new_dialog_id: int, kind: str = "full") -> None:
        if kind in ("glossary", "faq"):
            kb = feedback_short_answer(new_dialog_id, lang)
        else:
            kb = feedback_inline(new_dialog_id, lang)
        await _edit(body, reply_markup=kb)

    await run_qa_pipeline(
        question=question,
        session=session,
        user=user,
        lang=lang,
        set_status=set_status,
        set_final=set_final,
        skip_short_circuit=True,  # не вернуть тот же glossary-однострочник
    )


_ALLOWED_MODIFIERS = frozenset({"simplify", "example", "deepen"})


@router.callback_query(F.data.startswith("fu:"))
async def on_followup(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Тап по follow-up на PM-ответе. Шлём новый плейсхолдер ниже оригинала,
    стрим в него переформулированный ответ, цепляем свежую feedback+followup-клаву."""
    data = callback.data or ""
    parts = data.split(":", 2)
    if len(parts) != 3:
        await callback.answer()
        return
    try:
        dialog_id = int(parts[1])
    except ValueError:
        await callback.answer()
        return
    modifier = parts[2]
    if modifier not in _ALLOWED_MODIFIERS:
        await callback.answer()
        return

    # Быстрое подтверждение, чтобы кнопка не крутилась пока работаем.
    await callback.answer()

    msg = callback.message
    if msg is None or msg.bot is None or msg.chat is None:
        # Inline-message follow-up'ы пока не поддержаны — отвечаем тихо.
        return

    placeholder = await msg.answer(texts.tr(lang, texts.FU_PLACEHOLDER), parse_mode="HTML")
    last_sent = {"text": placeholder.text or ""}

    async def _edit(text: str, reply_markup=None) -> None:  # type: ignore[no-untyped-def]
        if text == last_sent["text"]:
            return
        try:
            await placeholder.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
            last_sent["text"] = text
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.1)
            with contextlib.suppress(Exception):
                await placeholder.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
                last_sent["text"] = text
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc).lower():
                log.warning("followup_edit_bad_request", exc=str(exc))
        except Exception as exc:
            log.warning("followup_edit_failed", exc_type=type(exc).__name__, exc=str(exc))

    async def set_status(text: str) -> None:
        await _edit(text)

    async def set_final(body: str, new_dialog_id: int, kind: str = "full") -> None:
        del kind  # follow-up всегда получает полную verbose-клавиатуру
        await _edit(body, reply_markup=feedback_inline(new_dialog_id, lang))

    ok = await run_followup_pipeline(
        dialog_id=dialog_id,
        modifier=modifier,
        session=session,
        user=user,
        lang=lang,
        set_status=set_status,
        set_final=set_final,
    )
    if not ok:
        with contextlib.suppress(Exception):
            await placeholder.edit_text(texts.tr(lang, texts.FU_EXPIRED))


@router.callback_query(F.data.startswith("fb:"))
async def on_feedback(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    _ = user
    _, dialog_id_str, rating_str = callback.data.split(":")  # type: ignore[union-attr]
    dialog_id = int(dialog_id_str)
    rating = int(rating_str)

    inserted = await record_feedback(
        session, dialog_id=dialog_id, user_id=user.id, rating=rating
    )
    # Маленький дофамин: 🎉-реакция на положительную оценку, только при первом
    # инсёрте — повторный rating не должен спамить реакциями.
    if inserted and rating >= 5 and callback.message is not None:
        with contextlib.suppress(Exception):
            await callback.bot.set_message_reaction(  # type: ignore[union-attr]
                chat_id=callback.message.chat.id,
                message_id=callback.message.message_id,
                reaction=[ReactionTypeEmoji(emoji="🎉")],
            )
    await callback.answer(
        texts.tr(lang, texts.FEEDBACK_THANKS if inserted else texts.FEEDBACK_ALREADY),
        show_alert=False,
    )

    # Callback приходит либо из PM (callback.message задан), либо из inline-сообщения
    # в чужом чате (callback.inline_message_id задан, callback.message=None).
    if callback.inline_message_id:
        # Inline-flow: дроп 👍/👎 ряда, но СОХРАНЯЕМ «Задать ещё вопрос»
        # (это switch_inline_query_current_chat, работает в любом чате
        # без членства бота).
        with contextlib.suppress(Exception):
            await callback.bot.edit_message_reply_markup(  # type: ignore[union-attr]
                inline_message_id=callback.inline_message_id,
                reply_markup=ask_only_inline(lang),
            )
        return

    # PM-flow: оставляем «Задать ещё вопрос» как shortcut.
    ask_only = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_ASK_ANOTHER),
                    callback_data="menu:ask",
                    style="primary",
                )
            ]
        ]
    )
    with contextlib.suppress(Exception):
        await callback.message.edit_reply_markup(reply_markup=ask_only)  # type: ignore[union-attr]
