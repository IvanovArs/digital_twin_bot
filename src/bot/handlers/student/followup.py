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
from src.bot.keyboards import (
    ask_only_inline,
    feedback_bare,
    feedback_brief,
    feedback_inline,
    feedback_short_answer,
)
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
    """Тап по follow-up на PM-ответе. Редактируем то же сообщение в месте —
    лента не засоряется новыми «карточками» на каждый «Проще / Пример /
    Подробнее». Если пользователь хочет историю переформулировок — она
    остаётся в /history (каждый follow-up создаёт новый Dialog в БД)."""
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
    inline_message_id = callback.inline_message_id
    bot = callback.bot
    # Поддерживаем оба пути: PM/группа (есть msg) и inline-сообщение
    # (есть только inline_message_id, msg=None). Без этого тапы по
    # «Проще/Пример/Подробнее» в inline-ответах молча уходили в никуда.
    if (msg is None or msg.bot is None or msg.chat is None) and not inline_message_id:
        return
    if bot is None and msg is not None:
        bot = msg.bot

    # Подтаскиваем исходный вопрос, чтобы держать blockquote сверху и во
    # время стрима. Без этого preview перетирает «<blockquote>q</blockquote>»
    # и формат сообщения «прыгает» между статусом и финалом.
    parent = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user.id)
        )
    ).scalar_one_or_none()
    raw_q = (parent.question or "") if parent is not None else ""
    raw_q = re.sub(r"^\[(?:simplify|example|deepen)\]\s*", "", raw_q)
    # Срезаем хвост со status-сообщением (старый баг — мог попасть в saved
    # question), чтобы blockquote сверху не показывал «вопрос ⌛ Ищу...».
    raw_q = re.sub(r"\s*[⌛⏳]\s.*$|\s*\n+.*$", "", raw_q, flags=re.DOTALL).strip()
    q_html = html.escape(raw_q) if raw_q else ""
    q_with_status = texts.tr(lang, texts.Q_WITH_STATUS)

    last_sent: dict[str, str] = {"text": (msg.text if msg is not None else "") or ""}

    async def _edit(text: str, reply_markup=None) -> None:  # type: ignore[no-untyped-def]
        if text == last_sent["text"]:
            return
        try:
            if inline_message_id is not None and bot is not None:
                await bot.edit_message_text(
                    inline_message_id=inline_message_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                )
            else:
                await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)  # type: ignore[union-attr]
            last_sent["text"] = text
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.1)
            with contextlib.suppress(Exception):
                if inline_message_id is not None and bot is not None:
                    await bot.edit_message_text(
                        inline_message_id=inline_message_id,
                        text=text,
                        parse_mode="HTML",
                        reply_markup=reply_markup,
                    )
                else:
                    await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)  # type: ignore[union-attr]
                last_sent["text"] = text
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc).lower():
                log.warning("followup_edit_bad_request", exc=str(exc))
        except Exception as exc:
            log.warning("followup_edit_failed", exc_type=type(exc).__name__, exc=str(exc))

    async def set_status(text: str) -> None:
        # Держим вопрос blockquote'ом сверху и во время стрима — финальный
        # `render_textbook_body` использует тот же шаблон Q_WITH_STATUS.
        if q_html:
            await _edit(q_with_status.format(q=q_html, status=text))
        else:
            await _edit(text)

    async def set_final(body: str, new_dialog_id: int, kind: str = "full") -> None:
        del kind  # follow-up всегда получает полную verbose-клавиатуру
        # Inline-mode-ответам нужна inline-friendly клавиатура с
        # switch_inline_query вместо callback_data="menu:ask" (который не
        # работает вне нашего PM).
        kb = (
            feedback_bare(new_dialog_id, lang)
            if inline_message_id is not None
            else feedback_inline(new_dialog_id, lang)
        )
        await _edit(body, reply_markup=kb)

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
            await msg.edit_text(texts.tr(lang, texts.FU_EXPIRED))


@router.callback_query(F.data.startswith("at:"))
async def on_ask_term(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Тап по «🔄 Да, про "Y"» в disambig-карточке.

    Перезапускаем pipeline уже с правильным термином, edit'им то же
    сообщение в месте — чтобы лента не засорялась.
    """
    data = callback.data or ""
    term = data[len("at:"):].strip()
    if not term or len(term) > 80:
        await callback.answer()
        return
    await callback.answer()
    msg = callback.message
    if msg is None or msg.bot is None:
        return

    # Перепакуем как «что такое <term>» — pipeline пройдёт обычным путём
    # с lexical-gate (который уже не сработает, термин в глоссарии).
    rephrased = f"что такое {term}"
    q_html = html.escape(rephrased)
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
                log.warning("ask_term_edit_bad_request", exc=str(exc))
        except Exception as exc:
            log.warning("ask_term_edit_failed", exc_type=type(exc).__name__, exc=str(exc))

    async def set_status(text: str) -> None:
        await _edit(q_with_status.format(q=q_html, status=text))

    async def set_final(body: str, new_dialog_id: int, kind: str = "full") -> None:
        if kind in ("glossary", "faq"):
            kb = feedback_short_answer(new_dialog_id, lang)
        elif kind == "disambig":
            # На fuzzy-suggested-термин снова не нашлось — крайне редкий путь;
            # оставим в feedback_brief, чтобы юзер мог как минимум 👎 ткнуть.
            kb = feedback_brief(new_dialog_id, lang)
        else:
            is_brief = (
                getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
            )
            kb = feedback_brief(new_dialog_id, lang) if is_brief else feedback_inline(new_dialog_id, lang)
        await _edit(body, reply_markup=kb)

    await run_qa_pipeline(
        question=rephrased,
        session=session,
        user=user,
        lang=lang,
        set_status=set_status,
        set_final=set_final,
        skip_short_circuit=True,  # disambig — это уже «не первый» запрос
    )


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
