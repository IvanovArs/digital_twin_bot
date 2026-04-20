"""Student flow: /ask button → question → live-status RAG answer → feedback."""

from __future__ import annotations

import contextlib
import html

import structlog
from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message, ReactionTypeEmoji
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import (
    ask_only_inline,
    ask_samples_inline,
    feedback_brief,
    feedback_inline,
    feedback_short_answer,
    main_inline,
    processing_keyboard,
)
from src.bot.services import processing_state
from src.bot.services.dialog_service import (
    record_feedback,
    set_favourite,
    user_dialogs_page,
)
from src.bot.services.qa_pipeline import run_followup_pipeline, run_qa_pipeline
from src.bot.states import StudentFlow
from src.db.models import Subject, User

log = structlog.get_logger(__name__)
router = Router(name="student")


# ---------- /ask entry ----------


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
    """Sample-question tapped → run the same pipeline as a typed question.

    We synthesise a Message with the sample text via Pydantic ``model_copy``
    so we don't have to re-implement the on_question flow. The callback's
    ``from_user`` is plugged in so logging / per-user throttling see the
    real student, not the bot.
    """
    await callback.answer()
    if callback.message is None:
        return
    try:
        idx = int((callback.data or "ask_sample:0").split(":", 1)[1])
        pair = texts.ASK_SAMPLES[idx]
    except (ValueError, IndexError):
        return
    sample = pair[0] if lang == "ru" else pair[1]

    # Echo the picked question as a chat line so the student sees what was sent.
    await callback.message.answer(f"🗣 <i>{html.escape(sample)}</i>", parse_mode="HTML")

    synthetic = callback.message.model_copy(
        update={"text": sample, "from_user": callback.from_user}
    )
    await state.set_state(StudentFlow.awaiting_question)
    await on_question(synthetic, state, session, user, lang)


# ---------- /subject ----------


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


# ---------- /history ----------


_HIST_PAGE_SIZE = 5


def _history_keyboard(
    dialogs, page: int, total: int, *, favourites_only: bool, search: str | None
) -> InlineKeyboardMarkup:
    """Star-toggle buttons per row + prev/next pagination.

    We pack star toggles and pagination into a single keyboard so each
    rendered history message is self-contained (no separate scroll-style
    flows). Callback payloads are short: ``hst:<page>:<fav>`` for navigation
    (``fav`` is 0/1) and ``star:<dialog_id>:<0|1>`` for toggles.
    """
    from aiogram.types import InlineKeyboardButton

    rows: list[list[InlineKeyboardButton]] = []
    for d in dialogs:
        marker = "⭐" if d.is_favourite else "☆"
        target = 0 if d.is_favourite else 1
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker} #{d.id}",
                    callback_data=f"star:{d.id}:{target}:{page}:{int(favourites_only)}",
                )
            ]
        )
    # Pagination row: prev/next only if there's more to show in that direction.
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="◀",
                callback_data=f"hst:{page - 1}:{int(favourites_only)}",
            )
        )
    if (page + 1) * _HIST_PAGE_SIZE < total:
        nav.append(
            InlineKeyboardButton(
                text="▶",
                callback_data=f"hst:{page + 1}:{int(favourites_only)}",
            )
        )
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _render_history(
    session: AsyncSession,
    user: User,
    lang: str,
    *,
    page: int,
    favourites_only: bool,
    search: str | None = None,
) -> tuple[str, InlineKeyboardMarkup | None]:
    dialogs, total = await user_dialogs_page(
        session,
        user_id=user.id,
        offset=page * _HIST_PAGE_SIZE,
        limit=_HIST_PAGE_SIZE,
        favourites_only=favourites_only,
        search=search,
    )
    if not dialogs:
        return texts.tr(lang, texts.HISTORY_EMPTY), None

    header = texts.tr(lang, texts.HISTORY_HEADER)
    if favourites_only:
        header = "⭐ " + header.lstrip("📜 ")
    if search:
        header += f"\n🔎 <i>поиск:</i> <code>{html.escape(search)}</code>\n"

    lines: list[str] = []
    for d in dialogs:
        title = texts.subject_title(d.subject, lang) if d.subject else "?"
        star = "⭐ " if d.is_favourite else ""
        lines.append(
            f"{star}" + texts.tr(lang, texts.HISTORY_ITEM).format(
                when=d.created_at.strftime("%d.%m %H:%M"),
                subject=title,
                q=_shorten(d.question, 120),
                a=_shorten(d.answer, 200),
            )
        )
    body = header + "\n\n".join(lines)
    kb = _history_keyboard(
        dialogs, page=page, total=total, favourites_only=favourites_only, search=search
    )
    return body, kb


@router.message(Command("history"))
async def on_history(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/history [page] — paginated history with ⭐ toggles.

    /history          first page, all dialogs
    /history 2        third page (0-indexed payload, 1-indexed UI)
    """
    page = 0
    arg = (command.args or "").strip()
    if arg:
        try:
            page = max(0, int(arg) - 1)
        except ValueError:
            page = 0
    body, kb = await _render_history(
        session, user, lang, page=page, favourites_only=False
    )
    await message.answer(
        body, parse_mode="HTML", reply_markup=kb or main_inline(lang)
    )


@router.message(Command("favourites"))
async def on_favourites(
    message: Message, session: AsyncSession, user: User, lang: str
) -> None:
    body, kb = await _render_history(
        session, user, lang, page=0, favourites_only=True
    )
    # `_render_history` returns `HISTORY_EMPTY` for both "no dialogs" and
    # "no favourites" — distinguish them so the user doesn't get the
    # wrong hint («Пока нет заданных вопросов» is misleading if they've
    # just never starred anything).
    if kb is None:
        body = texts.tr(lang, texts.FAVOURITES_EMPTY)
    await message.answer(
        body, parse_mode="HTML", reply_markup=kb or main_inline(lang)
    )


# ---------- /term — glossary-only lookup ----------


@router.message(Command("term"))
async def on_term(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/term <word> — direct glossary lookup. Fast reference for students
    who know they only want the definition, not a full RAG answer.
    No LLM involvement — teacher's curated definition or nothing."""
    from src.bot.services.glossary_upload import format_glossary_body, lookup_term

    query = (command.args or "").strip()
    if not query:
        await message.answer(
            "Использование: <code>/term &lt;термин&gt;</code>", parse_mode="HTML"
        )
        return
    hit = await lookup_term(session, question=query, subject_id=None)
    if hit is None:
        await message.answer(
            f"В глоссарии нет <code>«{html.escape(query)}»</code>. "
            "Задай обычный вопрос — поищу в учебниках.",
            parse_mode="HTML",
        )
        return
    await message.answer(format_glossary_body(hit, lang), parse_mode="HTML")


# ---------- /ref — sources for a past answer ----------


@router.message(Command("ref"))
async def on_ref(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/ref <dialog_id> — show the chunks that grounded a past answer.

    Students asked for a way to verify where an answer came from without
    rereading the whole textbook. We print the stored ``Dialog.sources``
    metadata (book, page, score) — that's enough to jump to the PDF page
    and read the original.
    """
    from src.db.models import Dialog as _Dialog

    arg = (command.args or "").strip()
    if not arg:
        await message.answer(
            "Использование: <code>/ref &lt;dialog_id&gt;</code>. "
            "ID есть рядом с ответом в /history.",
            parse_mode="HTML",
        )
        return
    try:
        dialog_id = int(arg)
    except ValueError:
        await message.answer("dialog_id должен быть числом.")
        return
    dialog = (
        await session.execute(
            select(_Dialog).where(
                _Dialog.id == dialog_id, _Dialog.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if dialog is None:
        await message.answer(
            f"Диалог <code>#{dialog_id}</code> не найден или не твой.",
            parse_mode="HTML",
        )
        return
    src_rows = dialog.sources or []
    if not src_rows:
        await message.answer(
            "У этого ответа нет учёта источников — возможно, он из глоссария "
            "или из FAQ преподавателя.",
        )
        return
    lines = [f"<b>Источники для #{dialog_id}</b>:"]
    for row in src_rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") == "web":
            title = html.escape(str(row.get("title") or "web"))
            url = html.escape(str(row.get("url") or ""), quote=True)
            lines.append(f"• 🌐 <a href=\"{url}\">{title}</a>")
        else:
            book = html.escape(str(row.get("book") or "?"))
            page = row.get("page")
            score = row.get("score")
            bits = [f"📚 {book}"]
            if page is not None:
                bits.append(f"стр. {page}")
            if score is not None:
                bits.append(f"совпадение {float(score):.2f}")
            lines.append("• " + " · ".join(bits))
    await message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)


# ---------- /export — own dialog history ----------


@router.message(Command("export"))
async def on_export(
    message: Message, session: AsyncSession, user: User, lang: str
) -> None:
    """/export — upload a plaintext file with the user's full Q&A history.

    No pagination, no LLM. Useful for students preparing for an exam who
    want offline notes of everything they've asked.
    """
    from aiogram.types import BufferedInputFile

    dialogs, _ = await user_dialogs_page(
        session, user_id=user.id, offset=0, limit=10_000
    )
    if not dialogs:
        await message.answer(texts.tr(lang, texts.HISTORY_EMPTY))
        return
    lines: list[str] = []
    for d in dialogs:
        when = d.created_at.strftime("%Y-%m-%d %H:%M")
        star = "⭐ " if d.is_favourite else ""
        subj = d.subject.title_ru if d.subject else "—"
        lines.append(f"{star}#{d.id}  {when}  [{subj}]")
        lines.append("Q: " + (d.question or "").strip())
        lines.append("A: " + (d.answer or "").strip())
        lines.append("")
    payload = "\n".join(lines).encode("utf-8")
    # Don't embed telegram_id in the filename — if the user forwards the
    # file to a group chat, their ID shouldn't leak. A plain ISO date is
    # enough for a student to keep multiple exports distinct.
    from datetime import datetime as _dt

    today = _dt.utcnow().strftime("%Y%m%d")
    await message.answer_document(
        BufferedInputFile(payload, filename=f"qa_history_{today}.txt"),
        caption=f"📦 {len(dialogs)} вопросов экспортировано.",
    )


@router.message(Command("find"))
async def on_find(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    query = (command.args or "").strip()
    if not query:
        await message.answer(
            "Использование: <code>/find &lt;подстрока&gt;</code> — ищет по своим вопросам.",
            parse_mode="HTML",
        )
        return
    body, kb = await _render_history(
        session, user, lang, page=0, favourites_only=False, search=query
    )
    if kb is None:
        body = texts.tr(lang, texts.FIND_EMPTY).format(q=html.escape(query))
    await message.answer(
        body, parse_mode="HTML", reply_markup=kb or main_inline(lang)
    )


@router.callback_query(F.data.startswith("hst:"))
async def on_history_page(
    callback: CallbackQuery, session: AsyncSession, user: User, lang: str
) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) < 3:
        await callback.answer()
        return
    try:
        page = int(parts[1])
        fav_only = bool(int(parts[2]))
    except ValueError:
        await callback.answer()
        return
    body, kb = await _render_history(
        session, user, lang, page=page, favourites_only=fav_only
    )
    await callback.answer()
    msg = callback.message
    if msg is None:
        return
    with contextlib.suppress(TelegramBadRequest):
        await msg.edit_text(body, parse_mode="HTML", reply_markup=kb)


@router.callback_query(F.data.startswith("star:"))
async def on_star_toggle(
    callback: CallbackQuery, session: AsyncSession, user: User, lang: str
) -> None:
    """Toggle ⭐ on a dialog the user owns; re-render the current history page."""
    parts = (callback.data or "").split(":")
    if len(parts) < 5:
        await callback.answer()
        return
    try:
        dialog_id = int(parts[1])
        to_fav = bool(int(parts[2]))
        page = int(parts[3])
        fav_only = bool(int(parts[4]))
    except ValueError:
        await callback.answer()
        return
    ok = await set_favourite(
        session, dialog_id=dialog_id, user_id=user.id, is_favourite=to_fav
    )
    await session.commit()
    await callback.answer(
        "⭐ в избранном" if to_fav and ok else ("☆ убрано" if ok else "Не твой диалог.")
    )
    body, kb = await _render_history(
        session, user, lang, page=page, favourites_only=fav_only
    )
    msg = callback.message
    if msg is None:
        return
    with contextlib.suppress(TelegramBadRequest):
        await msg.edit_text(body, parse_mode="HTML", reply_markup=kb)


# ---------- free-text question ----------


@router.message(StudentFlow.awaiting_question, F.text)
@router.message(F.text & ~F.text.startswith("/"))
async def on_question(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Thin handler — all RAG logic lives in `qa_pipeline.run_qa_pipeline`."""
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

    # Track the last-sent text so we skip no-op edits. Telegram otherwise
    # returns "Bad Request: message is not modified" and our edits silently
    # stop working — which used to manifest as "status doesn't change".
    last_sent = {"text": initial_status_text}

    async def _edit(new_text: str, reply_markup: object = None) -> None:
        # Dedupe purely by text — set_status always re-attaches the same
        # processing_keyboard, set_final flips both text and markup at once.
        # No realistic flow needs an "edit only the markup" path here.
        if new_text == last_sent["text"]:
            return  # dupe — would provoke "message is not modified"
        try:
            await status_msg.edit_text(new_text, parse_mode="HTML", reply_markup=reply_markup)
            last_sent["text"] = new_text
        except TelegramRetryAfter as exc:
            import asyncio as _asyncio

            await _asyncio.sleep(exc.retry_after + 0.1)
            try:
                await status_msg.edit_text(
                    new_text, parse_mode="HTML", reply_markup=reply_markup
                )
                last_sent["text"] = new_text
            except Exception as retry_exc:
                log.warning("status_edit_retry_failed", exc_type=type(retry_exc).__name__)
        except TelegramBadRequest as exc:
            # "message is not modified" is expected idempotency; everything
            # else (parse entities, message too long, blocked) is real.
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
        # The streaming previews end with the "▍" cursor — storing them
        # raw would dump answer fragments into the «Что сейчас делается?»
        # alert. Substitute a friendly label for the alert state, but
        # pass the real preview to the Telegram edit.
        alert_text = (
            texts.tr(lang, texts.STATUS_STREAMING)
            if status_text.rstrip().endswith("▍")
            else status_text
        )
        processing_state.set_status(rid, alert_text)
        # Re-attach the processing_keyboard on every status edit. Telegram's
        # editMessageText drops reply_markup if you don't pass one, so omitting
        # it here makes the «Что сейчас делается?» button vanish on the first
        # edit — which defeats the whole point.
        await _edit(
            q_with_status.format(q=q_html, status=status_text),
            reply_markup=processing_keyboard(rid, lang),
        )

    async def _react(emoji: str) -> None:
        # Visual heartbeat on the user's own bubble — survives status-message
        # edits so the student sees "received → answered" even after scrolling.
        if bot is None:
            return
        with contextlib.suppress(Exception):
            await bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message.message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )

    async def set_final(body: str, dialog_id: int, kind: str = "full") -> None:
        processing_state.clear(rid)  # kill the wh button mapping
        # Keyboard depends on the kind of answer the pipeline produced:
        #   "glossary"/"faq" → teacher-curated one-paragraph answer
        #     + «📖 Развёрнутый ответ» to run full RAG on demand.
        #   "full"/"web" in brief mode → only 👍/👎.
        #   "full"/"web" in verbose mode → follow-ups + «ask another».
        if kind in ("glossary", "faq"):
            kb = feedback_short_answer(dialog_id, lang)
        else:
            is_brief = (
                getattr(user, "answer_mode", None)
                and user.answer_mode.value == "brief"
            )
            kb = (
                feedback_brief(dialog_id, lang)
                if is_brief
                else feedback_inline(dialog_id, lang)
            )
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


# ---------- "what's happening" callback ----------


@router.callback_query(F.data.startswith("wh:"))
async def on_whats_happening(callback: CallbackQuery, lang: str) -> None:
    rid = (callback.data or "wh:").split(":", 1)[1]
    stage = processing_state.get_status(rid) or texts.tr(lang, texts.ALERT_STAGE_UNKNOWN)
    # Telegram caps callback alert text at 200 chars (UTF-16 code units).
    # The streaming preview easily exceeds that — truncate hard so we don't
    # crash with MESSAGE_TOO_LONG. Strip HTML tags too: alert is plain text,
    # so leftover &lt;b&gt; from the live preview look ugly.
    import re as _re

    plain = _re.sub(r"<[^>]+>", "", stage).replace("&lt;", "<").replace("&gt;", ">")
    if len(plain) > 190:
        plain = plain[:189] + "…"
    await callback.answer(text=plain, show_alert=True)


# ---------- expand a short (FAQ/glossary) answer into full RAG ----------


@router.callback_query(F.data.startswith("expand:"))
async def on_expand_answer(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """Student tapped «📖 Развёрнутый ответ» on a glossary/FAQ short
    answer. Re-run the full RAG pipeline on the same question stored in
    that Dialog row and **edit the same message in place** — the short
    definition turns into the full answer, no duplicate card below it.
    Status text during retrieval keeps the standard
    ``<blockquote>question</blockquote> ⌛ …`` framing so the student
    doesn't lose track of what's being looked up."""
    from src.bot.services.qa_pipeline import run_qa_pipeline
    from src.db.models import Dialog as _Dialog

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
            select(_Dialog).where(
                _Dialog.id == dialog_id, _Dialog.user_id == user.id
            )
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
            import asyncio as _asyncio

            await _asyncio.sleep(exc.retry_after + 0.1)
            with contextlib.suppress(Exception):
                await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
                last_sent["text"] = text
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc).lower():
                log.warning("expand_edit_bad_request", exc=str(exc))
        except Exception as exc:
            log.warning("expand_edit_failed", exc_type=type(exc).__name__, exc=str(exc))

    async def set_status(text: str) -> None:
        # Keep the question visible in a blockquote above the status —
        # same framing the regular /ask flow uses.
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
        skip_short_circuit=True,  # don't serve the same glossary line again
    )


# ---------- follow-up callback (Проще / Пример / Подробнее) ----------


_ALLOWED_MODIFIERS = frozenset({"simplify", "example", "deepen"})


@router.callback_query(F.data.startswith("fu:"))
async def on_followup(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """User tapped one of the follow-up buttons on a PM answer. Send a new
    placeholder below the original, stream a re-formulated answer into it,
    attach a fresh feedback+follow-up keyboard to the result."""
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

    # Quick acknowledgement so the button stops spinning while we work.
    await callback.answer()

    msg = callback.message
    if msg is None or msg.bot is None or msg.chat is None:
        # Inline-message follow-ups aren't wired for P2 — answer silently.
        return

    placeholder = await msg.answer(
        texts.tr(lang, texts.FU_PLACEHOLDER), parse_mode="HTML"
    )
    last_sent = {"text": placeholder.text or ""}

    async def _edit(text: str, reply_markup=None) -> None:  # type: ignore[no-untyped-def]
        if text == last_sent["text"]:
            return
        try:
            await placeholder.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)
            last_sent["text"] = text
        except TelegramRetryAfter as exc:
            import asyncio as _asyncio

            await _asyncio.sleep(exc.retry_after + 0.1)
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
        del kind  # follow-up output always gets the full verbose keyboard
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


# ---------- feedback callback ----------


@router.callback_query(F.data.startswith("fb:"))
async def on_feedback(
    callback: CallbackQuery,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    _ = user
    _, dialog_id_str, rating_str = callback.data.split(":")  # type: ignore[union-attr]
    dialog_id = int(dialog_id_str)
    rating = int(rating_str)

    inserted = await record_feedback(
        session, dialog_id=dialog_id, user_id=user.id, rating=rating
    )
    # Tiny dopamine hit: 🎉 reaction on a positive rating, runs only on the
    # first insert so a re-rate doesn't spam reactions.
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

    # Callback may come from a PM message (callback.message set) OR from an
    # inline-sent message in another chat (callback.inline_message_id set,
    # callback.message is None). Handle both.
    if callback.inline_message_id:
        # Inline flow: drop the 👍/👎 row but KEEP the «Задать ещё вопрос»
        # shortcut — it uses switch_inline_query_current_chat which works
        # in any chat without our bot being a member.
        with contextlib.suppress(Exception):
            await callback.bot.edit_message_reply_markup(  # type: ignore[union-attr]
                inline_message_id=callback.inline_message_id,
                reply_markup=ask_only_inline(lang),
            )
        return

    # PM flow: keep the «Задать ещё вопрос» button as a shortcut.
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


# ---------- helpers ----------


def _shorten(s: str, limit: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"
