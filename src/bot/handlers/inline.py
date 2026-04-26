"""Inline mode handlers — button tap and chosen_inline_result both feed run_qa_pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import html
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    CallbackQuery,
    ChosenInlineResult,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InlineQueryResultsButton,
    InputTextMessageContent,
    LinkPreviewOptions,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.keyboards import feedback_bare, processing_keyboard
from src.bot.services import processing_state
from src.bot.services.qa_pipeline import SetFinal, SetStatus, run_qa_pipeline
from src.db.models import User

log = structlog.get_logger(__name__)
router = Router(name="inline")

CACHE_TIME = 0

# In-process кэш id → исходный текст вопроса. Нужен потому, что
# Telegram-callback несёт только 64 байта callback_data, не сам вопрос.
_QUESTION_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_S = 600.0  # 10 минут — комфортный gap «подумал → ткнул»
# Hard cap: высокий поток inline-запросов (или популярный бот) не должен
# OOM-нить процесс. TTL GC удаляет stale на каждом чтении; этот cap включается
# только при потоке быстрее окна 10 мин. Удаляем старейших.
_CACHE_MAX_ENTRIES = 10_000

# Dedupe guard: как только пайплайн стартовал для inline_message_id, лочим его
# так, чтобы поздний chosen_inline_result или double-tap не запустили второй
# run, который бы дрался с первым за edit-слоты. Ключ — inline_message_id,
# значение — rid владельца. Используем dict.setdefault для атомарного
# check-and-claim — set дал бы классический TOCTOU между `in` и `add`.
# Claims с таймштампом, чтобы network split между _try_claim и _release не
# заклинил запись навечно — после _RUNNING_TTL_S stale-claim переотвоёвывается.
# Без этого следующий тап по тому же inline-сообщению зацикливался в «already
# running» до рестарта бота.
_RUNNING: dict[str, tuple[str, float]] = {}
_RUNNING_TTL_S = 120.0


def _try_claim(inline_message_id: str, rid: str) -> bool:
    """Атомарно резервируем ``inline_message_id`` за ``rid``. True — если
    мы первые (или предыдущий claim протух). dict-операции CPython
    атомарны, явный lock не нужен.
    """
    now = time.time()
    current = _RUNNING.get(inline_message_id)
    if current is None or now - current[1] > _RUNNING_TTL_S:
        _RUNNING[inline_message_id] = (rid, now)
        return True
    return current[0] == rid


def _release(inline_message_id: str) -> None:
    _RUNNING.pop(inline_message_id, None)


def _cache_gc() -> None:
    now = time.time()
    stale = [k for k, (_, ts) in _QUESTION_CACHE.items() if now - ts > _CACHE_TTL_S]
    for k in stale:
        _QUESTION_CACHE.pop(k, None)


def _cache_put(qid: str, question: str) -> None:
    _cache_gc()
    if len(_QUESTION_CACHE) >= _CACHE_MAX_ENTRIES:
        # Эвакуируем старейших 10%, чтобы не триммить на каждом insert.
        oldest = sorted(_QUESTION_CACHE.items(), key=lambda kv: kv[1][1])
        for k, _ in oldest[: max(1, _CACHE_MAX_ENTRIES // 10)]:
            _QUESTION_CACHE.pop(k, None)
    _QUESTION_CACHE[qid] = (question, time.time())


def _cache_get(qid: str) -> str | None:
    _cache_gc()
    hit = _QUESTION_CACHE.get(qid)
    return hit[0] if hit else None


def _results_button(lang: str) -> InlineQueryResultsButton:
    return InlineQueryResultsButton(
        text=texts.tr(lang, texts.BTN_INSTRUCTIONS),
        start_parameter="help",
    )


def _placeholder_message(q: str, lang: str) -> str:
    return texts.tr(lang, texts.Q_WITH_STATUS).format(
        q=html.escape(q),
        status=texts.tr(lang, texts.STATUS_RETRIEVING),
    )


def _placeholder_markup(qid: str, lang: str) -> InlineKeyboardMarkup:
    """Клавиатура, которую прикрепляем к каждому inline-результату.

    Две задачи:
      1. Telegram возвращает ``inline_message_id`` только для сообщений с
         markup — иначе путь редактирования недоступен.
      2. Если ``/setinlinefeedback`` выключен в BotFather, эта кнопка —
         единственный способ запустить пайплайн: тап → callback несёт
         ``inline_message_id`` → хендлер ниже работает.

    Payload ``iq:{qid}`` — ключ кэша для текста вопроса.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_GET_ANSWER),
                    callback_data=f"iq:{qid}",
                )
            ]
        ]
    )


# ---------- inline_query ----------


@router.inline_query()
async def on_inline_query(query: InlineQuery, lang: str) -> None:
    q = (query.query or "").strip()

    if not q:
        # Пустой запрос → 4 сэмпл-вопроса, которые студент может ткнуть и
        # отправить. Каждый — настоящий плейсхолдер с iq:{qid}-кнопкой, тап
        # запускает тот же пайплайн, что и набранный inline-запрос.
        results: list[InlineQueryResultArticle] = []
        idx = 0 if lang == "ru" else 1
        for ru_q, en_q in texts.ASK_SAMPLES:
            sample = (ru_q, en_q)[idx]
            qid = uuid.uuid4().hex[:16]
            _cache_put(qid, sample)
            results.append(
                InlineQueryResultArticle(
                    id=qid,
                    title=sample,
                    description=texts.tr(lang, texts.INLINE_PREVIEW_DESC),
                    input_message_content=InputTextMessageContent(
                        message_text=_placeholder_message(sample, lang),
                        parse_mode="HTML",
                    ),
                    reply_markup=_placeholder_markup(qid, lang),
                )
            )
        await query.answer(
            results=results,
            cache_time=CACHE_TIME,
            is_personal=True,
            button=_results_button(lang),
        )
        return

    # Short id that fits comfortably in the 64-byte callback_data budget.
    qid = uuid.uuid4().hex[:16]
    _cache_put(qid, q)

    title_tpl = texts.tr(lang, texts.INLINE_PREVIEW_TITLE)
    title = title_tpl.format(q=q[:60] + ("…" if len(q) > 60 else ""))
    description = texts.tr(lang, texts.INLINE_PREVIEW_DESC)

    await query.answer(
        results=[
            InlineQueryResultArticle(
                id=qid,
                title=title,
                description=description,
                input_message_content=InputTextMessageContent(
                    message_text=_placeholder_message(q, lang),
                    parse_mode="HTML",
                ),
                reply_markup=_placeholder_markup(qid, lang),
            )
        ],
        cache_time=CACHE_TIME,
        is_personal=True,
        button=_results_button(lang),
    )


# ---------- callback-driven trigger (works regardless of setinlinefeedback) ----


@router.callback_query(F.data.startswith("iq:"))
async def on_inline_ask(
    callback: CallbackQuery,
    bot: Bot,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    inline_message_id = callback.inline_message_id
    qid = (callback.data or "iq:").split(":", 1)[1]
    question = _cache_get(qid)

    if not inline_message_id:
        # Shouldn't happen in practice — callback_data only exists on inline
        # messages. Acknowledge gracefully so the client's spinner clears.
        await callback.answer(texts.tr(lang, texts.PROMPT_QUESTION))
        return

    if not question:
        await callback.answer(texts.tr(lang, texts.INLINE_EXPIRED), show_alert=True)
        return

    rid = processing_state.new_rid()
    # Atomic claim — a second concurrent tap with a *different* rid loses the
    # race and falls into the "already running" branch below.
    if not _try_claim(inline_message_id, rid):
        await callback.answer(texts.tr(lang, texts.STATUS_RETRIEVING), show_alert=False)
        return

    # Dismiss the tap's client-side spinner immediately; the pipeline's own
    # edit_message_text calls will update the message visibly.
    await callback.answer()

    processing_state.set_status(rid, texts.tr(lang, texts.STATUS_RETRIEVING))
    try:
        q_html = html.escape(question)
        # Swap the iq-button placeholder for the wh-button right away. The
        # first set_status would otherwise re-send the same text + a new
        # markup, and some Telegram clients still treat that as "not
        # modified", leaving the stale «Получить ответ» button on screen.
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(
                inline_message_id=inline_message_id,
                reply_markup=processing_keyboard(rid, lang),
            )
        set_status, set_final = _edit_via_inline_id(
            bot=bot, inline_message_id=inline_message_id, q_html=q_html, lang=lang, rid=rid
        )
        await run_qa_pipeline(
            question=question,
            session=session,
            user=user,
            lang=lang,
            set_status=set_status,
            set_final=set_final,
        )
    finally:
        processing_state.clear(rid)
        _release(inline_message_id)


# ---------- chosen_inline_result (optional fast-path) ----------


@router.chosen_inline_result()
async def on_chosen_inline_result(
    chosen: ChosenInlineResult,
    bot: Bot,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    inline_message_id = chosen.inline_message_id
    # Telegram отдаёт `query` = исходный текст inline_query — для sample-кнопок
    # на пустом query он тоже пустой, и автоматический запуск глох (юзер был
    # вынужден жать «Получить ответ»). Падаем на _QUESTION_CACHE через
    # result_id (= qid плейсхолдера), чтобы поднять текст вопроса.
    question = (chosen.query or "").strip()
    if not question:
        cached = _cache_get((chosen.result_id or "").strip())
        if cached:
            question = cached.strip()
    log.info(
        "chosen_inline_result_received",
        has_inline_id=bool(inline_message_id),
        question_len=len(question),
        recovered_from_cache=bool(question and not (chosen.query or "").strip()),
    )
    if not question:
        return

    q_html = html.escape(question)

    if inline_message_id:
        rid = processing_state.new_rid()
        # Atomic claim — if the callback-driven path already won, skip silently.
        if not _try_claim(inline_message_id, rid):
            return
        processing_state.set_status(rid, texts.tr(lang, texts.STATUS_RETRIEVING))
        # Same iq → wh swap as in on_inline_ask — see that comment.
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(
                inline_message_id=inline_message_id,
                reply_markup=processing_keyboard(rid, lang),
            )
        set_status, set_final = _edit_via_inline_id(
            bot=bot, inline_message_id=inline_message_id, q_html=q_html, lang=lang, rid=rid
        )
        try:
            await run_qa_pipeline(
                question=question,
                session=session,
                user=user,
                lang=lang,
                set_status=set_status,
                set_final=set_final,
            )
        finally:
            processing_state.clear(rid)
            _release(inline_message_id)
        return

    # Inline result chosen inside our own bot's PM: Telegram withholds the
    # inline_message_id *and* won't send us a Message. Send a fresh placeholder
    # and edit that one.
    chat_id = chosen.from_user.id
    rid = processing_state.new_rid()
    processing_state.set_status(rid, texts.tr(lang, texts.STATUS_RETRIEVING))
    try:
        placeholder = await bot.send_message(
            chat_id=chat_id,
            text=texts.tr(lang, texts.Q_WITH_STATUS).format(
                q=q_html, status=texts.tr(lang, texts.STATUS_RETRIEVING)
            ),
            parse_mode="HTML",
            reply_markup=processing_keyboard(rid, lang),
        )
    except Exception:
        processing_state.clear(rid)
        log.exception("inline_in_pm_send_placeholder_failed")
        return

    set_status, set_final = _edit_via_chat_message(
        bot=bot,
        chat_id=chat_id,
        message_id=placeholder.message_id,
        q_html=q_html,
        lang=lang,
        rid=rid,
    )
    try:
        await run_qa_pipeline(
            question=question,
            session=session,
            user=user,
            lang=lang,
            set_status=set_status,
            set_final=set_final,
        )
    finally:
        processing_state.clear(rid)


# ---------- edit-callback factories ----------


async def _safe_edit(
    coro_factory: Callable[[], Awaitable[object]],
    *,
    log_kw: dict[str, object],
) -> None:
    """Run an ``edit_message_text`` coroutine and translate the common
    Telegram failure modes into proper structured logs.

    * 429 ``TelegramRetryAfter``: sleep for the requested interval, retry once.
    * 400 ``TelegramBadRequest`` with "message is not modified": idempotent —
      swallow at debug level (we hit it whenever the throttle hands us the
      same preview twice).
    * Other 400s (parse entities, message too long, blocked): WARN with the
      exception text so we can see what Telegram is rejecting.
    """
    try:
        await coro_factory()
    except TelegramRetryAfter as exc:
        await asyncio.sleep(exc.retry_after + 0.1)
        try:
            await coro_factory()
        except Exception as retry_exc:
            log.warning(
                "inline_edit_retry_failed", exc_type=type(retry_exc).__name__, **log_kw
            )
    except TelegramBadRequest as exc:
        if "not modified" in str(exc).lower():
            return
        log.warning("inline_edit_bad_request", exc=str(exc), **log_kw)
    except Exception as exc:
        log.warning(
            "inline_edit_failed",
            exc_type=type(exc).__name__,
            exc=str(exc),
            **log_kw,
        )


def _edit_via_inline_id(
    *, bot: Bot, inline_message_id: str, q_html: str, lang: str, rid: str
) -> tuple[SetStatus, SetFinal]:
    q_with_status = texts.tr(lang, texts.Q_WITH_STATUS)

    async def set_status(text: str) -> None:
        # See student.py for why we substitute the alert state during
        # streaming — the live preview ends with "▍" and would otherwise
        # leak answer fragments into the «Что сейчас делается?» alert.
        alert_text = (
            texts.tr(lang, texts.STATUS_STREAMING)
            if text.rstrip().endswith("▍")
            else text
        )
        processing_state.set_status(rid, alert_text)
        await _safe_edit(
            lambda: bot.edit_message_text(
                inline_message_id=inline_message_id,
                text=q_with_status.format(q=q_html, status=text),
                parse_mode="HTML",
                reply_markup=processing_keyboard(rid, lang),
            ),
            log_kw={
                "via": "inline_message_id",
                "inline_message_id": inline_message_id,
                "status": text[:60],
            },
        )

    async def set_final(body: str, dialog_id: int, kind: str = "full") -> None:
        # Inline answers keep the bare keyboard regardless of kind —
        # inline results don't have space for a 3-row follow-up block.
        del kind
        processing_state.clear(rid)
        await _safe_edit(
            lambda: bot.edit_message_text(
                inline_message_id=inline_message_id,
                text=body,
                parse_mode="HTML",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                reply_markup=feedback_bare(dialog_id, lang),
            ),
            log_kw={
                "via": "inline_message_id",
                "inline_message_id": inline_message_id,
                "body_len": len(body),
                "dialog_id": dialog_id,
            },
        )

    return set_status, set_final


def _edit_via_chat_message(
    *, bot: Bot, chat_id: int, message_id: int, q_html: str, lang: str, rid: str
) -> tuple[SetStatus, SetFinal]:
    q_with_status = texts.tr(lang, texts.Q_WITH_STATUS)

    async def set_status(text: str) -> None:
        alert_text = (
            texts.tr(lang, texts.STATUS_STREAMING)
            if text.rstrip().endswith("▍")
            else text
        )
        processing_state.set_status(rid, alert_text)
        await _safe_edit(
            lambda: bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=q_with_status.format(q=q_html, status=text),
                parse_mode="HTML",
                reply_markup=processing_keyboard(rid, lang),
            ),
            log_kw={
                "via": "chat_message",
                "chat_id": chat_id,
                "message_id": message_id,
                "status": text[:60],
            },
        )

    async def set_final(body: str, dialog_id: int, kind: str = "full") -> None:
        # Inline answers keep the bare keyboard regardless of kind —
        # inline results don't have space for a 3-row follow-up block.
        del kind
        processing_state.clear(rid)
        await _safe_edit(
            lambda: bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=body,
                parse_mode="HTML",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                reply_markup=feedback_bare(dialog_id, lang),
            ),
            log_kw={
                "via": "chat_message",
                "chat_id": chat_id,
                "message_id": message_id,
                "body_len": len(body),
                "dialog_id": dialog_id,
            },
        )

    return set_status, set_final
