"""One RAG answer pipeline, shared by both the PM flow (student.py) and the
inline flow (inline.py).

Stages (the LLM-rewrite-and-retry middle step was removed — bge-m3 handles
typos well enough that an extra Qwen3 round-trip wasn't earning its 5–15 s):
  1. Retrieval + subject routing (single bge-m3 cosine pass).
  2. If top-1 score is below MIN_TOP_SCORE → fall through to web search.
  3. Stream the LLM answer to the caller's message, rendering a live preview.
  4. Persist the dialog and hand the final body back for the terminal edit.

The caller injects two async callbacks:

* ``set_status(text)`` — replace the in-flight status line (used during
  retrieval, web-search, and while the stream is still coming in).
* ``set_final(body, dialog_id)`` — the terminal edit once we have an answer.
  The dialog_id lets the caller attach the right feedback keyboard.

An optional ``typing_ping`` callback lets the PM flow also poke the
Telegram "typing…" indicator every few seconds; the inline flow skips it
because inline-result messages have no chat_id to send actions to.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

import structlog
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.services import followup_cache
from src.bot.services.dialog_service import save_dialog
from src.bot.services.faq_service import format_faq_body, lookup_faq
from src.bot.services.glossary_upload import format_glossary_body, lookup_term
from src.bot.services.safe_html import (
    safe_html as _safe_html,
)
from src.bot.services.safe_html import (
    truncate_for_telegram as _truncate_for_telegram,
)
from src.bot.services.warmup import MODELS_READY
from src.db.models import Dialog, User
from src.rag.answer_validator import validate_answer
from src.rag.circuit_breaker import CircuitOpenError
from src.rag.comparison import (
    build_comparison_messages,
    detect_comparison,
    merge_hits,
)
from src.rag.config import MIN_TOP_SCORE
from src.rag.llm import chat_stream, strip_think
from src.rag.pipeline import AskResult, resolve_subject
from src.rag.prompts import build_messages, build_web_messages
from src.rag.retriever import Hit
from src.rag.web_search import WebHit, search_web

log = structlog.get_logger(__name__)

SetStatus = Callable[[str], Awaitable[None]]
# Terminal edit. Pipeline passes ``(body, dialog_id, kind)``; the handler
# picks the keyboard based on ``kind``:
#   "full"      — regular RAG answer (verbose: follow-ups; brief: bare)
#   "glossary"  — teacher-curated short definition → "📖 Подробнее" button
#   "faq"       — teacher-fixed answer → same expand button as glossary
#   "web"       — web-fallback → regular feedback keyboard
# Legacy two-arg callers still work because the handler adapter supplies a
# default "full" kind.
SetFinal = Callable[..., Awaitable[None]]
TypingPing = Callable[[], Awaitable[None]]

# Telegram's hard ceiling for editMessageText is 1 edit/sec/message; tighter
# values reliably trigger 429 TelegramRetryAfter. 1.1 s leaves a small buffer
# so a brief LLM token burst doesn't queue up retry-after sleeps.
_STREAM_EDIT_INTERVAL_S = 1.1
# How often to poke the "typing…" indicator. Telegram auto-expires it after ~5s.
_TYPING_INTERVAL_S = 4.0
# Telegram's hard cap for sendMessage / editMessageText. We truncate the final
# body just below it so a verbose LLM answer never tanks the entire turn.
# ``_safe_html``, ``_balance_tags`` and ``_truncate_for_telegram`` now live
# in ``src.bot.services.safe_html`` so the FAQ and glossary short-circuit
# paths can sanitise teacher-typed content with the same rules.


async def run_qa_pipeline(
    *,
    question: str,
    session: AsyncSession,
    user: User,
    lang: str,
    set_status: SetStatus,
    set_final: SetFinal,
    typing_ping: TypingPing | None = None,
    skip_short_circuit: bool = False,
) -> None:
    """Retrieval → (optional rewrite + retry) → streaming LLM → final edit.

    ``skip_short_circuit`` skips the FAQ/glossary fast paths — used by the
    «📖 Развёрнутый ответ» button so tapping it on a glossary answer
    doesn't serve the same glossary one-liner again.
    """
    q_html = html.escape(question)
    t0 = time.monotonic()

    if typing_ping is not None:
        await typing_ping()

    # Ping the placeholder so we know edit works now, not 30 s later after
    # retrieval finishes. Also useful when the placeholder text is identical
    # to STATUS_RETRIEVING — Telegram will swallow the duplicate silently.
    await set_status(texts.tr(lang, texts.STATUS_RETRIEVING))

    # FAQ short-circuit: a teacher-curated answer for this exact question
    # (or a very close normalised form) skips retrieval and the LLM. Fast
    # path measured in milliseconds — no bge-m3, no Qwen3. Caller can opt
    # out via ``skip_short_circuit`` (e.g. «📖 Развёрнутый ответ» button).
    faq = None if skip_short_circuit else await lookup_faq(session, question=question, subject_id=None)
    if faq is not None:
        dialog = await save_dialog(
            session,
            user=user,
            question=question,
            result=AskResult(answer=faq.answer, subject=None, hits=[], route=None),  # type: ignore[arg-type]
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        body = format_faq_body(faq, lang, question=question)
        await set_final(body, dialog.id, "faq")
        log.info(
            "qa_answer_from_faq",
            faq_id=faq.id,
            dialog_id=dialog.id,
            subject_id=faq.subject_id,
        )
        return

    # Glossary short-circuit: if the normalised question is exactly a known
    # term («что такое стейкхолдер» → «стейкхолдер»), return the curated
    # definition. Less precise than FAQ (term-level, not question-level),
    # but catches the bulk of definitional queries students send.
    gloss = None if skip_short_circuit else await lookup_term(session, question=question, subject_id=None)
    if gloss is not None:
        dialog = await save_dialog(
            session,
            user=user,
            question=question,
            result=AskResult(answer=gloss.definition, subject=None, hits=[], route=None),  # type: ignore[arg-type]
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        body = format_glossary_body(gloss, lang, question=question)
        await set_final(body, dialog.id, "glossary")
        log.info(
            "qa_answer_from_glossary",
            term=gloss.term,
            dialog_id=dialog.id,
            subject_id=gloss.subject_id,
        )
        return

    # If the warm-up background task hasn't finished loading the models yet,
    # tell the user instead of blocking silently for 30–60 s (typical HF cold
    # download). One-off state — later questions fly past this wait instantly.
    if not MODELS_READY.is_set():
        await set_status(texts.tr(lang, texts.STATUS_WARMING))
        try:
            await asyncio.wait_for(MODELS_READY.wait(), timeout=120.0)
        except TimeoutError:
            log.warning("qa_warmup_timeout")
            await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
            return
        await set_status(texts.tr(lang, texts.STATUS_RETRIEVING))

    # Stage 1: retrieval + subject routing. If the user pinned a subject
    # via /subject, honour it — skip the global router and scope retrieval
    # to that subject only. If they asked a comparison («сравни X и Y»),
    # run two retrievals — one per term — and merge so the LLM sees
    # balanced context.
    pinned_slug = getattr(user, "current_subject_slug", None) or None
    cmp = detect_comparison(question)
    t_stage = time.monotonic()
    comparison_terms: tuple[str, str] | None = None
    try:
        if cmp is not None:
            term_a, term_b = cmp
            (subj_a, hits_a, _), (subj_b, hits_b, _) = await asyncio.gather(
                asyncio.to_thread(resolve_subject, term_a, pinned_slug),
                asyncio.to_thread(resolve_subject, term_b, pinned_slug),
            )
            hits = merge_hits(hits_a, hits_b, max_total=8)
            # Prefer a shared subject if both terms resolved into the same
            # one; otherwise whichever side has hits.
            if subj_a is not None and subj_b is not None and subj_a.slug == subj_b.slug:
                subject = subj_a
            else:
                subject = subj_a or subj_b
            comparison_terms = (term_a, term_b)
        else:
            subject, hits, _ = await asyncio.to_thread(
                resolve_subject, question, pinned_slug
            )
    except Exception:
        log.exception("qa_retrieval_failed")
        await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
        return
    top_score = hits[0].score if hits else 0.0
    log.info(
        "qa_stage_retrieval_done",
        hits=len(hits),
        top_score=round(top_score, 3),
        elapsed_ms=int((time.monotonic() - t_stage) * 1000),
    )

    # Stage 2: too weak → skip the expensive LLM-rewrite dance (5–15 s wasted
    # on Qwen3-4B, and bge-m3 is already robust to typos). Jump to web search.
    if not hits or subject is None or top_score < MIN_TOP_SCORE:
        log.info("qa_low_score_web_fallback", top_score=round(top_score, 3))
        await _web_fallback(
            session=session,
            user=user,
            question=question,
            q_html=q_html,
            lang=lang,
            set_status=set_status,
            set_final=set_final,
            typing_ping=typing_ping,
            t0=t0,
        )
        return

    # Stage 3: streaming answer from the textbook
    assert subject is not None and hits
    subj_title = texts.subject_title(subject, lang)
    # S2: surface what retrieval found so the user isn't staring at the same
    # "searching" line — hit count + top cosine score + subject.
    await set_status(
        texts.tr(lang, texts.STATUS_RETRIEVAL_FOUND).format(
            n=len(hits),
            score=f"{top_score:.2f}",
            subject=html.escape(subj_title),
        )
    )
    await set_status(texts.tr(lang, texts.STATUS_THINKING).format(subject=html.escape(subj_title)))

    brief = getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
    try:
        if comparison_terms is not None:
            messages = build_comparison_messages(
                comparison_terms[0],
                comparison_terms[1],
                hits,
                subject,
                lang=lang,
            )
        else:
            messages = build_messages(question, hits, subject, lang=lang, brief=brief)
        answer = await _stream_answer_to_ui(
            messages=messages,
            lang=lang,
            subject_title=subj_title,
            set_status=set_status,
            typing_ping=typing_ping,
        )
    except CircuitOpenError:
        # Breaker is open — llama-server is known-bad, fail fast with a
        # friendly message instead of sitting on a 300 s timeout.
        log.warning("qa_llm_circuit_open")
        await set_status(texts.tr(lang, texts.STATUS_BUSY))
        return
    except Exception:
        log.exception("qa_llm_failed")
        await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
        return

    # Belt-and-suspenders: even with a tight prompt Qwen3 sometimes invents
    # «(Surname, 1984)» or a made-up etymology. Strip anything not grounded
    # in the retrieved chunks before the body reaches the user.
    answer, validation = validate_answer(answer, [h.text for h in hits])
    if validation.total:
        log.info(
            "qa_answer_validator_stripped",
            attributions=validation.attributions_stripped,
            etymologies=validation.etymologies_stripped,
            foreign=validation.foreign_scripts_stripped,
        )

    latency_ms = int((time.monotonic() - t0) * 1000)

    result = AskResult(answer=answer, subject=subject, hits=hits, route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(
        session, user=user, question=question, result=result, latency_ms=latency_ms
    )

    if brief:
        sources = _format_textbook_sources_inline(hits)
        body = texts.tr(lang, texts.ANSWER_BODY_BRIEF).format(
            answer=_safe_html(answer), sources=sources
        )
    else:
        sources = _format_textbook_sources(hits, lang)
        body = texts.tr(lang, texts.ANSWER_BODY).format(
            q=q_html,
            subject=html.escape(subj_title),
            answer=_safe_html(answer),
            sources=sources,
        )
    body = _truncate_for_telegram(body)
    # Cache hits + subject so «⬇️ Проще / 💡 Пример / 📖 Подробнее» can re-run
    # the LLM without repeating retrieval. Keyed by the new dialog_id that's
    # embedded in the feedback keyboard. (Stored even in brief mode — the
    # user can switch to verbose later, and the cache is cheap to carry.)
    followup_cache.store(
        dialog.id,
        question=question,
        lang=lang,
        hits=hits,
        subject=subject,
        comparison_terms=comparison_terms,
    )
    await set_final(body, dialog.id, "full")
    log.info(
        "qa_answer_from_textbook",
        subject=subject.slug,
        top_score=round(hits[0].score, 3),
        latency_ms=latency_ms,
        dialog_id=dialog.id,
    )


async def _web_fallback(
    *,
    session: AsyncSession,
    user: User,
    question: str,
    q_html: str,
    lang: str,
    set_status: SetStatus,
    set_final: SetFinal,
    typing_ping: TypingPing | None,
    t0: float,
) -> None:
    await set_status(texts.tr(lang, texts.STATUS_WEB_SEARCH))
    if typing_ping is not None:
        await typing_ping()
    t_web = time.monotonic()
    try:
        web_hits = await asyncio.to_thread(search_web, question, 5, lang=lang)
    except Exception:
        log.exception("qa_web_search_failed")
        web_hits = []
    log.info(
        "qa_web_search_done",
        results=len(web_hits),
        elapsed_ms=int((time.monotonic() - t_web) * 1000),
    )

    if not web_hits:
        await set_status(texts.tr(lang, texts.NO_HITS))
        return

    # S7: summary — "found N sources: host1, host2, host3 — reading…"
    unique_hosts: list[str] = []
    for h in web_hits:
        if h.host and h.host not in unique_hosts:
            unique_hosts.append(h.host)
    await set_status(
        texts.tr(lang, texts.STATUS_WEB_FOUND).format(
            n=len(web_hits),
            hosts=", ".join(unique_hosts[:3]),
        )
    )

    # S6: cycle through the first ~3 hosts so the user literally sees which
    # domains we're pulling from. Spaced at _STREAM_EDIT_INTERVAL_S so we stay
    # well under Telegram's 1 edit/s/message cap. The last edit bleeds into
    # the "thinking" status below; overall we add at most ~3 edits here.
    for h in web_hits[:3]:
        await asyncio.sleep(_STREAM_EDIT_INTERVAL_S)
        await set_status(texts.tr(lang, texts.STATUS_WEB_VISITING).format(host=html.escape(h.host)))

    await set_status(texts.tr(lang, texts.STATUS_WEB_THINKING))
    try:
        messages = build_web_messages(question, web_hits, lang=lang)
        answer = await _stream_answer_to_ui(
            messages=messages,
            lang=lang,
            subject_title=None,
            set_status=set_status,
            typing_ping=typing_ping,
        )
    except Exception:
        log.exception("qa_web_llm_failed")
        await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
        return

    answer, validation = validate_answer(
        answer, [f"{h.title} {h.snippet}" for h in web_hits]
    )
    if validation.total:
        log.info(
            "qa_web_answer_validator_stripped",
            attributions=validation.attributions_stripped,
            etymologies=validation.etymologies_stripped,
            foreign=validation.foreign_scripts_stripped,
        )

    latency_ms = int((time.monotonic() - t0) * 1000)

    dialog = Dialog(
        user_id=user.id,
        subject_id=None,
        question=question,
        answer=answer,
        sources=[
            {"type": "web", "title": h.title, "url": h.url, "snippet": h.snippet[:300]}
            for h in web_hits
        ],
        latency_ms=latency_ms,
    )
    session.add(dialog)
    await session.flush()

    brief_web = getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
    if brief_web:
        sources = ", ".join(html.escape(h.host or h.url[:40]) for h in web_hits)
        body = texts.tr(lang, texts.ANSWER_BODY_BRIEF_WEB).format(
            answer=_safe_html(answer), sources=sources
        )
    else:
        sources = _format_web_sources(web_hits, lang)
        body = texts.tr(lang, texts.ANSWER_BODY_WEB).format(
            q=q_html,
            answer=_safe_html(answer),
            sources=sources,
        )
    body = _truncate_for_telegram(body)
    followup_cache.store(
        dialog.id,
        question=question,
        lang=lang,
        web_hits=list(web_hits),
    )
    await set_final(body, dialog.id, "web")


# ---------- streaming ----------


async def _stream_answer_to_ui(
    *,
    messages: list[dict[str, str]],
    lang: str,
    subject_title: str | None,
    set_status: SetStatus,
    typing_ping: TypingPing | None,
) -> str:
    """Consume the LLM SSE stream; re-edit the status message with a live
    running answer every ~1.2 s. Returns the full cleaned answer string.
    """
    accumulated: list[str] = []
    chunks_seen = 0
    edits_sent = 0
    first_edit_done = False
    stream_started = time.monotonic()
    last_edit = stream_started
    last_typing = stream_started

    # S4: prefill-gap heartbeat. Between STATUS_THINKING and the first token
    # there can be 5-10 s of silence while llama.cpp processes the prompt.
    # Tick "💭 Думаю… (Ns)" every ~1.2 s so the user sees something moving,
    # then cancel the instant the first chunk arrives.
    async def _heartbeat() -> None:
        secs = 0
        while True:
            await asyncio.sleep(_STREAM_EDIT_INTERVAL_S)
            secs += int(_STREAM_EDIT_INTERVAL_S)
            try:
                await set_status(texts.tr(lang, texts.STATUS_HEARTBEAT).format(s=secs))
            except TelegramRetryAfter:
                # Honour back-off but keep ticking — heartbeat is best-effort.
                await asyncio.sleep(1.0)
            except Exception:
                log.debug("heartbeat_edit_failed", exc_info=True)

    hb_task: asyncio.Task[None] | None = asyncio.create_task(_heartbeat())

    # Wrap the consumer in try/finally so a mid-stream exception (httpx
    # 5xx, cancellation, etc.) still tears down the heartbeat. Otherwise
    # it keeps editing the message every 1.1 s and overwrites the
    # INTERNAL_ERROR set_status the outer pipeline writes on failure.
    # Lock sampling to near-deterministic for factual answers: temperature
    # 0.1 + top_p 0.9 + repeat_penalty 1.1 virtually eliminates the "same
    # question, different invented author" drift we saw at the default
    # 0.7/0.8/1.05. Chain-of-thought is off via /no_think in the system
    # prompt anyway, so there's nothing creative we'd be clamping here.
    try:
        async for piece in chat_stream(
            messages,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
        ):
            if hb_task is not None:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await hb_task
                hb_task = None
                # Reset the edit clock so the first real preview fires
                # immediately, not after the heartbeat's residual interval.
                last_edit = time.monotonic() - _STREAM_EDIT_INTERVAL_S
            accumulated.append(piece)
            chunks_seen += 1
            now = time.monotonic()

            if typing_ping is not None and now - last_typing >= _TYPING_INTERVAL_S:
                try:
                    await typing_ping()
                except Exception:
                    # Typing is cosmetic — never let it kill the stream.
                    log.debug("typing_ping_failed", exc_info=True)
                last_typing = now

            # Fire the first edit as soon as anything arrives (so the user
            # sees the stream actually started), then throttle the rest.
            due_by_throttle = now - last_edit >= _STREAM_EDIT_INTERVAL_S
            if not first_edit_done or due_by_throttle:
                preview = _live_preview(
                    "".join(accumulated), lang=lang, subject_title=subject_title
                )
                try:
                    await set_status(preview)
                except TelegramRetryAfter as exc:
                    # Honour Telegram's back-off, then keep streaming.
                    # Sleeping bumps last_edit forward so we don't
                    # immediately retry.
                    await asyncio.sleep(exc.retry_after + 0.1)
                    last_edit = time.monotonic()
                    continue
                except TelegramBadRequest as exc:
                    # "message is not modified" is an idempotency signal we
                    # can swallow; other 400s mean the preview HTML is bad
                    # — log and keep streaming so the final body still has
                    # a chance.
                    if "not modified" not in str(exc).lower():
                        log.debug("stream_preview_bad_request", exc=str(exc))
                edits_sent += 1
                last_edit = now
                first_edit_done = True
    finally:
        if hb_task is not None:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await hb_task

    total_chars = sum(len(p) for p in accumulated)
    log.info(
        "stream_finished",
        chunks=chunks_seen,
        chars=total_chars,
        edits_sent=edits_sent,
        elapsed_s=round(time.monotonic() - stream_started, 2),
    )
    return strip_think("".join(accumulated))


def _live_preview(partial: str, *, lang: str, subject_title: str | None) -> str:
    """Render the in-flight partial answer as a status line (plain text)."""
    text = partial.strip()
    if len(text) > 800:
        text = "…" + text[-799:]
    prefix = ""
    if subject_title:
        prefix = (
            texts.tr(lang, texts.STATUS_THINKING).format(subject=html.escape(subject_title)) + "\n"
        )
    return f"{prefix}{html.escape(text)} ▍"


# ---------- follow-up re-run ----------


async def _reload_followup_from_dialog(
    session: AsyncSession,
    *,
    dialog_id: int,
    user: User,
    lang: str,
) -> followup_cache.FollowUpContext | None:
    """Rebuild a ``FollowUpContext`` for ``dialog_id`` by re-running
    retrieval on the question stored in the Dialog row. Used when the
    in-memory cache missed (TTL, bot restart, process kill). Returns
    ``None`` if the dialog is gone or doesn't belong to ``user``.
    """
    dialog = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user.id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        return None

    # Strip the `[simplify] `/`[example] `/`[deepen] ` prefix that was
    # prepended when a follow-up was saved — we want the original student
    # question for a fresh retrieval, not the tagged variant.
    import re as _re

    q = _re.sub(r"^\[(simplify|example|deepen)\]\s*", "", dialog.question or "", flags=_re.IGNORECASE)
    if not q.strip():
        return None

    # Detect whether this was a comparison and reproduce the same split.
    cmp = detect_comparison(q)
    if cmp is not None:
        term_a, term_b = cmp
        (subj_a, hits_a, _), (subj_b, hits_b, _) = await asyncio.gather(
            asyncio.to_thread(resolve_subject, term_a, None),
            asyncio.to_thread(resolve_subject, term_b, None),
        )
        hits = merge_hits(hits_a, hits_b, max_total=8)
        subject = (
            subj_a if (subj_a is not None and subj_b is not None and subj_a.slug == subj_b.slug)
            else (subj_a or subj_b)
        )
        ctx_obj = followup_cache.FollowUpContext(
            question=q,
            lang=lang,
            hits=hits,
            subject=subject,
            web_hits=None,
            created_at=0.0,
            comparison_terms=(term_a, term_b),
        )
        return ctx_obj

    subject, hits, _ = await asyncio.to_thread(resolve_subject, q, None)
    return followup_cache.FollowUpContext(
        question=q,
        lang=lang,
        hits=hits,
        subject=subject,
        web_hits=None,
        created_at=0.0,
        comparison_terms=None,
    )


async def run_followup_pipeline(
    *,
    dialog_id: int,
    modifier: str,
    session: AsyncSession,
    user: User,
    lang: str,
    set_status: SetStatus,
    set_final: SetFinal,
    typing_ping: TypingPing | None = None,
) -> bool:
    """Re-run the LLM on a previous answer's cached context with a prompt
    modifier («simplify» / «example» / «deepen»). If the in-memory
    follow-up cache missed (bot restarted, 24 h TTL elapsed), we fall
    back to reloading context from the Dialog row and re-running
    retrieval — so «Проще» on yesterday's answer still works.
    Returns False only if the dialog itself is gone / not owned by user.
    """
    ctx = followup_cache.get(dialog_id)
    if ctx is None:
        ctx = await _reload_followup_from_dialog(
            session, dialog_id=dialog_id, user=user, lang=lang
        )
        if ctx is None:
            return False
        # Rough status line while the rebuild is happening in-flight.
        await set_status(texts.tr(lang, texts.STATUS_RETRIEVING))

    t0 = time.monotonic()
    use_lang = ctx.lang or lang

    if ctx.is_web:
        assert ctx.web_hits is not None
        messages = build_web_messages(ctx.question, ctx.web_hits, lang=use_lang, modifier=modifier)
        corpus_texts = [f"{h.title} {h.snippet}" for h in ctx.web_hits]
        subject_title = None
    elif ctx.comparison_terms is not None:
        # Preserve the side-by-side comparison structure on follow-up —
        # without this, «Проще» on a «сравни X и Y» answer degraded to a
        # flat prose summary that lost the column-vs-column format.
        # ``modifier`` is passed through so «Проще / Пример / Подробнее»
        # still reshapes the tone, just without throwing away the table.
        ta, tb = ctx.comparison_terms
        messages = build_comparison_messages(
            ta, tb, ctx.hits, ctx.subject, lang=use_lang, modifier=modifier
        )
        corpus_texts = [h.text for h in ctx.hits]
        subject_title = (
            texts.subject_title(ctx.subject, use_lang) if ctx.subject is not None else None
        )
    else:
        messages = build_messages(
            ctx.question, ctx.hits, ctx.subject, lang=use_lang, modifier=modifier
        )
        corpus_texts = [h.text for h in ctx.hits]
        subject_title = (
            texts.subject_title(ctx.subject, use_lang) if ctx.subject is not None else None
        )

    try:
        answer = await _stream_answer_to_ui(
            messages=messages,
            lang=use_lang,
            subject_title=subject_title,
            set_status=set_status,
            typing_ping=typing_ping,
        )
    except Exception:
        log.exception("qa_followup_llm_failed")
        await set_status(texts.tr(use_lang, texts.INTERNAL_ERROR))
        return True  # cache was valid, just the LLM call blew up

    answer, validation = validate_answer(answer, corpus_texts)
    if validation.total:
        log.info(
            "qa_followup_validator_stripped",
            modifier=modifier,
            stripped=validation.total,
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    # Label the saved question so teachers inspecting the dialog table can
    # tell an original from a follow-up at a glance.
    question_tag = f"[{modifier}] {ctx.question}"
    q_html = html.escape(question_tag)

    if ctx.is_web:
        assert ctx.web_hits is not None
        new_dialog = Dialog(
            user_id=user.id,
            subject_id=None,
            question=question_tag,
            answer=answer,
            sources=[
                {"type": "web", "title": h.title, "url": h.url, "snippet": h.snippet[:300]}
                for h in ctx.web_hits
            ],
            latency_ms=latency_ms,
        )
        session.add(new_dialog)
        await session.flush()
        body = texts.tr(use_lang, texts.ANSWER_BODY_WEB).format(
            q=q_html,
            answer=_safe_html(answer),
            sources=_format_web_sources(ctx.web_hits, use_lang),
        )
    else:
        result = AskResult(answer=answer, subject=ctx.subject, hits=ctx.hits, route=None)  # type: ignore[arg-type]
        new_dialog = await save_dialog(
            session,
            user=user,
            question=question_tag,
            result=result,
            latency_ms=latency_ms,
        )
        body = texts.tr(use_lang, texts.ANSWER_BODY).format(
            q=q_html,
            subject=html.escape(subject_title or ""),
            answer=_safe_html(answer),
            sources=_format_textbook_sources(ctx.hits, use_lang),
        )

    body = _truncate_for_telegram(body)
    # Chain: re-cache under the NEW dialog_id so follow-ups can stack.
    # Preserve comparison_terms so a chain «Сравни X и Y → Проще → Пример»
    # keeps using the side-by-side prompt.
    followup_cache.store(
        new_dialog.id,
        question=ctx.question,
        lang=use_lang,
        hits=ctx.hits if not ctx.is_web else None,
        subject=ctx.subject,
        web_hits=ctx.web_hits if ctx.is_web else None,
        comparison_terms=ctx.comparison_terms,
    )
    await set_final(body, new_dialog.id, "web" if ctx.is_web else "full")
    log.info(
        "qa_followup_done",
        modifier=modifier,
        parent_dialog_id=dialog_id,
        new_dialog_id=new_dialog.id,
        latency_ms=latency_ms,
    )
    return True


# ---------- source formatting ----------


def _format_textbook_sources(hits: list[Hit], lang: str) -> str:
    by_book: dict[str, list[int]] = defaultdict(list)
    for h in hits:
        by_book[h.book].append(h.page)
    lines: list[str] = []
    for book, pages in by_book.items():
        pages_str = ", ".join(str(p) for p in sorted(set(pages)))
        lines.append(
            texts.tr(lang, texts.SOURCES_ITEM).format(book=html.escape(book), page=pages_str)
        )
    return "\n".join(lines)


def _format_textbook_sources_inline(hits: list[Hit]) -> str:
    """Single-line version used in brief-mode answers: «book p.12, 15; book2 p.3»."""
    by_book: dict[str, list[int]] = defaultdict(list)
    for h in hits:
        by_book[h.book].append(h.page)
    parts: list[str] = []
    for book, pages in by_book.items():
        pages_str = ", ".join(str(p) for p in sorted(set(pages)))
        parts.append(f"{html.escape(book)} (стр. {pages_str})")
    return "; ".join(parts)


def _format_web_sources(web_hits: list[WebHit], lang: str) -> str:
    return "\n".join(
        texts.tr(lang, texts.SOURCES_ITEM_WEB).format(
            url=html.escape(h.url, quote=True), title=html.escape(h.title)
        )
        for h in web_hits
    )
