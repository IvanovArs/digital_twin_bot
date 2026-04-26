"""RAG pipeline shared by PM and inline flows.

Stages: retrieval + routing → optional web fallback → streamed LLM answer
→ dialog persistence. Caller injects ``set_status`` (in-flight edits) and
``set_final`` (terminal edit + dialog_id for the keyboard).
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
import time
from collections.abc import Awaitable, Callable

import structlog
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.services import answer_cache, followup_cache
from src.bot.services.dialog_service import save_dialog
from src.bot.services.faq_service import format_faq_body, lookup_faq
from src.bot.services.glossary_upload import format_glossary_body, lookup_term
from src.bot.services.qa_renderer import (
    live_preview,
    render_textbook_body,
    render_web_body,
)
from src.bot.services.task_registry import tracked
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
from src.rag.web_search import search_web

log = structlog.get_logger(__name__)

# LLM refusal phrases — chain to web-fallback when matched at the start
# of a short answer (a longer answer that mentions "fragments don't cover X"
# as a caveat to a real answer must NOT trigger web).
_REFUSAL_PATTERNS_RU = (
    "в материалах курса этого прямо не нашлось",
    "в материалах курса нет ответа",
    "в материалах курса этого нет",
    "фрагменты не дают ответа",
    "фрагменты не покрывают",
    "во фрагментах нет",
    "в учебнике этого нет",
    "не нашёл ответа в материалах",
)
_REFUSAL_PATTERNS_EN = (
    "the course materials don't cover this",
    "the course materials do not cover this",
    "the fragments don't answer",
    "the fragments do not answer",
    "no answer in the materials",
    "not in the textbook",
)


def _looks_like_refusal(answer: str, lang: str) -> bool:
    head = answer.strip().lower()[:120]
    patterns = _REFUSAL_PATTERNS_EN if lang == "en" else _REFUSAL_PATTERNS_RU
    return any(p in head for p in patterns)


# Definitional-pattern: «что такое X», «определение X», «расскажи про X»,
# «X?» (одно слово). Если поймали — ``_extract_lookup_term`` возвращает X
# для проверки, реально ли X встречается во фрагментах. Иначе bge-m3
# матчит фонетически близкое («плейсхолдер» → «стейкхолдер») и LLM подменяет
# определение, не заметив подмены термина.
_DEFINITIONAL_RE = re.compile(
    r"^(?:что\s+такое|что\s+есть|определение|расскажи\s+(?:про|о)|объясни|"
    r"что\s+значит|кто\s+такой|кто\s+такая|кто\s+такие|"
    r"what\s+is|what\s+are|define|explain)\s+(.+?)$",
    flags=re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"[?.!,;:—\s]+$")
_SINGLE_WORD_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{2,}$")


def _extract_lookup_term(question: str) -> str | None:
    """Если вопрос вида «что такое X» (или одно слово-термин) — вернуть X.

    Иначе ``None`` — значит вопрос не lookup, lexical-gate не должен
    срабатывать (например, «как настроить SWOT-анализ» — про процесс,
    а не про слово).
    """
    q = (question or "").strip()
    q = _TRAILING_PUNCT.sub("", q)
    if not q:
        return None
    m = _DEFINITIONAL_RE.match(q)
    if m:
        term = _TRAILING_PUNCT.sub("", m.group(1).strip())
        return term or None
    if _SINGLE_WORD_RE.match(q):
        return q
    return None


def _term_present_in_hits(term: str, hits: list) -> bool:  # type: ignore[type-arg]
    """True если стем хоть одного слова термина встречается в каком-нибудь
    chunk-тексте. Стем общий с BM25 — «плейсхолдеру» матчится «плейсхолдер»,
    но НЕ «стейкхолдер» (разные стеммы).
    """
    from src.rag.hybrid import _stem, _tokenise

    term_stems = {_stem(t) for t in _tokenise(term) if len(t) >= 4}
    if not term_stems:
        return True  # слишком короткий термин (1-3 буквы) — не блокируем
    for h in hits:
        chunk_stems = {_stem(t) for t in _tokenise(getattr(h, "text", "") or "")}
        if term_stems & chunk_stems:
            return True
    return False


async def _find_fuzzy_term_in_corpus(
    session: AsyncSession, term: str, subject_id: int | None, *, threshold: float = 0.65
) -> str | None:
    """Найти близкий термин в глоссарии текущего предмета через difflib.

    Это страховка для типового кейса: студент помнит звучание, но забыл
    точное написание («плейсхолдер» при искомом «стейкхолдер», или наоборот).
    Если ratio ≥ ``threshold`` — предлагаем кнопкой «🔄 Да, про "Y"». Если
    глоссарий пуст или ничего не близко — возвращаем None, caller уйдёт в
    web-fallback. Threshold 0.65 эмпирический: «плейсхолдер»-«стейкхолдер»
    ≈ 0.83, «эмерджентность»-«эмерджентный» ≈ 0.95.
    """
    if subject_id is None:
        return None
    from difflib import SequenceMatcher

    from sqlalchemy import select as _select

    from src.db.models import GlossaryTerm

    rows = list(
        (
            await session.execute(
                _select(GlossaryTerm.term).where(GlossaryTerm.subject_id == subject_id)
            )
        ).scalars()
    )
    if not rows:
        return None
    q = term.lower().strip()
    best: str | None = None
    best_ratio = threshold
    for candidate in rows:
        c = candidate.lower().strip()
        if c == q:
            return candidate  # точное совпадение — хотя сюда мы попадаем только
            # если lexical-gate не нашёл стем; глоссарий-формы совпадают,
            # значит чанки реально не цитируют термин — отдаём как fuzzy.
        ratio = SequenceMatcher(None, q, c).ratio()
        if ratio > best_ratio:
            best = candidate
            best_ratio = ratio
    return best


SetStatus = Callable[[str], Awaitable[None]]
# (body, dialog_id, kind) — kind ∈ {"full","glossary","faq","web"} picks
# the feedback keyboard at the call site.
SetFinal = Callable[..., Awaitable[None]]
TypingPing = Callable[[], Awaitable[None]]

# Telegram cap is 1 edit/s/message; 1.1s avoids 429 bursts.
_STREAM_EDIT_INTERVAL_S = 1.1
_TYPING_INTERVAL_S = 4.0
_HEARTBEAT_DELAY_S = 4.0

# Per-modifier ceiling for follow-up answers. Default LLM_MAX_TOKENS=220 fits a
# short verbose answer but cuts «Подробнее»/«Пример» mid-thought — the user
# sees +1 sentence instead of a real expansion. Map intent → budget.
_FU_TOKEN_BUDGET: dict[str, int] = {"simplify": 260, "example": 480, "deepen": 700}


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
    """Retrieval → стриминг LLM → финальный edit + сохранение диалога.

    ``skip_short_circuit`` отключает FAQ/glossary/cache fast-path'ы — нужен
    кнопке «📖 Развёрнутый ответ», чтобы тап по glossary-ответу не вернул
    тот же glossary-однострочник.
    """
    with tracked():
        await _run_qa_pipeline_inner(
            question=question,
            session=session,
            user=user,
            lang=lang,
            set_status=set_status,
            set_final=set_final,
            typing_ping=typing_ping,
            skip_short_circuit=skip_short_circuit,
        )


async def _run_qa_pipeline_inner(
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
    q_html = html.escape(question)
    t0 = time.monotonic()

    if typing_ping is not None:
        await typing_ping()

    # LRU-кеш повторных вопросов — приоритетнее даже FAQ-сокращения, потому что
    # это нулевая работа: один dict-lookup, мгновенный edit. Скип, если юзер
    # явно жмёт «📖 Развёрнутый ответ» (skip_short_circuit=True) — там брифкеш
    # неуместен.
    brief_user = bool(
        getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
    )
    if not skip_short_circuit:
        cached = answer_cache.get(
            user_id=user.id, question=question, lang=lang, brief=brief_user
        )
        if cached is not None:
            await set_final(cached.body, cached.dialog_id, cached.kind)
            log.info(
                "qa_answer_from_cache",
                dialog_id=cached.dialog_id,
                cache_age_s=round(time.monotonic() - t0, 3),
            )
            return

    # Пинг плейсхолдера: убедимся, что edit работает уже сейчас, а не через 30с
    # после retrieval. Если текст совпадает с STATUS_RETRIEVING — Telegram
    # молча проглотит дубль.
    await set_status(texts.tr(lang, texts.STATUS_RETRIEVING))

    # FAQ-короткое замыкание: преподавательский ответ на этот вопрос
    # (или очень близкую нормализованную форму) пропускает retrieval и LLM.
    # Fast-path в миллисекундах — без bge-m3, без Qwen3. Caller отключает
    # через ``skip_short_circuit`` (кнопка «📖 Развёрнутый ответ»).
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

    # Glossary-короткое замыкание: если нормализованный вопрос точно совпал
    # с известным термином («что такое стейкхолдер» → «стейкхолдер»), вернуть
    # курированное определение. Менее точно, чем FAQ (термин, не вопрос),
    # но ловит большую часть definitional-запросов.
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

    # Если фоновый warm-up ещё не догрузил модели — сказать юзеру явно вместо
    # тихого блока на 30–60 с (типичный HF cold-download). Одноразовое
    # состояние — последующие вопросы пролетят мимо мгновенно.
    if not MODELS_READY.is_set():
        await set_status(texts.tr(lang, texts.STATUS_WARMING))
        try:
            await asyncio.wait_for(MODELS_READY.wait(), timeout=120.0)
        except TimeoutError:
            log.warning("qa_warmup_timeout")
            await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
            return
        await set_status(texts.tr(lang, texts.STATUS_RETRIEVING))

    # Стадия 1: retrieval + subject-routing. Если юзер закрепил предмет через
    # /subject — уважаем это, скипаем глобальный роутер и скоупим retrieval
    # только на этот предмет. Если был comparison-запрос («сравни X и Y»),
    # делаем два независимых retrieval — по одному на термин — и мержим,
    # чтобы LLM увидела сбалансированный контекст.
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
            # Предпочитаем общий предмет, если оба термина свелись к одному;
            # иначе — ту сторону, у которой есть hits.
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

    # Стадия 2: top-1 слишком слабый → пропускаем дорогой LLM-rewrite-цикл
    # (5–15с зря на Qwen3-4B, bge-m3 и так робастна к опечаткам), идём в web.
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

    # Lexical-gate для definitional-вопросов: если юзер спросил «что такое X»
    # (или просто «X»), а ни в одном из retrieved-чанков стем X не встречается —
    # это типовая фонетическая подмена (плейсхолдер vs стейкхолдер). LLM в
    # таком случае уверенно перефразирует «соседнее» определение под чужой
    # термин. Уходим в web вместо галлюцинации.
    if comparison_terms is None:
        # Локальная переменная не должна шейдить импорт ``lookup_term`` из
        # glossary_upload — Python видел бы её как local во всей функции и
        # ловил бы UnboundLocalError на FAQ-fast-path выше.
        _q_term = _extract_lookup_term(question)
        if _q_term and not _term_present_in_hits(_q_term, hits):
            # Студент мог помнить звучание, но забыть написание. Прежде чем
            # дёрнуть web — проверим fuzzy-близость в glossary текущего
            # предмета. Если есть похожий термин — покажем disambig-карточку
            # с кнопкой «🔄 Да, про "Y"» (один тап и pipeline стартует заново
            # с правильным словом).
            subj_id = getattr(subject, "id", None)
            fuzzy = await _find_fuzzy_term_in_corpus(session, _q_term, subj_id)
            if fuzzy:
                log.info("qa_lexical_gate_fuzzy_suggest", q=_q_term, suggested=fuzzy)
                disambig_body = texts.tr(lang, texts.DISAMBIG_BODY).format(
                    q=q_html,
                    subject=html.escape(texts.subject_title(subject, lang)),
                    q_term=html.escape(_q_term),
                    suggested=html.escape(fuzzy),
                )
                # Сохраняем в Dialog как «refused-with-suggestion» — для
                # коректного /history и /ref.
                refused_dialog = await save_dialog(
                    session,
                    user=user,
                    question=question,
                    result=AskResult(
                        answer=f"[no-match: suggested «{fuzzy}»]",
                        subject=subject,
                        hits=[],
                        route=None,
                    ),  # type: ignore[arg-type]
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
                await set_final(disambig_body, refused_dialog.id, "disambig")
                # Сохраняем suggestion в short-lived state, чтобы кнопка
                # `at:<term>` могла поднять ровно его, не доверяя
                # client-controlled callback_data слепо.
                followup_cache.store(
                    refused_dialog.id,
                    question=fuzzy,  # уже корректный термин
                    lang=lang,
                    hits=[],
                    subject=subject,
                )
                return
            log.info(
                "qa_lexical_gate_web_fallback",
                term=_q_term,
                top_score=round(top_score, 3),
            )
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

    # Стадия 3: стрим ответа по учебнику.
    assert subject is not None and hits
    subj_title = texts.subject_title(subject, lang)
    # Один edit перед стримом — объединяет «нашёл N + формулирую», чтобы не
    # жечь два edit'а через 50 мс друг от друга. На сотовой раньше бывал
    # видимый «прыжок» статуса.
    await set_status(
        texts.tr(lang, texts.STATUS_THINKING_FOUND).format(
            n=len(hits),
            subject=html.escape(subj_title),
        )
    )

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
            set_status=set_status,
            typing_ping=typing_ping,
        )
    except CircuitOpenError:
        # Breaker открыт — llama-server known-bad, отвечаем быстрым отказом
        # вместо ожидания 300с timeout'а.
        log.warning("qa_llm_circuit_open")
        await set_status(texts.tr(lang, texts.STATUS_BUSY))
        return
    except Exception:
        log.exception("qa_llm_failed")
        await set_status(texts.tr(lang, texts.INTERNAL_ERROR))
        return

    # Защита: даже с жёстким промптом Qwen3 иногда выдумывает «(Surname, 1984)»
    # или этимологию. Срезаем всё, что не подкреплено retrieved-чанками.
    answer, validation = validate_answer(answer, [h.text for h in hits])
    if validation.total:
        log.info(
            "qa_answer_validator_stripped",
            attributions=validation.attributions_stripped,
            etymologies=validation.etymologies_stripped,
            foreign=validation.foreign_scripts_stripped,
        )

    # Если модель сама сказала «фрагменты не покрывают» (мы её именно об
    # этом просим в system-prompt), прозрачно дёргаем web-fallback вместо
    # тупика. Детект — по короткому списку фраз; семантику не парсим.
    if _looks_like_refusal(answer, lang):
        log.info("qa_llm_refusal_web_fallback")
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

    latency_ms = int((time.monotonic() - t0) * 1000)

    result = AskResult(answer=answer, subject=subject, hits=hits, route=None)  # type: ignore[arg-type]
    dialog = await save_dialog(
        session, user=user, question=question, result=result, latency_ms=latency_ms
    )

    body = render_textbook_body(
        q_html=q_html,
        subject_title=subj_title,
        answer=answer,
        hits=hits,
        lang=lang,
        brief=bool(brief),
    )
    # Hits + subject в кэш: «⬇️ Проще / 💡 Пример / 📖 Подробнее» переезапускают
    # LLM без повторного retrieval. Ключ — новый dialog_id из feedback-клавиатуры.
    followup_cache.store(
        dialog.id,
        question=question,
        lang=lang,
        hits=hits,
        subject=subject,
        comparison_terms=comparison_terms,
    )
    # Кеш готового ответа — следующий тот же вопрос вернётся мгновенно.
    answer_cache.store(
        user_id=user.id,
        question=question,
        lang=lang,
        brief=bool(brief),
        body=body,
        kind="full",
        dialog_id=dialog.id,
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
        # Wall-clock cap: зависший DDG-backend не должен держать юзера на
        # «🌐 ищу в интернете…» минутами. ``DDGS(timeout=8)`` уже ограничивает
        # каждый backend-call; это общий потолок поверх ретраев
        # (~3 backend × 8с + 2 backoff = 30с worst case).
        web_hits = await asyncio.wait_for(
            asyncio.to_thread(search_web, question, 5, lang=lang),
            timeout=20.0,
        )
    except TimeoutError:
        log.warning("qa_web_search_timeout")
        web_hits = []
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

    # Сводка: «нашёл N источников: host1, host2, host3 — читаю…»
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

    # Прокручиваем первые ~3 хоста, чтобы юзер видел, откуда мы тянем.
    # Шаг — _STREAM_EDIT_INTERVAL_S, чтобы держаться ниже потолка Telegram
    # (1 edit/s/msg). Последний edit перетекает в «формулирую» ниже.
    for h in web_hits[:3]:
        await asyncio.sleep(_STREAM_EDIT_INTERVAL_S)
        await set_status(texts.tr(lang, texts.STATUS_WEB_VISITING).format(host=html.escape(h.host)))

    await set_status(texts.tr(lang, texts.STATUS_WEB_THINKING))
    try:
        messages = build_web_messages(question, web_hits, lang=lang)
        answer = await _stream_answer_to_ui(
            messages=messages,
            lang=lang,
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

    brief_web = bool(
        getattr(user, "answer_mode", None) and user.answer_mode.value == "brief"
    )
    body = render_web_body(
        q_html=q_html,
        answer=answer,
        web_hits=list(web_hits),
        lang=lang,
        brief=brief_web,
    )
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
    set_status: SetStatus,
    typing_ping: TypingPing | None,
    max_tokens: int | None = None,
) -> str:
    """Потребляем SSE-стрим LLM; каждые ~1.2с переписываем status-сообщение
    с живым превью ответа. Возвращает финальную очищенную строку.
    """
    accumulated: list[str] = []
    chunks_seen = 0
    edits_sent = 0
    first_edit_done = False
    stream_started = time.monotonic()
    last_edit = stream_started
    last_typing = stream_started

    # Один спокойный heartbeat через 4с тишины — раньше был тикер каждые 1.1с,
    # видимо мерцавший. Если первый токен прилетит раньше 4с (теплый кеш,
    # короткий промпт) — юзер heartbeat'а не увидит вовсе.
    async def _heartbeat() -> None:
        await asyncio.sleep(_HEARTBEAT_DELAY_S)
        try:
            await set_status(texts.tr(lang, texts.STATUS_ANALYSING))
        except (TelegramBadRequest, TelegramRetryAfter):
            # not modified / rate limit — норм, best-effort.
            pass
        except Exception:
            log.debug("heartbeat_edit_failed", exc_info=True)

    hb_task: asyncio.Task[None] | None = asyncio.create_task(_heartbeat())

    # try/finally чтобы mid-stream exception (httpx 5xx, отмена) корректно
    # завершил heartbeat — иначе он продолжит редактировать каждые 1.1с,
    # перезаписывая INTERNAL_ERROR-статус, который пишет внешний pipeline.
    # Sampling зафиксирован near-deterministic для факт-ответов:
    # temp=0.1 + top_p=0.9 + repeat_penalty=1.1 убивает дрейф «одинаковый
    # вопрос — разные выдуманные авторы», который был на дефолтных 0.7/0.8/1.05.
    # CoT уже отключён через /no_think в system-promtp'е.
    try:
        async for piece in chat_stream(
            messages,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=max_tokens,
        ):
            if hb_task is not None:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await hb_task
                hb_task = None
                # Сбрасываем edit-таймер, чтобы первое настоящее превью ушло
                # мгновенно, а не через остаток heartbeat-интервала.
                last_edit = time.monotonic() - _STREAM_EDIT_INTERVAL_S
            accumulated.append(piece)
            chunks_seen += 1
            now = time.monotonic()

            if typing_ping is not None and now - last_typing >= _TYPING_INTERVAL_S:
                try:
                    await typing_ping()
                except Exception:
                    # Typing — косметика, не должна валить стрим.
                    log.debug("typing_ping_failed", exc_info=True)
                last_typing = now

            # Первый edit — сразу как только что-то прилетело (юзер видит,
            # что стрим пошёл). Дальше — троттлинг.
            due_by_throttle = now - last_edit >= _STREAM_EDIT_INTERVAL_S
            if not first_edit_done or due_by_throttle:
                preview = live_preview("".join(accumulated))
                try:
                    await set_status(preview)
                except TelegramRetryAfter as exc:
                    # Уважаем back-off Telegram, потом продолжаем стрим.
                    # Sleep сдвигает last_edit, чтоб не повторить сразу.
                    await asyncio.sleep(exc.retry_after + 0.1)
                    last_edit = time.monotonic()
                    continue
                except TelegramBadRequest as exc:
                    # «message is not modified» — идемпотентный сигнал,
                    # глотаем. Другие 400 → битый preview-HTML, логируем,
                    # стрим продолжаем (финальный body ещё спасётся).
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
    # Дополнительно срезаем мусор из status-сообщения, который мог попасть
    # в saved-question у старых записей (баг до 2026-04-27).
    import re as _re

    q = _re.sub(
        r"^\[(simplify|example|deepen)\]\s*",
        "",
        dialog.question or "",
        flags=_re.IGNORECASE,
    )
    q = _re.sub(r"\s*[⌛⏳]\s.*$|\s*\n+.*$", "", q, flags=_re.DOTALL).strip()
    if not q:
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

    fu_max_tokens = _FU_TOKEN_BUDGET.get(modifier)

    try:
        answer = await _stream_answer_to_ui(
            messages=messages,
            lang=use_lang,
            set_status=set_status,
            typing_ping=typing_ping,
            max_tokens=fu_max_tokens,
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
        body = render_web_body(
            q_html=q_html,
            answer=answer,
            web_hits=list(ctx.web_hits),
            lang=use_lang,
            brief=False,
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
        body = render_textbook_body(
            q_html=q_html,
            subject_title=subject_title or "",
            answer=answer,
            hits=ctx.hits,
            lang=use_lang,
            brief=False,
        )

    # Под новым dialog_id, чтобы цепочка follow-up'ов работала. comparison_terms
    # сохраняем — иначе «Сравни X и Y → Проще → Пример» теряет side-by-side.
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


# Источник-форматтеры теперь в src/bot/services/qa_renderer.py.
