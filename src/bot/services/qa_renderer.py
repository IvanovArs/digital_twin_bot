"""Telegram-форматирование ответов RAG-пайплайна.

Здесь собрано всё, что превращает чистый ``answer + hits`` в готовый HTML
для ``editMessageText``: безопасный escape, склейка источников, обрезка
до 4096 символов, live-preview во время стриминга.

Модуль не знает про сессии БД, LLM и aiogram — только про текст. Это
позволяет переиспользовать его из CLI / web-фронта без правок.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict

from src.bot import texts
from src.bot.services.safe_html import safe_html as _safe_html
from src.bot.services.safe_html import truncate_for_telegram as _truncate_for_telegram
from src.rag.retriever import Hit
from src.rag.web_search import WebHit

_TAG_RE = re.compile(r"<[^>]+>")


def render_textbook_sources(hits: list[Hit], lang: str) -> str:
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


def render_textbook_sources_inline(hits: list[Hit]) -> str:
    """Однострочные источники для brief-режима: «book p.12, 15; book2 p.3»."""
    by_book: dict[str, list[int]] = defaultdict(list)
    for h in hits:
        by_book[h.book].append(h.page)
    parts: list[str] = []
    for book, pages in by_book.items():
        pages_str = ", ".join(str(p) for p in sorted(set(pages)))
        parts.append(f"{html.escape(book)} (стр. {pages_str})")
    return "; ".join(parts)


def render_web_sources(web_hits: list[WebHit], lang: str) -> str:
    return "\n".join(
        texts.tr(lang, texts.SOURCES_ITEM_WEB).format(
            url=html.escape(h.url, quote=True), title=html.escape(h.title)
        )
        for h in web_hits
    )


def render_textbook_body(
    *,
    q_html: str,
    subject_title: str,
    answer: str,
    hits: list[Hit],
    lang: str,
    brief: bool,
) -> str:
    if brief:
        body = texts.tr(lang, texts.ANSWER_BODY_BRIEF).format(
            answer=_safe_html(answer),
            sources=render_textbook_sources_inline(hits),
        )
    else:
        body = texts.tr(lang, texts.ANSWER_BODY).format(
            q=q_html,
            subject=html.escape(subject_title),
            answer=_safe_html(answer),
            sources=render_textbook_sources(hits, lang),
        )
    return _truncate_for_telegram(body)


def render_web_body(
    *,
    q_html: str,
    answer: str,
    web_hits: list[WebHit],
    lang: str,
    brief: bool,
) -> str:
    if brief:
        sources_inline = ", ".join(html.escape(h.host or h.url[:40]) for h in web_hits)
        body = texts.tr(lang, texts.ANSWER_BODY_BRIEF_WEB).format(
            answer=_safe_html(answer),
            sources=sources_inline,
        )
    else:
        body = texts.tr(lang, texts.ANSWER_BODY_WEB).format(
            q=q_html,
            answer=_safe_html(answer),
            sources=render_web_sources(web_hits, lang),
        )
    return _truncate_for_telegram(body)


def live_preview(partial: str) -> str:
    """In-flight превью стрима: HTML-теги вырезаются, добавляется курсор `▍`.

    Стриппинг (а не escape) — обязателен: незакрытый ``<b>`` посреди стрима
    под parse_mode=HTML вернул бы 400 BadRequest.
    """
    text = _TAG_RE.sub("", partial).strip()
    if len(text) > 800:
        text = "…" + text[-799:]
    return f"{html.escape(text)} ▍"
