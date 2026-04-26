"""Тесты Telegram-форматтера ответов RAG."""

from __future__ import annotations

from src.bot.services.qa_renderer import (
    live_preview,
    render_textbook_body,
    render_textbook_sources,
    render_textbook_sources_inline,
    render_web_body,
    render_web_sources,
)
from src.rag.retriever import Hit
from src.rag.web_search import WebHit


def _h(book: str, page: int, score: float = 0.8) -> Hit:
    return Hit(text="x", subject_slug="tos", book=book, page=page, score=score)


def test_render_textbook_sources_groups_pages_per_book() -> None:
    out = render_textbook_sources([_h("a.pdf", 12), _h("a.pdf", 5), _h("b.pdf", 3)], lang="ru")
    assert "a.pdf" in out and "5, 12" in out
    assert "b.pdf" in out and "3" in out


def test_render_textbook_sources_inline_compact() -> None:
    out = render_textbook_sources_inline([_h("a.pdf", 12), _h("a.pdf", 12), _h("b.pdf", 1)])
    assert "a.pdf (стр. 12)" in out
    assert "b.pdf (стр. 1)" in out
    assert ";" in out


def test_render_textbook_body_brief_skips_subject_block() -> None:
    body = render_textbook_body(
        q_html="?",
        subject_title="Теория систем",
        answer="ответ",
        hits=[_h("a.pdf", 1)],
        lang="ru",
        brief=True,
    )
    assert "ответ" in body
    # В brief нет header'а с курсом — только inline "Источники:".
    assert "Курс:" not in body and "📚" not in body
    assert "a.pdf" in body


def test_render_textbook_body_verbose_has_subject_and_sources() -> None:
    body = render_textbook_body(
        q_html="?",
        subject_title="Теория систем",
        answer="<b>X</b> — это Y",
        hits=[_h("a.pdf", 7)],
        lang="ru",
        brief=False,
    )
    assert "Теория систем" in body
    assert "Источники" in body
    assert "стр. 7" in body


def test_render_web_sources_renders_html_anchor() -> None:
    out = render_web_sources(
        [WebHit(title="Wiki", url="https://ru.wikipedia.org/x", snippet="...")],
        lang="ru",
    )
    assert '<a href="https://ru.wikipedia.org/x">Wiki</a>' in out


def test_render_web_body_brief_uses_host_list() -> None:
    body = render_web_body(
        q_html="?",
        answer="ответ",
        web_hits=[WebHit(title="A", url="https://wikipedia.org/a", snippet="...")],
        lang="ru",
        brief=True,
    )
    assert "wikipedia.org" in body


def test_live_preview_strips_partial_html_tags() -> None:
    # Незакрытый <b> не должен сломать parse_mode=HTML — стрипаем.
    out = live_preview("<b>Стейкхолдер</b> — это <i>заин")
    assert "<b>" not in out and "</b>" not in out
    assert "Стейкхолдер" in out
    assert out.endswith("▍")


def test_live_preview_keeps_tail_for_long_partial() -> None:
    long = "x" * 1500
    out = live_preview(long)
    assert out.startswith("…")
    assert out.endswith("▍")
    assert len(out) < 1000
