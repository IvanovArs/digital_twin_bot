"""Тесты фильтра качества web-сниппетов и хост-блок-листа."""

from __future__ import annotations

from unittest.mock import patch

from src.rag import web_search


def _ddg_result(title: str, url: str, body: str) -> dict[str, str]:
    return {"title": title, "href": url, "body": body}


def test_drops_thin_snippet_under_60_chars() -> None:
    raw = [_ddg_result("Stakeholder — Wikipedia", "https://en.wikipedia.org/x", "short text.")]
    with patch.object(web_search, "_try_backend", return_value=raw):
        out = web_search.search_web("stakeholder", k=3)
    assert out == []


def test_drops_snippet_with_too_few_words() -> None:
    raw = [_ddg_result("OK", "https://example.com/x", "слово1 слово2 слово3 слово4 слово5 слово6 слово7")]
    with patch.object(web_search, "_try_backend", return_value=raw):
        out = web_search.search_web("q", k=3)
    assert out == []


def test_drops_when_snippet_is_substring_of_title() -> None:
    raw = [
        _ddg_result(
            "Stakeholder Wikipedia free encyclopedia entry article",
            "https://en.wikipedia.org/x",
            "wikipedia free encyclopedia entry",
        )
    ]
    with patch.object(web_search, "_try_backend", return_value=raw):
        out = web_search.search_web("q", k=3)
    assert out == []


def test_keeps_substantive_snippet() -> None:
    raw = [
        _ddg_result(
            "Stakeholder definition",
            "https://example.org/article",
            "Stakeholder is a person or organisation that has an interest in a project, "
            "either internal or external.",
        )
    ]
    with patch.object(web_search, "_try_backend", return_value=raw):
        out = web_search.search_web("stakeholder", k=3)
    assert len(out) == 1
    assert out[0].title == "Stakeholder definition"


def test_returns_empty_when_all_backends_fail() -> None:
    with patch.object(web_search, "_try_backend", side_effect=Exception("boom")):
        out = web_search.search_web("q", k=3)
    assert out == []
