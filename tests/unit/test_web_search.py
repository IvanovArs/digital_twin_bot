"""Tests for the WebHit.host property and the host blocklist filter."""

from __future__ import annotations

from src.rag.web_search import WebHit, _host_blocked


def test_host_strips_www() -> None:
    h = WebHit(title="t", url="https://www.ru.wikipedia.org/wiki/x", snippet="s")
    assert h.host == "ru.wikipedia.org"


def test_host_handles_no_scheme() -> None:
    h = WebHit(title="t", url="malformed-url-no-host", snippet="s")
    # Falls back to a slice of the URL when parse yields nothing useful.
    assert h.host  # never empty


def test_host_lowercases() -> None:
    h = WebHit(title="t", url="https://EXAMPLE.COM/x", snippet="s")
    assert h.host == "example.com"


def test_blocklist_blocks_known_spam() -> None:
    assert _host_blocked("ru-stat.com")
    assert _host_blocked("otvet.mail.ru")
    assert _host_blocked("www.znanija.com")  # www. stripped


def test_blocklist_allows_legit() -> None:
    assert not _host_blocked("ru.wikipedia.org")
    assert not _host_blocked("cyberleninka.ru")
    assert not _host_blocked("")
