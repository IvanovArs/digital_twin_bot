"""Тесты handlers/inline.py: TTL-кеш, dedupe, helpers + inline_query handler."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.handlers import inline as inline_mod
from src.bot.handlers.inline import (
    _cache_gc,
    _cache_get,
    _cache_put,
    _placeholder_markup,
    _placeholder_message,
    _release,
    _results_button,
    _try_claim,
    on_inline_query,
)


def setup_function(_func) -> None:  # type: ignore[no-untyped-def]
    inline_mod._QUESTION_CACHE.clear()
    inline_mod._RUNNING.clear()


# ---------- TTL cache ----------


def test_cache_put_and_get_same_key() -> None:
    _cache_put("abc", "что такое стейкхолдер")
    assert _cache_get("abc") == "что такое стейкхолдер"


def test_cache_get_unknown_returns_none() -> None:
    assert _cache_get("missing") is None


def test_cache_gc_drops_stale_entries(monkeypatch) -> None:
    monkeypatch.setattr(inline_mod, "_CACHE_TTL_S", 0.01)
    _cache_put("k", "v")
    time.sleep(0.02)
    _cache_gc()
    assert _cache_get("k") is None


def test_cache_max_entries_evicts_oldest(monkeypatch) -> None:
    monkeypatch.setattr(inline_mod, "_CACHE_MAX_ENTRIES", 5)
    for i in range(6):
        _cache_put(f"k{i}", f"v{i}")
    # Должно быть ≤ 5 (после эвакуации 10%).
    assert len(inline_mod._QUESTION_CACHE) <= 5


# ---------- dedupe claim ----------


def test_try_claim_first_caller_wins() -> None:
    assert _try_claim("inline-1", "rid-A") is True
    assert _try_claim("inline-1", "rid-B") is False


def test_try_claim_same_rid_idempotent() -> None:
    assert _try_claim("inline-2", "rid-X") is True
    assert _try_claim("inline-2", "rid-X") is True


def test_release_allows_reclaim() -> None:
    assert _try_claim("inline-3", "rid-1") is True
    _release("inline-3")
    assert _try_claim("inline-3", "rid-2") is True


def test_claim_ttl_allows_reclaim_after_timeout(monkeypatch) -> None:
    monkeypatch.setattr(inline_mod, "_RUNNING_TTL_S", 0.01)
    _try_claim("inline-4", "rid-A")
    time.sleep(0.02)
    assert _try_claim("inline-4", "rid-B") is True


# ---------- helpers ----------


def test_placeholder_message_includes_status_and_quoted_question() -> None:
    body = _placeholder_message("что такое X?", "ru")
    assert "что такое X?" in body
    assert "Ищу" in body or "Searching" in body


def test_placeholder_markup_carries_qid() -> None:
    kb = _placeholder_markup("qid42", "ru")
    cbs = {b.callback_data for row in kb.inline_keyboard for b in row if b.callback_data}
    assert cbs == {"iq:qid42"}


def test_results_button_returns_help_deeplink() -> None:
    btn = _results_button("ru")
    assert btn.start_parameter == "help"


# ---------- on_inline_query ----------


@pytest.mark.asyncio
async def test_inline_query_empty_returns_samples() -> None:
    iq = MagicMock()
    iq.query = ""
    iq.answer = AsyncMock()
    iq.from_user = SimpleNamespace(language_code="ru")
    await on_inline_query(iq, "ru")
    iq.answer.assert_awaited()
    kwargs = iq.answer.call_args.kwargs
    assert kwargs.get("is_personal") is True
    results = kwargs.get("results") or iq.answer.call_args.args[0]
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_inline_query_real_question_caches_id() -> None:
    iq = MagicMock()
    iq.query = "что такое эмерджентность"
    iq.answer = AsyncMock()
    iq.from_user = SimpleNamespace(language_code="ru")
    await on_inline_query(iq, "ru")
    # В кеше осел ровно один qid с этим текстом.
    cached = list(inline_mod._QUESTION_CACHE.values())
    assert any("эмерджентность" in v[0] for v in cached)
