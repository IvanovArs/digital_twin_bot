"""Тесты LRU-кеша ответов на повторные вопросы."""

from __future__ import annotations

import time

from src.bot.services import answer_cache


def setup_function(_func) -> None:  # type: ignore[no-untyped-def]
    answer_cache.invalidate_all()


def test_normalize_collapses_whitespace_and_punct() -> None:
    assert answer_cache.normalize("  Что такое  Стейкхолдер? ") == "что такое  стейкхолдер".replace(
        "  ", " "
    )
    # Случайная пунктуация на краях стирается.
    assert answer_cache.normalize("Стейкхолдер!!") == "стейкхолдер"
    assert answer_cache.normalize('"система"?') == "система"


def test_get_returns_none_on_empty_cache() -> None:
    assert (
        answer_cache.get(user_id=1, question="что-то", lang="ru", brief=False) is None
    )


def test_store_and_retrieve_same_question() -> None:
    answer_cache.store(
        user_id=1,
        question="что такое стейкхолдер?",
        lang="ru",
        brief=False,
        body="<b>Стейкхолдер</b> — ...",
        kind="full",
        dialog_id=42,
    )
    hit = answer_cache.get(
        user_id=1, question="ЧТО такое  стейкхолдер  ?", lang="ru", brief=False
    )
    assert hit is not None
    assert hit.dialog_id == 42
    assert hit.kind == "full"


def test_cache_isolated_per_user() -> None:
    answer_cache.store(
        user_id=1, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=1
    )
    assert (
        answer_cache.get(user_id=2, question="q", lang="ru", brief=False) is None
    )


def test_cache_isolated_per_lang_and_brief() -> None:
    answer_cache.store(
        user_id=1, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=1
    )
    assert (
        answer_cache.get(user_id=1, question="q", lang="en", brief=False) is None
    )
    assert (
        answer_cache.get(user_id=1, question="q", lang="ru", brief=True) is None
    )


def test_ttl_expires_old_entry(monkeypatch) -> None:
    answer_cache.store(
        user_id=1, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=1
    )
    monkeypatch.setattr(answer_cache, "_TTL_S", 0.01)
    time.sleep(0.02)
    assert answer_cache.get(user_id=1, question="q", lang="ru", brief=False) is None


def test_invalidate_user_drops_only_that_user() -> None:
    answer_cache.store(
        user_id=1, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=1
    )
    answer_cache.store(
        user_id=2, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=2
    )
    n = answer_cache.invalidate_user(1)
    assert n == 1
    assert answer_cache.get(user_id=1, question="q", lang="ru", brief=False) is None
    assert answer_cache.get(user_id=2, question="q", lang="ru", brief=False) is not None


def test_invalidate_all_returns_count() -> None:
    answer_cache.store(
        user_id=1, question="q", lang="ru", brief=False, body="b", kind="full", dialog_id=1
    )
    answer_cache.store(
        user_id=2, question="q2", lang="ru", brief=False, body="b", kind="full", dialog_id=2
    )
    assert answer_cache.invalidate_all() == 2
    assert answer_cache.stats()["entries"] == 0


def test_max_entries_evicts_oldest(monkeypatch) -> None:
    monkeypatch.setattr(answer_cache, "_MAX_ENTRIES", 10)
    for i in range(11):
        answer_cache.store(
            user_id=i,
            question=f"q{i}",
            lang="ru",
            brief=False,
            body=f"b{i}",
            kind="full",
            dialog_id=i,
        )
    # Эвакуация удалила ~10% самых старых.
    assert answer_cache.stats()["entries"] <= 10
