"""Tests for the follow-up feature: cache + prompt modifier.

The full callback → streaming → save_dialog path needs a real aiogram +
DB, so we unit-test the two pure layers:

1. ``followup_cache`` — store/get semantics and TTL eviction.
2. ``apply_followup_modifier`` + ``build_messages(modifier=...)`` — the
   modifier lands inside the user-role message without disturbing grounding.
"""

from __future__ import annotations

import pytest

from src.bot.services import followup_cache
from src.rag.prompts import apply_followup_modifier, build_messages, build_web_messages
from src.rag.retriever import Hit
from src.rag.web_search import WebHit
from src.subjects.schema import Subject


@pytest.fixture(autouse=True)
def _reset_cache():
    followup_cache.clear()
    yield
    followup_cache.clear()


def _hit(text: str = "дефиниция системы из учебника") -> Hit:
    return Hit(text=text, subject_slug="tos", book="x.pdf", page=1, score=0.8)


def _subject() -> Subject:
    return Subject(slug="tos", title_ru="Теория систем", title_en="Theory of Systems")


# ---------- cache ----------


def test_cache_store_then_get_returns_same_context() -> None:
    followup_cache.store(
        42,
        question="что такое система",
        lang="ru",
        hits=[_hit()],
        subject=_subject(),
    )
    ctx = followup_cache.get(42)
    assert ctx is not None
    assert ctx.question == "что такое система"
    assert ctx.lang == "ru"
    assert len(ctx.hits) == 1
    assert ctx.subject is not None and ctx.subject.slug == "tos"
    assert ctx.web_hits is None
    assert ctx.is_web is False


def test_cache_miss_returns_none() -> None:
    assert followup_cache.get(999) is None


def test_cache_web_variant_is_flagged() -> None:
    web = [WebHit(title="x", url="https://e.x/a", snippet="s")]
    followup_cache.store(7, question="q", lang="en", web_hits=web)
    ctx = followup_cache.get(7)
    assert ctx is not None
    assert ctx.is_web is True
    assert ctx.web_hits == web
    assert ctx.hits == []


def test_cache_evicts_stale_entries(monkeypatch) -> None:
    # Patch time.time in the cache module to return a fixed epoch so we can
    # later advance it deterministically. Advance past the full TTL —
    # P15 bumped that from 30 min to 24 h.
    base = 10_000_000.0
    monkeypatch.setattr(followup_cache.time, "time", lambda: base)
    followup_cache.store(1, question="q", lang="ru", hits=[_hit()])
    assert followup_cache.get(1) is not None
    monkeypatch.setattr(
        followup_cache.time, "time", lambda: base + followup_cache._TTL_SECONDS + 10
    )
    assert followup_cache.get(1) is None


def test_cache_stored_under_different_ids_do_not_collide() -> None:
    followup_cache.store(1, question="a", lang="ru", hits=[_hit("A")])
    followup_cache.store(2, question="b", lang="ru", hits=[_hit("B")])
    ctx1 = followup_cache.get(1)
    ctx2 = followup_cache.get(2)
    assert ctx1 is not None and ctx2 is not None
    assert ctx1.question == "a"
    assert ctx2.question == "b"


# ---------- prompt modifier ----------


@pytest.mark.parametrize("mod", ["simplify", "example", "deepen"])
def test_apply_modifier_ru_appends_instruction_before_otvet(mod: str) -> None:
    base = "some context\n\nВопрос студента: X\n\nОтвет:"
    out = apply_followup_modifier(base, mod, "ru")
    # The modifier tail must appear inside the user message — and must sit
    # BEFORE the trailing "Ответ:" so it's the last instruction the LLM reads.
    assert "Дополнительная инструкция" in out
    assert out.endswith("\n\nОтвет:")


@pytest.mark.parametrize("mod", ["simplify", "example", "deepen"])
def test_apply_modifier_en(mod: str) -> None:
    base = "snippets\n\nStudent's question: X\n\nAnswer:"
    out = apply_followup_modifier(base, mod, "en")
    assert "Extra" in out
    assert out.endswith("\n\nAnswer:")


def test_apply_modifier_unknown_passes_through() -> None:
    base = "abc\n\nОтвет:"
    assert apply_followup_modifier(base, "no_such_mod", "ru") == base


def test_build_messages_with_modifier_ru() -> None:
    hits = [_hit()]
    msgs = build_messages("что такое X", hits, _subject(), lang="ru", modifier="simplify")
    user = msgs[1]["content"]
    assert "Дополнительная инструкция" in user
    # Grounding text from the fragment survives.
    assert "дефиниция системы из учебника" in user


def test_build_messages_without_modifier_stable() -> None:
    hits = [_hit()]
    msgs = build_messages("что такое X", hits, _subject(), lang="ru")
    assert "Дополнительная инструкция" not in msgs[1]["content"]


def test_build_web_messages_with_modifier() -> None:
    web = [WebHit(title="t", url="https://e.x/a", snippet="s")]
    msgs = build_web_messages("q", web, lang="ru", modifier="example")
    assert "Дополнительная инструкция" in msgs[1]["content"]


# ---------- texts ----------


def test_followup_buttons_have_both_ru_and_en() -> None:
    from src.bot import texts as t

    for key in (t.BTN_FU_SIMPLIFY, t.BTN_FU_EXAMPLE, t.BTN_FU_DEEPEN):
        assert key["ru"] and key["en"]
        assert key["ru"] != key["en"]


def test_keyboard_feedback_has_followup_row() -> None:
    from src.bot.keyboards import feedback_inline

    kb = feedback_inline(dialog_id=123, lang="ru")
    # feedback row (2) + follow-up row (3) + «ask another» row (1) = 3 rows,
    # exactly 3 buttons in the middle row.
    rows = kb.inline_keyboard
    assert len(rows) == 3
    assert len(rows[1]) == 3
    callback_data = {b.callback_data for b in rows[1] if b.callback_data}
    assert callback_data == {
        "fu:123:simplify",
        "fu:123:example",
        "fu:123:deepen",
    }
