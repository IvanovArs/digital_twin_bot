"""Тесты сборки inline-клавиатур: правильные callback_data, структура рядов."""

from __future__ import annotations

from src.bot.keyboards import (
    ask_only_inline,
    ask_samples_inline,
    feedback_bare,
    feedback_brief,
    feedback_inline,
    feedback_short_answer,
    main_inline,
    processing_keyboard,
)


def _all_callbacks(kb) -> set[str]:  # type: ignore[no-untyped-def]
    out: set[str] = set()
    for row in kb.inline_keyboard:
        for btn in row:
            if btn.callback_data:
                out.add(btn.callback_data)
    return out


def test_main_inline_has_three_buttons() -> None:
    kb = main_inline("ru")
    cbs = _all_callbacks(kb)
    assert {"menu:ask", "menu:subjects", "menu:help"} <= cbs


def test_main_inline_works_for_en() -> None:
    kb = main_inline("en")
    cbs = _all_callbacks(kb)
    assert {"menu:ask", "menu:subjects", "menu:help"} <= cbs


def test_ask_samples_emits_one_button_per_sample() -> None:
    from src.bot.texts import ASK_SAMPLES

    kb = ask_samples_inline("ru")
    rows = kb.inline_keyboard
    assert len(rows) == len(ASK_SAMPLES)
    cbs = _all_callbacks(kb)
    assert all(c.startswith("ask_sample:") for c in cbs)


def test_feedback_inline_has_followup_and_thumbs_rows() -> None:
    kb = feedback_inline(dialog_id=42, lang="ru")
    rows = kb.inline_keyboard
    assert len(rows) == 3  # сэмпл / followup / feedback
    cbs = _all_callbacks(kb)
    assert {"fb:42:5", "fb:42:1"} <= cbs
    assert {"fu:42:simplify", "fu:42:example", "fu:42:deepen"} <= cbs


def test_feedback_brief_is_just_thumbs() -> None:
    kb = feedback_brief(dialog_id=99, lang="ru")
    assert len(kb.inline_keyboard) == 1
    cbs = _all_callbacks(kb)
    assert cbs == {"fb:99:5", "fb:99:1"}


def test_feedback_bare_has_followup_row() -> None:
    """Inline-ответы теперь умеют редактироваться по `inline_message_id`,
    так что в bare-клавиатуру вернули ряд follow-up'ов — студент в группе
    получает то же поведение «Проще / Пример / Подробнее» что и в личке."""
    kb = feedback_bare(dialog_id=7, lang="ru")
    cbs = _all_callbacks(kb)
    assert "fb:7:5" in cbs and "fb:7:1" in cbs
    assert {"fu:7:simplify", "fu:7:example", "fu:7:deepen"}.issubset(cbs)


def test_feedback_short_answer_has_expand_button() -> None:
    kb = feedback_short_answer(dialog_id=11, lang="ru")
    cbs = _all_callbacks(kb)
    assert any(c.startswith("expand:") for c in cbs)


def test_processing_keyboard_carries_rid() -> None:
    kb = processing_keyboard(rid="abc-123", lang="ru")
    cbs = _all_callbacks(kb)
    assert any("abc-123" in c for c in cbs)


def test_ask_only_inline_has_switch_inline_button() -> None:
    """«Задать ещё вопрос» — это switch_inline_query_current_chat,
    у него нет callback_data, проверяем по другому полю."""
    kb = ask_only_inline("ru")
    rows = kb.inline_keyboard
    assert len(rows) == 1 and len(rows[0]) == 1
    btn = rows[0][0]
    assert btn.switch_inline_query_current_chat is not None
