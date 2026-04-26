"""Тесты чистых helper'ов qa_pipeline (без сети)."""

from __future__ import annotations

from src.bot.services.qa_pipeline import _looks_like_refusal


def test_refusal_detected_at_start_of_short_answer_ru() -> None:
    txt = "В материалах курса этого прямо не нашлось."
    assert _looks_like_refusal(txt, "ru") is True


def test_refusal_not_detected_when_phrase_is_caveat_inside_long_answer() -> None:
    # Длинный полноценный ответ с упоминанием паттерна как оговорки в конце —
    # НЕ должен триггерить web-fallback.
    txt = (
        "Стейкхолдер — это лицо или группа, заинтересованные в проекте. "
        "Делятся на внутренних и внешних. Внутренние: акционеры, сотрудники. "
        "Внешние: клиенты, поставщики, регуляторы. "
        "Хотя в некоторых учебниках фрагменты не покрывают подкатегории."
    )
    assert _looks_like_refusal(txt, "ru") is False


def test_refusal_pattern_at_end_of_short_answer_not_detected() -> None:
    # Только в первых 120 символах — справедливая оговорка в конце не должна
    # триггерить.
    txt = "Стейкхолдер — заинтересованное лицо проекта. " + "x " * 60 + "во фрагментах нет."
    assert _looks_like_refusal(txt, "ru") is False


def test_refusal_detected_en() -> None:
    txt = "The course materials don't cover this directly."
    assert _looks_like_refusal(txt, "en") is True


def test_normal_answer_is_not_refusal() -> None:
    assert _looks_like_refusal("Стейкхолдер — это X.", "ru") is False
    assert _looks_like_refusal("A stakeholder is X.", "en") is False
