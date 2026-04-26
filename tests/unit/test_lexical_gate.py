"""Тесты lexical-gate: detection «что такое X» + проверка наличия термина в hits."""

from __future__ import annotations

from src.bot.services.qa_pipeline import (
    _extract_lookup_term,
    _term_present_in_hits,
)
from src.rag.retriever import Hit


def _h(text: str) -> Hit:
    return Hit(text=text, subject_slug="tos", book="x.pdf", page=1, score=0.5)


# ---------- _extract_lookup_term ----------


def test_extract_what_is_pattern() -> None:
    assert _extract_lookup_term("что такое плейсхолдер") == "плейсхолдер"
    assert _extract_lookup_term("Что такое стейкхолдер?") == "стейкхолдер"
    assert _extract_lookup_term("что есть эмерджентность") == "эмерджентность"


def test_extract_definition_pattern() -> None:
    assert _extract_lookup_term("определение системы") == "системы"
    assert _extract_lookup_term("расскажи про подсистему") == "подсистему"
    assert _extract_lookup_term("объясни декомпозицию") == "декомпозицию"


def test_extract_who_pattern() -> None:
    assert _extract_lookup_term("кто такой стейкхолдер") == "стейкхолдер"


def test_extract_english_patterns() -> None:
    assert _extract_lookup_term("what is placeholder") == "placeholder"
    assert _extract_lookup_term("define stakeholder") == "stakeholder"


def test_single_word_returned_as_is() -> None:
    assert _extract_lookup_term("эмерджентность") == "эмерджентность"
    assert _extract_lookup_term("placeholder") == "placeholder"


def test_non_definitional_returns_none() -> None:
    """Процедурные/контекстные вопросы не должны триггерить gate."""
    assert _extract_lookup_term("как настроить SWOT-анализ?") is None
    assert _extract_lookup_term("в каком году появилась теория систем") is None
    assert _extract_lookup_term("сравни X и Y") is None


def test_empty_returns_none() -> None:
    assert _extract_lookup_term("") is None
    assert _extract_lookup_term("   ") is None
    assert _extract_lookup_term("?") is None


def test_short_single_letter_not_term() -> None:
    """Одна буква — не считаем термином."""
    assert _extract_lookup_term("X") is None


# ---------- _term_present_in_hits ----------


def test_term_present_when_exact_match() -> None:
    hits = [_h("Стейкхолдер — это лицо или организация...")]
    assert _term_present_in_hits("стейкхолдер", hits) is True


def test_term_present_inflected_form() -> None:
    """Стеммер сводит «стейкхолдеру» и «стейкхолдер» к одному."""
    hits = [_h("...стейкхолдеру важна стабильность поставок...")]
    assert _term_present_in_hits("стейкхолдер", hits) is True


def test_term_absent_phonetically_close_word_not_match() -> None:
    """Главный кейс: «плейсхолдер» НЕ должен матчиться к «стейкхолдер»."""
    hits = [_h("Стейкхолдер — это заинтересованное лицо.")]
    assert _term_present_in_hits("плейсхолдер", hits) is False


def test_term_absent_completely() -> None:
    hits = [_h("Эмерджентность — свойство целого, отсутствующее у частей.")]
    assert _term_present_in_hits("стейкхолдер", hits) is False


def test_term_absent_returns_false_with_multiple_hits() -> None:
    hits = [
        _h("Эмерджентность — свойство целого."),
        _h("Декомпозиция — разделение на подсистемы."),
        _h("Системный анализ — методика исследования."),
    ]
    assert _term_present_in_hits("плейсхолдер", hits) is False


def test_short_term_passes_through() -> None:
    """Термины ≤3 символов слишком общие — gate не блокирует.

    Иначе бы заблокировали валидные вопросы про IT, ER, ОС и т.п.
    """
    assert _term_present_in_hits("ОС", [_h("совсем другой текст")]) is True
    assert _term_present_in_hits("X", []) is True


def test_multi_word_term_any_word_matches() -> None:
    """Если хоть одно слово многословного термина в hits — matched."""
    hits = [_h("...методика декомпозиции широко применяется...")]
    assert _term_present_in_hits("методика Кошарского", hits) is True


def test_empty_hits_returns_false() -> None:
    assert _term_present_in_hits("стейкхолдер", []) is False
