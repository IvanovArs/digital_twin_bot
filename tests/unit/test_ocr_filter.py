"""Tests for the OCR-garbage heuristic used in ingest + retrieval."""

from __future__ import annotations

from src.rag.ocr_filter import (
    detect_ru_dominant,
    is_ocr_garbage,
    is_ocr_garbage_auto,
)

CLEAN_RU = (
    "Стейкхолдер — это физическое или юридическое лицо, которое может "
    "повлиять на достижение организацией своих целей или на работу "
    "организации в целом. Заинтересованные стороны включают акционеров, "
    "поставщиков, клиентов, сотрудников и государственные органы."
)

CLEAN_RU_WITH_ACRONYMS = (
    "Для внешней среды применяют STEP-анализ (PEST-анализ) или его "
    "расширенный вариант — PESTELанализ. Также используется SWOT-анализ "
    "в сочетании с теорией заинтересованных сторон. В. Г. Колосов "
    "предложил альтернативную методику описания среды организации."
)

CLEAN_RU_WITH_URL = (
    "Основные требования к иерархическим структурам: в структурах не "
    "должно быть «вырожденных» ветвей, когда у родительского элемента "
    "отсутствуют дочерние. Источник: "
    "https://elib.spbstu.ru/dl/3/2024/vr/vr24-1408.pdf/info"
)

OCR_MIXED = (
    "Gow Бергаланфи определил экаифинатьность как pearpasse в мирытых "
    "системах. Wenartpompyeaie noxxon ‘Tor термин предельных состояний "
    "fades ‘oro Mow системы при полностмо начальных условиях зависяще."
)

OCR_LATIN_DEBRIS = (
    "„шт == some До ==YASS же И ownsere een conc i emer eter mee Feira "
    "cog es cemmenra eet a me te n nness na эффективность Soe ой риа "
    "ералосеор cere eed ae ea a"
)

CLEAN_EN = (
    "A stakeholder is a person or organisation that can affect the "
    "achievement of the organisation's objectives or is affected by the "
    "activities of the organisation. Stakeholders include shareholders, "
    "suppliers, customers, employees, and government regulators."
)


def test_clean_russian_prose_is_not_garbage() -> None:
    assert not is_ocr_garbage(CLEAN_RU)


def test_acronym_compound_is_not_garbage() -> None:
    assert not is_ocr_garbage(CLEAN_RU_WITH_ACRONYMS)


def test_text_with_urls_is_not_garbage() -> None:
    assert not is_ocr_garbage(CLEAN_RU_WITH_URL)


def test_mixed_alphabet_garbage_is_flagged() -> None:
    assert is_ocr_garbage(OCR_MIXED)


def test_latin_dominant_debris_in_ru_index_is_flagged() -> None:
    assert is_ocr_garbage(OCR_LATIN_DEBRIS)


def test_clean_english_is_not_garbage_in_en_mode() -> None:
    assert not is_ocr_garbage(CLEAN_EN, ru_dominant=False)


def test_clean_english_looks_like_garbage_in_ru_mode() -> None:
    """Sanity check for the ``ru_dominant`` flag: English prose would be
    falsely rejected by the RU-index rules, which is exactly why the C-ticket
    auto-detect exists."""
    assert is_ocr_garbage(CLEAN_EN, ru_dominant=True)


def test_auto_detect_handles_english_prose() -> None:
    """Full auto pipeline: detect English, apply mirror rules, accept it."""
    assert not is_ocr_garbage_auto(CLEAN_EN)


def test_auto_detect_handles_russian_prose() -> None:
    assert not is_ocr_garbage_auto(CLEAN_RU)


def test_detect_ru_dominant_on_russian_text() -> None:
    assert detect_ru_dominant(CLEAN_RU) is True


def test_detect_ru_dominant_on_english_text() -> None:
    assert detect_ru_dominant(CLEAN_EN) is False


def test_short_chunk_is_not_flagged() -> None:
    """Too-short snippets (headers, page numbers) can't be judged reliably
    — let them through rather than false-reject a legitimate fragment."""
    assert not is_ocr_garbage("Глава 1. Введение")
