from __future__ import annotations

from src.bot import texts


def test_normalize_lang_direct_match():
    assert texts.normalize_lang("ru") == "ru"
    assert texts.normalize_lang("en") == "en"


def test_normalize_lang_with_region():
    assert texts.normalize_lang("ru-RU") == "ru"
    assert texts.normalize_lang("EN-US") == "en"


def test_normalize_lang_ex_ussr_falls_back_to_ru():
    assert texts.normalize_lang("uk") == "ru"
    assert texts.normalize_lang("kk") == "ru"
    assert texts.normalize_lang("be") == "ru"


def test_normalize_lang_other_languages_fall_back_to_en():
    assert texts.normalize_lang("de") == "en"
    assert texts.normalize_lang("fr") == "en"
    assert texts.normalize_lang("zh") == "en"


def test_normalize_lang_missing_or_blank():
    assert texts.normalize_lang(None) == "ru"
    assert texts.normalize_lang("") == "ru"


def test_tr_picks_requested_language():
    t: texts.Tr = {"ru": "Привет", "en": "Hi"}
    assert texts.tr("ru", t) == "Привет"
    assert texts.tr("en", t) == "Hi"


def test_tr_falls_back_when_missing():
    only_ru: texts.Tr = {"ru": "только_ru"}
    assert texts.tr("en", only_ru) == "только_ru"

    only_en: texts.Tr = {"en": "only_en"}
    assert texts.tr("ru", only_en) == "only_en"


def test_tr_all_returns_distinct_variants():
    assert texts.tr_all(texts.BTN_ASK) == {"💬 Задать вопрос", "💬 Ask a question"}


def test_all_ui_keys_have_both_ru_and_en():
    """Hard commitment: every UI constant must exist in both languages."""
    required = [
        texts.HELLO,
        texts.PROMPT_QUESTION,
        texts.BTN_ASK,
        texts.BTN_SUBJECTS,
        texts.BTN_HELP,
        texts.HELP_STUDENT,
        texts.HELP_ADMIN_EXTRA,
        texts.NO_HITS,
        texts.Q_WITH_STATUS,
        texts.STATUS_RETRIEVING,
        texts.STATUS_THINKING,
        texts.ANSWER_BODY,
        texts.SOURCES_ITEM,
        texts.BTN_FEEDBACK_UP,
        texts.BTN_FEEDBACK_DOWN,
        texts.BTN_ASK_ANOTHER,
        texts.FEEDBACK_THANKS,
        texts.INTERNAL_ERROR,
        texts.GLOSSARY_EMPTY,
        texts.INLINE_EMPTY_TITLE,
        texts.INLINE_PREVIEW_TITLE,
        texts.INLINE_PREVIEW_DESC,
        texts.INLINE_MESSAGE_TEXT,
        texts.BTN_INSTRUCTIONS,
        texts.ACCESS_DENIED,
    ]
    for d in required:
        assert d.get("ru"), f"missing ru: {d!r}"
        assert d.get("en"), f"missing en: {d!r}"


def test_subject_title_picks_en_or_falls_back():
    class S:
        title_ru = "Теория систем"
        title_en = "Theory of Systems"

    assert texts.subject_title(S, "ru") == "Теория систем"
    assert texts.subject_title(S, "en") == "Theory of Systems"

    class NoEn:
        title_ru = "Только рус"
        title_en = ""

    assert texts.subject_title(NoEn, "en") == "Только рус"
