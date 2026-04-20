"""Inline keyboards for the bot.

Bot API 9.4 styles (Feb 2026):
  "primary" = blue, "success" = green, "danger" = red.
Older Telegram clients silently ignore the ``style`` field.

All interactive buttons live *under the message* as inline buttons — no
persistent reply-keyboard strip at the bottom of the chat.
"""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from src.bot import texts


def main_inline(lang: str) -> InlineKeyboardMarkup:
    """Primary menu row: ask / subjects / help. Attached to /start, /help, /subjects."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_ASK),
                    callback_data="menu:ask",
                    style="primary",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_SUBJECTS),
                    callback_data="menu:subjects",
                    style="primary",
                ),
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_MORE),
                    callback_data="menu:help",
                    style="primary",
                ),
            ],
        ],
    )


def ask_samples_inline(lang: str) -> InlineKeyboardMarkup:
    """Sample-question buttons shown under /ask.

    Each tap fires ``ask_sample:<idx>`` → student.py reuses the normal
    free-text question flow with the sample text plugged in.
    """
    rows: list[list[InlineKeyboardButton]] = []
    idx_lang = 0 if lang == "ru" else 1
    for i, pair in enumerate(texts.ASK_SAMPLES):
        rows.append(
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_SAMPLE_PREFIX) + pair[idx_lang],
                    callback_data=f"ask_sample:{i}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _followup_row(dialog_id: int, lang: str) -> list[InlineKeyboardButton]:
    """Follow-up row: «Проще / Пример / Подробнее». Each button re-runs the
    LLM on the cached hits with a prompt modifier — no second retrieval."""
    return [
        InlineKeyboardButton(
            text=texts.tr(lang, texts.BTN_FU_SIMPLIFY),
            callback_data=f"fu:{dialog_id}:simplify",
        ),
        InlineKeyboardButton(
            text=texts.tr(lang, texts.BTN_FU_EXAMPLE),
            callback_data=f"fu:{dialog_id}:example",
        ),
        InlineKeyboardButton(
            text=texts.tr(lang, texts.BTN_FU_DEEPEN),
            callback_data=f"fu:{dialog_id}:deepen",
        ),
    ]


def feedback_inline(dialog_id: int, lang: str) -> InlineKeyboardMarkup:
    """Full feedback markup for PM answers: 👍/👎 + follow-ups + «Задать ещё вопрос»."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_UP),
                    callback_data=f"fb:{dialog_id}:5",
                    style="success",
                ),
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_DOWN),
                    callback_data=f"fb:{dialog_id}:1",
                    style="danger",
                ),
            ],
            _followup_row(dialog_id, lang),
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_ASK_ANOTHER),
                    callback_data="menu:ask",
                    style="primary",
                ),
            ],
        ],
    )


def processing_keyboard(rid: str, lang: str) -> InlineKeyboardMarkup:
    """Attached to the placeholder while ``run_qa_pipeline`` is running.

    Tapping the button fires a callback that pulls the latest stage label out
    of ``processing_state`` and returns it as a Telegram alert — a quick "what
    is the bot doing right now?" peek without spamming chat edits.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_WHATS_HAPPENING),
                    callback_data=f"wh:{rid}",
                )
            ]
        ]
    )


def _ask_another_inline_button(lang: str) -> InlineKeyboardButton:
    """«💬 Задать ещё вопрос» button that prefills the chat input with
    ``@<bot_username> `` via Telegram's ``switch_inline_query_current_chat``
    primitive. Works in groups, channels and PMs without our bot being a
    member of the chat — just opens inline mode in place.
    """
    return InlineKeyboardButton(
        text=texts.tr(lang, texts.BTN_ASK_ANOTHER),
        switch_inline_query_current_chat="",
        style="primary",
    )


def feedback_bare(dialog_id: int, lang: str) -> InlineKeyboardMarkup:
    """Inline-sent answer keyboard: 👍/👎 + «Задать ещё вопрос»."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_UP),
                    callback_data=f"fb:{dialog_id}:5",
                    style="success",
                ),
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_DOWN),
                    callback_data=f"fb:{dialog_id}:1",
                    style="danger",
                ),
            ],
            [_ask_another_inline_button(lang)],
        ],
    )


def ask_only_inline(lang: str) -> InlineKeyboardMarkup:
    """Post-rating keyboard for inline-sent answers.

    After feedback we strip the 👍/👎 row but **keep** the «Задать ещё
    вопрос» shortcut so the student can fire a follow-up right away.
    Used by ``on_feedback`` in the inline callback path.
    """
    return InlineKeyboardMarkup(inline_keyboard=[[_ask_another_inline_button(lang)]])


def feedback_short_answer(dialog_id: int, lang: str) -> InlineKeyboardMarkup:
    """Keyboard for a glossary/FAQ short-circuit answer: 👍/👎 + a
    «📖 Развёрнутый ответ» button that re-runs the full RAG on the same
    question. Rationale: curated FAQ/glossary answers are one-paragraph,
    by design — but sometimes the student wants the full textbook
    treatment, and the one-tap expand is the nicest way to offer it.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_UP),
                    callback_data=f"fb:{dialog_id}:5",
                    style="success",
                ),
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_DOWN),
                    callback_data=f"fb:{dialog_id}:1",
                    style="danger",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_EXPAND_SHORT),
                    callback_data=f"expand:{dialog_id}",
                    style="primary",
                ),
            ],
        ],
    )


def feedback_brief(dialog_id: int, lang: str) -> InlineKeyboardMarkup:
    """Minimal keyboard for brief-mode answers: just 👍/👎, no follow-ups.

    Students in brief mode came for a textbook lookup, not a chat — one
    row of feedback is the most UI we attach; follow-up buttons and
    «ask another» would be noise in that mode.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_UP),
                    callback_data=f"fb:{dialog_id}:5",
                    style="success",
                ),
                InlineKeyboardButton(
                    text=texts.tr(lang, texts.BTN_FEEDBACK_DOWN),
                    callback_data=f"fb:{dialog_id}:1",
                    style="danger",
                ),
            ]
        ],
    )


