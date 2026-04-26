"""Tests for the brief-mode preference + prompt + keyboard."""

from __future__ import annotations

from src.bot.keyboards import feedback_brief, feedback_inline
from src.db.models import AnswerMode, User, UserRole
from src.rag.prompts import build_messages
from src.rag.retriever import Hit
from src.subjects.schema import Subject


def _hit(text: str = "фрагмент учебника") -> Hit:
    return Hit(text=text, subject_slug="tos", book="x.pdf", page=1, score=0.8)


def _subject() -> Subject:
    return Subject(slug="tos", title_en="Theory of Systems", title_ru="Теория систем")


# ---------- AnswerMode enum & User default ----------


def test_user_defaults_to_brief_mode() -> None:
    """Дефолт PM-ответа — brief: меньше визуального шума, мгновенно читается."""
    u = User(telegram_id=1, full_name="U")
    assert u.answer_mode is AnswerMode.brief


def test_user_role_and_mode_are_independent() -> None:
    u = User(telegram_id=1, full_name="U")
    u.role = UserRole.teacher
    u.answer_mode = AnswerMode.brief
    assert u.role is UserRole.teacher
    assert u.answer_mode is AnswerMode.brief


def test_answer_mode_enum_has_both_values() -> None:
    assert {m.value for m in AnswerMode} == {"verbose", "brief"}


# ---------- prompt ----------


def test_brief_mode_produces_single_paragraph_ru_instruction() -> None:
    msgs = build_messages("что такое X", [_hit()], _subject(), lang="ru", brief=True)
    user = msgs[1]["content"]
    assert "ОДНИМ" in user or "одним" in user
    assert "Без буллетов" in user
    # The three-part structure instruction from verbose mode must NOT be here.
    assert "1) определение" not in user


def test_brief_mode_en_instruction() -> None:
    msgs = build_messages("what is X", [_hit()], _subject(), lang="en", brief=True)
    user = msgs[1]["content"]
    assert "ONE compact paragraph" in user
    assert "No bullets" in user


def test_verbose_mode_allows_conditional_bullets() -> None:
    """Verbose mode no longer mandates a rigid "3-5 bullets" structure (that
    was forcing the model to invent filler when fragments were thin). It now
    just *allows* bullets when the fragments themselves enumerate things."""
    msgs = build_messages("что такое X", [_hit()], _subject(), lang="ru", brief=False)
    user = msgs[1]["content"]
    assert "Буллеты" in user or "буллеты" in user
    assert "перечисление" in user
    # Brief mode's "no bullets" hard ban must NOT leak into verbose.
    assert "Без буллетов" not in user


def test_brief_mode_preserves_grounding_system_prompt() -> None:
    """Brief mode must NOT loosen the no-invention / no-CJK / no-guess rules
    — those live in the system message, not the user tail."""
    msgs = build_messages("q", [_hit()], _subject(), lang="ru", brief=True)
    system = msgs[0]["content"]
    assert "/no_think" in system
    # The anti-hallucination block must still ground answers in fragments.
    assert "ТОЛЬКО на основе фрагментов" in system
    assert "Не додумывай" in system or "выдумывать" in system


# ---------- keyboard ----------


def test_feedback_brief_has_only_feedback_row() -> None:
    kb = feedback_brief(dialog_id=42, lang="ru")
    assert len(kb.inline_keyboard) == 1
    assert len(kb.inline_keyboard[0]) == 2  # 👍 + 👎
    cbs = {b.callback_data for b in kb.inline_keyboard[0] if b.callback_data}
    assert cbs == {"fb:42:5", "fb:42:1"}


def test_feedback_verbose_still_has_followup_row() -> None:
    kb = feedback_inline(dialog_id=42, lang="ru")
    # Sanity: the verbose keyboard we didn't touch still has 3 rows.
    assert len(kb.inline_keyboard) == 3
    cbs = {b.callback_data for b in kb.inline_keyboard[1] if b.callback_data}
    assert cbs == {
        "fu:42:simplify",
        "fu:42:example",
        "fu:42:deepen",
    }
