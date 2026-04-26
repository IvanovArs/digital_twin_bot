"""Студенческий поток: вопросы, история, источники, фидбек.

Раньше всё лежало в одном 935-строчном `student.py`. Сейчас разнесено по темам:

- ``_common.py``     — shorten (хелпер для UI-листов)
- ``entry.py``       — /ask, /subject + sample-кнопки
- ``history.py``     — /history, /favourites, /find, /export + пагинация и ⭐
- ``quick_lookup.py``— /term, /ref
- ``question.py``    — on_question (главный entry в RAG)
- ``followup.py``    — 👍/👎, «Проще/Пример/Подробнее», expand, «что сейчас?»

Главный ``router`` объединяет под-роутеры — bot/main.py подключает только его.
"""

from __future__ import annotations

from aiogram import Router

from src.bot.handlers.student import entry, followup, history, question, quick_lookup

# Обратная совместимость с тестами и старыми импортами.
from src.bot.handlers.student._common import shorten as _shorten  # noqa: F401
from src.bot.handlers.student.entry import (  # noqa: F401
    on_ask,
    on_ask_sample,
    on_subject_command,
)
from src.bot.handlers.student.followup import (  # noqa: F401
    on_ask_term,
    on_expand_answer,
    on_feedback,
    on_followup,
    on_whats_happening,
)
from src.bot.handlers.student.history import (  # noqa: F401
    _history_keyboard,
    _render_history,
    on_export,
    on_favourites,
    on_find,
    on_history,
    on_history_page,
    on_star_toggle,
)
from src.bot.handlers.student.question import on_question  # noqa: F401
from src.bot.handlers.student.quick_lookup import on_ref, on_term  # noqa: F401

router = Router(name="student")
router.include_routers(
    entry.router,
    quick_lookup.router,
    history.router,
    question.router,
    followup.router,
)

__all__ = ["router"]
