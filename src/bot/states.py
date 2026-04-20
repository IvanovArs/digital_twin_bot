"""FSM states."""

from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class StudentFlow(StatesGroup):
    awaiting_question = State()


class AdminFlow(StatesGroup):
    uploading_material = State()
    fixing_answer = State()
    uploading_glossary = State()
