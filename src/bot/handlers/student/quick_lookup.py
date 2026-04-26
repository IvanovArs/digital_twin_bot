"""Быстрые справочные команды: /term (глоссарий) и /ref (источники)."""

from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Dialog, User

router = Router(name="student_quick")


@router.message(Command("term"))
async def on_term(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/term <слово> — прямой lookup в глоссарии. Без LLM, только курированное
    определение преподавателя или ничего."""
    from src.bot.services.glossary_upload import format_glossary_body, lookup_term

    query = (command.args or "").strip()
    if not query:
        await message.answer("Использование: <code>/term &lt;термин&gt;</code>", parse_mode="HTML")
        return
    hit = await lookup_term(session, question=query, subject_id=None)
    if hit is None:
        await message.answer(
            f"В глоссарии нет <code>«{html.escape(query)}»</code>. "
            "Задай обычный вопрос — поищу в учебниках.",
            parse_mode="HTML",
        )
        return
    await message.answer(format_glossary_body(hit, lang), parse_mode="HTML")


@router.message(Command("ref"))
async def on_ref(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/ref <dialog_id> — показать источники, на которых был построен ответ.

    Студенты просили способ проверить, откуда взят ответ, без перечитывания
    учебника. Печатаем сохранённый ``Dialog.sources`` (книга, страница, score)
    — этого хватит, чтобы открыть PDF и прочесть оригинал.
    """
    arg = (command.args or "").strip()
    if not arg:
        await message.answer(
            "Использование: <code>/ref &lt;dialog_id&gt;</code>. "
            "ID есть рядом с ответом в /history.",
            parse_mode="HTML",
        )
        return
    try:
        dialog_id = int(arg)
    except ValueError:
        await message.answer("dialog_id должен быть числом.")
        return
    dialog = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user.id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        await message.answer(
            f"Диалог <code>#{dialog_id}</code> не найден или не твой.", parse_mode="HTML"
        )
        return
    src_rows = dialog.sources or []
    if not src_rows:
        await message.answer(
            "У этого ответа нет учёта источников — возможно, он из глоссария "
            "или из FAQ преподавателя.",
        )
        return
    lines = [f"<b>Источники для #{dialog_id}</b>:"]
    for row in src_rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") == "web":
            title = html.escape(str(row.get("title") or "web"))
            url = html.escape(str(row.get("url") or ""), quote=True)
            lines.append(f'• 🌐 <a href="{url}">{title}</a>')
        else:
            book = html.escape(str(row.get("book") or "?"))
            page = row.get("page")
            score = row.get("score")
            bits = [f"📚 {book}"]
            if page is not None:
                bits.append(f"стр. {page}")
            if score is not None:
                bits.append(f"совпадение {float(score):.2f}")
            lines.append("• " + " · ".join(bits))
    await message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)
