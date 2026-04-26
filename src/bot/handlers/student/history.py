"""История диалогов: /history, /favourites, /find, /export + пагинация и ⭐."""

from __future__ import annotations

import contextlib
import html

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot import texts
from src.bot.handlers.student._common import shorten
from src.bot.keyboards import main_inline
from src.bot.services.dialog_service import set_favourite, user_dialogs_page
from src.db.models import User

router = Router(name="student_history")

_HIST_PAGE_SIZE = 5


def _history_keyboard(
    dialogs, page: int, total: int, *, favourites_only: bool, search: str | None
) -> InlineKeyboardMarkup:
    """Кнопки ⭐-toggle на каждой строке + пагинация prev/next.

    Пакуем всё в одну клавиатуру, чтобы каждое история-сообщение было
    самодостаточным. Payload компактный: ``hst:<page>:<fav>`` для навигации
    (``fav`` — 0/1) и ``star:<dialog_id>:<0|1>:<page>:<fav>`` для toggle.
    """
    rows: list[list[InlineKeyboardButton]] = []
    for d in dialogs:
        marker = "⭐" if d.is_favourite else "☆"
        target = 0 if d.is_favourite else 1
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker} #{d.id}",
                    callback_data=f"star:{d.id}:{target}:{page}:{int(favourites_only)}",
                )
            ]
        )
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(text="◀", callback_data=f"hst:{page - 1}:{int(favourites_only)}")
        )
    if (page + 1) * _HIST_PAGE_SIZE < total:
        nav.append(
            InlineKeyboardButton(text="▶", callback_data=f"hst:{page + 1}:{int(favourites_only)}")
        )
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _render_history(
    session: AsyncSession,
    user: User,
    lang: str,
    *,
    page: int,
    favourites_only: bool,
    search: str | None = None,
) -> tuple[str, InlineKeyboardMarkup | None]:
    dialogs, total = await user_dialogs_page(
        session,
        user_id=user.id,
        offset=page * _HIST_PAGE_SIZE,
        limit=_HIST_PAGE_SIZE,
        favourites_only=favourites_only,
        search=search,
    )
    if not dialogs:
        return texts.tr(lang, texts.HISTORY_EMPTY), None

    header = texts.tr(lang, texts.HISTORY_HEADER)
    if favourites_only:
        header = "⭐ " + header.lstrip("📜 ")
    if search:
        header += f"\n🔎 <i>поиск:</i> <code>{html.escape(search)}</code>\n"

    lines: list[str] = []
    for d in dialogs:
        title = texts.subject_title(d.subject, lang) if d.subject else "?"
        star = "⭐ " if d.is_favourite else ""
        lines.append(
            f"{star}" + texts.tr(lang, texts.HISTORY_ITEM).format(
                when=d.created_at.strftime("%d.%m %H:%M"),
                subject=title,
                q=shorten(d.question, 120),
                a=shorten(d.answer, 200),
            )
        )
    body = header + "\n\n".join(lines)
    kb = _history_keyboard(
        dialogs, page=page, total=total, favourites_only=favourites_only, search=search
    )
    return body, kb


@router.message(Command("history"))
async def on_history(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    """/history [page] — постраничная история с ⭐-toggle.

    /history          первая страница, все диалоги
    /history 2        третья страница (UI 1-индексирован, payload — 0)
    """
    page = 0
    arg = (command.args or "").strip()
    if arg:
        try:
            page = max(0, int(arg) - 1)
        except ValueError:
            page = 0
    body, kb = await _render_history(session, user, lang, page=page, favourites_only=False)
    await message.answer(body, parse_mode="HTML", reply_markup=kb or main_inline(lang))


@router.message(Command("favourites"))
async def on_favourites(
    message: Message, session: AsyncSession, user: User, lang: str
) -> None:
    body, kb = await _render_history(session, user, lang, page=0, favourites_only=True)
    # `_render_history` возвращает HISTORY_EMPTY и для «нет диалогов», и для
    # «нет favourites» — различаем, чтобы не вводить пользователя в заблуждение
    # сообщением «Пока нет заданных вопросов», когда он просто никогда не
    # ставил ⭐.
    if kb is None:
        body = texts.tr(lang, texts.FAVOURITES_EMPTY)
    await message.answer(body, parse_mode="HTML", reply_markup=kb or main_inline(lang))


@router.message(Command("find"))
async def on_find(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
    lang: str,
) -> None:
    query = (command.args or "").strip()
    if not query:
        await message.answer(
            "Использование: <code>/find &lt;подстрока&gt;</code> — ищет по своим вопросам.",
            parse_mode="HTML",
        )
        return
    body, kb = await _render_history(
        session, user, lang, page=0, favourites_only=False, search=query
    )
    if kb is None:
        body = texts.tr(lang, texts.FIND_EMPTY).format(q=html.escape(query))
    await message.answer(body, parse_mode="HTML", reply_markup=kb or main_inline(lang))


@router.message(Command("export"))
async def on_export(
    message: Message, session: AsyncSession, user: User, lang: str
) -> None:
    """/export — выгрузить всю историю Q&A в .txt-файл.

    Без пагинации, без LLM. Полезно перед экзаменом — оффлайн-конспект
    всего, что когда-либо спрашивал.
    """
    dialogs, _ = await user_dialogs_page(session, user_id=user.id, offset=0, limit=10_000)
    if not dialogs:
        await message.answer(texts.tr(lang, texts.HISTORY_EMPTY))
        return
    lines: list[str] = []
    for d in dialogs:
        when = d.created_at.strftime("%Y-%m-%d %H:%M")
        star = "⭐ " if d.is_favourite else ""
        subj = d.subject.title_ru if d.subject else "—"
        lines.append(f"{star}#{d.id}  {when}  [{subj}]")
        lines.append("Q: " + (d.question or "").strip())
        lines.append("A: " + (d.answer or "").strip())
        lines.append("")
    payload = "\n".join(lines).encode("utf-8")
    # Не кладём telegram_id в имя файла — если юзер форварднёт его в группу,
    # его ID не должен утечь. ISO-дата достаточна для различения экспортов.
    from datetime import datetime as _dt

    today = _dt.utcnow().strftime("%Y%m%d")
    await message.answer_document(
        BufferedInputFile(payload, filename=f"qa_history_{today}.txt"),
        caption=f"📦 {len(dialogs)} вопросов экспортировано.",
    )


@router.callback_query(F.data.startswith("hst:"))
async def on_history_page(
    callback: CallbackQuery, session: AsyncSession, user: User, lang: str
) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) < 3:
        await callback.answer()
        return
    try:
        page = int(parts[1])
        fav_only = bool(int(parts[2]))
    except ValueError:
        await callback.answer()
        return
    body, kb = await _render_history(session, user, lang, page=page, favourites_only=fav_only)
    await callback.answer()
    msg = callback.message
    if msg is None:
        return
    with contextlib.suppress(TelegramBadRequest):
        await msg.edit_text(body, parse_mode="HTML", reply_markup=kb)


@router.callback_query(F.data.startswith("star:"))
async def on_star_toggle(
    callback: CallbackQuery, session: AsyncSession, user: User, lang: str
) -> None:
    """Toggle ⭐ на собственном диалоге; перерендериваем текущую страницу."""
    parts = (callback.data or "").split(":")
    if len(parts) < 5:
        await callback.answer()
        return
    try:
        dialog_id = int(parts[1])
        to_fav = bool(int(parts[2]))
        page = int(parts[3])
        fav_only = bool(int(parts[4]))
    except ValueError:
        await callback.answer()
        return
    ok = await set_favourite(session, dialog_id=dialog_id, user_id=user.id, is_favourite=to_fav)
    await session.commit()
    await callback.answer(
        "⭐ в избранном" if to_fav and ok else ("☆ убрано" if ok else "Не твой диалог.")
    )
    body, kb = await _render_history(session, user, lang, page=page, favourites_only=fav_only)
    msg = callback.message
    if msg is None:
        return
    with contextlib.suppress(TelegramBadRequest):
        await msg.edit_text(body, parse_mode="HTML", reply_markup=kb)
