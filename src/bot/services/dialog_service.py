"""Запись RAG-диалогов в БД."""

from __future__ import annotations

from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.db.models import Dialog, Feedback, Subject, User
from src.rag.pipeline import AskResult


async def save_dialog(
    session: AsyncSession,
    *,
    user: User,
    question: str,
    result: AskResult,
    latency_ms: int | None = None,
) -> Dialog:
    subject_id: int | None = None
    if result.subject is not None:
        row = (
            await session.execute(select(Subject).where(Subject.slug == result.subject.slug))
        ).scalar_one_or_none()
        subject_id = row.id if row else None

    sources = [
        {"book": h.book, "page": h.page, "subject": h.subject_slug, "score": round(h.score, 4)}
        for h in result.hits
    ]

    dialog = Dialog(
        user_id=user.id,
        subject_id=subject_id,
        question=question,
        answer=result.answer,
        sources=sources,
        latency_ms=latency_ms,
    )
    session.add(dialog)
    await session.flush()
    return dialog


async def record_feedback(
    session: AsyncSession,
    *,
    dialog_id: int,
    user_id: int,
    rating: int,
) -> bool:
    """Записать feedback для ``dialog_id``, если он принадлежит ``user_id``.

    Возвращает ``True`` при свежем insert'е, ``False`` если диалог не
    принадлежит caller'у (IDOR-защита — callback_data контролируется клиентом,
    проверяем владение здесь) или если feedback уже существует.

    Раньше double-tap по 👍/👎 гонял SELECT-then-INSERT и падал на unique-
    констрейнте → INTERNAL_ERROR юзеру. Используем INSERT … ON CONFLICT
    DO NOTHING на Postgres; на SQLite — try/except вокруг IntegrityError
    как безопасный fallback (нет dialect-helper'а).
    """
    dialog = (
        await session.execute(
            select(Dialog.id).where(Dialog.id == dialog_id, Dialog.user_id == user_id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        return False

    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        stmt = (
            pg_insert(Feedback)
            .values(dialog_id=dialog_id, rating=rating)
            .on_conflict_do_nothing(index_elements=["dialog_id"])
            .returning(Feedback.id)
        )
        inserted = (await session.execute(stmt)).scalar_one_or_none()
        await session.flush()
        return inserted is not None

    # Savepoint, чтобы IntegrityError на гонке не откатил всю transaction'у
    # вместе с предыдущей feedback-записью (только под SQLite — на Postgres
    # path выше возвращает rowcount без savepoint).
    try:
        async with session.begin_nested():
            session.add(Feedback(dialog_id=dialog_id, rating=rating))
    except IntegrityError:
        return False
    return True


async def recent_dialogs(session: AsyncSession, *, user_id: int, limit: int = 5) -> list[Dialog]:
    stmt = (
        select(Dialog)
        .options(joinedload(Dialog.subject))
        .where(Dialog.user_id == user_id)
        .order_by(desc(Dialog.created_at))
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


async def user_dialogs_page(
    session: AsyncSession,
    *,
    user_id: int,
    offset: int = 0,
    limit: int = 5,
    favourites_only: bool = False,
    search: str | None = None,
) -> tuple[list[Dialog], int]:
    """Постраничная история.

    Возвращает ``(rows, total)``. На Postgres substring-фильтр уезжает в SQL
    через ``ILIKE`` (UTF-8 case-insensitive, включая кириллицу) — power-юзер
    с тысячами строк не перекачивает всю историю на каждый keystroke. На
    SQLite (dev) — fallback на Python: ``LOWER`` тут ASCII-only и молча
    промахнулся бы по «Стейкхолдер». Личные dev-истории маленькие — цена
    незначительна.
    """
    base = select(Dialog).options(joinedload(Dialog.subject)).where(Dialog.user_id == user_id)
    count_base = select(func.count(Dialog.id)).where(Dialog.user_id == user_id)
    if favourites_only:
        base = base.where(Dialog.is_favourite.is_(True))
        count_base = count_base.where(Dialog.is_favourite.is_(True))

    needle = (search or "").strip()
    dialect = session.bind.dialect.name if session.bind is not None else ""

    if needle and dialect == "postgresql":
        pat = f"%{needle}%"
        base = base.where(Dialog.question.ilike(pat))
        count_base = count_base.where(Dialog.question.ilike(pat))

    ordered = base.order_by(desc(Dialog.created_at))

    if needle and dialect != "postgresql":
        all_rows = list((await session.execute(ordered)).scalars().unique())
        low = needle.lower()
        matching = [d for d in all_rows if low in (d.question or "").lower()]
        return matching[offset : offset + limit], len(matching)

    total_count = (await session.execute(count_base)).scalar_one()
    rows = list(
        (
            await session.execute(ordered.offset(offset).limit(limit))
        )
        .scalars()
        .unique()
    )
    return rows, int(total_count or 0)


async def set_favourite(
    session: AsyncSession,
    *,
    dialog_id: int,
    user_id: int,
    is_favourite: bool,
) -> bool:
    """Toggle ⭐-флага на диалоге, которым владеет caller. Возвращает,
    был ли обновлён ряд (False — если диалог не принадлежит ``user_id``)."""
    dialog = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user_id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        return False
    dialog.is_favourite = is_favourite
    await session.flush()
    return True
