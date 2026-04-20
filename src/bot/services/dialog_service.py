"""Persist RAG dialogs to the DB."""

from __future__ import annotations

from sqlalchemy import desc, select
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
    """Insert feedback for ``dialog_id`` if it belongs to ``user_id``.

    Returns ``True`` on a fresh insert, ``False`` if the dialog isn't owned by
    the caller (IDOR protection — callback_data is client-controlled, so we
    must verify ownership here) or if a feedback row already exists.
    """
    dialog = (
        await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user_id)
        )
    ).scalar_one_or_none()
    if dialog is None:
        return False

    existing = (
        await session.execute(select(Feedback).where(Feedback.dialog_id == dialog_id))
    ).scalar_one_or_none()
    if existing is not None:
        return False

    session.add(Feedback(dialog_id=dialog_id, rating=rating))
    await session.flush()
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
    """Paginated history.

    Returns ``(rows, total)``. ``search`` does a case-insensitive substring
    match against the question text, done in Python so Cyrillic queries
    work on both SQLite (dev) and Postgres (prod) — SQLite's ``LIKE NOCASE``
    is ASCII-only. Personal history is small (hundreds of rows, not
    millions), so the cost is negligible.
    """
    base = select(Dialog).options(joinedload(Dialog.subject)).where(Dialog.user_id == user_id)
    if favourites_only:
        base = base.where(Dialog.is_favourite.is_(True))

    ordered = base.order_by(desc(Dialog.created_at))

    if search:
        needle = search.strip().lower()
        all_rows = list(
            (await session.execute(ordered)).scalars().unique()
        )
        matching = [d for d in all_rows if needle in (d.question or "").lower()]
        total_count = len(matching)
        page = matching[offset : offset + limit]
        return page, total_count

    total_count = len(
        (
            await session.execute(
                select(Dialog.id).where(
                    *(
                        [Dialog.user_id == user_id]
                        + ([Dialog.is_favourite.is_(True)] if favourites_only else [])
                    )
                )
            )
        ).all()
    )
    rows = list(
        (
            await session.execute(ordered.offset(offset).limit(limit))
        )
        .scalars()
        .unique()
    )
    return rows, total_count


async def set_favourite(
    session: AsyncSession,
    *,
    dialog_id: int,
    user_id: int,
    is_favourite: bool,
) -> bool:
    """Toggle the ⭐ flag on a dialog the caller owns. Returns whether the
    row was updated (False if the dialog doesn't belong to ``user_id``)."""
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
