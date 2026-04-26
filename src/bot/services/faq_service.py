"""Teacher-curated FAQ: the short-circuit layer between a question and RAG.

Flow:
1. Student gets an answer they rate 👎.
2. Teacher opens ``/teacher_review`` — sees pending low-rated dialogs.
3. Teacher picks one via ``/teacher_fix <dialog_id>``, types a corrected
   answer; we save it as an ``FAQEntry`` keyed on the normalised question.
4. Next student to ask the same normalised question (within the same
   subject or globally) gets the teacher's answer verbatim — no LLM, no
   retrieval, so the teacher's phrasing is authoritative.

We re-use ``teacher_stats._normalise_question`` so the grouping logic is
consistent with what teachers see in the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.services.safe_html import safe_html
from src.bot.services.teacher_stats import _normalise_question
from src.db.models import Dialog, FAQEntry, Feedback, User

_FAQ_PREFIX_RU = "📌 <b>Ответ преподавателя</b>\n"
_FAQ_PREFIX_EN = "📌 <b>Teacher's answer</b>\n"


@dataclass
class PendingReview:
    dialog_id: int
    question: str
    answer: str
    subject_slug: str | None
    rating: int
    feedback_at: datetime


async def pending_reviews(
    session: AsyncSession,
    *,
    limit: int = 20,
) -> list[PendingReview]:
    """Return 👎-rated dialogs that don't yet have a FAQ answer under the
    same normalised question. Oldest-first so the review queue drains in
    order.
    """
    rows = (
        await session.execute(
            select(Dialog, Feedback)
            .join(Feedback, Feedback.dialog_id == Dialog.id)
            .where(Feedback.rating <= 2)
            .order_by(Feedback.created_at.asc())
        )
    ).all()

    # Build set of (subject_id, normalised_question) already in FAQ so we
    # can skip dialogs the teacher has already addressed. Cheap in-memory
    # filter because FAQ rows are small.
    faq_rows = list(
        (await session.execute(select(FAQEntry.subject_id, FAQEntry.question_normalised))).all()
    )
    faq_keys = {(sid, qn) for sid, qn in faq_rows}

    out: list[PendingReview] = []
    for dialog, feedback in rows:
        key = (dialog.subject_id, _normalise_question(dialog.question))
        if key in faq_keys:
            continue
        subj_slug = dialog.subject.slug if dialog.subject else None
        out.append(
            PendingReview(
                dialog_id=dialog.id,
                question=dialog.question,
                answer=dialog.answer,
                subject_slug=subj_slug,
                rating=feedback.rating,
                feedback_at=feedback.created_at,
            )
        )
        if len(out) >= limit:
            break
    return out


async def save_faq_from_dialog(
    session: AsyncSession,
    *,
    dialog_id: int,
    answer: str,
    teacher: User,
) -> FAQEntry | None:
    """Turn a corrected answer into a FAQ entry keyed on the original
    dialog's normalised question. Returns ``None`` if the dialog is gone.
    """
    dialog = (
        await session.execute(select(Dialog).where(Dialog.id == dialog_id))
    ).scalar_one_or_none()
    if dialog is None:
        return None

    q_norm = _normalise_question(dialog.question)
    if not q_norm:
        return None

    # Upsert by (subject_id, q_norm): if the teacher re-answers the same
    # question we overwrite, not duplicate.
    existing = (
        await session.execute(
            select(FAQEntry).where(
                FAQEntry.subject_id == dialog.subject_id,
                FAQEntry.question_normalised == q_norm,
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        existing.answer = answer
        existing.parent_dialog_id = dialog.id
        existing.created_by_user_id = teacher.id
        existing.created_at = datetime.now(existing.created_at.tzinfo)
        entry = existing
    else:
        entry = FAQEntry(
            subject_id=dialog.subject_id,
            question_normalised=q_norm,
            question_original=dialog.question.strip(),
            answer=answer,
            parent_dialog_id=dialog.id,
            created_by_user_id=teacher.id,
        )
        session.add(entry)
    await session.flush()
    return entry


async def lookup_faq(
    session: AsyncSession,
    *,
    question: str,
    subject_id: int | None,
) -> FAQEntry | None:
    """Return the best-matching FAQ for this question, or ``None``.

    Match rule: exact equality on normalised text. Subject-scoped FAQ beats
    global one (teacher's subject-specific correction is more precise than
    a generic answer).
    """
    q_norm = _normalise_question(question)
    if not q_norm:
        return None
    # Try subject-scoped first, fall back to global (subject_id IS NULL).
    stmt = select(FAQEntry).where(FAQEntry.question_normalised == q_norm)
    if subject_id is not None:
        stmt = stmt.where(or_(FAQEntry.subject_id == subject_id, FAQEntry.subject_id.is_(None)))
    else:
        stmt = stmt.where(FAQEntry.subject_id.is_(None))
    rows = list((await session.execute(stmt)).scalars())
    if not rows:
        return None
    # Subject-scoped rows win.
    rows.sort(key=lambda r: (0 if r.subject_id == subject_id else 1, -r.id))
    return rows[0]


def format_faq_body(entry: FAQEntry, lang: str, question: str | None = None) -> str:
    """Render a teacher-curated answer for Telegram ``parse_mode="HTML"``.

    The teacher's prose may include light HTML (<b>, <code>, <blockquote>);
    everything else is escaped. Unbalanced tags are dropped so a typo like
    «</b>» at the end of a submission doesn't break the render for every
    student who asks the same question afterward.

    When ``question`` is provided, it's quoted in a ``<blockquote>`` at
    the top so the short answer keeps the same «question + answer»
    framing as full RAG responses.
    """
    import html as _html_mod

    prefix = _FAQ_PREFIX_EN if lang == "en" else _FAQ_PREFIX_RU
    q_block = (
        f"<blockquote>{_html_mod.escape(question.strip())}</blockquote>\n\n" if question else ""
    )
    return q_block + prefix + safe_html(entry.answer)
