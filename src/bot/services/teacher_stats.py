"""Per-subject stats and coverage-gap reports for teachers.

Two headline functions:

* ``subject_stats`` — volume, feedback distribution, latency percentiles,
  web-fallback share, top repeat-questions for a single subject over a
  rolling window. Feeds ``/teacher_stats``.

* ``coverage_gaps`` — questions that fell through to the web-fallback path
  (``Dialog.subject_id IS NULL``) grouped by normalised form, most-frequent
  first. Feeds ``/teacher_gaps`` — a literal TODO list for the course
  author: these are the topics students ask about but the textbook doesn't
  cover.

Both run in Python for percentiles/grouping so they work on SQLite (dev)
and Postgres (prod) without vendor-specific SQL.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Dialog, Feedback, Subject

# Follow-up entries are already saved with a "[simplify] original question"
# prefix; strip it so they don't inflate the top-questions list and so the
# coverage-gaps view shows the real student phrasing.
_FOLLOWUP_TAG = re.compile(r"^\[(simplify|example|deepen)\]\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")
# Leading greeting/question-word noise that makes otherwise-identical
# questions look distinct. Strip only when it's at the very start so we
# don't mangle the middle of a phrase.
_LEAD_JUNK = re.compile(
    r"^(что такое|что есть|расскажи про|расскажи о|объясни|что значит|определение)\s+",
    re.IGNORECASE,
)


def _normalise_question(q: str) -> str:
    """Collapse minor variations so «Что такое система», «что есть система?»
    and «Система — это?» group together in top-questions and gaps reports."""
    s = _FOLLOWUP_TAG.sub("", q or "").strip().lower()
    s = _LEAD_JUNK.sub("", s).strip()
    s = s.rstrip("?.,!;: —-")
    s = _WS.sub(" ", s)
    return s


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(round(p * (len(s) - 1))))
    return s[idx]


@dataclass
class SubjectStats:
    subject_slug: str
    subject_title: str
    window_days: int
    dialogs_total: int
    dialogs_window: int
    up_count: int
    down_count: int
    avg_rating: float | None
    p50_latency_ms: int | None
    p95_latency_ms: int | None
    web_fallback_count: int
    web_fallback_ratio: float
    top_questions: list[tuple[str, int]] = field(default_factory=list)


async def subject_stats(
    session: AsyncSession,
    *,
    subject_slug: str,
    days: int = 7,
    top_n: int = 10,
) -> SubjectStats | None:
    """Return rolling stats for one subject; ``None`` if slug is unknown."""
    subj = (
        await session.execute(select(Subject).where(Subject.slug == subject_slug))
    ).scalar_one_or_none()
    if subj is None:
        return None

    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Total (lifetime) diagogs for this subject.
    total = (
        await session.execute(
            select(func.count(Dialog.id)).where(Dialog.subject_id == subj.id)
        )
    ).scalar_one() or 0

    # Windowed dialogs + latencies + questions.
    windowed = list(
        (
            await session.execute(
                select(Dialog).where(
                    Dialog.subject_id == subj.id,
                    Dialog.created_at >= cutoff,
                )
            )
        ).scalars()
    )
    latencies = [int(d.latency_ms) for d in windowed if d.latency_ms is not None]

    # Feedback on this subject's windowed dialogs.
    ratings = list(
        (
            await session.execute(
                select(Feedback.rating)
                .join(Dialog, Feedback.dialog_id == Dialog.id)
                .where(
                    Dialog.subject_id == subj.id,
                    Feedback.created_at >= cutoff,
                )
            )
        ).scalars()
    )
    up = sum(1 for r in ratings if r >= 4)
    down = sum(1 for r in ratings if r <= 2)
    avg = (sum(ratings) / len(ratings)) if ratings else None

    # Web-fallback ratio: dialogs where the user asked a question assigned
    # to this subject by the router but retrieval failed. We can't read a
    # historic "intended subject" off a NULL subject_id, so we approximate
    # via the user's ``current_subject_slug`` — good enough for a rolling
    # coverage trend. See ``coverage_gaps`` for the global view.
    web_rows = list(
        (
            await session.execute(
                select(Dialog)
                .join(Dialog.user)
                .where(
                    Dialog.subject_id.is_(None),
                    Dialog.created_at >= cutoff,
                )
            )
        ).scalars()
    )
    web_fallback_count = sum(
        1
        for d in web_rows
        if getattr(d.user, "current_subject_slug", None) == subj.slug
    )
    denom = len(windowed) + web_fallback_count
    web_ratio = (web_fallback_count / denom) if denom else 0.0

    # Top repeated questions (normalised).
    counter: Counter[str] = Counter()
    originals: dict[str, str] = {}
    for d in windowed:
        key = _normalise_question(d.question)
        if not key:
            continue
        counter[key] += 1
        originals.setdefault(key, d.question.strip())
    top = [
        (originals[k], v) for k, v in counter.most_common(top_n) if v > 1
    ]

    return SubjectStats(
        subject_slug=subj.slug,
        subject_title=subj.title_ru,
        window_days=days,
        dialogs_total=int(total),
        dialogs_window=len(windowed),
        up_count=up,
        down_count=down,
        avg_rating=avg,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        web_fallback_count=web_fallback_count,
        web_fallback_ratio=web_ratio,
        top_questions=top,
    )


@dataclass
class GapEntry:
    question: str
    count: int
    last_seen: datetime


async def coverage_gaps(
    session: AsyncSession,
    *,
    days: int = 30,
    limit: int = 50,
) -> list[GapEntry]:
    """Questions that hit the web-fallback path, grouped by normalised form.

    The point of this list is a TODO for the course author: «эти темы
    студенты спрашивают, но учебник не отвечает — либо дополни материал,
    либо исключи из программы».
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    rows = list(
        (
            await session.execute(
                select(Dialog.question, Dialog.created_at).where(
                    Dialog.subject_id.is_(None),
                    Dialog.created_at >= cutoff,
                )
            )
        ).all()
    )
    counter: Counter[str] = Counter()
    representative: dict[str, str] = {}
    last_seen: dict[str, datetime] = {}
    for question, created_at in rows:
        key = _normalise_question(question or "")
        if not key:
            continue
        counter[key] += 1
        representative.setdefault(key, (question or "").strip())
        if key not in last_seen or created_at > last_seen[key]:
            last_seen[key] = created_at

    ranked = counter.most_common(limit)
    return [
        GapEntry(
            question=representative[k],
            count=v,
            last_seen=last_seen[k],
        )
        for k, v in ranked
    ]
