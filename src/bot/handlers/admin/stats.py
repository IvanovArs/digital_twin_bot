"""Статистика для админа и преподавателя: /admin_stats, /teacher_stats, /teacher_gaps."""

from __future__ import annotations

from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.admin._common import deny, is_admin
from src.bot.services.admin_service import stats_24h
from src.bot.services.teacher_stats import coverage_gaps, subject_stats
from src.db.models import User

router = Router(name="admin_stats")


@router.message(Command("admin_stats"))
async def on_admin_stats(message: Message, session: AsyncSession, user: User) -> None:
    if not is_admin(user):
        await deny(message)
        return
    s = await stats_24h(session)
    avg = f"{s['avg_rating']:.2f}" if s["avg_rating"] is not None else "—"
    p95 = f"{s['p95_latency_ms']} ms" if s["p95_latency_ms"] is not None else "—"
    await message.answer(
        f"<b>Статистика</b>\n"
        f"• Диалогов всего: {s['dialogs_total']}\n"
        f"• За 24 ч: {s['dialogs_24h']}\n"
        f"• Средняя оценка: {avg}\n"
        f"• p95 latency: {p95}",
        parse_mode="HTML",
    )


@router.message(Command("teacher_stats"))
async def on_teacher_stats(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_stats <slug> [days=7] — статистика по предмету."""
    if not is_admin(user):
        await deny(message)
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/teacher_stats &lt;slug&gt; [days]</code>", parse_mode="HTML"
        )
        return
    slug = args[0]
    try:
        days = int(args[1]) if len(args) > 1 else 7
    except ValueError:
        await message.answer("days должен быть числом.")
        return
    days = max(1, min(days, 365))
    s = await subject_stats(session, subject_slug=slug, days=days)
    if s is None:
        await message.answer(f"Неизвестный slug: <code>{slug}</code>", parse_mode="HTML")
        return

    avg = f"{s.avg_rating:.2f}" if s.avg_rating is not None else "—"
    p50 = f"{s.p50_latency_ms} ms" if s.p50_latency_ms is not None else "—"
    p95 = f"{s.p95_latency_ms} ms" if s.p95_latency_ms is not None else "—"
    lines = [
        f"<b>📊 {s.subject_title}</b> (окно {s.window_days} дн.)",
        f"• Диалогов в окне: <b>{s.dialogs_window}</b> (всего: {s.dialogs_total})",
        f"• 👍 {s.up_count}  ·  👎 {s.down_count}  ·  средняя: {avg}",
        f"• latency p50: {p50}  ·  p95: {p95}",
        f"• web-fallback: {s.web_fallback_count}  ({s.web_fallback_ratio * 100:.0f}% от запросов)",
    ]
    if s.top_questions:
        lines.append("")
        lines.append("<b>Топ повторов:</b>")
        for q, n in s.top_questions:
            preview = q.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:79] + "…"
            lines.append(f"• {escape(preview)} — <b>{n}×</b>")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("teacher_gaps"))
async def on_teacher_gaps(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_gaps [days=30] [limit=30] — вопросы, ушедшие в web-fallback.
    То есть пробелы покрытия учебника. TODO-лист для авторов курса."""
    if not is_admin(user):
        await deny(message)
        return
    args = (command.args or "").split()
    try:
        days = int(args[0]) if len(args) > 0 else 30
        limit = int(args[1]) if len(args) > 1 else 30
    except ValueError:
        await message.answer("days и limit должны быть числами.")
        return
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 100))
    gaps = await coverage_gaps(session, days=days, limit=limit)
    if not gaps:
        await message.answer(
            f"За последние {days} дн. веб-fallback не сработал ни разу — "
            "учебник покрывает всё, что спрашивают. 🎉"
        )
        return
    lines = [
        f"<b>🕳️ Пробелы в материалах</b> (последние {days} дн., топ-{len(gaps)})",
        "Это вопросы, где RAG не нашёл ответ в учебнике и ушёл в интернет:",
        "",
    ]
    for g in gaps:
        preview = g.question.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:89] + "…"
        lines.append(f"• {escape(preview)} — <b>{g.count}×</b>")
    await message.answer("\n".join(lines), parse_mode="HTML")
