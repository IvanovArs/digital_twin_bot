"""Admin/teacher commands: manage subjects and materials, view stats."""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import Document, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.services.admin_service import (
    reindex_subject_in_background,
    save_material,
    stats_24h,
)
from src.bot.services.faq_service import pending_reviews, save_faq_from_dialog
from src.bot.services.glossary_upload import parse_glossary_payload, replace_glossary
from src.bot.services.teacher_stats import coverage_gaps, subject_stats
from src.bot.services.upload_guard import check_magic, check_size
from src.bot.states import AdminFlow
from src.config import settings
from src.db.models import Subject, SubjectMaterial, User, UserRole
from src.db.session import get_sessionmaker

log = structlog.get_logger(__name__)
router = Router(name="admin")


def _is_admin(user: User) -> bool:
    return user.role in (UserRole.admin, UserRole.teacher)


async def _deny(message: Message) -> None:
    """Visible «access denied» reply. Silent ``return`` on a /command makes
    the bot look broken — students kept trying the same admin command
    three times in a row. A one-line explicit reply is kinder."""
    await message.answer("⛔ Команда доступна только преподавателям и админам.")


def _is_superadmin(user: User) -> bool:
    """Superadmins are those listed in ADMIN_TELEGRAM_IDS. Only they may
    change another user's role — DB-level admins can upload and reindex but
    cannot grant roles, so that a compromised DB-admin can't self-promote
    or escalate a confederate past the env-var gate set by the server owner.
    """
    return user.telegram_id in settings.admin_ids


def _parse_role(arg: str | None, default: UserRole = UserRole.teacher) -> UserRole | None:
    if not arg:
        return default
    name = arg.strip().lower()
    for role in UserRole:
        if role.value == name or role.name == name:
            return role
    return None


@router.message(Command("admin_subjects"))
async def on_admin_subjects(message: Message, session: AsyncSession, user: User) -> None:
    if not _is_admin(user):
        await _deny(message)
        return

    subjects = list((await session.execute(select(Subject).order_by(Subject.title_ru))).scalars())
    if not subjects:
        await message.answer("Нет предметов в courses.yaml.")
        return

    lines = ["<b>Предметы:</b>"]
    for s in subjects:
        mats = list(
            (
                await session.execute(
                    select(SubjectMaterial).where(SubjectMaterial.subject_id == s.id)
                )
            ).scalars()
        )
        indexed = sum(1 for m in mats if m.indexed_at is not None)
        active = "🟢" if s.is_active else "⚪"
        lines.append(
            f"{active} <code>{s.slug}</code> — {s.title_ru} "
            f"(материалов: {len(mats)}, проиндексировано: {indexed})"
        )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("admin_upload"))
async def on_admin_upload(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    if not _is_admin(user):
        await _deny(message)
        return

    slug = (command.args or "").strip()
    if not slug:
        await message.answer("Использование: /admin_upload &lt;slug&gt;", parse_mode="HTML")
        return

    subj = (await session.execute(select(Subject).where(Subject.slug == slug))).scalar_one_or_none()
    if subj is None:
        await message.answer(f"Неизвестный предмет: <code>{slug}</code>", parse_mode="HTML")
        return

    await state.set_state(AdminFlow.uploading_material)
    await state.update_data(subject_slug=slug)
    await message.answer(
        f"Пришли файл для предмета <b>{subj.title_ru}</b>.\n"
        f"Поддерживаются: <code>.pdf .docx .md .txt</code>. Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.uploading_material)
async def on_admin_upload_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Загрузка отменена.")


@router.message(AdminFlow.uploading_material, F.document)
async def on_admin_upload_doc(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    # try/finally: any exception on the download-or-save path left the
    # teacher stuck in ``AdminFlow.uploading_material``, with every next
    # message (including their /cancel) treated as a new document.
    try:
        data = await state.get_data()
        slug = str(data.get("subject_slug") or "")
        subj = (
            await session.execute(select(Subject).where(Subject.slug == slug))
        ).scalar_one_or_none()
        if subj is None:
            await message.answer(
                "Предмет пропал. Начни заново: /admin_upload &lt;slug&gt;",
                parse_mode="HTML",
            )
            return

        doc: Document = message.document  # type: ignore[assignment]
        # ``doc.file_name`` is client-controlled. Strip any directory components
        # so an admin client can't write outside BOOKS_DIR via "../../etc/foo.pdf"
        # or an absolute POSIX path like "/etc/passwd". PurePosixPath is safe
        # to use as a parser regardless of host OS.
        from pathlib import PurePosixPath

        raw_name = doc.file_name or "unnamed"
        safe_name = PurePosixPath(raw_name).name
        if (
            not safe_name
            or safe_name.startswith(".")
            or "/" in safe_name
            or "\\" in safe_name
        ):
            await message.answer("Недопустимое имя файла.")
            return
        fname_lower = safe_name.lower()
        if not fname_lower.endswith((".pdf", ".docx", ".md", ".txt")):
            await message.answer("Поддерживаются: .pdf .docx .md .txt")
            return
        size_err = check_size(doc.file_size)
        if size_err is not None:
            await message.answer(size_err)
            return

        if message.bot is None:
            await message.answer("Не удалось связаться с Telegram API.")
            return
        file = await message.bot.download(doc)
        if file is None:
            await message.answer("Не удалось скачать файл.")
            return
        payload = file.read()
        magic_err = check_magic(safe_name, payload)
        if magic_err is not None:
            await message.answer(magic_err)
            return

        try:
            await save_material(
                session,
                subject=subj,
                filename=safe_name,
                payload=payload,
                uploader=user,
                target_dir=settings.BOOKS_DIR,
            )
        except Exception:
            log.exception("save_material_failed", subject=subj.slug, filename=safe_name)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить файл — попробуй позже.")
            return
        await message.answer(
            f"✅ Файл сохранён. Запусти /admin_reindex <code>{subj.slug}</code>, "
            "чтобы включить его в индекс.",
            parse_mode="HTML",
        )
    finally:
        await state.clear()


@router.message(Command("admin_reindex"))
async def on_admin_reindex(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    if not _is_admin(user):
        await _deny(message)
        return

    slug_arg = (command.args or "").strip() or None
    if slug_arg is not None:
        exists = (
            await session.execute(select(Subject).where(Subject.slug == slug_arg))
        ).scalar_one_or_none()
        if exists is None:
            await message.answer(f"Неизвестный предмет: <code>{slug_arg}</code>", parse_mode="HTML")
            return

    scope = slug_arg or "все предметы"
    await message.answer(f"🔄 Запускаю переиндексацию ({scope})… Это может занять пару минут.")

    import asyncio

    asyncio.create_task(  # noqa: RUF006
        _reindex_and_report(message, slug_arg)
    )


async def _reindex_and_report(message: Message, slug: str | None) -> None:
    try:
        await reindex_subject_in_background(get_sessionmaker(), subject_slug=slug)
        await message.answer("✅ Переиндексация завершена.")
    except Exception:
        # Don't leak internal exception text to the chat — paths and stack
        # info help attackers more than admins. Full trace is in structlog.
        log.exception("reindex_task_failed")
        try:
            await message.answer("⚠️ Ошибка переиндексации. Подробности — в логах.")
        except Exception:
            log.exception("reindex_task_report_failed")


@router.message(Command("admin_stats"))
async def on_admin_stats(message: Message, session: AsyncSession, user: User) -> None:
    if not _is_admin(user):
        await _deny(message)
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


# ---------- teacher dashboard ----------


@router.message(Command("teacher_stats"))
async def on_teacher_stats(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_stats <slug> [days=7] — per-subject rolling stats."""
    if not _is_admin(user):
        await _deny(message)
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/teacher_stats &lt;slug&gt; [days]</code>",
            parse_mode="HTML",
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
        f"• web-fallback: {s.web_fallback_count}  "
        f"({s.web_fallback_ratio * 100:.0f}% от запросов)",
    ]
    if s.top_questions:
        lines.append("")
        lines.append("<b>Топ повторов:</b>")
        for q, n in s.top_questions:
            preview = q.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:79] + "…"
            from html import escape

            lines.append(f"• {escape(preview)} — <b>{n}×</b>")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("teacher_gaps"))
async def on_teacher_gaps(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_gaps [days=30] [limit=30] — questions that hit web-fallback
    (→ missing from textbooks). A TODO list for course authors."""
    if not _is_admin(user):
        await _deny(message)
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
    from html import escape

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


# ---------- teacher glossary upload ----------


@router.message(Command("teacher_glossary_upload"))
async def on_teacher_glossary_upload(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_glossary_upload <slug> — accept a CSV/YAML with term,definition
    pairs and replace the subject's glossary with the uploaded set."""
    if not _is_admin(user):
        await _deny(message)
        return
    slug = (command.args or "").strip()
    if not slug:
        await message.answer(
            "Использование: <code>/teacher_glossary_upload &lt;slug&gt;</code>",
            parse_mode="HTML",
        )
        return
    subj = (
        await session.execute(select(Subject).where(Subject.slug == slug))
    ).scalar_one_or_none()
    if subj is None:
        await message.answer(f"Неизвестный slug: <code>{slug}</code>", parse_mode="HTML")
        return
    await state.set_state(AdminFlow.uploading_glossary)
    await state.update_data(subject_slug=slug)
    await message.answer(
        f"Пришли CSV или YAML с глоссарием для <b>{subj.title_ru}</b>. "
        "CSV: <code>term,definition</code>. "
        "YAML: <code>terms: [{term: …, definition: …}]</code>. "
        "Файл <b>заменит</b> существующий глоссарий предмета. Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.uploading_glossary)
async def on_glossary_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.")


@router.message(AdminFlow.uploading_glossary, F.document)
async def on_glossary_doc(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    try:
        data = await state.get_data()
        slug = str(data.get("subject_slug") or "")
        subj = (
            await session.execute(select(Subject).where(Subject.slug == slug))
        ).scalar_one_or_none()
        if subj is None:
            await message.answer(
                "Предмет пропал. Начни заново: /teacher_glossary_upload"
            )
            return

        doc = message.document
        if doc is None:
            return
        fname = (doc.file_name or "glossary").lower()
        if not fname.endswith((".csv", ".yaml", ".yml")):
            await message.answer("Поддерживаются только .csv / .yaml / .yml.")
            return
        size_err = check_size(doc.file_size)
        if size_err is not None:
            await message.answer(size_err)
            return
        if message.bot is None:
            await message.answer("Не удалось связаться с Telegram API.")
            return
        file = await message.bot.download(doc)
        if file is None:
            await message.answer("Не удалось скачать файл.")
            return
        payload = file.read()
        magic_err = check_magic(fname, payload)
        if magic_err is not None:
            await message.answer(magic_err)
            return
        try:
            entries = parse_glossary_payload(payload, fname)
        except ValueError as exc:
            await message.answer(f"Ошибка разбора: {exc}")
            return
        if not entries:
            await message.answer("В файле не нашлось ни одной пары term/definition.")
            return

        try:
            result = await replace_glossary(session, subject=subj, entries=entries)
            await session.commit()
        except Exception:
            log.exception("glossary_save_failed", subject=subj.slug)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить глоссарий — попробуй позже.")
            return
        log.info(
            "glossary_uploaded",
            subject=subj.slug,
            inserted=result.inserted,
            replaced=result.replaced_previous,
            skipped=result.skipped_empty,
            by_user=user.telegram_id,
        )
        await message.answer(
            f"✅ Глоссарий обновлён для <b>{subj.title_ru}</b>:\n"
            f"• добавлено: <b>{result.inserted}</b>\n"
            f"• заменено предыдущих: {result.replaced_previous}\n"
            f"• пропущено пустых/дубликатов: {result.skipped_empty}",
            parse_mode="HTML",
        )
    finally:
        await state.clear()


# ---------- feedback moderation ----------


@router.message(Command("teacher_review"))
async def on_teacher_review(
    message: Message,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_review — show up to 20 pending 👎-rated dialogs that don't
    yet have a teacher-curated FAQ. Each entry shows the question, the
    bot's answer (trimmed), and the dialog_id needed for /teacher_fix."""
    if not _is_admin(user):
        await _deny(message)
        return
    pending = await pending_reviews(session, limit=20)
    if not pending:
        await message.answer(
            "Очередь модерации пуста — негативных оценок без ответа препода нет. 🎉"
        )
        return
    from html import escape

    lines = [f"<b>🧑‍🏫 Очередь модерации</b> ({len(pending)}):"]
    for pr in pending:
        q_preview = pr.question.replace("\n", " ")
        if len(q_preview) > 120:
            q_preview = q_preview[:119] + "…"
        a_preview = (pr.answer or "").replace("\n", " ")
        if len(a_preview) > 160:
            a_preview = a_preview[:159] + "…"
        subj = f" [<code>{escape(pr.subject_slug or 'web')}</code>]"
        # Show the student's rating as 👎 for 1-2 stars and the bot's
        # prior answer preview — teachers kept asking for context on WHY
        # the student was unhappy.
        rating_icon = "👎" if pr.rating <= 2 else f"{pr.rating}⭐"
        lines.append(
            f"\n• <code>#{pr.dialog_id}</code>{subj} {rating_icon}\n"
            f"<b>В:</b> {escape(q_preview)}\n"
            f"<i>Бот ответил:</i> {escape(a_preview)}"
        )
    lines.append(
        "\nПравь: <code>/teacher_fix &lt;id&gt;</code> — дальше бот спросит правильный ответ."
    )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("teacher_fix"))
async def on_teacher_fix(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    """/teacher_fix <dialog_id> — enter FSM; next message from teacher
    becomes the FAQ answer for that dialog's normalised question."""
    if not _is_admin(user):
        await _deny(message)
        return
    arg = (command.args or "").strip()
    if not arg:
        await message.answer(
            "Использование: <code>/teacher_fix &lt;dialog_id&gt;</code>",
            parse_mode="HTML",
        )
        return
    try:
        dialog_id = int(arg)
    except ValueError:
        await message.answer("dialog_id должен быть числом.")
        return
    from src.db.models import Dialog as _Dialog

    dialog = (
        await session.execute(select(_Dialog).where(_Dialog.id == dialog_id))
    ).scalar_one_or_none()
    if dialog is None:
        await message.answer(f"Диалог <code>#{dialog_id}</code> не найден.", parse_mode="HTML")
        return

    from html import escape

    await state.set_state(AdminFlow.fixing_answer)
    await state.update_data(dialog_id=dialog_id)
    q_preview = dialog.question.replace("\n", " ")
    if len(q_preview) > 200:
        q_preview = q_preview[:199] + "…"
    await message.answer(
        f"<b>Вопрос студента:</b>\n{escape(q_preview)}\n\n"
        "Напиши правильный ответ одним сообщением. "
        "Можно HTML: <code>&lt;b&gt;…&lt;/b&gt;</code>, маркер «• ». "
        "Отмена — /cancel",
        parse_mode="HTML",
    )


@router.message(Command("cancel"), AdminFlow.fixing_answer)
async def on_fix_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.")


@router.message(AdminFlow.fixing_answer, F.text)
async def on_fix_receive(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    # try/finally guarantees the FSM is cleared even if the DB commit or
    # the reply roundtrip raises — without it a transient failure left
    # teachers stuck in ``AdminFlow.fixing_answer`` forever, with every
    # next command interpreted as another FAQ body.
    try:
        data = await state.get_data()
        dialog_id = int(data.get("dialog_id") or 0)
        answer = (message.text or "").strip()
        if not answer:
            await message.answer("Пустой ответ — напиши текст или /cancel.")
            return
        if len(answer) > 3500:
            await message.answer("Слишком длинный ответ (лимит 3500 символов).")
            return
        try:
            entry = await save_faq_from_dialog(
                session, dialog_id=dialog_id, answer=answer, teacher=user
            )
            await session.commit()
        except Exception:
            log.exception("faq_save_failed", dialog_id=dialog_id)
            await session.rollback()
            await message.answer("⚠️ Не удалось сохранить — попробуй позже.")
            return
        if entry is None:
            await message.answer("Не удалось сохранить — диалог пропал или пуст.")
            return
        log.info(
            "faq_saved",
            faq_id=entry.id,
            parent_dialog=dialog_id,
            by_user=user.telegram_id,
        )
        await message.answer(
            f"✅ Ответ сохранён как FAQ (<code>#{entry.id}</code>). "
            "Следующие студенты с таким же вопросом получат его без LLM.",
            parse_mode="HTML",
        )
    finally:
        await state.clear()


# ---------- role management (superadmin only) ----------


@router.message(Command("whoami"))
async def on_whoami(message: Message, user: User) -> None:
    """Every user can see their telegram_id + current role. Needed so a
    prospective teacher can send their id to the superadmin for promotion."""
    tag = "superadmin" if _is_superadmin(user) else user.role.value
    await message.answer(
        f"<b>Вы:</b> {user.full_name or '—'}\n"
        f"• Telegram ID: <code>{user.telegram_id}</code>\n"
        f"• Роль: <b>{tag}</b>",
        parse_mode="HTML",
    )


@router.message(Command("admin_promote"))
async def on_admin_promote(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_promote <telegram_id> [teacher|admin|student]

    Default role is ``teacher`` — the most common case. The target must have
    spoken to the bot at least once (so their User row exists)."""
    if not _is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/admin_promote &lt;telegram_id&gt; [teacher|admin|student]</code>\n"
            "Пользователь должен хотя бы раз нажать /start у бота.",
            parse_mode="HTML",
        )
        return
    try:
        target_tg = int(args[0])
    except ValueError:
        await message.answer("Telegram ID должен быть числом.")
        return
    role = _parse_role(args[1] if len(args) > 1 else None)
    if role is None:
        await message.answer(
            "Неизвестная роль. Допустимо: teacher, admin, student."
        )
        return
    target = (
        await session.execute(select(User).where(User.telegram_id == target_tg))
    ).scalar_one_or_none()
    if target is None:
        await message.answer(
            f"Пользователь <code>{target_tg}</code> не найден в базе. "
            "Попроси его написать боту /start и повтори.",
            parse_mode="HTML",
        )
        return
    if target.role is role:
        await message.answer(
            f"У <code>{target_tg}</code> уже роль <b>{role.value}</b>.",
            parse_mode="HTML",
        )
        return
    previous = target.role
    target.role = role
    await session.commit()
    log.info(
        "role_promoted",
        by_user=user.telegram_id,
        target=target_tg,
        from_role=previous.value,
        to_role=role.value,
    )
    await message.answer(
        f"✅ <code>{target_tg}</code> ({target.full_name or '—'}) "
        f"теперь <b>{role.value}</b> (было: {previous.value}).",
        parse_mode="HTML",
    )


@router.message(Command("admin_demote"))
async def on_admin_demote(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_demote <telegram_id> — reset role to student."""
    if not _is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    args = (command.args or "").split()
    if not args:
        await message.answer(
            "Использование: <code>/admin_demote &lt;telegram_id&gt;</code>",
            parse_mode="HTML",
        )
        return
    try:
        target_tg = int(args[0])
    except ValueError:
        await message.answer("Telegram ID должен быть числом.")
        return
    target = (
        await session.execute(select(User).where(User.telegram_id == target_tg))
    ).scalar_one_or_none()
    if target is None:
        await message.answer(
            f"Пользователь <code>{target_tg}</code> не найден.", parse_mode="HTML"
        )
        return
    if target.role is UserRole.student:
        await message.answer(
            f"<code>{target_tg}</code> и так student.", parse_mode="HTML"
        )
        return
    previous = target.role
    target.role = UserRole.student
    await session.commit()
    log.info(
        "role_demoted",
        by_user=user.telegram_id,
        target=target_tg,
        from_role=previous.value,
    )
    await message.answer(
        f"✅ <code>{target_tg}</code> ({target.full_name or '—'}) "
        f"теперь <b>student</b> (было: {previous.value}).",
        parse_mode="HTML",
    )


@router.message(Command("admin_users"))
async def on_admin_users(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    user: User,
) -> None:
    """/admin_users [teacher|admin|student]  — список пользователей,
    опционально отфильтрованный по роли. Только superadmin — чтобы личные
    данные не утекали DB-админу-teacher."""
    if not _is_superadmin(user):
        await message.answer("⛔ Только для суперадмина.")
        return
    filter_role = _parse_role((command.args or "").strip() or None, default=None)  # type: ignore[arg-type]
    stmt = select(User).order_by(User.role, User.telegram_id)
    if filter_role is not None:
        stmt = stmt.where(User.role == filter_role)
    users = list((await session.execute(stmt)).scalars())
    if not users:
        await message.answer("Нет пользователей под этот фильтр.")
        return
    # Truncate to avoid blowing Telegram's 4096-char cap on a large DB.
    lines = [f"<b>Пользователи</b> ({len(users)}):"]
    for u in users[:80]:
        lines.append(
            f"• <code>{u.telegram_id}</code> — {u.full_name or '—'} ({u.role.value})"
        )
    if len(users) > 80:
        lines.append(f"… и ещё {len(users) - 80}")
    await message.answer("\n".join(lines), parse_mode="HTML")
