"""Admin-сервисы: загрузка материалов + переиндексация в фоне."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.db.models import Dialog, Feedback, Subject, SubjectMaterial, User
from src.rag.ingest import build_index

log = structlog.get_logger(__name__)


async def save_material(
    session: AsyncSession,
    *,
    subject: Subject,
    filename: str,
    payload: bytes,
    uploader: User,
    target_dir: Path,
) -> SubjectMaterial:
    """Сохранить загруженный файл на диск и upsert-нуть строку material'а.

    Если DB-flush упал после write на диск, удаляем orphan-файл — иначе
    повторные ретраи накопят полу-закоммиченные uploads. ``dst.unlink(missing_ok=True)``
    безопасен и при перезаписи существующего material'а: предыдущий
    checksum в DB, можно переcкачать.
    """
    subject_dir = target_dir / subject.slug
    subject_dir.mkdir(parents=True, exist_ok=True)
    dst = subject_dir / filename
    # Запомним, создавали ли файл, чтобы rollback знал, что чистить.
    # Если файл уже существовал (препод переcкачивает то же имя) —
    # не удаляем при DB-failure: предыдущий payload — committed-state.
    newly_created = not dst.exists()
    dst.write_bytes(payload)

    checksum = hashlib.sha256(payload).hexdigest()[:16]

    try:
        existing = (
            await session.execute(
                select(SubjectMaterial).where(
                    SubjectMaterial.subject_id == subject.id,
                    SubjectMaterial.filename == filename,
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            row = SubjectMaterial(
                subject_id=subject.id,
                filename=filename,
                uploaded_by_user_id=uploader.id,
                checksum=checksum,
            )
            session.add(row)
        else:
            existing.checksum = checksum
            existing.uploaded_by_user_id = uploader.id
            existing.indexed_at = None  # форсим reindex
            row = existing

        await session.flush()
    except Exception:
        # DB упал после write на диск — caller сделает rollback сессии,
        # выровняем файловую систему, удалив orphan (только если файл
        # был свежесозданным).
        if newly_created:
            with contextlib.suppress(OSError):
                dst.unlink(missing_ok=True)
        raise
    log.info("material_saved", subject=subject.slug, filename=filename, size=len(payload))
    return row


def _invalidate_retrieval_caches() -> None:
    """Сбросить все @lru_cache, кэширующие chunks.jsonl / embeddings.npy.

    ``build_index`` перезаписывает файлы на диске, но in-process lru-кэши
    были созданы ДО write'а и продолжали отдавать stale-snapshot до
    рестарта бота — студенты получали retrieval-результаты по чанкам,
    которых уже нет в индексе. Чистим, чтобы следующий retrieval
    подхватил свежие артефакты.
    """
    # Отложенный импорт: admin_service зовётся из handler-контекста; не
    # хотим тянуть sentence-transformers в call-graph на module import
    # (~5с и ~500 МБ при первом touch).
    from src.rag.hybrid import _bm25, _fingerprint_index
    from src.rag.retriever import _encode_query_cached, _load_index, load_chunks

    for fn in (
        load_chunks,
        _load_index,
        _encode_query_cached,
        _bm25,
        _fingerprint_index,
    ):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


async def reindex_subject_in_background(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    subject_slug: str | None = None,
) -> None:
    """Запустить ingest в worker-thread'е, потом проставить SubjectMaterial.indexed_at.

    Вызывается через asyncio.create_task, чтобы Telegram-handler'ы не блокировались.
    """
    log.info("reindex_started", subject=subject_slug or "*")
    try:
        await asyncio.to_thread(build_index, subject_slug)
    except Exception:
        log.exception("reindex_failed", subject=subject_slug)
        return

    # Сначала кэши — любой concurrent-вопрос студента, прилетевший после
    # build_index, должен видеть новые чанки.
    _invalidate_retrieval_caches()

    async with sessionmaker() as session:
        q = select(SubjectMaterial)
        if subject_slug is not None:
            q = q.join(Subject).where(Subject.slug == subject_slug)
        materials = list((await session.execute(q)).scalars())
        now = datetime.now(UTC)
        for m in materials:
            m.indexed_at = now
        await session.commit()
    log.info("reindex_done", subject=subject_slug or "*", updated=len(materials))


async def stats_24h(session: AsyncSession) -> dict[str, object]:
    """Компактный stats-payload для /admin_stats.

    Работает на SQLite (dev) и PostgreSQL (prod): 24-часовой cutoff
    считаем в Python (SQLite не умеет вычитать interval из NOW()), p95
    тоже Python-side — у SQLite нет percentile_cont.
    """
    total_dialogs = (await session.execute(select(func.count(Dialog.id)))).scalar_one()

    cutoff = datetime.now(UTC) - timedelta(days=1)
    recent_dialogs = (
        await session.execute(select(func.count(Dialog.id)).where(Dialog.created_at >= cutoff))
    ).scalar()

    avg_rating = (await session.execute(select(func.avg(Feedback.rating)))).scalar_one()

    latencies = [
        int(v)
        for v in (
            await session.execute(select(Dialog.latency_ms).where(Dialog.latency_ms.is_not(None)))
        ).scalars()
        if v is not None
    ]
    p95: int | None = None
    if latencies:
        latencies.sort()
        idx = max(0, int(round(0.95 * (len(latencies) - 1))))
        p95 = latencies[idx]

    return {
        "dialogs_total": int(total_dialogs or 0),
        "dialogs_24h": int(recent_dialogs or 0),
        "avg_rating": float(avg_rating) if avg_rating is not None else None,
        "p95_latency_ms": p95,
    }
