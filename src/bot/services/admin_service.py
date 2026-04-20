"""Admin services: upload materials + reindex in the background."""

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
    """Save an uploaded file on disk and upsert the materials row.

    If the DB flush fails after the file has already been written, we
    remove the orphan file so successive retries don't leave the disk
    full of partially-committed uploads. ``dst.unlink(missing_ok=True)``
    is safe even if the user is overwriting an existing material — the
    previous checksum is recorded in the DB, and they can re-upload.
    """
    subject_dir = target_dir / subject.slug
    subject_dir.mkdir(parents=True, exist_ok=True)
    dst = subject_dir / filename
    # Track whether we just created the file so rollback knows to clean
    # it up. If it already existed (teacher re-uploading the same name),
    # we don't delete it on DB failure — the previous payload is still
    # the committed state.
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
            existing.indexed_at = None  # force reindex
            row = existing

        await session.flush()
    except Exception:
        # DB side failed after we wrote the payload to disk — a caller
        # will rollback the session, so make the filesystem match by
        # removing the orphan file (only if we created it fresh).
        if newly_created:
            with contextlib.suppress(OSError):
                dst.unlink(missing_ok=True)
        raise
    log.info("material_saved", subject=subject.slug, filename=filename, size=len(payload))
    return row


def _invalidate_retrieval_caches() -> None:
    """Drop every @lru_cache that snapshots chunks.jsonl / embeddings.npy.

    ``build_index`` rewrites those files on disk, but the in-process lru
    caches were minted before the write and would happily serve the stale
    snapshot until the next bot restart — students ended up getting
    retrieval results for chunks that no longer exist in the index. Clear
    them here so the very next retrieval re-loads the freshly rebuilt
    artefacts.
    """
    # Deferred imports: admin_service is called from handler context; we
    # don't want to pull sentence-transformers into the call graph at
    # module import (it takes ~5 s and ~500 MB on first touch).
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
    """Run ingest in a worker thread, then update SubjectMaterial.indexed_at rows.

    Called via asyncio.create_task so Telegram handlers don't block.
    """
    log.info("reindex_started", subject=subject_slug or "*")
    try:
        await asyncio.to_thread(build_index, subject_slug)
    except Exception:
        log.exception("reindex_failed", subject=subject_slug)
        return

    # Caches first — any concurrent student question racing through here
    # after build_index returned must see the new chunks.
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
    log.info(
        "reindex_done", subject=subject_slug or "*", updated=len(materials)
    )


async def stats_24h(session: AsyncSession) -> dict[str, object]:
    """Return a compact stats payload for /admin_stats.

    Works on both SQLite (dev) and PostgreSQL (prod): the 24-hour cutoff is
    computed in Python (SQLite can't subtract an interval from NOW()), and p95
    is also Python-side since SQLite has no percentile_cont.
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
