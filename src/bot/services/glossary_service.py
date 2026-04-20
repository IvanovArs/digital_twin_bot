"""Glossary sync (YAML → DB) + student-facing queries.

Each subject can have a `data/glossary/<slug>.yaml` file with a list of terms.
On bot startup we upsert those rows into `glossary_terms`.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import GlossaryTerm, Subject

log = structlog.get_logger(__name__)


async def sync_glossary_from_yaml(session: AsyncSession, glossary_dir: Path) -> None:
    """For every `<slug>.yaml` in ``glossary_dir``, upsert terms into DB.

    Silently skips files whose slug has no matching Subject (courses.yaml not
    yet updated). Does NOT delete terms missing from YAML — admins may add
    terms out-of-band via bot commands later.
    """
    if not glossary_dir.is_dir():
        return

    for yaml_path in sorted(glossary_dir.glob("*.yaml")):
        slug = yaml_path.stem
        subject = (
            await session.execute(select(Subject).where(Subject.slug == slug))
        ).scalar_one_or_none()
        if subject is None:
            log.warning("glossary_yaml_for_unknown_subject", slug=slug, path=str(yaml_path))
            continue

        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            log.error("glossary_yaml_invalid", path=str(yaml_path), error=str(exc))
            continue

        entries = raw.get("terms") or []
        if not isinstance(entries, list):
            log.error("glossary_yaml_terms_not_list", path=str(yaml_path))
            continue

        existing = {
            row.term: row
            for row in (
                await session.execute(
                    select(GlossaryTerm).where(GlossaryTerm.subject_id == subject.id)
                )
            ).scalars()
        }

        added = updated = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term") or "").strip()
            definition = str(entry.get("definition") or "").strip()
            category = (entry.get("category") or None) or None
            if not term or not definition:
                continue

            row = existing.get(term)
            if row is None:
                session.add(
                    GlossaryTerm(
                        subject_id=subject.id,
                        term=term,
                        definition=definition,
                        category=category,
                    )
                )
                added += 1
            elif row.definition != definition or row.category != category:
                row.definition = definition
                row.category = category
                updated += 1

        await session.flush()
        log.info("glossary_sync_done", subject=slug, added=added, updated=updated)


async def list_terms(
    session: AsyncSession,
    *,
    subject_slug: str | None = None,
) -> list[GlossaryTerm]:
    """All glossary terms (optionally filtered to one subject), sorted alphabetically."""
    stmt = select(GlossaryTerm)
    if subject_slug is not None:
        stmt = stmt.join(Subject).where(Subject.slug == subject_slug)
    stmt = stmt.order_by(GlossaryTerm.term)
    return list((await session.execute(stmt)).scalars())


async def find_term(
    session: AsyncSession,
    query: str,
    *,
    subject_slug: str | None = None,
) -> list[GlossaryTerm]:
    """Case-insensitive substring match on term name, within a subject if given."""
    stmt = select(GlossaryTerm).where(GlossaryTerm.term.ilike(f"%{query}%"))
    if subject_slug is not None:
        stmt = stmt.join(Subject).where(Subject.slug == subject_slug)
    stmt = stmt.order_by(GlossaryTerm.term).limit(20)
    return list((await session.execute(stmt)).scalars())
