"""Mirror `courses.yaml` → `subjects` table at bot startup.

Idempotent: adds new subjects, updates titles/descriptions on existing ones,
marks subjects missing from courses.yaml as inactive (does NOT delete them —
we want to keep historical FK references from dialogs/materials).
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Subject as SubjectRow
from src.subjects import Catalog
from src.subjects import Subject as SubjectCfg

log = structlog.get_logger(__name__)


async def sync_subjects(session: AsyncSession, catalog: Catalog) -> None:
    """Sync courses.yaml catalog into the subjects table (upsert + deactivate-missing)."""
    yaml_slugs: set[str] = set(catalog.slugs())

    existing: dict[str, SubjectRow] = {
        row.slug: row for row in (await session.execute(select(SubjectRow))).scalars()
    }

    added = updated = deactivated = 0

    for subj_cfg in catalog:
        row = existing.get(subj_cfg.slug)
        if row is None:
            row = _cfg_to_row(subj_cfg)
            session.add(row)
            added += 1
        else:
            _apply_cfg_to_row(subj_cfg, row)
            updated += 1

    for slug, row in existing.items():
        if slug not in yaml_slugs and row.is_active:
            row.is_active = False
            deactivated += 1

    await session.flush()
    log.info(
        "subjects_sync_done",
        added=added,
        updated=updated,
        deactivated=deactivated,
        total_active=len(yaml_slugs),
    )


def _cfg_to_row(cfg: SubjectCfg) -> SubjectRow:
    return SubjectRow(
        slug=cfg.slug,
        title_en=cfg.title_en,
        title_ru=cfg.title_ru,
        description_ru=cfg.description_ru or None,
        description_en=cfg.description_en or None,
        voice=cfg.voice,
        is_active=cfg.is_active,
    )


def _apply_cfg_to_row(cfg: SubjectCfg, row: SubjectRow) -> None:
    row.title_en = cfg.title_en
    row.title_ru = cfg.title_ru
    row.description_ru = cfg.description_ru or None
    row.description_en = cfg.description_en or None
    row.voice = cfg.voice
    row.is_active = cfg.is_active
