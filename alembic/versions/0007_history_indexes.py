"""Композитные индексы для горячих запросов /history и /favourites.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-26
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # /favourites: WHERE user_id=? AND is_favourite=true ORDER BY created_at DESC.
    # Старый ix_dialogs_user_favourite не покрывал ORDER BY → Postgres делал Sort.
    op.create_index(
        "ix_dialogs_user_fav_created",
        "dialogs",
        ["user_id", "is_favourite", sa.text("created_at DESC")],
    )
    # stats_24h: WHERE created_at >= cutoff. Без индекса — full scan.
    op.create_index(
        "ix_dialogs_created_at",
        "dialogs",
        [sa.text("created_at DESC")],
    )
    # FAQ-очередь по 👎-отзывам — ORDER BY feedback.created_at.
    op.create_index(
        "ix_feedback_created",
        "feedback",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_created", table_name="feedback")
    op.drop_index("ix_dialogs_created_at", table_name="dialogs")
    op.drop_index("ix_dialogs_user_fav_created", table_name="dialogs")
