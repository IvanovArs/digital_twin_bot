"""users.answer_mode preference

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-20

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "answer_mode",
            sa.String(length=16),
            nullable=False,
            server_default="verbose",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "answer_mode")
