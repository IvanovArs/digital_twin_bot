"""dialogs.is_favourite flag

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-20

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "dialogs",
        sa.Column(
            "is_favourite",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Index lets ``/favourites`` skip the full-table scan once the dialog
    # table grows past a few thousand rows.
    op.create_index(
        "ix_dialogs_user_favourite",
        "dialogs",
        ["user_id", "is_favourite"],
    )


def downgrade() -> None:
    op.drop_index("ix_dialogs_user_favourite", table_name="dialogs")
    op.drop_column("dialogs", "is_favourite")
