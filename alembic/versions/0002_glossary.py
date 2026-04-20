"""glossary_terms table

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-18

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "glossary_terms",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "subject_id",
            sa.Integer(),
            sa.ForeignKey("subjects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("term", sa.String(length=200), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.UniqueConstraint("subject_id", "term", name="uq_glossary_subject_term"),
    )
    op.create_index("ix_glossary_subject", "glossary_terms", ["subject_id"])


def downgrade() -> None:
    op.drop_index("ix_glossary_subject", table_name="glossary_terms")
    op.drop_table("glossary_terms")
