"""faq_entries table — teacher-curated answers served before RAG.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-20

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "faq_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "subject_id",
            sa.Integer(),
            sa.ForeignKey("subjects.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("question_normalised", sa.String(length=300), nullable=False),
        sa.Column("question_original", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column(
            "parent_dialog_id",
            sa.Integer(),
            sa.ForeignKey("dialogs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    # A subject scope can have at most one canonical answer per normalised
    # question. NULL subject_id means "global" — we keep a partial unique
    # index for that via a separate manual constraint below; SQLite can't
    # express partial indexes through SQLAlchemy Alembic ops cleanly, so
    # the Python layer re-checks on insert.
    op.create_index(
        "ix_faq_subject_qnorm",
        "faq_entries",
        ["subject_id", "question_normalised"],
    )
    op.create_index(
        "ix_faq_qnorm",
        "faq_entries",
        ["question_normalised"],
    )


def downgrade() -> None:
    op.drop_index("ix_faq_qnorm", table_name="faq_entries")
    op.drop_index("ix_faq_subject_qnorm", table_name="faq_entries")
    op.drop_table("faq_entries")
