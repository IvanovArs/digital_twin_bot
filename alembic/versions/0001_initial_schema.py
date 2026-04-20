"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-18

Uses dialect-neutral types so the same migration works on both SQLite (dev)
and PostgreSQL (prod).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("full_name", sa.String(length=150), nullable=True),
        sa.Column(
            "role",
            sa.Enum("student", "teacher", "admin", name="user_role", native_enum=False, length=16),
            nullable=False,
            server_default="student",
        ),
        sa.Column("group_name", sa.String(length=50), nullable=True),
        sa.Column("current_subject_slug", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index("ix_users_telegram_id", "users", ["telegram_id"], unique=True)

    op.create_table(
        "subjects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slug", sa.String(length=64), nullable=False, unique=True),
        sa.Column("title_en", sa.String(length=200), nullable=False),
        sa.Column("title_ru", sa.String(length=200), nullable=False),
        sa.Column("description_ru", sa.Text(), nullable=True),
        sa.Column("description_en", sa.Text(), nullable=True),
        sa.Column(
            "teacher_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("voice", sa.String(length=40), nullable=False, server_default="academic_ru"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index("ix_subjects_slug", "subjects", ["slug"], unique=True)
    op.create_index("ix_subjects_active", "subjects", ["is_active"])

    op.create_table(
        "subject_materials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "subject_id",
            sa.Integer(),
            sa.ForeignKey("subjects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column(
            "uploaded_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.UniqueConstraint("subject_id", "filename", name="uq_subject_materials_file"),
    )
    op.create_index("ix_subject_materials_subject", "subject_materials", ["subject_id"])

    op.create_table(
        "dialogs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "subject_id",
            sa.Integer(),
            sa.ForeignKey("subjects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index("ix_dialogs_user_created", "dialogs", ["user_id", "created_at"])
    op.create_index("ix_dialogs_subject_created", "dialogs", ["subject_id", "created_at"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "dialog_id",
            sa.Integer(),
            sa.ForeignKey("dialogs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating_range"),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_index("ix_dialogs_subject_created", table_name="dialogs")
    op.drop_index("ix_dialogs_user_created", table_name="dialogs")
    op.drop_table("dialogs")
    op.drop_index("ix_subject_materials_subject", table_name="subject_materials")
    op.drop_table("subject_materials")
    op.drop_index("ix_subjects_active", table_name="subjects")
    op.drop_index("ix_subjects_slug", table_name="subjects")
    op.drop_table("subjects")
    op.drop_index("ix_users_telegram_id", table_name="users")
    op.drop_table("users")
