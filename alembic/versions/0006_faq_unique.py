"""UniqueConstraint on FAQEntry(subject_id, question_normalised)

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-20

Without this constraint, two teachers racing to fix the same dialog
could each see "no existing FAQ" in their separate sessions and both
INSERT — producing duplicate rows under the same normalised question,
and the subsequent ``lookup_faq`` would non-deterministically pick
between them on each student query.

``subject_id`` may be NULL for global FAQ. SQLite and Postgres both
treat NULL as distinct for UNIQUE, so two global FAQs for the same
normalised question can still race past the constraint; the race is
rare enough that we accept it (and ``save_faq_from_dialog`` upserts by
the same key, so the second writer will overwrite the first rather
than duplicate). The constraint closes the subject-scoped case, which
is the common scenario.
"""

from __future__ import annotations

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("faq_entries") as batch:
        batch.create_unique_constraint(
            "uq_faq_subject_qnorm",
            ["subject_id", "question_normalised"],
        )


def downgrade() -> None:
    with op.batch_alter_table("faq_entries") as batch:
        batch.drop_constraint("uq_faq_subject_qnorm", type_="unique")
