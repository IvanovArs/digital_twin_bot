"""SQLAlchemy 2.0 declarative models.

Postgres-first. No legacy tables from the old SQL dump — this is a fresh schema
aligned with the multi-subject RAG architecture.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column, relationship


class Base(MappedAsDataclass, DeclarativeBase):
    """Common declarative base."""


class UserRole(str, enum.Enum):
    student = "student"
    teacher = "teacher"
    admin = "admin"


class AnswerMode(str, enum.Enum):
    """Per-user preference for how much prose the bot wraps around an answer.

    ``verbose`` — default; follow-up buttons, header with the question,
    «Источники» block, rich structure.
    ``brief`` — just the answer body + one-line sources, no follow-up row,
    no feedback chrome. For users who want a textbook lookup, not a chat.
    """

    verbose = "verbose"
    brief = "brief"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(150), default=None)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", native_enum=False, length=16),
        default=UserRole.student,
    )
    answer_mode: Mapped[AnswerMode] = mapped_column(
        Enum(AnswerMode, name="answer_mode", native_enum=False, length=16),
        default=AnswerMode.verbose,
    )
    group_name: Mapped[str | None] = mapped_column(String(50), default=None)
    current_subject_slug: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)

    dialogs: Mapped[list[Dialog]] = relationship(
        back_populates="user",
        init=False,
        lazy="selectin",
    )


class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (Index("ix_subjects_active", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title_en: Mapped[str] = mapped_column(String(200))
    title_ru: Mapped[str] = mapped_column(String(200))
    description_ru: Mapped[str | None] = mapped_column(Text, default=None)
    description_en: Mapped[str | None] = mapped_column(Text, default=None)
    teacher_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        default=None,
    )
    voice: Mapped[str] = mapped_column(String(40), default="academic_ru")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)

    materials: Mapped[list[SubjectMaterial]] = relationship(
        back_populates="subject",
        init=False,
        lazy="selectin",
        cascade="all, delete-orphan",
    )


class SubjectMaterial(Base):
    __tablename__ = "subject_materials"
    __table_args__ = (
        UniqueConstraint("subject_id", "filename", name="uq_subject_materials_file"),
        Index("ix_subject_materials_subject", "subject_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"),
    )
    filename: Mapped[str] = mapped_column(String(255))
    uploaded_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        default=None,
    )
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str | None] = mapped_column(String(64), default=None)
    indexed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)

    subject: Mapped[Subject] = relationship(back_populates="materials", init=False)


class Dialog(Base):
    __tablename__ = "dialogs"
    __table_args__ = (
        Index("ix_dialogs_user_created", "user_id", "created_at"),
        Index("ix_dialogs_subject_created", "subject_id", "created_at"),
        Index("ix_dialogs_user_favourite", "user_id", "is_favourite"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"),
        default=None,
    )
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, default=None)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    is_favourite: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)

    user: Mapped[User] = relationship(back_populates="dialogs", init=False)
    subject: Mapped[Subject | None] = relationship(init=False, lazy="selectin")
    feedback: Mapped[Feedback | None] = relationship(
        back_populates="dialog",
        init=False,
        lazy="joined",
        uselist=False,
    )


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating_range"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    dialog_id: Mapped[int] = mapped_column(
        ForeignKey("dialogs.id", ondelete="CASCADE"),
        unique=True,
    )
    rating: Mapped[int] = mapped_column(SmallInteger)
    comment: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)

    dialog: Mapped[Dialog] = relationship(back_populates="feedback", init=False)


class GlossaryTerm(Base):
    __tablename__ = "glossary_terms"
    __table_args__ = (
        UniqueConstraint("subject_id", "term", name="uq_glossary_subject_term"),
        Index("ix_glossary_subject", "subject_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    term: Mapped[str] = mapped_column(String(200))
    definition: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(80), default=None)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)


class FAQEntry(Base):
    """Teacher-curated answer served *before* RAG when the question matches.

    Populated when a teacher reviews a 👎-rated dialog via ``/teacher_fix``
    and writes a corrected answer. Subsequent students asking the same
    normalised question get the teacher's answer verbatim — the LLM isn't
    involved, so latency drops to a single DB lookup and the teacher's
    phrasing is preserved. ``subject_id`` may be NULL for a global FAQ.
    """

    __tablename__ = "faq_entries"
    __table_args__ = (
        UniqueConstraint(
            "subject_id",
            "question_normalised",
            name="uq_faq_subject_qnorm",
        ),
        Index("ix_faq_subject_qnorm", "subject_id", "question_normalised"),
        Index("ix_faq_qnorm", "question_normalised"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False, autoincrement=True)
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"),
        default=None,
    )
    question_normalised: Mapped[str] = mapped_column(String(300), default="")
    question_original: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    parent_dialog_id: Mapped[int | None] = mapped_column(
        ForeignKey("dialogs.id", ondelete="SET NULL"),
        default=None,
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)
