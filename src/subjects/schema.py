"""Pydantic schema for a subject entry in courses.yaml."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}[a-z0-9]$")


class Subject(BaseModel):
    """A single course subject. Mirrors one entry from courses.yaml."""

    slug: str = Field(..., description="URL-safe identifier, e.g. 'theory_of_systems'")
    title_en: str = Field(..., min_length=1, max_length=120)
    title_ru: str = Field(..., min_length=1, max_length=120)
    description_ru: str = Field(default="", max_length=2000)
    description_en: str = Field(default="", max_length=2000)
    teacher_telegram_id: int | None = None
    voice: str = Field(default="academic_ru")
    is_active: bool = True

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError(
                "slug must be lowercase, 3-64 chars, start with a letter, "
                "use only [a-z0-9_], end with a letter/digit"
            )
        return value

    @property
    def display_ru(self) -> str:
        return self.title_ru or self.title_en or self.slug
