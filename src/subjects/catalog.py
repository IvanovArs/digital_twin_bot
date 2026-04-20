"""Load and validate courses.yaml — the subjects catalog."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.subjects.schema import Subject


class CatalogError(Exception):
    """Raised when courses.yaml is malformed or duplicates slugs."""


class Catalog:
    """In-memory catalog of subjects, keyed by slug."""

    def __init__(self, subjects: Iterable[Subject]):
        self._by_slug: dict[str, Subject] = {}
        for s in subjects:
            if s.slug in self._by_slug:
                raise CatalogError(f"Duplicate subject slug in courses.yaml: {s.slug!r}")
            self._by_slug[s.slug] = s

    def __iter__(self) -> Iterator[Subject]:
        return iter(self._by_slug.values())

    def __len__(self) -> int:
        return len(self._by_slug)

    def __contains__(self, slug: object) -> bool:
        return isinstance(slug, str) and slug in self._by_slug

    def get(self, slug: str) -> Subject | None:
        return self._by_slug.get(slug)

    def require(self, slug: str) -> Subject:
        subj = self._by_slug.get(slug)
        if subj is None:
            raise KeyError(f"Unknown subject slug: {slug!r}")
        return subj

    def active(self) -> list[Subject]:
        return [s for s in self._by_slug.values() if s.is_active]

    def slugs(self) -> list[str]:
        return list(self._by_slug.keys())


def load_catalog(path: Path) -> Catalog:
    """Parse courses.yaml and return a validated Catalog.

    Raises CatalogError if the file is missing, malformed, or empty.
    """
    if not path.exists():
        raise CatalogError(f"courses.yaml not found at {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"courses.yaml is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict) or "subjects" not in raw:
        raise CatalogError("courses.yaml must have a top-level 'subjects:' list")

    entries = raw.get("subjects") or []
    if not isinstance(entries, list):
        raise CatalogError("'subjects' must be a list")

    subjects: list[Subject] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CatalogError(f"subjects[{i}] must be a mapping")
        try:
            subjects.append(Subject(**entry))
        except ValidationError as exc:
            raise CatalogError(f"subjects[{i}] invalid: {exc}") from exc

    if not subjects:
        raise CatalogError("courses.yaml has no subjects — add at least one")

    return Catalog(subjects)
