from __future__ import annotations

from pathlib import Path

import pytest

from src.subjects.catalog import CatalogError, load_catalog


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "courses.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_catalog(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
subjects:
  - slug: theory_of_systems
    title_en: "Theory of Systems"
    title_ru: "Теория систем"
  - slug: discrete_math
    title_en: "Discrete Math"
    title_ru: "Дискретная математика"
""",
    )
    cat = load_catalog(p)
    assert cat.slugs() == ["theory_of_systems", "discrete_math"]
    assert cat.require("theory_of_systems").title_ru == "Теория систем"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CatalogError):
        load_catalog(tmp_path / "missing.yaml")


def test_duplicate_slug_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
subjects:
  - slug: dup
    title_en: a
    title_ru: a
  - slug: dup
    title_en: b
    title_ru: b
""",
    )
    with pytest.raises(CatalogError, match="Duplicate"):
        load_catalog(p)


def test_invalid_slug_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
subjects:
  - slug: Bad-Slug!
    title_en: x
    title_ru: x
""",
    )
    with pytest.raises(CatalogError):
        load_catalog(p)


def test_empty_subjects_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "subjects: []\n")
    with pytest.raises(CatalogError, match="no subjects"):
        load_catalog(p)


def test_active_filter(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
subjects:
  - slug: alpha_one
    title_en: A
    title_ru: А
    is_active: false
  - slug: beta_two
    title_en: B
    title_ru: Б
""",
    )
    cat = load_catalog(p)
    assert [s.slug for s in cat.active()] == ["beta_two"]
