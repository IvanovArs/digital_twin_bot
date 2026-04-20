"""Tests for glossary-based query expansion."""

from __future__ import annotations

import pytest

from src.rag import glossary
from src.rag.glossary import _MAX_EXPANSIONS, expand_query


@pytest.fixture(autouse=True)
def _patch_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the YAML-loaded index with a deterministic one for these tests.

    We patch the lru_cache'd loader by clearing its cache and monkeypatching
    the function it wraps — easier than mocking yaml + filesystem.
    """
    fake = {
        "стейкхолдер": ["лицо или группа, влияющие на цели организации"],
        "система": ["множество элементов в отношениях"],
        "методика": [
            "методика Кошарского-Уёмова — двойственное определение",
            "методика Волковой-Четверикова — концепция деятельности",
        ],
        "культура": ["корпоративная культура — артефакты и ценности (Шейн)"],
    }

    def _fake_index() -> dict[str, list[str]]:
        return fake

    monkeypatch.setattr(glossary, "_glossary_index", _fake_index)


def test_expand_matches_exact_term() -> None:
    out = expand_query("что такое стейкхолдер")
    assert "лицо или группа" in out
    assert out.startswith("что такое стейкхолдер")


def test_expand_matches_inflected_form() -> None:
    out = expand_query("кто такие стейкхолдеры")
    assert "лицо или группа" in out


def test_expand_skips_unknown_query() -> None:
    out = expand_query("сегодня хорошая погода")
    assert out == "сегодня хорошая погода"


def test_expand_caps_at_max() -> None:
    # «методика» has two definitions — both should appear, but no more than
    # _MAX_EXPANSIONS (currently 2) total; «культура» gets dropped.
    out = expand_query("методика Кошарского и культура")
    new_text = out[len("методика Кошарского и культура") :]
    # Each defn appended with a leading space → counting strict matches.
    matches = sum(1 for d in [
        "Кошарского-Уёмова",
        "Волковой-Четверикова",
        "корпоративная культура",
    ] if d in new_text)
    assert matches <= _MAX_EXPANSIONS


def test_expand_dedupes_definitions() -> None:
    # Same root word appearing twice in the query shouldn't add the same
    # definition twice.
    out = expand_query("система система система")
    assert out.count("множество элементов") == 1
