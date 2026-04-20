"""Tests for the comparison detector + merge + prompt builder."""

from __future__ import annotations

import pytest

from src.rag.comparison import (
    build_comparison_messages,
    detect_comparison,
    merge_hits,
)
from src.rag.retriever import Hit
from src.subjects.schema import Subject


def _hit(text: str, book: str = "x.pdf", page: int = 1) -> Hit:
    return Hit(text=text, subject_slug="tos", book=book, page=page, score=0.7)


# ---------- detect_comparison ----------


@pytest.mark.parametrize(
    "q,expected",
    [
        ("сравни стейкхолдера и акционера", ("стейкхолдера", "акционера")),
        ("Сравни SWOT и PEST анализ", ("SWOT", "PEST анализ")),
        ("сравните онтологию с таксономией", ("онтологию", "таксономией")),
        ("разница между системой и подсистемой", ("системой", "подсистемой")),
        (
            "чем отличается эмерджентность от иерархичности?",
            ("эмерджентность", "иерархичности"),
        ),
        ("отличия системы от подсистемы", ("системы", "подсистемы")),
        ("различия система и модель", ("система", "модель")),
        ("compare ontology and taxonomy", ("ontology", "taxonomy")),
        ("difference between SWOT and PEST", ("SWOT", "PEST")),
        ("SWOT vs PEST", ("SWOT", "PEST")),
    ],
)
def test_detect_matches_canonical_patterns(q: str, expected: tuple[str, str]) -> None:
    result = detect_comparison(q)
    assert result is not None
    assert result[0].lower() == expected[0].lower()
    assert result[1].lower() == expected[1].lower()


@pytest.mark.parametrize(
    "q",
    [
        "что такое стейкхолдер",
        "расскажи про эмерджентность",
        "",
        "?",
        "сравни",
        "сравни стейкхолдера",  # only one term
    ],
)
def test_detect_returns_none_for_non_comparisons(q: str) -> None:
    assert detect_comparison(q) is None


def test_detect_rejects_same_term_twice() -> None:
    """«сравни онтологию и онтологию» shouldn't trigger."""
    assert detect_comparison("сравни онтологию и онтологию") is None


def test_detect_strips_leading_what_is() -> None:
    out = detect_comparison("сравни что такое система и что такое модель")
    assert out is not None
    assert out[0] == "система"
    assert out[1] == "модель"


# ---------- merge_hits ----------


def test_merge_interleaves_and_deduplicates() -> None:
    a = [_hit("text A1", book="a.pdf"), _hit("text A2", book="a.pdf")]
    b = [_hit("text B1", book="b.pdf"), _hit("text A1", book="b.pdf")]  # dupe text
    merged = merge_hits(a, b, max_total=10)
    # Order: a[0], b[0], a[1]. b[1] is a dupe of a[0] by text.
    assert [h.text for h in merged] == ["text A1", "text B1", "text A2"]


def test_merge_respects_cap() -> None:
    a = [_hit(f"A{i}") for i in range(5)]
    b = [_hit(f"B{i}") for i in range(5)]
    assert len(merge_hits(a, b, max_total=3)) == 3
    assert len(merge_hits(a, b, max_total=100)) == 10


def test_merge_handles_empty_lists() -> None:
    a = [_hit("solo")]
    assert merge_hits(a, []) == a
    assert merge_hits([], a) == a
    assert merge_hits([], []) == []


# ---------- build_comparison_messages ----------


def _subject() -> Subject:
    return Subject(slug="tos", title_en="Theory of Systems", title_ru="Теория систем")


def test_comparison_prompt_contains_both_terms_and_context_ru() -> None:
    hits = [_hit("ракурс про стейкхолдеров"), _hit("про акционеров")]
    msgs = build_comparison_messages("стейкхолдер", "акционер", hits, _subject())
    assert len(msgs) == 2
    user = msgs[1]["content"]
    assert "стейкхолдер" in user and "акционер" in user
    # Context from hits lands in the message body.
    assert "ракурс про стейкхолдеров" in user
    assert "про акционеров" in user
    # Structure instruction is present.
    assert "сравни" in user.lower()


def test_comparison_prompt_en() -> None:
    hits = [_hit("about SWOT"), _hit("about PEST")]
    msgs = build_comparison_messages("SWOT", "PEST", hits, _subject(), lang="en")
    user = msgs[1]["content"]
    assert "Compare" in user
    assert "SWOT" in user and "PEST" in user


def test_comparison_system_prompt_preserved() -> None:
    """System prompt is the regular one (no_think, structure rules) so the
    comparison variant doesn't regress grounding / anti-hallucination."""
    hits = [_hit("fragment")]
    msgs = build_comparison_messages("a", "b", hits, _subject())
    system = msgs[0]["content"]
    assert system.startswith("/no_think")
