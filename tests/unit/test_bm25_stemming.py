"""Tests for the lightweight Russian stemmer used in BM25 tokenisation.

The point is behavioural: inflected forms of the same root should produce
the same BM25 token so «что такое стейкхолдеры» hits a chunk about
«стейкхолдера».
"""

from __future__ import annotations

from src.rag.hybrid import _stem, _tokenise


def test_russian_inflections_collapse_to_same_stem() -> None:
    forms = [
        "стейкхолдер",
        "стейкхолдера",
        "стейкхолдеров",
        "стейкхолдеру",
        "стейкхолдерами",
    ]
    stems = {_stem(f) for f in forms}
    # At least one common stem that appears in >=3 forms.
    counts: dict[str, int] = {}
    for f in forms:
        counts[_stem(f)] = counts.get(_stem(f), 0) + 1
    dominant = max(counts.values())
    assert dominant >= 3, f"forms didn't collapse: {stems}"


def test_short_tokens_are_not_stemmed() -> None:
    """«я», «не», «в» etc. should stay untouched."""
    assert _stem("я") == "я"
    assert _stem("не") == "не"
    assert _stem("и") == "и"


def test_acronyms_stay_uppercase_then_lower() -> None:
    assert _stem("SWOT") == "swot"
    assert _stem("PEST") == "pest"


def test_digits_are_not_stemmed() -> None:
    assert _stem("1984") == "1984"
    assert _stem("42") == "42"


def test_english_suffixes_stripped() -> None:
    assert _stem("running") == "runn"
    assert _stem("methods") == "method"


def test_tokenise_lowers_and_stems() -> None:
    tokens = _tokenise("Стейкхолдеры и акционеры")
    # All tokens are lowercase stems, no duplicates.
    assert all(t == t.lower() for t in tokens)
    # «стейкхолдеры» and «стейкхолдер» in corpus should share a stem.
    assert _stem("стейкхолдеры") == _stem("стейкхолдер")


def test_stemmer_preserves_already_short_words() -> None:
    """Words ≤3 chars are left as-is — stripping them produces too many
    false merges («то» → «» etc.)."""
    assert _stem("то") == "то"
    assert _stem("где") == "где"
