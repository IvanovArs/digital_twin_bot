from __future__ import annotations

from src.rag.retriever import Hit
from src.rag.router import detect_subject


def _hit(subject: str, score: float) -> Hit:
    return Hit(text="", subject_slug=subject, book="b", page=1, score=score)


def test_empty_hits() -> None:
    r = detect_subject([])
    assert r.subject_slug is None
    assert r.ranked == []


def test_single_subject_dominates() -> None:
    hits = [_hit("a", 0.8), _hit("a", 0.7), _hit("a", 0.6)]
    r = detect_subject(hits)
    assert r.subject_slug == "a"
    assert not r.ambiguous


def test_clear_winner() -> None:
    hits = [_hit("a", 0.9), _hit("a", 0.8), _hit("b", 0.1)]
    r = detect_subject(hits, margin_threshold=0.1)
    assert r.subject_slug == "a"
    assert not r.ambiguous


def test_ambiguous_when_close() -> None:
    hits = [_hit("a", 0.5), _hit("b", 0.48)]
    r = detect_subject(hits, margin_threshold=0.1)
    assert r.subject_slug == "a"
    assert r.ambiguous


def test_negative_scores_clamped() -> None:
    # an outlier with negative similarity shouldn't flip the vote
    hits = [_hit("a", 0.6), _hit("a", 0.5), _hit("b", -0.4)]
    r = detect_subject(hits)
    assert r.subject_slug == "a"
    ranked = dict(r.ranked)
    assert ranked["b"] == 0.0


def test_ranked_order() -> None:
    hits = [_hit("c", 0.2), _hit("a", 0.9), _hit("b", 0.4)]
    r = detect_subject(hits)
    slugs = [s for s, _ in r.ranked]
    assert slugs == ["a", "b", "c"]
