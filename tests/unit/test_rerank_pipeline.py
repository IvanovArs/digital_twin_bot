"""Tests that the pipeline wires the reranker in front of the final top-k.

We don't exercise the real bge-reranker model (too heavy for a unit test);
instead we patch ``src.rag.reranker.rerank`` with a canned scorer and check
that (a) the pipeline calls it, (b) reordering respects the scores,
(c) the pipeline falls back gracefully when the reranker raises.
"""

from __future__ import annotations

import pytest

from src.rag import pipeline as pipeline_mod
from src.rag.retriever import Hit


def _hit(text: str, score: float, subject: str = "tos") -> Hit:
    return Hit(text=text, subject_slug=subject, book="x.pdf", page=1, score=score)


def test_rerank_final_respects_scorer(monkeypatch) -> None:
    """Cosine top-3 is A > B > C. The mocked reranker flips to C > A > B."""
    a = _hit("A", 0.9)
    b = _hit("B", 0.8)
    c = _hit("C", 0.7)
    pool = [a, b, c]

    def fake_rerank(_q: str, texts: list[str]) -> list[float]:
        mapping = {"A": 0.5, "B": 0.4, "C": 0.9}
        return [mapping[t] for t in texts]

    import src.rag.reranker as rr_module

    monkeypatch.setattr(rr_module, "rerank", fake_rerank)

    ordered = pipeline_mod._rerank_final("q", pool, k=3)
    assert [h.text for h in ordered] == ["C", "A", "B"]


def test_rerank_final_keeps_cosine_scores_on_hits(monkeypatch) -> None:
    """The Hit.score field must NOT be overwritten by the reranker — downstream
    gating relies on it being cosine similarity."""
    a = _hit("A", 0.91)
    b = _hit("B", 0.77)

    import src.rag.reranker as rr_module

    monkeypatch.setattr(rr_module, "rerank", lambda _q, _t: [0.1, 0.9])
    ordered = pipeline_mod._rerank_final("q", [a, b], k=2)
    assert ordered[0].score == 0.77  # B — cosine score preserved
    assert ordered[1].score == 0.91  # A


def test_rerank_final_falls_back_when_reranker_raises(monkeypatch) -> None:
    """A broken reranker must not kill the turn — degrade to cosine order."""
    a = _hit("A", 0.9)
    b = _hit("B", 0.8)

    def exploding_rerank(_q: str, _t: list[str]) -> list[float]:
        raise RuntimeError("model download failed")

    import src.rag.reranker as rr_module

    monkeypatch.setattr(rr_module, "rerank", exploding_rerank)
    ordered = pipeline_mod._rerank_final("q", [a, b], k=2)
    assert [h.text for h in ordered] == ["A", "B"]  # unchanged cosine order


def test_rerank_final_respects_k(monkeypatch) -> None:
    pool = [_hit(f"T{i}", 0.9 - i * 0.05) for i in range(10)]

    import src.rag.reranker as rr_module

    monkeypatch.setattr(rr_module, "rerank", lambda _q, texts: list(range(len(texts))))
    top3 = pipeline_mod._rerank_final("q", pool, k=3)
    assert len(top3) == 3


def test_rerank_final_noop_on_single_hit() -> None:
    a = _hit("solo", 0.5)
    # Even without patching — the helper short-circuits for len<=1.
    assert pipeline_mod._rerank_final("q", [a], k=5) == [a]


def test_rerank_final_noop_on_empty() -> None:
    assert pipeline_mod._rerank_final("q", [], k=5) == []


@pytest.mark.asyncio
async def test_resolve_subject_with_explicit_slug_applies_rerank(monkeypatch) -> None:
    """When the user pins a subject slug, the retriever returns a pool and
    the pipeline reranks it before slicing to top-k."""
    from src.rag import pipeline

    hits_pool = [
        _hit("noise 1", 0.91, subject="tos"),
        _hit("the real answer about системах", 0.88, subject="tos"),
        _hit("noise 2", 0.82, subject="tos"),
    ]

    def fake_search(q, k=None, subject_slug=None):
        return hits_pool

    # The pipeline now uses search_hybrid (BM25 + dense fused). Patch both
    # names the pipeline might bind to so the stubbed pool reaches the
    # reranker regardless of import path.
    monkeypatch.setattr(pipeline, "search_hybrid", fake_search)
    monkeypatch.setattr(pipeline_mod, "search_hybrid", fake_search)

    import src.rag.reranker as rr_module

    # Only the 2nd chunk is actually on-topic.
    monkeypatch.setattr(rr_module, "rerank", lambda _q, t: [0.1, 0.9, 0.2])

    # Stub catalog.require / detect_subject so we don't need a real catalog.
    class _Subj:
        slug = "tos"

    monkeypatch.setattr(
        pipeline, "_catalog", lambda: type("C", (), {"require": lambda self, _s: _Subj()})()
    )
    monkeypatch.setattr(
        pipeline, "detect_subject", lambda hits: type("R", (), {"subject_slug": "tos"})()
    )

    subj, hits, _ = pipeline.resolve_subject("вопрос", explicit_slug="tos", k=3)
    assert subj is not None
    assert hits[0].text == "the real answer about системах"
