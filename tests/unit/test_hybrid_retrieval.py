"""Tests for the hybrid BM25 + dense retrieval + RRF module.

We don't want to exercise the real bge-m3 + full chunks.jsonl here, so we
patch ``search`` / ``search_bm25`` / ``_load_index`` / ``_encode_query``
with stubs and check:
* RRF correctly fuses two rankings.
* Dense-only chunks keep cosine score.
* BM25-only chunks get a freshly-computed cosine so the gate keeps working.
* Empty-side fast paths are correct.
"""

from __future__ import annotations

import numpy as np

from src.rag import hybrid
from src.rag.retriever import Hit


def _hit(text: str, score: float, book: str = "a.pdf") -> Hit:
    return Hit(text=text, subject_slug="tos", book=book, page=1, score=score)


# ---------- _tokenise ----------


def test_tokeniser_handles_mixed_alphabet() -> None:
    toks = hybrid._tokenise("Фриман 1984, Freeman's ontology")
    assert "фриман" in toks
    assert "1984" in toks
    assert "freeman" in toks
    assert "ontology" in toks


def test_tokeniser_drops_punct_and_spaces() -> None:
    toks = hybrid._tokenise("a , b — c; d.e")
    assert toks == ["a", "b", "c", "d", "e"]


# ---------- _rrf_ranks ----------


def test_rrf_prefers_items_ranked_high_in_both_lists() -> None:
    r1 = ["A", "B", "C"]
    r2 = ["C", "A", "D"]
    fused = hybrid._rrf_ranks(r1, r2)
    ordered = sorted(fused.items(), key=lambda x: -x[1])
    # A is top-2 in both → wins. C is top-1 in r2 and top-3 in r1.
    assert ordered[0][0] == "A"
    # D is bottom — lowest score.
    assert ordered[-1][0] == "D"


def test_rrf_monotonic_in_rank() -> None:
    """Higher rank positions get less weight."""
    fused = hybrid._rrf_ranks(["A", "B", "C"])
    assert fused["A"] > fused["B"] > fused["C"]


# ---------- search_hybrid ----------


def test_search_hybrid_returns_dense_only_when_bm25_empty(monkeypatch) -> None:
    dense = [_hit("x", 0.9), _hit("y", 0.8)]
    monkeypatch.setattr(hybrid, "search", lambda *a, **kw: dense)
    monkeypatch.setattr(hybrid, "search_bm25", lambda *a, **kw: [])
    out = hybrid.search_hybrid("q", k=5)
    assert out == dense


def test_search_hybrid_enriches_bm25_only_hits_with_cosine(monkeypatch) -> None:
    """When dense finds nothing, BM25 hits are returned with a fresh
    cosine score (not the raw BM25 number)."""
    bm25 = [_hit("a", 5.2), _hit("b", 3.1)]  # BM25 scores
    monkeypatch.setattr(hybrid, "search", lambda *a, **kw: [])
    monkeypatch.setattr(hybrid, "search_bm25", lambda *a, **kw: bm25)
    # Stub the embedding infra: matrix row dot query = [0.77, 0.42].
    fake_matrix = np.array([[1.0], [0.5]], dtype=np.float32)
    fake_q = np.array([0.77], dtype=np.float32)
    fake_chunks = [
        {"text": "a", "subject_slug": "tos", "book": "b", "page": 1},
        {"text": "b", "subject_slug": "tos", "book": "b", "page": 1},
    ]
    monkeypatch.setattr(hybrid, "_load_index", lambda: (fake_matrix, fake_chunks))
    monkeypatch.setattr(hybrid, "_encode_query", lambda _q: fake_q)
    monkeypatch.setattr(hybrid, "_bm25", lambda: (None, fake_chunks))
    out = hybrid.search_hybrid("q", k=5)
    assert len(out) == 2
    # Scores were upgraded from raw BM25 to real cosine (lookup dot product).
    assert abs(out[0].score - 0.77) < 1e-6
    assert abs(out[1].score - 0.77 * 0.5) < 1e-6


def test_search_hybrid_rrf_merges_both_rankings(monkeypatch) -> None:
    """A chunk in both rankings should beat chunks that only one side
    has — the core RRF payoff."""
    common = _hit("common-text shared by both", 0.7)
    dense_only = _hit("dense-only", 0.8)
    bm25_only = _hit("bm25-only", 4.2)
    dense = [dense_only, common]
    bm25 = [common, bm25_only]

    monkeypatch.setattr(hybrid, "search", lambda *a, **kw: dense)
    monkeypatch.setattr(hybrid, "search_bm25", lambda *a, **kw: bm25)
    # Stub cosine lookup for the BM25-only chunk.
    fake_matrix = np.array([[1.0]], dtype=np.float32)
    fake_q = np.array([0.55], dtype=np.float32)
    fake_chunks = [{"text": "bm25-only", "subject_slug": "tos", "book": "b", "page": 1}]
    monkeypatch.setattr(hybrid, "_load_index", lambda: (fake_matrix, fake_chunks))
    monkeypatch.setattr(hybrid, "_encode_query", lambda _q: fake_q)
    monkeypatch.setattr(hybrid, "_bm25", lambda: (None, fake_chunks))

    out = hybrid.search_hybrid("q", k=3)
    assert next(h.text for h in out) == "common-text shared by both"
    assert len(out) == 3


def test_search_hybrid_preserves_dense_cosine_scores(monkeypatch) -> None:
    """A chunk that came from dense must carry its cosine score unchanged —
    the MIN_TOP_SCORE gate in the pipeline relies on that."""
    h_a = _hit("A", 0.93)
    h_b = _hit("B", 0.81)
    pool = [h_a, h_b]
    monkeypatch.setattr(hybrid, "search", lambda *args, **kw: pool)
    monkeypatch.setattr(hybrid, "search_bm25", lambda *args, **kw: pool)
    out = hybrid.search_hybrid("q", k=2)
    scores_by_text = {h.text: h.score for h in out}
    assert abs(scores_by_text["A"] - 0.93) < 1e-9
    assert abs(scores_by_text["B"] - 0.81) < 1e-9
