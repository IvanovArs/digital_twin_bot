"""Тесты search() / search_rerank() с подменённым индексом и mock-моделью."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.rag import config as rag_config
from src.rag import retriever as ret_mod


@pytest.fixture
def fake_index(tmp_path: Path, monkeypatch) -> tuple[np.ndarray, list[dict]]:
    """Кладёт фейковый embedding-индекс на диск и подсовывает его модулю."""
    chunks = [
        {"text": "стейкхолдер — это лицо", "subject_slug": "tos", "book": "a.pdf", "page": 1},
        {"text": "эмерджентность — свойство целого", "subject_slug": "tos", "book": "a.pdf", "page": 2},
        {"text": "интеграл — площадь под кривой", "subject_slug": "math", "book": "b.pdf", "page": 7},
    ]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chunks_file = tmp_path / "chunks.jsonl"
    embeds_file = tmp_path / "embeddings.npy"
    chunks_file.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    np.save(embeds_file, matrix)

    monkeypatch.setattr(rag_config, "CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(rag_config, "EMBEDDINGS_FILE", embeds_file)
    monkeypatch.setattr(ret_mod, "CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(ret_mod, "EMBEDDINGS_FILE", embeds_file)

    def _safe_clear() -> None:
        for fn in (ret_mod.load_chunks, ret_mod._load_index, ret_mod._encode_query_cached):
            cc = getattr(fn, "cache_clear", None)
            if cc is not None:
                cc()

    _safe_clear()
    yield matrix, chunks
    _safe_clear()


def _stub_encoder(monkeypatch, vec: np.ndarray) -> None:
    """Подменяет _encode_query_cached, минуя bge-m3 (тяжёлая модель)."""
    monkeypatch.setattr(ret_mod, "_encode_query_cached", lambda _q: vec.astype(np.float32))


def test_search_returns_top_match_by_cosine(fake_index, monkeypatch) -> None:
    matrix, chunks = fake_index
    _stub_encoder(monkeypatch, np.array([0.9, 0.1, 0.0]))
    out = ret_mod.search("стейкхолдер", k=1)
    assert len(out) == 1
    assert "стейкхолдер" in out[0].text


def test_search_filters_by_subject(fake_index, monkeypatch) -> None:
    _stub_encoder(monkeypatch, np.array([0.0, 0.0, 1.0]))
    out = ret_mod.search("интеграл", k=2, subject_slug="tos")
    # math-чанк отфильтрован, лучшие — из tos.
    assert all(h.subject_slug == "tos" for h in out)


def test_search_returns_empty_when_subject_has_no_chunks(fake_index, monkeypatch) -> None:
    _stub_encoder(monkeypatch, np.array([1.0, 0.0, 0.0]))
    out = ret_mod.search("q", k=2, subject_slug="ghost")
    assert out == []


def test_top_indices_orders_by_descending_similarity(fake_index, monkeypatch) -> None:
    matrix, chunks = fake_index
    sims = np.array([0.1, 0.9, 0.5])
    top = ret_mod._top_indices(sims, k=3, subject_slug=None, chunks=chunks)
    assert list(top) == [1, 2, 0]


def test_load_chunks_raises_clear_error_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ret_mod, "CHUNKS_FILE", tmp_path / "nope.jsonl")
    ret_mod.load_chunks.cache_clear()
    with pytest.raises(FileNotFoundError, match="Сначала"):
        ret_mod.load_chunks()
    ret_mod.load_chunks.cache_clear()


def test_load_index_raises_when_dimensions_mismatch(tmp_path, monkeypatch) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    embeds_file = tmp_path / "embeddings.npy"
    # 1 chunk но 2 embedding-строки → mismatch.
    chunks_file.write_text(
        json.dumps({"text": "x", "subject_slug": "s", "book": "b", "page": 1}),
        encoding="utf-8",
    )
    np.save(embeds_file, np.zeros((2, 3), dtype=np.float32))
    monkeypatch.setattr(ret_mod, "CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(ret_mod, "EMBEDDINGS_FILE", embeds_file)
    ret_mod.load_chunks.cache_clear()
    ret_mod._load_index.cache_clear()
    with pytest.raises(RuntimeError, match="Пересобери индекс"):
        ret_mod._load_index()
    ret_mod.load_chunks.cache_clear()
    ret_mod._load_index.cache_clear()


def test_mmr_select_picks_diverse_set() -> None:
    # Три почти параллельных вектора + один ортогональный → MMR должен
    # включить ортогональный во второе место несмотря на меньшую relevance.
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    relevance = np.array([0.95, 0.92, 0.90, 0.50], dtype=np.float32)
    picked = ret_mod._mmr_select(embeddings, relevance, k=2, lambda_=0.5)
    assert 0 in picked
    assert 3 in picked  # diversity bonus вытаскивает ортогональный


def test_mmr_handles_empty_input() -> None:
    out = ret_mod._mmr_select(np.empty((0, 2)), np.array([]), k=5)
    assert out == []


def test_search_rerank_uses_provided_scorer(fake_index, monkeypatch) -> None:
    """search_rerank на mock-реранкере: возвращает чанк с самым высоким
    rerank-score (после фильтра RERANK_MIN_SCORE)."""
    _stub_encoder(monkeypatch, np.array([1.0, 0.0, 0.0]))

    def stub_rerank(_q, texts):  # type: ignore[no-untyped-def]
        # Самый высокий score у того, где есть «эмерджентность».
        return [5.0 if "эмерджент" in t else 1.6 for t in texts]

    import sys
    import types

    fake_mod = types.ModuleType("src.rag.reranker")
    fake_mod.rerank = stub_rerank  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.rag.reranker", fake_mod)
    out = ret_mod.search_rerank("эмерджентность", k=1)
    assert len(out) == 1
    assert "эмерджент" in out[0].text
