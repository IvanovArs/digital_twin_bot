"""Tests that reindex_subject_in_background clears the retrieval lru caches."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.bot.services import admin_service


def test_invalidate_retrieval_caches_calls_cache_clear(monkeypatch) -> None:
    """The function must walk the five lru-cached loaders and call
    ``cache_clear`` on each so no stale chunks survive a reindex."""
    from src.rag import hybrid as hybrid_mod
    from src.rag import retriever as retriever_mod

    calls: list[str] = []

    def _mk(name: str) -> MagicMock:
        m = MagicMock()
        m.cache_clear = MagicMock(side_effect=lambda: calls.append(name))
        return m

    monkeypatch.setattr(retriever_mod, "load_chunks", _mk("load_chunks"))
    monkeypatch.setattr(retriever_mod, "_load_index", _mk("_load_index"))
    monkeypatch.setattr(
        retriever_mod, "_encode_query_cached", _mk("_encode_query_cached")
    )
    monkeypatch.setattr(hybrid_mod, "_bm25", _mk("_bm25"))
    monkeypatch.setattr(hybrid_mod, "_fingerprint_index", _mk("_fingerprint_index"))

    admin_service._invalidate_retrieval_caches()
    assert set(calls) == {
        "load_chunks",
        "_load_index",
        "_encode_query_cached",
        "_bm25",
        "_fingerprint_index",
    }


def test_invalidate_ignores_targets_without_cache_clear(monkeypatch) -> None:
    """If a future refactor drops ``@lru_cache`` from one of the loaders,
    the invalidator should silently skip — not crash the reindex flow."""
    from src.rag import hybrid as hybrid_mod
    from src.rag import retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "load_chunks", lambda: None)  # plain fn
    monkeypatch.setattr(retriever_mod, "_load_index", MagicMock(cache_clear=MagicMock()))
    monkeypatch.setattr(
        retriever_mod, "_encode_query_cached", MagicMock(cache_clear=MagicMock())
    )
    monkeypatch.setattr(hybrid_mod, "_bm25", MagicMock(cache_clear=MagicMock()))
    monkeypatch.setattr(
        hybrid_mod, "_fingerprint_index", MagicMock(cache_clear=MagicMock())
    )

    # Must not raise.
    admin_service._invalidate_retrieval_caches()
