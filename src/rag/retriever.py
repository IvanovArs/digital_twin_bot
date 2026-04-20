"""Top-k semantic search over the unified RAG index.

Two-stage retrieval:
  1. ``search()`` — bi-encoder (bge-m3) cosine over a pool of RERANK_POOL
     candidates (cheap, wide).
  2. ``search_rerank()`` — cross-encoder (bge-reranker-v2-m3) rescores the
     pool, then MMR picks a diverse TOP_K for the LLM context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.rag.config import (
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    MMR_LAMBDA,
    RERANK_MIN_SCORE,
    RERANK_POOL,
    TOP_K,
)


@dataclass(frozen=True)
class Hit:
    text: str
    subject_slug: str
    book: str
    page: int
    score: float


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # dtype=float16 + low_cpu_mem_usage=True: HF оборачивает создание модуля
    # в set_default_dtype(float16), поэтому nn.Embedding(250002, 1024)
    # аллоцирует ~500 МБ вместо ~1 ГБ. На Windows с почти заполненной RAM
    # (llama-server + другие Python-процессы) без этого ловили MemoryError на
    # torch.empty(...) на старте. fp16 не бьёт качество retrieval: cosine
    # по L2-нормированным векторам устойчив к fp16-округлению.
    return SentenceTransformer(
        EMBEDDING_MODEL,
        model_kwargs={"dtype": torch.float16, "low_cpu_mem_usage": True},
    )


@lru_cache(maxsize=1)
def load_chunks() -> list[dict[str, object]]:
    """Single source of truth for the chunk table.

    Previously ``_load_index`` and ``hybrid._bm25`` each parsed
    chunks.jsonl separately — ~80 MB duplicated in memory on an 15 k-chunk
    corpus. Making one ``load_chunks()`` the shared loader drops that.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Индекс не найден. Сначала: python -m src.rag.ingest\n"
            f"Ожидался файл: {CHUNKS_FILE}"
        )
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def _load_index() -> tuple[np.ndarray, list[dict[str, object]]]:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Эмбеддинги не найдены. Сначала: python -m src.rag.ingest\n"
            f"Ожидался файл: {EMBEDDINGS_FILE}"
        )
    chunks = load_chunks()
    matrix = np.load(EMBEDDINGS_FILE)
    if matrix.shape[0] != len(chunks):
        raise RuntimeError(
            f"Размер эмбеддингов ({matrix.shape[0]}) ≠ числу чанков ({len(chunks)}). "
            "Пересобери индекс: python -m src.rag.ingest"
        )
    return matrix, chunks


@lru_cache(maxsize=128)
def _encode_query_cached(query: str) -> np.ndarray:
    """Cached variant — `resolve_subject` calls `_encode_query` twice on
    the same question (wide pass + hybrid pass). With an LRU we skip the
    second bge-m3 forward (~50 ms each). Cache is keyed by exact query
    text, which is fine because `expand_query` is deterministic."""
    from src.rag.glossary import expand_query

    expanded = expand_query(query)
    encoded = (
        _model()
        .encode([expanded], normalize_embeddings=True, convert_to_numpy=True)
        .astype(np.float32)[0]
    )
    return np.asarray(encoded)


def _encode_query(query: str) -> np.ndarray:
    # Glossary-based expansion: short questions like «что такое X» are
    # cosine-penalised against fat 200-token chunks. Appending the known
    # definition of X concentrates content tokens and lifts top_score
    # measurably without changing the index. No-op when the term is unknown.
    return _encode_query_cached(query)


def _top_indices(
    sims: np.ndarray, k: int, subject_slug: str | None, chunks: list[dict[str, object]]
) -> np.ndarray:
    """Return absolute chunk indices of the top-k similarities, sorted desc."""
    if subject_slug is not None:
        mask = np.array([c["subject_slug"] == subject_slug for c in chunks], dtype=bool)
        if not mask.any():
            return np.empty(0, dtype=np.int64)
        cand = np.where(mask)[0]
        cand_sims = sims[cand]
        k_actual = min(k, len(cand))
        top_local = np.argpartition(-cand_sims, k_actual - 1)[:k_actual]
        top_local = top_local[np.argsort(-cand_sims[top_local])]
        return cand[top_local]
    k_actual = min(k, len(sims))
    top = np.argpartition(-sims, k_actual - 1)[:k_actual]
    return top[np.argsort(-sims[top])]


def _hit_from_chunk(chunk: dict[str, object], score: float) -> Hit:
    page = chunk["page"]
    return Hit(
        text=str(chunk["text"]),
        subject_slug=str(chunk["subject_slug"]),
        book=str(chunk["book"]),
        page=int(page) if isinstance(page, int | str) else 0,
        score=score,
    )


def search(query: str, k: int = TOP_K, subject_slug: str | None = None) -> list[Hit]:
    """Return top-k hits by cosine similarity, optionally restricted to a subject."""
    matrix, chunks = _load_index()
    if matrix.size == 0:
        return []

    q = _encode_query(query)
    sims = matrix @ q  # all vectors are L2-normalized → cosine similarity
    top = _top_indices(sims, k, subject_slug, chunks)
    return [_hit_from_chunk(chunks[int(i)], float(sims[i])) for i in top]


def _mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[int]:
    """Greedy MMR: pick k indices balancing relevance and diversity.

    Assumes rows of ``embeddings`` are L2-normalized so similarity = dot.
    """
    n = len(relevance)
    if n == 0:
        return []
    k = min(k, n)
    picked: list[int] = []
    remaining = set(range(n))
    # First pick is always the most relevant — MMR becomes trivial otherwise.
    first = int(np.argmax(relevance))
    picked.append(first)
    remaining.remove(first)

    while len(picked) < k and remaining:
        picked_mat = embeddings[picked]  # [p, D]
        best_idx = -1
        best_score = -np.inf
        for i in remaining:
            diversity_penalty = float(np.max(picked_mat @ embeddings[i]))
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        picked.append(best_idx)
        remaining.remove(best_idx)
    return picked


def search_rerank(
    query: str,
    k: int = TOP_K,
    subject_slug: str | None = None,
    *,
    pool: int = RERANK_POOL,
) -> list[Hit]:
    """Two-stage retrieval: bi-encoder pool → cross-encoder rerank → MMR.

    The returned hits' ``score`` field still carries **cosine similarity** so
    existing callers that gate on MIN_TOP_SCORE (calibrated to cosine) keep
    working. The reranker score only drives ORDERING of the returned list.
    """
    # Import locally so a retriever-only consumer (e.g. cli_search) doesn't
    # pay the reranker cold-start if it never calls this function.
    from src.rag.reranker import rerank as _rerank

    matrix, chunks = _load_index()
    if matrix.size == 0:
        return []

    q = _encode_query(query)
    sims = matrix @ q
    pool_idx = _top_indices(sims, pool, subject_slug, chunks)
    if pool_idx.size == 0:
        return []

    pool_texts = [str(chunks[int(i)]["text"]) for i in pool_idx]
    rr_scores = np.asarray(_rerank(query, pool_texts), dtype=np.float32)

    # Drop obvious noise before spending an MMR step on it. If nothing passes
    # the gate we still return the single best chunk — the caller applies its
    # own MIN_TOP_SCORE check on the cosine score for the final decision.
    keep = rr_scores >= RERANK_MIN_SCORE
    if not keep.any():
        best = int(np.argmax(rr_scores))
        return [_hit_from_chunk(chunks[int(pool_idx[best])], float(sims[pool_idx[best]]))]

    filtered_idx = pool_idx[keep]
    filtered_rr = rr_scores[keep]
    filtered_embeds = matrix[filtered_idx]

    picked_local = _mmr_select(filtered_embeds, filtered_rr, k=k)
    return [
        _hit_from_chunk(chunks[int(filtered_idx[i])], float(sims[filtered_idx[i]]))
        for i in picked_local
    ]


__all__ = ["Hit", "replace", "search", "search_rerank"]
