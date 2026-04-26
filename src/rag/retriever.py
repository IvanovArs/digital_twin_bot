"""Top-k семантический поиск по унифицированному RAG-индексу.

Двухстадийный retrieval:
  1. ``search()`` — bi-encoder (bge-m3) cosine по пулу из RERANK_POOL
     кандидатов (дёшево, широко).
  2. ``search_rerank()`` — cross-encoder (bge-reranker-v2-m3) rescore'ит
     пул, MMR выбирает диверсифицированный TOP_K для LLM-контекста.
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
    """Единый источник правды для chunk-таблицы.

    Раньше ``_load_index`` и ``hybrid._bm25`` парсили chunks.jsonl
    независимо — ~80 МБ дублированного JSON в памяти на корпусе 15k. Один
    ``load_chunks()`` как общий loader устраняет это.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Индекс не найден. Сначала: python -m src.rag.ingest\n" f"Ожидался файл: {CHUNKS_FILE}"
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
    """Кэшированная версия — `resolve_subject` вызывает `_encode_query`
    дважды на одном вопросе (wide-pass + hybrid-pass). LRU позволяет
    скипнуть второй bge-m3-forward (~50 мс). Ключ — точный текст запроса,
    что ок: `expand_query` детерминирован."""
    from src.rag.glossary import expand_query

    expanded = expand_query(query)
    encoded = (
        _model()
        .encode([expanded], normalize_embeddings=True, convert_to_numpy=True)
        .astype(np.float32)[0]
    )
    return np.asarray(encoded)


def _encode_query(query: str) -> np.ndarray:
    # Glossary-расширение: короткие вопросы вроде «что такое X» проигрывают
    # cosine'ом жирным 200-токеновым чанкам. Если приклеить известное
    # определение X — content-токены концентрируются, top_score заметно
    # растёт без перестройки индекса. No-op если термин неизвестен.
    return _encode_query_cached(query)


def _top_indices(
    sims: np.ndarray, k: int, subject_slug: str | None, chunks: list[dict[str, object]]
) -> np.ndarray:
    """Абсолютные индексы top-k сходств, отсортированы по убыванию."""
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
    """Top-k хитов по cosine similarity, опционально ограничено предметом."""
    matrix, chunks = _load_index()
    if matrix.size == 0:
        return []

    q = _encode_query(query)
    sims = matrix @ q  # все вектора L2-нормированы → dot = cosine similarity
    top = _top_indices(sims, k, subject_slug, chunks)
    return [_hit_from_chunk(chunks[int(i)], float(sims[i])) for i in top]


def _mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[int]:
    """Жадный MMR: выбирает k индексов, балансируя релевантность и разнообразие.

    Предполагает, что строки ``embeddings`` L2-нормированы (similarity = dot).
    """
    n = len(relevance)
    if n == 0:
        return []
    k = min(k, n)
    picked: list[int] = []
    remaining = set(range(n))
    # Первый — всегда самый релевантный, иначе MMR вырождается.
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
    """Двухстадийный retrieval: bi-encoder pool → cross-encoder rerank → MMR.

    Поле ``score`` возвращаемых хитов несёт **cosine similarity** — старые
    caller'ы с MIN_TOP_SCORE-гейтом (откалиброванным под cosine) продолжают
    работать. Реранкер влияет только на ПОРЯДОК.
    """
    # Импорт локальный — retriever-only consumer (например, cli_search) не
    # должен платить cold-start реранкера, если эта функция не зовётся.
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

    # Дропаем явный шум до MMR. Если никто не прошёл — возвращаем единственный
    # лучший чанк; caller сам применит MIN_TOP_SCORE по cosine для финального решения.
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
