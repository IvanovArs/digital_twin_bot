"""BM25 + dense retrieval fused with RRF (Cormack et al., 2009).

BM25 catches rare lexical hits (surnames, GOST numbers) that cosine
normalises away; dense catches paraphrases BM25 misses. Fusion is by
rank, not score, since BM25 and cosine aren't on the same scale.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.retriever import (
    Hit,
    _encode_query,
    _load_index,
    load_chunks,
    search,
)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

# Лёгкий стриппер русских суффиксов — тот же класс, что в glossary expander.
# Достаточно, чтобы «стейкхолдер», «стейкхолдера», «стейкхолдеры»,
# «стейкхолдеров» сводились к одному BM25-токену — лексический поиск
# перестаёт промахиваться по словоформам. Работает ~1 мкс/токен; pymorphy2
# добавил бы ~80 МБ и GPL-зависимость.
_RU_SUFFIX = re.compile(
    r"(ами|ями|ыми|ими|ого|ему|ому|ыми|ого|ой|ою|ую|ев|ов|ах|ях|ем|ой|"
    r"ие|ия|ии|ой|ей|ью|ей|ь|ы|и|а|я|е|у|ю|й|й)$"
)
_EN_SUFFIX = re.compile(r"(ing|ed|ly|es|s)$")


def _stem(tok: str) -> str:
    # Не трогаем токены короче 4 символов — false positives доминируют.
    if len(tok) <= 3:
        return tok
    # Цифры и акронимы оставляем как есть.
    if tok[0].isdigit() or tok.isupper():
        return tok.lower()
    low = tok.lower()
    if _RU_SUFFIX.search(low):
        return _RU_SUFFIX.sub("", low) or low
    if _EN_SUFFIX.search(low):
        return _EN_SUFFIX.sub("", low) or low
    return low


def _tokenise(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text or "")]


@lru_cache(maxsize=1)
def _bm25() -> tuple[BM25Okapi | None, list[dict[str, object]]]:
    """Строим BM25-индекс один раз на процесс. Шарим ``load_chunks`` с
    dense-retriever'ом — chunks.jsonl парсится один раз, на корпусе 15k
    это экономит ~80 МБ дублированного JSON в памяти.

    Возвращает ``(None, [])`` если индекса нет (свежий deploy до
    ``python -m src.rag.ingest``). Caller'ы трактуют это как пустой
    результат и уходят в web-fallback, а не валят всю turn'у.
    """
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        return None, []
    tokenised = [_tokenise(str(c["text"])) for c in chunks]
    return BM25Okapi(tokenised), chunks


@lru_cache(maxsize=1)
def _fingerprint_index() -> dict[str, int]:
    """Карта ``_fingerprint(text) → chunk_idx`` — BM25-only хиты находят свою
    cosine-строку за O(1) вместо линейного скана. Строится один раз на процесс."""
    _, chunks = _bm25()
    return {_fingerprint(str(c["text"])): i for i, c in enumerate(chunks)}


def search_bm25(
    query: str,
    *,
    k: int = 20,
    subject_slug: str | None = None,
) -> list[Hit]:
    """Top-k BM25-хиты. Score — сырой BM25, несопоставим с cosine."""
    bm25, chunks = _bm25()
    if bm25 is None or not chunks:
        return []  # индекса ещё нет — мягкий degrade до dense-only
    q_tokens = _tokenise(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    if subject_slug is not None:
        mask = [i for i, c in enumerate(chunks) if c["subject_slug"] == subject_slug]
        if not mask:
            return []
        ranked = sorted(mask, key=lambda i: -scores[i])[:k]
    else:
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    out: list[Hit] = []
    for i in ranked:
        c = chunks[i]
        out.append(
            Hit(
                text=str(c["text"]),
                subject_slug=str(c["subject_slug"]),
                book=str(c["book"]),
                page=int(c["page"]),
                score=float(scores[i]),
            )
        )
    return out


def _fingerprint(text: str) -> str:
    """Стабильный identity-key для RRF-дедупа.

    Раньше брали первые 200 символов — дёшево, но коллизия на главах
    учебника с общим intro-boilerplate'ом («Глава 3. Системный анализ.
    Введение. В данной главе рассматривается…»). Две разных главы
    мержились в одну, cosine-score одной молча перезаписывал другой.
    Полный md5 от тела устраняет коллизию; стоимость ~1 мкс на чанк.
    """
    return hashlib.md5((text or "").encode("utf-8"), usedforsecurity=False).hexdigest()


def _rrf_ranks(*rankings: list[str], rrf_k: float = 60.0) -> dict[str, float]:
    """Reciprocal Rank Fusion. Стандартный k=60 из TREC-paper'а — устойчив
    по доменам, тюнить под наш масштаб смысла нет."""
    agg: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            agg[key] = agg.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return agg


def _cosine_for_chunk(query_vec: np.ndarray, matrix: np.ndarray, idx: int) -> float:
    """Cosine similarity для одного чанка прямой матричной выборкой.

    Дёшево — один dot-product 1024-d векторов. Нужно для хитов, пришедших
    только из BM25: downstream-гейт ``MIN_TOP_SCORE`` ожидает реальный
    cosine на каждом возвращённом хите.
    """
    return float(matrix[idx] @ query_vec)


def search_hybrid(
    query: str,
    *,
    k: int = 20,
    subject_slug: str | None = None,
) -> list[Hit]:
    """BM25 + dense, объединение через RRF, возвращает top-k Hit'ов.

    Каждый Hit несёт настоящий cosine-``score`` (не BM25 и не RRF), поэтому
    существующий ``MIN_TOP_SCORE``-гейт и fallback-логика работают без правок.

    Если задан ``subject_slug`` — оба retriever'а скоупятся на этот предмет.
    """
    dense_hits = search(query, k=k, subject_slug=subject_slug)
    bm25_hits = search_bm25(query, k=k, subject_slug=subject_slug)

    # Fast-path: одна сторона пустая → RRF не нужен.
    if not bm25_hits:
        return dense_hits
    if not dense_hits:
        # Dense промахнулся, BM25 нашёл. Обогащаем BM25-хиты cosine-score'ом,
        # чтобы downstream-гейт работал. O(1)-lookup через preуd fingerprint-кэш.
        try:
            matrix, _ = _load_index()
        except FileNotFoundError:
            return bm25_hits[:k]  # cosine не посчитать — возвращаем BM25 как есть
        q_vec = _encode_query(query)
        fp_index = _fingerprint_index()
        from dataclasses import replace

        enriched: list[Hit] = []
        for h in bm25_hits:
            idx = fp_index.get(_fingerprint(h.text))
            if idx is None:
                enriched.append(h)
                continue
            enriched.append(replace(h, score=_cosine_for_chunk(q_vec, matrix, idx)))
        return enriched[:k]

    # Обычный путь: RRF merge. Сохраняем cosine от dense как Hit.score
    # эталонный; только BM25-only чанкам считаем cosine заново.
    dense_by_fp = {_fingerprint(h.text): h for h in dense_hits}
    bm25_by_fp = {_fingerprint(h.text): h for h in bm25_hits}

    dense_ranking = [_fingerprint(h.text) for h in dense_hits]
    bm25_ranking = [_fingerprint(h.text) for h in bm25_hits]
    fused = _rrf_ranks(dense_ranking, bm25_ranking)

    # Cosine для BM25-only-чанков считаем лениво. Fingerprint→idx-кэш
    # собран один раз на процесс, поэтому lookup — O(1) даже на корпусе 100k.
    cached_q: np.ndarray | None = None
    cached_matrix: np.ndarray | None = None
    cached_fp_index: dict[str, int] | None = None

    def _cosine_for(fp: str, bm25_hit: Hit) -> float:
        nonlocal cached_q, cached_matrix, cached_fp_index
        if cached_q is None:
            try:
                cached_matrix, _ = _load_index()
            except FileNotFoundError:
                return 0.0
            cached_q = _encode_query(query)
            cached_fp_index = _fingerprint_index()
        assert cached_fp_index is not None and cached_matrix is not None
        idx = cached_fp_index.get(fp)
        if idx is None:
            return 0.0
        return _cosine_for_chunk(cached_q, cached_matrix, idx)

    merged: list[Hit] = []
    from dataclasses import replace

    for fp, _rrf_score in sorted(fused.items(), key=lambda x: -x[1]):
        if fp in dense_by_fp:
            merged.append(dense_by_fp[fp])
        else:
            # BM25-only: upgrade score from raw BM25 to cosine so the
            # MIN_TOP_SCORE gate speaks the same language across hits.
            bm = bm25_by_fp[fp]
            merged.append(replace(bm, score=_cosine_for(fp, bm)))
        if len(merged) >= k:
            break
    return merged
