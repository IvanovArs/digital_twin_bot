"""Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (RRF).

Why both:

* bge-m3 (dense, bi-encoder) is great on semantics — finds chunks that
  paraphrase the question. But it normalises away exact-token matches. A
  student asking «кто автор теории систем» doesn't pull the Berталанфи
  page up if the page phrases it as «один из основоположников общей теории
  систем» rather than «автор теории систем».

* BM25 (sparse, lexical) is the opposite: it thrives on rare exact tokens
  (surnames, years, acronyms, GOST numbers) that embedding cosine treats
  as noise. On «кто такой Фримен» it finds the exact page immediately.

Together, merged via Reciprocal Rank Fusion (Cormack et al., 2009), you
get the best of both — semantic coverage + literal precision. RRF ignores
raw scores (which aren't comparable between BM25 and cosine) and fuses
by *rank* alone, which is robust to scale differences.

One process-wide BM25 index is built lazily from ``chunks.jsonl``; it
shares its chunk order with ``retriever._load_index`` so we can map
indices both ways.
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

# Lightweight Russian suffix stripper — same class used in the glossary
# expander. Enough to align «стейкхолдер», «стейкхолдера», «стейкхолдеры»,
# «стейкхолдеров» under one BM25 token so lexical search stops missing
# inflected forms. Runs in ~1 µs per token; no extra dependency
# (pymorphy2 would add ~80 MB and a GPL licence surface).
_RU_SUFFIX = re.compile(
    r"(ами|ями|ыми|ими|ого|ему|ому|ыми|ого|ой|ою|ую|ев|ов|ах|ях|ем|ой|"
    r"ие|ия|ии|ой|ей|ью|ей|ь|ы|и|а|я|е|у|ю|й|й)$"
)
_EN_SUFFIX = re.compile(r"(ing|ed|ly|es|s)$")


def _stem(tok: str) -> str:
    # Don't strip tokens shorter than 4 chars — false positives dominate.
    if len(tok) <= 3:
        return tok
    # Keep digits and acronyms intact.
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
    """Build the BM25 index once per process. Shares ``load_chunks`` with
    the dense retriever so chunks.jsonl isn't parsed twice — on a 15 k
    corpus that saves ~80 MB of duplicated JSON in memory.

    Returns ``(None, [])`` if the index is missing (e.g., fresh deploy
    before ``python -m src.rag.ingest`` ran). Callers treat that as an
    empty result set and fall through to the web-fallback path instead
    of crashing the whole turn.
    """
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        return None, []
    tokenised = [_tokenise(str(c["text"])) for c in chunks]
    return BM25Okapi(tokenised), chunks


@lru_cache(maxsize=1)
def _fingerprint_index() -> dict[str, int]:
    """Map ``_fingerprint(text) → chunk_idx`` so BM25-only hits look up
    their cosine row in O(1) instead of scanning the whole chunks list.
    Builds once per process, after the chunk table is loaded."""
    _, chunks = _bm25()
    return {_fingerprint(str(c["text"])): i for i, c in enumerate(chunks)}


def search_bm25(
    query: str,
    *,
    k: int = 20,
    subject_slug: str | None = None,
) -> list[Hit]:
    """Top-k BM25 hits. Score is raw BM25, not comparable to cosine."""
    bm25, chunks = _bm25()
    if bm25 is None or not chunks:
        return []  # index not built yet — graceful degrade to dense-only
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
    """Stable identity key for RRF deduplication.

    Previously used the first 200 chars of ``text`` — cheap but collided
    on textbook chapters that share a common intro boilerplate («Глава 3.
    Системный анализ. Введение. В данной главе рассматривается…»). Two
    different chapters with the same opener would be merged and one
    chunk's cosine score would silently overwrite the other. A full md5
    of the chunk body eliminates the collision; the hash cost is
    negligible (~1 µs per chunk).
    """
    return hashlib.md5((text or "").encode("utf-8"), usedforsecurity=False).hexdigest()


def _rrf_ranks(*rankings: list[str], rrf_k: float = 60.0) -> dict[str, float]:
    """Reciprocal Rank Fusion. Standard k=60 from the TREC paper — robust
    across domains, not worth tuning for our scale."""
    agg: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            agg[key] = agg.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return agg


def _cosine_for_chunk(query_vec: np.ndarray, matrix: np.ndarray, idx: int) -> float:
    """Compute cosine similarity for a single chunk by direct matrix lookup.

    Cheap: it's one dot product of 1024-d vectors. We need this for hits
    that came from BM25 but not dense — ``MIN_TOP_SCORE`` gating downstream
    expects every returned hit to carry a real cosine score.
    """
    return float(matrix[idx] @ query_vec)


def search_hybrid(
    query: str,
    *,
    k: int = 20,
    subject_slug: str | None = None,
) -> list[Hit]:
    """Run BM25 + dense, fuse via RRF, return top-k Hit objects.

    Every returned Hit carries a real cosine ``score`` (not BM25 or RRF),
    so the existing ``MIN_TOP_SCORE`` gate and the reranker-independent
    fallback logic keep working unchanged.

    If ``subject_slug`` is passed, both retrievers are subject-scoped.
    """
    dense_hits = search(query, k=k, subject_slug=subject_slug)
    bm25_hits = search_bm25(query, k=k, subject_slug=subject_slug)

    # Fast path: one side is empty or identical → skip the RRF overhead.
    if not bm25_hits:
        return dense_hits
    if not dense_hits:
        # Dense missed entirely but BM25 found something. Enrich BM25 hits
        # with real cosine scores so the downstream gate works. O(1) idx
        # lookup via the prebuilt fingerprint cache.
        try:
            matrix, _ = _load_index()
        except FileNotFoundError:
            return bm25_hits[:k]  # can't compute cosine — return BM25 as-is
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

    # Normal path: RRF merge. Keep dense's cosine scores as the Hit.score
    # of record; only BM25-only chunks need a freshly computed cosine.
    dense_by_fp = {_fingerprint(h.text): h for h in dense_hits}
    bm25_by_fp = {_fingerprint(h.text): h for h in bm25_hits}

    dense_ranking = [_fingerprint(h.text) for h in dense_hits]
    bm25_ranking = [_fingerprint(h.text) for h in bm25_hits]
    fused = _rrf_ranks(dense_ranking, bm25_ranking)

    # Resolve cosine for any BM25-only chunks lazily. The fingerprint→idx
    # cache is prebuilt once per process, so lookups stay O(1) even on a
    # 100 k-chunk corpus.
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
