"""End-to-end RAG-пайплайн: вопрос → retrieval → subject-routing → LLM → ответ.

Использование из кода:

    from src.rag.pipeline import ask

    result = ask("что такое логарифм?")        # автоматический роутинг
    result = ask("что такое система?", subject_slug="theory_of_systems")

CLI-entrypoint — `src.rag.cli_ask`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.rag.config import COURSES_YAML, RERANK_MIN_SCORE, RERANK_POOL, TOP_K
from src.rag.hybrid import _fingerprint, _fingerprint_index, search_hybrid
from src.rag.llm import chat
from src.rag.prompts import build_messages
from src.rag.retriever import Hit, _load_index, _mmr_select, replace, search
from src.rag.router import RouteResult, detect_subject
from src.subjects import Catalog, Subject, load_catalog


@dataclass(frozen=True)
class AskResult:
    answer: str
    subject: Subject | None
    hits: list[Hit]
    # Может быть None для коротких путей (FAQ/glossary/disambig/web), где
    # роутер вообще не задействован. Полный RAG-поток всегда заполняет.
    route: RouteResult | None = None


@lru_cache(maxsize=1)
def _catalog() -> Catalog:
    return load_catalog(Path(COURSES_YAML))


def _rerank_final(query: str, pool: list[Hit], k: int) -> list[Hit]:
    """Cross-encoder rescore on ``pool``, drop noise, then MMR for diversity.

    Returns hits keeping their original cosine score (callers gate on it for
    MIN_TOP_SCORE), only the ORDERING changes — driven by reranker logits and
    a final greedy MMR pass that pushes near-duplicate chunks (same page,
    same paragraph) out of TOP_K. Quietly no-ops to cosine order if the
    reranker can't load (offline first run, OOM, etc.).
    """
    if len(pool) <= 1:
        return pool[:k]
    try:
        from src.rag.reranker import rerank as _rerank_fn
    except Exception:
        return pool[:k]
    try:
        scores = np.asarray(_rerank_fn(query, [h.text for h in pool]), dtype=np.float32)
    except Exception:
        # Network hiccup downloading weights, OOM, etc. — degrade gracefully.
        return pool[:k]

    # Drop chunks below RERANK_MIN_SCORE — bge-reranker-v2-m3 emits logits
    # where ~+1.5 is the empirical "I'm reasonably sure this is relevant"
    # cutoff on RU; below that the model is guessing. If everything fails
    # the gate, return the single least-bad chunk so downstream's MIN_TOP_SCORE
    # can decide between answering weakly and falling through to web.
    keep_mask = scores >= RERANK_MIN_SCORE
    if not keep_mask.any():
        best = int(np.argmax(scores))
        return [pool[best]]
    kept_pool = [pool[i] for i, k_flag in enumerate(keep_mask) if k_flag]
    kept_scores = scores[keep_mask]

    # MMR over the kept set — 5 chunks from the same page were eating TOP_K.
    # Resolve chunk → embedding via the lazy fingerprint map shared with
    # hybrid.py (md5 of full body), so it's O(pool) instead of the linear
    # scan it used to be.
    try:
        matrix, _ = _load_index()
        fp_idx = _fingerprint_index()
        embed_rows: list[np.ndarray] = []
        for h in kept_pool:
            idx = fp_idx.get(_fingerprint(h.text))
            if idx is None:
                raise KeyError("chunk_not_in_index")
            embed_rows.append(matrix[idx])
        kept_embeds = np.stack(embed_rows)
    except Exception:
        scored = sorted(
            zip(kept_pool, kept_scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [h for h, _ in scored[:k]]

    picked = _mmr_select(kept_embeds, kept_scores, k=k)
    return [kept_pool[i] for i in picked]


def resolve_subject(
    question: str,
    explicit_slug: str | None,
    *,
    k: int = TOP_K,
) -> tuple[Subject | None, list[Hit], RouteResult]:
    """Найти релевантные хиты и решить, по какому предмету отвечаем.

    Если задан ``explicit_slug`` — поиск скоупится на этот предмет, роутер
    только подтверждает. Иначе сначала один wide-cosine-pass выбирает
    предмет, потом реранкер bge-reranker-v2-m3 переcортирует пул победителя
    по реальной query-vs-chunk-релевантности — bi-encoder-cosine семантичен,
    но грубоват; cross-encoder различает near-synonyms, которые все
    выглядят «достаточно близко» в embedding-пространстве и иначе смешались бы.

    Реранкер крутится в lru-кэшированном singleton'е (см. ``reranker.py``):
    первая query платит 300–500 мс на загрузку модели, дальше каждый
    вызов — ~30 мс на CPU для пула в 20 чанков. Оборачиваем в try/except —
    если веса реранкера недоступны, градуально падаем в чистый cosine-order,
    не валя всю turn'у.
    """
    catalog = _catalog()

    if explicit_slug is not None:
        subj = catalog.require(explicit_slug)
        # Hybrid-pool: dense-семантика + BM25-лексика, мерджим через RRF.
        # Cross-encoder реранкер делает финальный cut по merged-set'у — чанк,
        # пробивший только один сигнал, всё равно может стать топом, если
        # реранкер согласен.
        pool = search_hybrid(question, k=max(k, RERANK_POOL), subject_slug=explicit_slug)
        route = detect_subject(pool)
        hits = _rerank_final(question, pool, k=k)
        return subj, hits, route

    # Один широкий cosine-pass — выбирает предмет голосованием по ~2k-пулу.
    # Subject-routing остаётся на чистом dense-cosine: BM25 слишком
    # чувствителен к редким токенам, голосование становится нестабильным —
    # один чанк с совпавшей фамилией может перевесить предмет с реальным
    # семантическим покрытием.
    wide_hits = search(question, k=max(k * 4, 20))
    route = detect_subject(wide_hits)
    if route.subject_slug is None:
        return None, [], route

    subject = catalog.get(route.subject_slug)
    # Когда предмет зафиксирован — подключаем BM25 в пул для реранка.
    pool = search_hybrid(question, k=max(k, RERANK_POOL), subject_slug=route.subject_slug)
    hits = _rerank_final(question, pool, k=k)
    return subject, hits, route


# Re-export, чтобы старые call-site'ы с ``from src.rag.pipeline import replace``
# продолжали работать. ``replace`` — алиас на dataclass-replace из retriever.Hit,
# полезен для сборки синтетических Hit-копий в тестах.
__all__ = ["AskResult", "ask", "replace", "resolve_subject"]


def ask(
    question: str,
    *,
    subject_slug: str | None = None,
    k: int = TOP_K,
    llm_override: dict[str, object] | None = None,
) -> AskResult:
    """Ответить на ``question``, опционально ограниченный ``subject_slug``.

    ``llm_override`` пробрасывается в LLM-клиент (например, {"temperature": 0.0}).
    """
    subject, hits, route = resolve_subject(question, subject_slug, k=k)
    if not hits or subject is None:
        return AskResult(
            answer="В материалах курсов нет ответа на этот вопрос. Уточните у преподавателя.",
            subject=subject,
            hits=[],
            route=route,
        )

    messages = build_messages(question, hits, subject)
    extra: dict[str, object] = llm_override or {}
    answer = chat(messages, **extra)  # type: ignore[arg-type]
    return AskResult(answer=answer, subject=subject, hits=hits, route=route)
