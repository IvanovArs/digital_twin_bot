"""End-to-end RAG pipeline: question → retrieval → subject routing → LLM → answer.

Usage from code:

    from src.rag.pipeline import ask

    result = ask("что такое логарифм?")        # auto-route
    result = ask("что такое система?", subject_slug="theory_of_systems")

See `src.rag.cli_ask` for a command-line entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.rag.config import COURSES_YAML, RERANK_POOL, TOP_K
from src.rag.hybrid import search_hybrid
from src.rag.llm import chat
from src.rag.prompts import build_messages
from src.rag.retriever import Hit, replace, search
from src.rag.router import RouteResult, detect_subject
from src.subjects import Catalog, Subject, load_catalog


@dataclass(frozen=True)
class AskResult:
    answer: str
    subject: Subject | None
    hits: list[Hit]
    route: RouteResult


@lru_cache(maxsize=1)
def _catalog() -> Catalog:
    return load_catalog(Path(COURSES_YAML))


def _rerank_final(query: str, pool: list[Hit], k: int) -> list[Hit]:
    """Apply the cross-encoder to ``pool`` and return the top-k by rerank
    score. The returned hits keep their original cosine score (callers gate
    on it for MIN_TOP_SCORE), only the ordering changes. Quietly no-ops if
    the reranker isn't installed/loadable — we fall back to cosine order.
    """
    if len(pool) <= 1:
        return pool[:k]
    try:
        from src.rag.reranker import rerank as _rerank_fn
    except Exception:
        return pool[:k]
    try:
        scores = _rerank_fn(query, [h.text for h in pool])
    except Exception:
        # Network hiccup downloading weights, OOM, etc. — degrade gracefully.
        return pool[:k]
    scored = sorted(zip(pool, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [h for h, _ in scored[:k]]


def resolve_subject(
    question: str,
    explicit_slug: str | None,
    *,
    k: int = TOP_K,
) -> tuple[Subject | None, list[Hit], RouteResult]:
    """Find relevant hits and decide which subject we're answering from.

    If ``explicit_slug`` is passed, the search is scoped to that subject
    and the router just confirms it. Otherwise a single wide cosine pass
    picks the subject first, then we re-rank the winning subject's pool
    with bge-reranker-v2-m3 to order the final top-k by true
    query-vs-chunk relevance — bi-encoder cosine is semantic but coarse;
    the cross-encoder distinguishes near-synonyms that all look "close
    enough" in embedding space and would otherwise mix in.

    The reranker runs in a lru-cached singleton (see ``reranker.py``), so
    first-query cold-start pays the 300–500 ms model load, then every
    subsequent call is ~30 ms on CPU for a 20-chunk pool. We wrap it in
    try/except — if the reranker weights are unavailable we degrade to
    plain cosine order rather than failing the whole turn.
    """
    catalog = _catalog()

    if explicit_slug is not None:
        subj = catalog.require(explicit_slug)
        # Hybrid pool: dense semantic + BM25 lexical, fused via RRF. The
        # cross-encoder reranker then takes the final cut from the merged
        # set — so a chunk that scored high only on one signal still has
        # a shot to be the top answer if the reranker agrees.
        pool = search_hybrid(
            question, k=max(k, RERANK_POOL), subject_slug=explicit_slug
        )
        route = detect_subject(pool)
        hits = _rerank_final(question, pool, k=k)
        return subj, hits, route

    # Single wide cosine pass first — picks the subject via majority vote
    # over a ~2k pool. Subject routing stays on dense cosine alone because
    # BM25 is too sensitive to rare tokens for a majority vote to be
    # stable: one chunk with a matching surname can outvote a subject with
    # genuine semantic coverage.
    wide_hits = search(question, k=max(k * 4, 20))
    route = detect_subject(wide_hits)
    if route.subject_slug is None:
        return None, [], route

    subject = catalog.get(route.subject_slug)
    # Once the subject is fixed, bring BM25 into the pool for the rerank.
    pool = search_hybrid(
        question, k=max(k, RERANK_POOL), subject_slug=route.subject_slug
    )
    hits = _rerank_final(question, pool, k=k)
    return subject, hits, route


# Re-export so call sites that previously did ``from src.rag.pipeline import
# replace`` keep working. ``replace`` is aliased from retriever.Hit's
# dataclass and useful for building synthetic Hit copies in tests.
__all__ = ["AskResult", "ask", "replace", "resolve_subject"]


def ask(
    question: str,
    *,
    subject_slug: str | None = None,
    k: int = TOP_K,
    llm_override: dict[str, object] | None = None,
) -> AskResult:
    """Answer ``question``, optionally constrained to ``subject_slug``.

    ``llm_override`` is forwarded to the LLM client (e.g. {"temperature": 0.0}).
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
