"""In-memory кэш, чтобы follow-up-кнопки скипали retrieval.

Когда ``run_qa_pipeline`` отрабатывает, кладём ``hits``/``web_hits`` и
немного routing-метаданных под новый ``dialog_id``. Тап по
«Проще / Пример / Подробнее» забирает тот же контекст и переcпускает LLM
с prompt-модификатором — без второго bge-m3-pass, без второго web-search;
follow-up прилетает за ~2× streaming-latency вместо ~2×retrieval+stream.

Кэш per-process: при рестарте бота старые диалоги теряют follow-up.
Это ок — их кнопки вернут friendly «контекст устарел, спроси заново».
TTL подобран под внимание юзера к одному ответу, не к cross-session workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.rag.retriever import Hit
from src.rag.web_search import WebHit
from src.subjects import Subject

# TTL 24 ч: тап «Проще» по вчерашнему ответу должен ещё работать, а не
# падать в «контекст устарел». На промахе кэша (или после рестарта бота)
# пайплайн перезагружает оригинальный вопрос из Dialog'а и заново делает
# retrieval. См. fallback-путь в ``qa_pipeline.run_followup_pipeline``.
_TTL_SECONDS = 24 * 60 * 60
# Поднят вместе с TTL — суточная активность студентов помещается.
_MAX_ENTRIES = 5_000


@dataclass
class FollowUpContext:
    question: str
    lang: str
    hits: list[Hit]
    subject: Subject | None
    web_hits: list[WebHit] | None
    created_at: float
    # Заполняется, если исходный вопрос был «сравни X и Y» — follow-up'ам
    # нужно это, чтобы «Проще» на comparison-ответе re-run'ил comparison-
    # промпт, а не плоский single-term build (который иначе вернёт
    # «определение X, потом определение Y»-прозой).
    comparison_terms: tuple[str, str] | None = None

    @property
    def is_web(self) -> bool:
        return self.subject is None and self.web_hits is not None


_CACHE: dict[int, FollowUpContext] = {}


def _evict_stale(now: float) -> None:
    stale = [did for did, ctx in _CACHE.items() if now - ctx.created_at > _TTL_SECONDS]
    for did in stale:
        _CACHE.pop(did, None)
    # Если всё ещё над лимитом — дропаем старейших по created_at.
    if len(_CACHE) > _MAX_ENTRIES:
        overflow = len(_CACHE) - _MAX_ENTRIES
        for did in sorted(_CACHE, key=lambda d: _CACHE[d].created_at)[:overflow]:
            _CACHE.pop(did, None)


def store(
    dialog_id: int,
    *,
    question: str,
    lang: str,
    hits: list[Hit] | None = None,
    subject: Subject | None = None,
    web_hits: list[WebHit] | None = None,
    comparison_terms: tuple[str, str] | None = None,
) -> None:
    now = time.time()
    _evict_stale(now)
    _CACHE[dialog_id] = FollowUpContext(
        question=question,
        lang=lang,
        hits=hits or [],
        subject=subject,
        web_hits=web_hits,
        created_at=now,
        comparison_terms=comparison_terms,
    )


def get(dialog_id: int) -> FollowUpContext | None:
    ctx = _CACHE.get(dialog_id)
    if ctx is None:
        return None
    if time.time() - ctx.created_at > _TTL_SECONDS:
        _CACHE.pop(dialog_id, None)
        return None
    return ctx


def clear() -> None:
    """Тестовый хук — стирает in-memory state между параметризованными прогонами."""
    _CACHE.clear()
