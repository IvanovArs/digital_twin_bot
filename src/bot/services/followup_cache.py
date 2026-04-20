"""In-memory cache that lets the follow-up buttons skip retrieval.

When ``run_qa_pipeline`` finishes, we stash the ``hits``/``web_hits`` plus a
bit of routing metadata under the new ``dialog_id``. A tap on
«Упрости / Дай пример / Подробнее» retrieves the same context and re-runs
the LLM with a prompt modifier — no second bge-m3 pass, no second web
search, so the follow-up lands in ~2× streaming latency instead of
~2×retrieval+stream.

Cache is per-process: if the bot restarts, old dialogs lose their
follow-up. That's fine — their buttons just return a friendly "context
expired, ask fresh" alert. TTL matches the expected user attention span on
a single answer, not a cross-session workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.rag.retriever import Hit
from src.rag.web_search import WebHit
from src.subjects import Subject

# 24 h TTL: a student clicking «Проще» on yesterday's answer should still
# work, not hit «контекст устарел». When the cache does miss (or the bot
# restarted), the pipeline falls back to reloading the original question
# from the Dialog row and re-running retrieval. See
# ``qa_pipeline.run_followup_pipeline`` for the fallback path.
_TTL_SECONDS = 24 * 60 * 60
# Bumped along with TTL so a day's worth of active students still fits.
_MAX_ENTRIES = 5_000


@dataclass
class FollowUpContext:
    question: str
    lang: str
    hits: list[Hit]
    subject: Subject | None
    web_hits: list[WebHit] | None
    created_at: float
    # Set when the original question was a «сравни X и Y» — follow-ups
    # need this so «Проще» on a comparison answer re-runs the comparison
    # prompt instead of a flat single-term build (which would otherwise
    # produce «определение X, потом определение Y» prose).
    comparison_terms: tuple[str, str] | None = None

    @property
    def is_web(self) -> bool:
        return self.subject is None and self.web_hits is not None


_CACHE: dict[int, FollowUpContext] = {}


def _evict_stale(now: float) -> None:
    stale = [did for did, ctx in _CACHE.items() if now - ctx.created_at > _TTL_SECONDS]
    for did in stale:
        _CACHE.pop(did, None)
    # If still over the cap, drop the oldest by created_at.
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
    """Test hook — wipes the in-memory state between parametrised runs."""
    _CACHE.clear()
