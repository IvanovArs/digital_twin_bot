"""Subject-router: по top-k хитам из unified-индекса решает, к какому предмету
вероятнее всего относится вопрос студента.

Стратегия: голосование, взвешенное по score.
  score_sum[s] = сумма hit.score для hit в hits, где hit.subject_slug == s
  winner       = argmax(score_sum)
  confidence   = (top1 - top2) / top1   (насколько лидер доминирует)

Caller решает, доверять ли победителю или попросить юзера disambiguate,
по `margin < ROUTER_CONFIDENCE_MARGIN`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.rag.config import ROUTER_CONFIDENCE_MARGIN
from src.rag.retriever import Hit


@dataclass(frozen=True)
class RouteResult:
    subject_slug: str | None
    """None если хитов не было."""
    margin: float
    """Относительный разрыв между top-1 и top-2 score'ами предметов (0..1)."""
    ambiguous: bool
    """True если margin < ROUTER_CONFIDENCE_MARGIN и конкурировали ≥ 2 предмета."""
    ranked: list[tuple[str, float]]
    """(subject_slug, score_sum), отсортирован по убыванию score_sum."""


def detect_subject(
    hits: list[Hit],
    *,
    margin_threshold: float = ROUTER_CONFIDENCE_MARGIN,
) -> RouteResult:
    if not hits:
        return RouteResult(subject_slug=None, margin=0.0, ambiguous=False, ranked=[])

    totals: dict[str, float] = defaultdict(float)
    for h in hits:
        # отрицательный cosine клипуем в 0 — outlier'ы не должны перекосить голосование
        totals[h.subject_slug] += max(0.0, h.score)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top1_slug, top1_score = ranked[0]

    if len(ranked) == 1 or top1_score <= 0:
        return RouteResult(
            subject_slug=top1_slug,
            margin=1.0 if top1_score > 0 else 0.0,
            ambiguous=False,
            ranked=ranked,
        )

    top2_score = ranked[1][1]
    margin = (top1_score - top2_score) / top1_score
    ambiguous = margin < margin_threshold

    return RouteResult(
        subject_slug=top1_slug,
        margin=margin,
        ambiguous=ambiguous,
        ranked=ranked,
    )
