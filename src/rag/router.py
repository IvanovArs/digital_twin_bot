"""Subject router: given top-k hits from the unified index, decide which
subject the student's question likely belongs to.

Strategy: score-weighted voting.
  score_sum[s] = sum of hit.score for hit in hits if hit.subject_slug == s
  winner     = argmax(score_sum)
  confidence = (top1 - top2) / top1   (how dominant the leader is)

The caller decides whether to trust the winner or ask the user to
disambiguate, based on `margin < ROUTER_CONFIDENCE_MARGIN`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.rag.config import ROUTER_CONFIDENCE_MARGIN
from src.rag.retriever import Hit


@dataclass(frozen=True)
class RouteResult:
    subject_slug: str | None
    """None iff there were no hits."""
    margin: float
    """Relative gap between top-1 and top-2 subject scores (0..1)."""
    ambiguous: bool
    """True when margin < ROUTER_CONFIDENCE_MARGIN and ≥ 2 subjects competed."""
    ranked: list[tuple[str, float]]
    """(subject_slug, score_sum) sorted by score_sum desc."""


def detect_subject(
    hits: list[Hit],
    *,
    margin_threshold: float = ROUTER_CONFIDENCE_MARGIN,
) -> RouteResult:
    if not hits:
        return RouteResult(subject_slug=None, margin=0.0, ambiguous=False, ranked=[])

    totals: dict[str, float] = defaultdict(float)
    for h in hits:
        # clamp negative cosine to 0 so outliers don't tilt the vote the wrong way
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
