"""Cross-encoder reranker that reorders retrieval hits by true relevance.

bge-m3 (a bi-encoder) is fast at fetching a wide candidate pool, but it ranks
by embedding cosine — which often stalls at synonyms and loses to a proper
question-vs-chunk cross-encoder on actual precision. We keep bge-m3 for the
cheap top-30 fetch and let bge-reranker-v2-m3 pick the final top-k.

On CPU this model runs ~30 ms for 30 (query, text) pairs — the win is worth
the latency in every measured run.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.rag.config import RERANKER_MODEL


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load bge-reranker-v2-m3 once per process.

    Uses CUDA when available (6-10× faster than CPU on a 15-chunk pool:
    ~3-5 ms vs ~30 ms), falls back to CPU when torch can't see a GPU so
    the dev laptop path still works. Model weights are ~560 MB — loads
    once into VRAM and stays there for the whole process lifetime.
    """
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:  # noqa: S110
        # torch import or probe can raise on broken installs; stay on CPU.
        pass
    return CrossEncoder(RERANKER_MODEL, device=device)


def rerank(query: str, texts: list[str]) -> list[float]:
    """Return a relevance score per text, in the same order as input.

    Higher = more relevant. Scores are raw logits — callers compare them
    relatively, not against an absolute threshold unless they know the model.
    """
    if not texts:
        return []
    pairs = [(query, t) for t in texts]
    scores = _model().predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    return [float(s) for s in scores]
