"""Cross-encoder реранкер: переcортировка retrieval-хитов по реальной релевантности.

bge-m3 (bi-encoder) быстро тянет широкий candidate-pool, но ранжирует по
embedding-cosine — часто упирается в синонимы и проигрывает нормальному
question-vs-chunk cross-encoder'у по реальному precision'у. Bge-m3 оставляем
для дешёвого top-30 fetch'а, bge-reranker-v2-m3 — для финального top-k.

На CPU модель отрабатывает ~30 мс на 30 (query, text)-пар — выигрыш стоит
этой латентности в каждом измеренном прогоне.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.rag.config import RERANKER_MODEL


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Загрузить bge-reranker-v2-m3 один раз на процесс.

    Использует CUDA если есть (6–10× быстрее CPU на пуле в 15 чанков:
    ~3–5 мс vs ~30 мс), иначе fallback на CPU — dev-ноутбук тоже работает.
    Веса ~560 МБ, грузятся один раз в VRAM и живут весь процесс.
    """
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:  # noqa: S110
        # torch import или probe могут упасть на битой установке; остаёмся на CPU.
        pass
    return CrossEncoder(RERANKER_MODEL, device=device)


def rerank(query: str, texts: list[str]) -> list[float]:
    """Вернуть relevance-score на каждый текст, в том же порядке.

    Больше = релевантнее. Score'ы — raw logits, caller сравнивает их
    относительно, а не против абсолютного порога (если только не знает модель).
    """
    if not texts:
        return []
    pairs = [(query, t) for t in texts]
    scores = _model().predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    return [float(s) for s in scores]
