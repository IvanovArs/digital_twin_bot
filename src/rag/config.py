"""Paths and RAG constants.

For env-driven settings (LLM base URL, model, etc.) see `src/config.py` (Phase 3).
This module keeps the path layout stable so that ingest/retriever/router can
import it without touching pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COURSES_YAML = ROOT / "courses.yaml"

BOOKS_DIR = ROOT / "data" / "books"
INDEX_DIR = ROOT / "data" / "index"
MODELS_DIR = ROOT / "data" / "models"

# Global unified index (all subjects in one matrix + one JSONL of chunks).
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
INDEX_META_FILE = INDEX_DIR / "meta.json"  # embedding model, chunk params, created_at

EMBEDDING_MODEL = "BAAI/bge-m3"
# Cross-encoder reranker that takes (query, chunk) → relevance score. Cheap on
# CPU (~30 ms for 30 pairs) and lifts precision of the top-k meaningfully.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE = 650
CHUNK_OVERLAP = 130
# Top-k fetched from the index per question. 5 is the sweet spot after
# dropping the reranker: more hits mostly bloat the prompt with weaker chunks
# (verified — top_score for 6-8 ranks usually drops below MIN_TOP_SCORE).
TOP_K = 5
# How many candidates the reranker rescores before we cut to TOP_K. With
# TOP_K=5 the bottom 10 of a pool=30 rarely bubble up after hybrid RRF;
# 15 covers the realistic quality ceiling while halving reranker CPU per
# turn (~15 ms vs ~30 ms on CPU). If prod metrics show recall@5 dropping,
# tick back up to 20.
RERANK_POOL = 15
# Below this reranker score a chunk is treated as noise — never sent to LLM.
# bge-reranker-v2-m3 emits logits; ~0 means uncertain, negatives mean wrong.
RERANK_MIN_SCORE = 0.0
# MMR λ balances relevance (1.0) vs diversity (0.0). 0.7 = mostly relevant,
# but drops near-duplicate chunks that inflate the context without adding info.
MMR_LAMBDA = 0.7
# Minimum cosine of top-1 to not give up (below → rewrite question + retry).
# bge-m3: ≥0.55 strong on-topic, 0.35–0.55 tangential-but-useful, <0.35 noise.
MIN_TOP_SCORE = 0.35
# Above this score we're confident enough to skip the LLM-rewrite step entirely
# (saves one LLM round-trip on clean questions).
MIN_TOP_SCORE_CONFIDENT = 0.55

# Confidence margin between top-1 and top-2 subjects (0..1).
# If below this threshold the router returns ambiguous=True.
ROUTER_CONFIDENCE_MARGIN = 0.08

# Dev-only: local llama-server for generation. Prefer 4B if downloaded
# (fits fully on a 4 GB VRAM GPU → ~10× faster than partial-offload 8B).
LLM_MODEL_FILE_4B = MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf"
LLM_MODEL_FILE_8B = MODELS_DIR / "Qwen3-8B-Q4_K_M.gguf"
# Legacy alias kept for backwards-compat with existing scripts.
LLM_MODEL_FILE_DEV = LLM_MODEL_FILE_4B if LLM_MODEL_FILE_4B.exists() else LLM_MODEL_FILE_8B
LLAMA_SERVER_EXE_DEV = Path(r"C:\Users\danya\WebstormProjects\exeProject\runtime\llama-server.exe")
