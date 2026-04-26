"""Application settings sourced from environment variables (`.env` in dev).

All configuration flows through this module. Do not read os.environ directly
from other modules — import `settings` from here instead.

See `.env.example` for the full list of supported variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Telegram ---
    BOT_TOKEN: SecretStr = Field(default=SecretStr(""))
    ADMIN_TELEGRAM_IDS: str = Field(default="")
    # Comma-separated Telegram user IDs allowed to use the bot. Empty = open to all.
    ALLOWED_TELEGRAM_IDS: str = Field(default="")

    # --- Health probe HTTP ---
    # We always serve /healthz + /readyz on this port so docker-compose / k8s
    # can probe liveness regardless of polling vs webhook mode. No webhook
    # support yet — this is purely the health endpoint.
    # Default 0.0.0.0 so the container's own port-mapping works; bind to
    # 127.0.0.1 if you want host-only.
    HEALTH_LISTEN: str = "0.0.0.0"  # noqa: S104
    HEALTH_PORT: int = 8081

    # --- Database ---
    # SQLite by default — ноль настройки для локального dev.
    # Прод в docker-compose переопределяет на postgresql+asyncpg://…
    DB_URL: str = Field(default=f"sqlite+aiosqlite:///{ROOT / 'data' / 'dev.db'}")

    # --- LLM (local llama.cpp) ---
    LLM_PROVIDER: Literal["openai_compat"] = "openai_compat"
    LLM_BASE_URL: str = "http://127.0.0.1:8089/v1"
    LLM_API_KEY: SecretStr = Field(default=SecretStr(""))
    LLM_MODEL: str = "qwen"
    LLM_TIMEOUT_SECONDS: float = 300.0
    # Qwen3 non-think recipe (official): temp 0.7 / top_p 0.8 / top_k 20 / min_p 0.
    # With /no_think in the system prompt this gives warm, grounded answers
    # without the reasoning token waste of the think mode.
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.8
    LLM_TOP_K: int = 20
    LLM_MIN_P: float = 0.0
    LLM_REPEAT_PENALTY: float = 1.05
    # 220 tokens fits a 2–5 short-sentence answer (the new free-form prompt;
    # the old "3–5 bullets" mandate is gone). Tighter cap = faster decode on
    # CPU (~2–3 s saved at 10 tok/s) AND less room for the model to invent
    # filler when the fragments are thin. Raise via env if you need longer.
    LLM_MAX_TOKENS: int = 220

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # --- RAG ---
    # Stable retrieval constants live in src.rag.config (TOP_K, MIN_TOP_SCORE,
    # CHUNK_SIZE, …). These env overrides are kept for operator knobs only.
    CHUNK_SIZE: int = 650
    CHUNK_OVERLAP: int = 130
    ROUTER_CONFIDENCE_MARGIN: float = 0.08

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # --- Paths ---
    BOOKS_DIR: Path = ROOT / "data" / "books"
    INDEX_DIR: Path = ROOT / "data" / "index"
    MODELS_DIR: Path = ROOT / "data" / "models"
    COURSES_YAML: Path = ROOT / "courses.yaml"

    @field_validator("ADMIN_TELEGRAM_IDS", "ALLOWED_TELEGRAM_IDS")
    @classmethod
    def _strip_commas(cls, v: str) -> str:
        return ",".join(p.strip() for p in v.split(",") if p.strip())

    @property
    def admin_ids(self) -> set[int]:
        return {int(x) for x in self.ADMIN_TELEGRAM_IDS.split(",") if x}

    @property
    def allowed_ids(self) -> set[int]:
        """Empty set ⇒ no allowlist (bot open to everyone)."""
        return {int(x) for x in self.ALLOWED_TELEGRAM_IDS.split(",") if x}


settings = Settings()
