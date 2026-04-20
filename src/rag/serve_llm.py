"""Launch `llama-server.exe` with a local GGUF for dev.

Designed for the user's laptop:
  - RTX 3050 Ti Laptop (4 GB VRAM)
  - 16 GB RAM
  - llama.cpp runtime already at
    C:\\Users\\danya\\WebstormProjects\\exeProject\\runtime\\

On production VPS we instead run the `ghcr.io/ggml-org/llama.cpp:server`
container via docker compose — no need for this script there.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from urllib.parse import urlparse

from src.config import settings
from src.rag.config import LLAMA_SERVER_EXE_DEV, LLM_MODEL_FILE_4B, LLM_MODEL_FILE_8B


def main() -> None:
    if not LLAMA_SERVER_EXE_DEV.exists():
        print(f"llama-server.exe не найден: {LLAMA_SERVER_EXE_DEV}", file=sys.stderr)
        sys.exit(1)

    if LLM_MODEL_FILE_4B.exists():
        model_path = LLM_MODEL_FILE_4B
        n_gpu_layers = "999"
        ctx_size = "4096"
    elif LLM_MODEL_FILE_8B.exists():
        model_path = LLM_MODEL_FILE_8B
        n_gpu_layers = "14"
        ctx_size = "4096"
    else:
        print(
            "Ни Qwen3-4B, ни Qwen3-8B в data/models/ не найдено.",
            file=sys.stderr,
        )
        sys.exit(1)

    # mmap ON (no --no-mmap) → OS lazy-loads the GGUF from disk, no 2.5–5 GB
    # RAM-spike at startup. batch-size 128 is plenty for --parallel 1.
    port = urlparse(settings.LLM_BASE_URL).port or 8089
    cmd = [
        str(LLAMA_SERVER_EXE_DEV),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        ctx_size,
        "--n-gpu-layers",
        n_gpu_layers,
        "--threads",
        "6",
        "--batch-size",
        "128",
        "--parallel",
        "1",
        "--flash-attn",
        "on",
    ]
    print("Запуск llama-server:")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    env = os.environ.copy()
    env["PATH"] = str(LLAMA_SERVER_EXE_DEV.parent) + os.pathsep + env.get("PATH", "")
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
