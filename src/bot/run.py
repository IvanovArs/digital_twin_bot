"""One-command launcher: llama-server + bot.

Starts llama-server.exe as a background subprocess, waits until its /health
replies, then boots the aiogram bot in the current process. Ctrl+C kills both.

Run with:
    python -m src.bot.run
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx
import structlog

from src.bot.main import main as bot_main
from src.config import ROOT, settings
from src.logging_conf import configure_logging
from src.rag.config import LLAMA_SERVER_EXE_DEV, LLM_MODEL_FILE_4B, LLM_MODEL_FILE_8B

log = structlog.get_logger(__name__)

HEALTH_TIMEOUT = 120  # seconds to wait for llama-server to come up
POLL_EVERY = 1.5


def _llama_port() -> int:
    """Pick the port llama-server should listen on — from LLM_BASE_URL in .env."""
    parsed = urlparse(settings.LLM_BASE_URL)
    return parsed.port or 8089


def _llama_cmd() -> list[str]:
    """Pick model + flags for best speed on an RTX 3050 Ti Laptop (4 GB VRAM),
    while leaving enough headroom so the rest of the system doesn't freeze.

    * Qwen3-4B Q4 (~2.5 GB): all layers on GPU, ctx 3072 (top-k=5 fits easily)
    * Qwen3-8B Q4 (~5 GB): only 14 layers on GPU (frees ~300 MB VRAM)
    * mmap ON (no --no-mmap) — lazy load avoids the startup RAM spike
    * batch/ubatch 512 — prefill is the dominant cost; bigger batch = ~3-4×
      faster prompt ingestion on this GPU
    * KV cache q8_0 — halves KV-cache VRAM, lossless for Qwen3 Q4 answers,
      lets us keep all layers offloaded at the new ctx
    * --mlock pins the 2.5 GB model in RAM (kills cold re-read spikes)
    * --no-warmup skips the dummy decode on boot (we have MODELS_READY anyway)
    """
    if LLM_MODEL_FILE_4B.exists():
        model_path = LLM_MODEL_FILE_4B
        n_gpu_layers = "999"
        ctx_size = "3072"
    else:
        model_path = LLM_MODEL_FILE_8B
        n_gpu_layers = "14"
        ctx_size = "3072"

    return [
        str(LLAMA_SERVER_EXE_DEV),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(_llama_port()),
        "--ctx-size",
        ctx_size,
        "--n-gpu-layers",
        n_gpu_layers,
        "--threads",
        "4",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--parallel",
        "1",
        "--cont-batching",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--mlock",
        "--no-warmup",
        "--prio",
        "2",
    ]


def _llama_health_url() -> str:
    base = settings.LLM_BASE_URL.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    return f"{root}/health"


LLAMA_LOG = ROOT / "data" / "llama-server.log"


def _start_llama_server() -> subprocess.Popen[bytes]:
    if not LLAMA_SERVER_EXE_DEV.exists():
        raise FileNotFoundError(f"llama-server.exe not found: {LLAMA_SERVER_EXE_DEV}")
    if not (LLM_MODEL_FILE_4B.exists() or LLM_MODEL_FILE_8B.exists()):
        raise FileNotFoundError(
            f"No GGUF model found. Expected one of:\n"
            f"  {LLM_MODEL_FILE_4B}\n  {LLM_MODEL_FILE_8B}"
        )

    env = os.environ.copy()
    env["PATH"] = str(LLAMA_SERVER_EXE_DEV.parent) + os.pathsep + env.get("PATH", "")

    LLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = LLAMA_LOG.open("wb")
    log.info("starting_llama_server", exe=str(LLAMA_SERVER_EXE_DEV), log=str(LLAMA_LOG))

    creationflags = 0
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP → we can send CTRL_BREAK_EVENT to stop cleanly.
        # BELOW_NORMAL_PRIORITY_CLASS → Windows schedulers give other apps priority,
        # so loading / running the 2.5 GB model doesn't freeze the desktop.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS

    proc = subprocess.Popen(
        _llama_cmd(),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    # keep the file handle alive on the proc object so it stays open
    proc._log_file = logf  # type: ignore[attr-defined]
    return proc


def _tail_log(n_bytes: int = 4000) -> str:
    try:
        data = LLAMA_LOG.read_bytes()[-n_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return "<no log file>"


def _wait_healthy(proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    url = _llama_health_url()
    log.info("waiting_for_llama_health", url=url)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _tail_log()
            raise RuntimeError(
                f"llama-server exited early with code {proc.returncode}.\n"
                f"--- last log output ({LLAMA_LOG}) ---\n{tail}"
            )
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                log.info("llama_server_ready")
                return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_EVERY)
    raise TimeoutError(f"llama-server did not become healthy within {HEALTH_TIMEOUT}s")


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    log.info("stopping_llama_server", pid=proc.pid)
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        log.exception("failed_to_stop_llama_server")


async def _run() -> None:
    configure_logging()
    proc = _start_llama_server()

    # Docker / systemd signal shutdown with SIGTERM, not SIGINT. Without
    # an explicit handler, asyncio on POSIX exits the loop without ever
    # running our finally-block → llama-server turns into an orphan.
    # We wire SIGTERM to the same shutdown path as SIGINT so `docker stop`
    # and Ctrl+C behave identically.
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown(_signum: int | None = None, _frame: object = None) -> None:
        log.info("shutdown_signal_received")
        shutdown_event.set()

    if sys.platform != "win32":
        # add_signal_handler is POSIX-only; on Windows signal.signal still
        # delivers SIGINT, and SIGTERM isn't a real concept anyway (docker
        # desktop uses CTRL_BREAK_EVENT which we already handle in _stop).
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
            loop.add_signal_handler(signal.SIGINT, _request_shutdown)

    try:
        await asyncio.to_thread(_wait_healthy, proc)
        main_task = asyncio.create_task(bot_main())
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        done, _ = await asyncio.wait(
            {main_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_task in done and not main_task.done():
            main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await main_task
    finally:
        _stop(proc)


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
