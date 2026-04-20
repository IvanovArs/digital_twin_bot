"""Tiny aiohttp app that serves liveness + readiness probes on HEALTH_PORT.

Runs alongside ``dp.start_polling`` in ``main.py``. Orchestrators
(docker-compose, k8s) use these endpoints to decide "kill and restart this
container" (liveness) and "is it OK to send traffic" (readiness).

* ``/healthz`` — 200 as long as the process is up and the asyncio loop spins.
* ``/readyz``  — 200 iff models warmed AND llama-server replies to /health.
"""

from __future__ import annotations

import asyncio

import structlog
from aiohttp import web

from src.bot.services.warmup import MODELS_READY
from src.config import settings
from src.rag.llm import ping as llm_ping

log = structlog.get_logger(__name__)


async def _liveness(_request: web.Request) -> web.Response:
    # Liveness only proves the loop is alive enough to answer HTTP. Don't
    # call llama-server / Postgres here — they have their own healthchecks.
    return web.json_response({"status": "ok"})


async def _readiness(_request: web.Request) -> web.Response:
    """Reports the slow dependencies. Returns 503 until they're up."""
    models_ok = MODELS_READY.is_set()
    # llm_ping is sync httpx; bounce to a thread so we don't block the loop.
    try:
        llm_ok = await asyncio.wait_for(asyncio.to_thread(llm_ping), timeout=2.0)
    except (TimeoutError, Exception):
        llm_ok = False

    payload = {"models": models_ok, "llm": llm_ok}
    code = 200 if models_ok and llm_ok else 503
    return web.json_response(payload, status=code)


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/healthz", _liveness)
    app.router.add_get("/readyz", _readiness)
    return app


async def serve_forever(app: web.Application) -> None:
    """Run the aiohttp app until the surrounding loop is cancelled."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, settings.HEALTH_LISTEN, settings.HEALTH_PORT)
    await site.start()
    log.info("health_server_started", host=settings.HEALTH_LISTEN, port=settings.HEALTH_PORT)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


__all__ = ["build_app", "serve_forever"]
