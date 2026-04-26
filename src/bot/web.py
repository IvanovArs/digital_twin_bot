"""Маленькое aiohttp-приложение с liveness/readiness-пробами на HEALTH_PORT.

Крутится параллельно ``dp.start_polling`` в ``main.py``. Оркестраторы
(docker-compose, k8s) дёргают эти endpoint'ы, чтобы решать «убить-
перезапустить контейнер» (liveness) и «можно ли слать трафик» (readiness).

* ``/healthz`` — 200, пока процесс жив и asyncio-loop крутится.
* ``/readyz``  — 200 если модели прогреты И llama-server отвечает на /health.
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
    # Liveness доказывает только то, что loop жив достаточно, чтобы ответить
    # HTTP. Не дёргаем llama-server / Postgres — у них свои healthcheck'и.
    return web.json_response({"status": "ok"})


async def _readiness(_request: web.Request) -> web.Response:
    """Репортит медленные зависимости. Возвращает 503, пока они не подняты."""
    models_ok = MODELS_READY.is_set()
    # llm_ping — sync httpx; уводим в thread, чтобы не блокировать loop.
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
    """Крутить aiohttp-приложение, пока внешний loop не отменён."""
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
