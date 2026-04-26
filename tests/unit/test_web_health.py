"""Тесты health-endpoint'ов /healthz и /readyz."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.bot import web as bot_web
from src.bot.services.warmup import MODELS_READY


@pytest.mark.asyncio
async def test_healthz_always_200() -> None:
    app = bot_web.build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.json()
        assert body == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_503_when_models_not_ready(monkeypatch) -> None:
    MODELS_READY.clear()
    monkeypatch.setattr(bot_web, "llm_ping", lambda: True)
    app = bot_web.build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/readyz")
        assert resp.status == 503
        body = await resp.json()
        assert body["models"] is False


@pytest.mark.asyncio
async def test_readyz_503_when_llm_down(monkeypatch) -> None:
    MODELS_READY.set()
    monkeypatch.setattr(bot_web, "llm_ping", lambda: False)
    try:
        app = bot_web.build_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/readyz")
            assert resp.status == 503
            body = await resp.json()
            assert body["models"] is True
            assert body["llm"] is False
    finally:
        MODELS_READY.clear()


@pytest.mark.asyncio
async def test_readyz_200_when_all_green(monkeypatch) -> None:
    MODELS_READY.set()
    monkeypatch.setattr(bot_web, "llm_ping", lambda: True)
    try:
        app = bot_web.build_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/readyz")
            assert resp.status == 200
            body = await resp.json()
            assert body == {"models": True, "llm": True}
    finally:
        MODELS_READY.clear()
