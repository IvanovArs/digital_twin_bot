"""Тесты SSE-стрима chat_stream и health-эндпоинта ping."""

from __future__ import annotations

import json

import httpx
import pytest

from src.rag import llm as llm_mod


def _install_async_mock(
    monkeypatch: pytest.MonkeyPatch,
    handler,  # type: ignore[no-untyped-def]
) -> None:
    real_async = httpx.AsyncClient

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("transport", None)
        return real_async(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(llm_mod, "_ASYNC_CLIENT", None)
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", factory)


def _sse_lines(*pieces: str) -> bytes:
    out: list[str] = []
    for piece in pieces:
        chunk = {"choices": [{"delta": {"content": piece}}]}
        out.append(f"data: {json.dumps(chunk)}\n")
    out.append("data: [DONE]\n")
    return ("\n".join(out) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_chat_stream_yields_each_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_lines("Привет", ", ", "мир!"))

    _install_async_mock(monkeypatch, handler)
    chunks: list[str] = []
    async for piece in llm_mod.chat_stream([{"role": "user", "content": "q"}]):
        chunks.append(piece)
    assert "".join(chunks) == "Привет, мир!"


@pytest.mark.asyncio
async def test_chat_stream_skips_empty_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = (
            b"\n"  # пустая строка
            b"data: \n"  # пустой data
            b"data: [DONE]\n"  # sentinel
            b"data: " + json.dumps({"choices": [{"delta": {"content": "X"}}]}).encode() + b"\n"
            b"\n"
        )
        return httpx.Response(200, content=body)

    _install_async_mock(monkeypatch, handler)
    chunks = [c async for c in llm_mod.chat_stream([{"role": "user", "content": "q"}])]
    assert chunks == ["X"]


@pytest.mark.asyncio
async def test_chat_stream_ignores_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = b"data: {not valid json}\n" b"data: " + json.dumps(
            {"choices": [{"delta": {"content": "ok"}}]}
        ).encode() + b"\n"
        return httpx.Response(200, content=body)

    _install_async_mock(monkeypatch, handler)
    chunks = [c async for c in llm_mod.chat_stream([{"role": "user", "content": "q"}])]
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_chat_stream_records_breaker_failure_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.rag.circuit_breaker import llm_breaker

    llm_breaker.record_success()  # сброс
    initial = llm_breaker._state.consecutive_failures

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _install_async_mock(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in llm_mod.chat_stream([{"role": "user", "content": "q"}]):
            pass
    assert llm_breaker._state.consecutive_failures > initial
    llm_breaker.record_success()  # восстановим


def test_ping_returns_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client = httpx.Client

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("transport", None)
        return real_client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
            **kwargs,
        )

    monkeypatch.setattr(llm_mod.httpx, "Client", factory)
    assert llm_mod.ping() is True


def test_ping_returns_false_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    real_client = httpx.Client

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(boom), **kwargs)

    monkeypatch.setattr(llm_mod.httpx, "Client", factory)
    assert llm_mod.ping() is False
