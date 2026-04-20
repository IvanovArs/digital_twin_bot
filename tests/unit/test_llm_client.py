from __future__ import annotations

import httpx
import pytest

from src.rag import llm as llm_mod


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,  # type: ignore[no-untyped-def]
) -> None:
    """Replace ``httpx.Client`` inside the llm module with a factory that
    routes all traffic through a MockTransport. Uses the real class captured
    *before* patching to avoid infinite recursion.

    Also resets the module-level singleton client so a previous test's cached
    instance doesn't leak into this one.
    """
    real_client_cls = httpx.Client

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("transport", None)
        return real_client_cls(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(llm_mod, "_SYNC_CLIENT", None)
    monkeypatch.setattr(llm_mod.httpx, "Client", factory)


def test_chat_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": "привет!"}}]})

    _install_mock_transport(monkeypatch, handler)

    assert llm_mod.chat([{"role": "user", "content": "ping"}]) == "привет!"


def test_chat_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _install_mock_transport(monkeypatch, handler)

    assert llm_mod.chat([{"role": "user", "content": "ping"}]) == "ok"
    assert calls["n"] == 3


def test_chat_strips_qwen3_think_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3 may emit <think>…</think> before the answer — we must drop it."""

    def handler(req: httpx.Request) -> httpx.Response:
        content = (
            "<think>\nЭто рассуждение модели, не для пользователя.\n</think>\n\n"
            "Настоящий ответ, который должен увидеть студент."
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    _install_mock_transport(monkeypatch, handler)

    out = llm_mod.chat([{"role": "user", "content": "q"}])
    assert "<think>" not in out
    assert "рассуждение" not in out
    assert out.startswith("Настоящий ответ")


def test_chat_keeps_content_when_only_think_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the whole response is thinking, fall back to the raw text so the user
    at least sees something rather than an empty message."""

    def handler(req: httpx.Request) -> httpx.Response:
        content = "<think>\nТолько размышления.\n</think>"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    _install_mock_transport(monkeypatch, handler)

    out = llm_mod.chat([{"role": "user", "content": "q"}])
    assert out  # not empty
    assert "Только размышления" in out
