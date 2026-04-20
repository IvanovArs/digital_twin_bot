"""OpenAI-compatible HTTP client to a local llama.cpp server.

``chat()``        — synchronous one-shot, used for short helpers
                    (query rewriting, ambiguous-subject hints).
``chat_stream()`` — async generator over SSE ``data: {…}`` chunks for the
                    main answer, so the user sees tokens appear progressively.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import re
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.rag.circuit_breaker import llm_breaker

# Qwen3 reasons inside <think>...</think> before giving the user-facing answer.
# If it ever slips through (e.g. when /no_think didn't apply), drop it so the
# student sees the answer, not the chain-of-thought.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove any ``<think>…</think>`` block and trim whitespace."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    return cleaned or text.strip()


def _headers() -> dict[str, str]:
    key = settings.LLM_API_KEY.get_secret_value() or "no-key"
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _url(path: str) -> str:
    return f"{settings.LLM_BASE_URL.rstrip('/')}{path}"


# ---------- reusable HTTP clients ----------
# One connection pool per process instead of tearing down httpx on every
# request. llama.cpp is a single local endpoint — the savings are modest but
# matter on bursts (retry loops, streaming re-connects).

_SYNC_CLIENT: httpx.Client | None = None
_ASYNC_CLIENT: httpx.AsyncClient | None = None


def _sync_client(timeout_s: float) -> httpx.Client:
    global _SYNC_CLIENT
    if _SYNC_CLIENT is None:
        _SYNC_CLIENT = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
    return _SYNC_CLIENT


def _async_client(timeout_s: float) -> httpx.AsyncClient:
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        _ASYNC_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0))
    return _ASYNC_CLIENT


@atexit.register
def _close_clients() -> None:
    global _SYNC_CLIENT
    if _SYNC_CLIENT is not None:
        with contextlib.suppress(Exception):
            _SYNC_CLIENT.close()
        _SYNC_CLIENT = None
    # AsyncClient needs an event loop to close — skip; main.main() does that
    # explicitly via aclose_async_client() before its loop exits.


async def aclose_async_client() -> None:
    """Close the singleton ``httpx.AsyncClient`` from inside the running loop.

    Call this from ``main()``'s shutdown path to avoid the
    ``Unclosed client session`` warning and the Windows-specific 30-second
    hang that httpx's background GC exhibits when the loop tears down with
    pending connections.
    """
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        client = _ASYNC_CLIENT
        _ASYNC_CLIENT = None
        with contextlib.suppress(Exception):
            await client.aclose()


# ---------- synchronous one-shot ----------


def _build_payload(
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    repeat_penalty: float | None,
    max_tokens: int | None,
    stream: bool,
) -> dict[str, object]:
    """Compose the OpenAI-compatible body llama.cpp accepts, with Qwen3 knobs."""
    return {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
        "top_p": top_p if top_p is not None else settings.LLM_TOP_P,
        "top_k": top_k if top_k is not None else settings.LLM_TOP_K,
        "min_p": min_p if min_p is not None else settings.LLM_MIN_P,
        "repeat_penalty": (
            repeat_penalty if repeat_penalty is not None else settings.LLM_REPEAT_PENALTY
        ),
        "max_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        "stream": stream,
    }


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
)
def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> str:
    """Single-shot chat completion. Retries up to 3× on 5xx / transport errors.

    Raises ``CircuitOpenError`` if the breaker is open (llama-server
    flagged as unhealthy) — the caller should surface a fast failure to
    the user rather than wait 300 s for a timeout.
    """
    llm_breaker.guard()
    payload = _build_payload(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        stream=False,
    )
    client = _sync_client(timeout or settings.LLM_TIMEOUT_SECONDS)
    try:
        r = client.post(_url("/chat/completions"), headers=_headers(), json=payload)
        r.raise_for_status()
    except (httpx.HTTPStatusError, httpx.TransportError):
        llm_breaker.record_failure()
        raise
    data = r.json()
    raw = str(data["choices"][0]["message"]["content"])
    llm_breaker.record_success()
    return strip_think(raw)


# ---------- async streaming ----------


async def chat_stream(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> AsyncIterator[str]:
    """Yield ``delta.content`` strings as llama-server streams them.

    The caller is responsible for accumulating chunks and throttling the UI
    updates (Telegram allows ~1 edit/sec per message). ``<think>`` blocks are
    NOT stripped here — they're rare with ``/no_think`` in the prompt, but
    if they do appear the caller should strip them from the accumulated text.
    """
    llm_breaker.guard()
    payload = _build_payload(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        stream=True,
    )
    client = _async_client(timeout or settings.LLM_TIMEOUT_SECONDS)
    # The breaker fires on transport errors (no connection) and on
    # non-2xx responses. We can't catch mid-stream cancellations as
    # breaker signal without false positives, so only the handshake
    # matters for tripping — once the stream is flowing, we trust it.
    try:
        async with client.stream(
            "POST",
            _url("/chat/completions"),
            headers=_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_raw = line[len("data:") :].strip()
                if not data_raw or data_raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_raw)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
    except (httpx.HTTPStatusError, httpx.TransportError):
        llm_breaker.record_failure()
        raise
    else:
        llm_breaker.record_success()


# ---------- health ----------


def ping() -> bool:
    """Health-check llama-server's /health (at root, not /v1)."""
    base = settings.LLM_BASE_URL.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    try:
        with httpx.Client(timeout=2.0) as client:
            return client.get(f"{root}/health").status_code == 200
    except httpx.HTTPError:
        return False
