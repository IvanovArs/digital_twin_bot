"""OpenAI-совместимый HTTP-клиент к локальному llama.cpp-серверу.

``chat()``        — синхронный one-shot для коротких хелперов
                    (rewrite запроса, подсказки по subject).
``chat_stream()`` — async-генератор поверх SSE ``data: {…}`` чанков для
                    основного ответа, чтобы юзер видел токены по мере появления.
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

# Qwen3 рассуждает внутри <think>...</think> перед ответом юзеру. Если блок
# просочился (например, /no_think не сработал) — режем его, чтобы студент
# видел ответ, а не chain-of-thought.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Удалить любой ``<think>…</think>``-блок и обрезать пробелы по краям."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    return cleaned or text.strip()


def _headers() -> dict[str, str]:
    key = settings.LLM_API_KEY.get_secret_value() or "no-key"
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _url(path: str) -> str:
    return f"{settings.LLM_BASE_URL.rstrip('/')}{path}"


# ---------- переиспользуемые HTTP-клиенты ----------
# Один connection-pool на процесс, чтобы не пересоздавать httpx на каждый
# запрос. llama.cpp — один локальный endpoint, экономия скромная, но важна
# на burst'ах (retry-циклы, переподключения стрима).

_SYNC_CLIENT: httpx.Client | None = None
_ASYNC_CLIENT: httpx.AsyncClient | None = None

# llama-server — один локальный endpoint. Default httpx-pool (10 keepalive /
# 100 max) для нас сильно избыточен; явные маленькие лимиты держат одно
# тёплое соединение между burst'ами (retry-циклы, цепочки стрим-reconnect).
_HTTPX_LIMITS = httpx.Limits(
    max_keepalive_connections=4,
    max_connections=8,
    keepalive_expiry=120.0,
)


def _sync_client(timeout_s: float) -> httpx.Client:
    global _SYNC_CLIENT
    if _SYNC_CLIENT is None:
        _SYNC_CLIENT = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=_HTTPX_LIMITS,
        )
    return _SYNC_CLIENT


def _async_client(timeout_s: float) -> httpx.AsyncClient:
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        _ASYNC_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=_HTTPX_LIMITS,
        )
    return _ASYNC_CLIENT


@atexit.register
def _close_clients() -> None:
    global _SYNC_CLIENT
    if _SYNC_CLIENT is not None:
        with contextlib.suppress(Exception):
            _SYNC_CLIENT.close()
        _SYNC_CLIENT = None
    # AsyncClient требует живой event-loop для закрытия — пропускаем; main.main()
    # делает это явно через aclose_async_client() перед выходом из loop'а.


async def aclose_async_client() -> None:
    """Закрыть singleton ``httpx.AsyncClient`` изнутри живого loop'а.

    Зови из shutdown-пути ``main()``, чтобы избежать предупреждения
    ``Unclosed client session`` и Windows-специфичного 30-секундного hang'а,
    который background-GC httpx устраивает при teardown'е loop'а с pending-соединениями.
    """
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        client = _ASYNC_CLIENT
        _ASYNC_CLIENT = None
        with contextlib.suppress(Exception):
            await client.aclose()


# ---------- синхронный one-shot ----------


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
    """Собрать OpenAI-совместимое тело, которое принимает llama.cpp, с Qwen3-knob'ами."""
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
    """One-shot chat completion. До 3 попыток при 5xx / транспортных ошибках.

    Бросает ``CircuitOpenError`` если breaker открыт (llama-server помечен как
    unhealthy) — caller должен показать быстрый отказ пользователю, а не
    ждать 300с-таймаут.
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


# ---------- async-стриминг ----------


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
    """Yield-ит строки ``delta.content`` по мере стриминга от llama-server'а.

    Caller сам аккумулирует чанки и троттлит UI-edit'ы (Telegram = ~1 edit/sec
    на сообщение). ``<think>``-блоки здесь НЕ срезаются — с ``/no_think`` в
    промпте они редки, но если попали — caller должен срезать на накопленном.
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
    # Breaker срабатывает на транспортных ошибках (нет соединения) и не-2xx
    # ответах. Mid-stream-отмены не ловим как сигнал для breaker'а (слишком
    # много false-positive'ов), так что важен только handshake — раз стрим
    # пошёл, доверяем ему.
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
    """Health-check на /health у llama-server (он в root'е, не /v1)."""
    base = settings.LLM_BASE_URL.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    try:
        with httpx.Client(timeout=2.0) as client:
            return client.get(f"{root}/health").status_code == 200
    except httpx.HTTPError:
        return False
