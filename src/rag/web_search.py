"""DuckDuckGo text-search — fallback, когда в учебнике ответа нет.

Без API-key и аккаунта. Один HTTP round-trip на вызов; держим top-``k``
результатов (title, url, snippet) и кормим их локальной LLM для синтеза.
Вопрос студента покидает машину только после провала retrieval + LLM-
rewrite, и только в DuckDuckGo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import structlog
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

log = structlog.get_logger(__name__)

# JSON-endpoint DDG агрессивно rate-limit'ит. HTML-backend скрейпит SERP
# напрямую — гораздо терпимее к частым запросам.
_BACKENDS = ("html", "lite", "api")
_MAX_ATTEMPTS = 3

# SEO-спам / агрегаторы, часто всплывающие в DDG-результатах высоко, но
# не несущие ценности (или активно дезинформирующие). Дропаем после
# search'а — LLM не видит их как «авторитет». Список держим коротким —
# false-positive'ы тут молча скрывают хорошие результаты от студента.
_HOST_BLOCKLIST = frozenset(
    {
        "ru-stat.com",
        "otvet.mail.ru",
        "znanija.com",
        "znanija.org",
        "znanio.ru",
        "answers.yahoo.com",
        "spishi.ru",
    }
)


def _host_blocked(host: str) -> bool:
    if not host:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    return h in _HOST_BLOCKLIST


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str

    @property
    def host(self) -> str:
        """Голый hostname для status-UI: «ru.wikipedia.org» из полного URL.

        Fallback на срез URL если parsing упал — пустая строка в status'е
        нам не нужна никогда.
        """
        try:
            h = (urlparse(self.url).hostname or "").lower()
            return h[4:] if h.startswith("www.") else h or self.url[:40]
        except Exception:
            return self.url[:40]


def _try_backend(query: str, k: int, region: str, backend: str) -> list[dict[str, str]]:
    # 8 с на backend — DuckDuckGo не должен висеть бесконечно. Без этого
    # застрявший DDG-endpoint блокирует asyncio.to_thread-worker до OS
    # socket-таймаута (часто минуты), юзер сидит на «🌐 ищу в интернете…»
    # with no progress, no recovery.
    with DDGS(timeout=8) as ddgs:
        return list(
            ddgs.text(
                query,
                region=region,
                safesearch="moderate",
                max_results=k,
                backend=backend,
            )
        )


def search_web(query: str, k: int = 5, *, lang: str = "ru") -> list[WebHit]:
    """До ``k`` DDG-результатов; пробует несколько backend'ов + малый backoff.

    Возвращает ``[]`` при полном fail'е (rate-limit на каждом backend, сеть и т.п.).
    """
    region = "ru-ru" if lang == "ru" else "wt-wt"
    raw: list[dict[str, str]] = []

    for attempt in range(_MAX_ATTEMPTS):
        for backend in _BACKENDS:
            try:
                raw = _try_backend(query, k, region, backend)
                if raw:
                    log.info("ddg_ok", backend=backend, attempt=attempt + 1, results=len(raw))
                    break
            except RatelimitException:
                log.warning("ddg_ratelimit", backend=backend, attempt=attempt + 1)
                continue
            except Exception:
                log.exception("ddg_backend_failed", backend=backend, attempt=attempt + 1)
                continue
        if raw:
            break
        time.sleep(1.5 * (attempt + 1))  # backoff 1.5с, 3с

    if not raw:
        log.error("ddg_all_backends_failed", query=query)
        return []

    out: list[WebHit] = []
    for r in raw:
        title = str(r.get("title") or "").strip()
        url = str(r.get("href") or r.get("url") or "").strip()
        snippet = str(r.get("body") or r.get("snippet") or "").strip()
        if not title or not url:
            continue
        # Пропускаем почти-пустые сниппеты — DDG иногда возвращает только title.
        # Без текста LLM «доздаёт» контент из parametric-memory и цитирует
        # источник, как будто тот это сказал. ≥60 chars + ≥8 word-токенов —
        # эмпирический порог: одно-предложенные определения проходят, чистый
        # title-шум отсекается.
        if len(snippet) < 60 or len(snippet.split()) < 8 or snippet.lower() in title.lower():
            log.info("ddg_drop_thin_snippet", host=urlparse(url).hostname or "", chars=len(snippet))
            continue
        hit = WebHit(title=title, url=url, snippet=snippet)
        if _host_blocked(hit.host):
            log.info("ddg_drop_blocked_host", host=hit.host, url=url)
            continue
        out.append(hit)
    return out
