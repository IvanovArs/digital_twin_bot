"""DuckDuckGo text search — fallback when the textbook has no answer.

No API key, no account. One HTTP round-trip per call; we keep the top-``k``
results with (title, url, snippet) and feed them to the local LLM for
synthesis. The student's question leaves the box only after retrieval + LLM
rewrite both failed, and only to DuckDuckGo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import structlog
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

log = structlog.get_logger(__name__)

# DDG's JSON endpoint rate-limits aggressively. The HTML backend scrapes the
# SERP page directly — much more tolerant of frequent queries.
_BACKENDS = ("html", "lite", "api")
_MAX_ATTEMPTS = 3

# SEO spam / aggregator hosts that frequently appear high in DDG results but
# add no value (or actively misinform). We drop them after the search returns
# so the LLM never sees them as "authoritative". Keep this list short — false
# positives here silently hide good results from the student.
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
        """Bare hostname for status UI: ``ru.wikipedia.org`` from a full URL.

        Falls back to a slice of the URL if parsing fails — we never want the
        status line to render an empty string.
        """
        try:
            h = (urlparse(self.url).hostname or "").lower()
            return h[4:] if h.startswith("www.") else h or self.url[:40]
        except Exception:
            return self.url[:40]


def _try_backend(query: str, k: int, region: str, backend: str) -> list[dict[str, str]]:
    with DDGS() as ddgs:
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
    """Return up to ``k`` DDG results; tries several backends + small backoff.

    Returns ``[]`` on total failure (rate-limit on every backend, network, etc.).
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
        time.sleep(1.5 * (attempt + 1))  # backoff 1.5s, 3s

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
        hit = WebHit(title=title, url=url, snippet=snippet)
        if _host_blocked(hit.host):
            log.info("ddg_drop_blocked_host", host=hit.host, url=url)
            continue
        out.append(hit)
    return out
