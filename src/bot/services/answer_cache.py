"""Кеш готовых ответов на повторные вопросы.

Студенты часто задают один и тот же вопрос в разных формулировках.
Нормализуем строку (lower + collapse whitespace + strip), храним последние
N пар (question, lang, brief) → (body, dialog_id_template). При попадании
в кеш юзер видит ответ мгновенно — без retrieval, без LLM, без edit-цикла.

Кеш изолирован per-user: чужие диалоги не светятся (для приватности и для
правильного follow-up'а — кэшим dialog_id, который ссылается на конкретного
владельца).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_NORM_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CachedAnswer:
    body: str
    kind: str  # "full" | "web" | "glossary" | "faq"
    dialog_id: int
    cached_at: float


def normalize(question: str) -> str:
    """Нормализация для ключа: lower + collapse whitespace + strip пунктуации
    по краям. «Что такое стейкхолдер?» и «что такое стейкхолдер»
    попадают в один ключ.
    """
    s = (question or "").strip().lower()
    s = _NORM_RE.sub(" ", s)
    return s.strip(" .?!,;:—-\"'`")


_CACHE: dict[tuple[int, str, str, bool], CachedAnswer] = {}
_TTL_S = 30 * 60.0  # 30 минут — преподаватель за это время не успеет переписать ответ
_MAX_ENTRIES = 2_000


def get(*, user_id: int, question: str, lang: str, brief: bool) -> CachedAnswer | None:
    key = (user_id, normalize(question), lang, brief)
    hit = _CACHE.get(key)
    if hit is None:
        return None
    if time.time() - hit.cached_at > _TTL_S:
        _CACHE.pop(key, None)
        return None
    return hit


def store(
    *,
    user_id: int,
    question: str,
    lang: str,
    brief: bool,
    body: str,
    kind: str,
    dialog_id: int,
) -> None:
    if len(_CACHE) >= _MAX_ENTRIES:
        # Дроп старейших 10% — амортизация O(N), но N=2k, mem-cheap.
        oldest = sorted(_CACHE.items(), key=lambda kv: kv[1].cached_at)
        for k, _ in oldest[: max(1, _MAX_ENTRIES // 10)]:
            _CACHE.pop(k, None)
    key = (user_id, normalize(question), lang, brief)
    _CACHE[key] = CachedAnswer(body=body, kind=kind, dialog_id=dialog_id, cached_at=time.time())


def invalidate_user(user_id: int) -> int:
    """Сбросить весь кеш юзера (например, после смены закреплённого предмета).
    Возвращает число удалённых записей.
    """
    keys = [k for k in _CACHE if k[0] == user_id]
    for k in keys:
        _CACHE.pop(k, None)
    return len(keys)


def invalidate_all() -> int:
    """Полная инвалидация (например, после reindex предмета — старые ответы
    могут ссылаться на удалённые чанки)."""
    n = len(_CACHE)
    _CACHE.clear()
    return n


def stats() -> dict[str, int]:
    return {"entries": len(_CACHE), "max": _MAX_ENTRIES}
