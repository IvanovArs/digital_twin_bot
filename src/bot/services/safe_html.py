"""Санитайзер для любого Telegram-``parse_mode="HTML"`` тела, источник
которого частично контролируется юзером (LLM-вывод, FAQ от препода,
определения глоссария).

Правила:

* Выживает только короткий whitelist Telegram-inline-тегов.
* Всё остальное — ``html.escape``.
* Open/close-теги балансируются — незакрытый ``<b>`` иначе сделает так,
  что Telegram отвергнет весь edit с «can't parse entities».
* ``truncate_for_telegram`` режет тело ниже 4096-char-потолка, не
  разрезая тег пополам.

Используется RAG-renderer'ом и short-circuit-путями FAQ/глоссария.
Преподаватели пишут ответы прозой или лёгким HTML; санитайзим на
write-to-user — кривой ``</b>`` не сломает рендеринг для всех студентов,
задающих тот же вопрос.
"""

from __future__ import annotations

import html
import re

# Список короткий — каждая запись — место, где кривой тег источника
# может сломать parse_mode="HTML" для всего сообщения.
ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s", "code", "pre", "blockquote")

_ALLOWED_TAG_RE = re.compile(
    r"&lt;(/?)(" + "|".join(ALLOWED_TAGS) + r")&gt;",
    flags=re.IGNORECASE,
)
_TAG_TOKEN_RE = re.compile(
    r"<(/?)(" + "|".join(ALLOWED_TAGS) + r")>",
    flags=re.IGNORECASE,
)

TG_MESSAGE_LIMIT = 4096


def balance_tags(text: str) -> str:
    """Дроп orphan-close и незакрытых open-тегов — Telegram принимает HTML.
    LLM и руководящие inputs иногда эмитят пол-тегов ("<b>foo" без
    закрытия, "</b>" без открытия). Идём по матчам как по стеку — orphan'ы
    тихо выкидываются, окружающий текст не трогается.
    """
    matches = list(_TAG_TOKEN_RE.finditer(text))
    if not matches:
        return text
    drop: set[int] = set()
    stack: list[tuple[str, int]] = []
    for i, m in enumerate(matches):
        is_close = bool(m.group(1))
        tag = m.group(2).lower()
        if is_close:
            if stack and stack[-1][0] == tag:
                stack.pop()
            else:
                drop.add(i)
        else:
            stack.append((tag, i))
    for _, i in stack:
        drop.add(i)

    if not drop:
        return text
    parts: list[str] = []
    cursor = 0
    for i, m in enumerate(matches):
        parts.append(text[cursor : m.start()])
        if i not in drop:
            parts.append(m.group(0))
        cursor = m.end()
    parts.append(text[cursor:])
    return "".join(parts)


def safe_html(text: str) -> str:
    """Экранировать ``text`` для Telegram, сохранив whitelist-теги.

    Компромисс между raw-escape (теряет всё форматирование) и no-escape
    (случайный ``<`` ломает сообщение): экранируем всё, восстанавливаем
    allowed-теги regex'ом, балансируем — незакрытый ``<b>`` не валит edit.
    """
    restored = _ALLOWED_TAG_RE.sub(r"<\1\2>", html.escape(text))
    return balance_tags(restored)


def truncate_for_telegram(body: str) -> str:
    """Обрезать ``body`` до 4096-char-лимита editMessageText в Telegram.

    Срезаем на 16 chars ниже потолка и ре-балансируем теги — open-``<b>``
    разрезанный посреди слова не отвергнет всё сообщение.
    """
    if len(body) <= TG_MESSAGE_LIMIT:
        return body
    return balance_tags(body[: TG_MESSAGE_LIMIT - 16] + "\n…")
