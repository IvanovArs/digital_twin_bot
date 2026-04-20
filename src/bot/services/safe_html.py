"""Sanitiser for any Telegram ``parse_mode="HTML"`` message body whose
source is partially user-controlled (LLM output, teacher-typed FAQ,
glossary definitions).

The rules:

* Only a short whitelist of Telegram-supported inline tags survives.
* Everything else is ``html.escape``-d.
* Open/close tags are balanced — an unclosed ``<b>`` would otherwise make
  Telegram reject the whole edit with «can't parse entities».
* ``_truncate_for_telegram`` caps the body below the 4096-char ceiling
  without cutting a tag in half.

Used by the RAG answer renderer and by the FAQ/glossary short-circuit
paths. Teachers type answers in plain prose or light HTML; we sanitise on
write-to-user so a stray ``</b>`` can't break the render for every
student who asks the same question.
"""

from __future__ import annotations

import html
import re

# Keep the list short — every entry is a place where a malformed tag from
# the source can break parse_mode="HTML" for the whole message.
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
    """Drop orphan close tags and unclosed open tags so Telegram accepts
    the HTML. LLMs and hand-typed inputs occasionally emit half-tags
    ("<b>foo" with no closer, or "</b>" with no opener). We walk matches
    as a stack — orphans are quietly dropped and the surrounding text is
    kept untouched.
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
    """Escape ``text`` for Telegram but preserve the whitelisted tags.

    Compromise between raw-escape (loses all formatting) and no-escape
    (stray ``<`` breaks the message): escape everything, re-enable allowed
    tags via regex, then balance them so an unclosed ``<b>`` doesn't kill
    the whole edit.
    """
    restored = _ALLOWED_TAG_RE.sub(r"<\1\2>", html.escape(text))
    return balance_tags(restored)


def truncate_for_telegram(body: str) -> str:
    """Cap ``body`` to Telegram's 4096-char editMessageText limit.

    Slices 16 chars below the cap and re-balances tags so an open ``<b>``
    cut mid-word doesn't reject the entire message.
    """
    if len(body) <= TG_MESSAGE_LIMIT:
        return body
    return balance_tags(body[: TG_MESSAGE_LIMIT - 16] + "\n…")
