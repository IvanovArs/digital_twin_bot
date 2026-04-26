"""Общие хелперы для студенческих хендлеров."""

from __future__ import annotations


def shorten(s: str, limit: int) -> str:
    """Однострочное превью с многоточием для UI-листов."""
    s = (s or "").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"
