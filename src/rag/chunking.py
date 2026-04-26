"""Чистая логика чанкинга (без I/O).

Жёстко ограничивает длину каждого чанка ``chunk_size`` даже когда в
исходном тексте нет нормальных границ предложений.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    text: str
    subject_slug: str
    book: str
    page: int
    chunk_idx: int


def _pre_split_long_sentences(sentences: list[str], hard_limit: int) -> list[str]:
    """Разбить любое «предложение» длиннее ``hard_limit`` жёстким резом."""
    out: list[str] = []
    for s in sentences:
        if len(s) <= hard_limit:
            out.append(s)
        else:
            for i in range(0, len(s), hard_limit):
                out.append(s[i : i + hard_limit])
    return out


def split_pages_into_chunks(
    pages: list[tuple[int, str]],
    *,
    subject_slug: str,
    book: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    global_idx = 0

    for page_num, text in pages:
        if len(text) <= chunk_size:
            chunks.append(
                Chunk(
                    text=text,
                    subject_slug=subject_slug,
                    book=book,
                    page=page_num,
                    chunk_idx=global_idx,
                )
            )
            global_idx += 1
            continue

        sentences = _pre_split_long_sentences(_SENTENCE_END.split(text), chunk_size)
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= chunk_size:
                buf = f"{buf} {sent}".strip() if buf else sent
            else:
                if buf:
                    chunks.append(
                        Chunk(
                            text=buf,
                            subject_slug=subject_slug,
                            book=book,
                            page=page_num,
                            chunk_idx=global_idx,
                        )
                    )
                    global_idx += 1
                tail = buf[-overlap:] if buf and overlap and len(buf) > overlap else ""
                candidate = (tail + " " + sent).strip() if tail else sent
                buf = candidate if len(candidate) <= chunk_size else sent[:chunk_size]
        if buf:
            chunks.append(
                Chunk(
                    text=buf,
                    subject_slug=subject_slug,
                    book=book,
                    page=page_num,
                    chunk_idx=global_idx,
                )
            )
            global_idx += 1

    return chunks
