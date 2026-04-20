"""Glossary-based query expansion.

Bi-encoders penalise short queries: cosine between a 3-token "что такое X"
and a 200-token textbook chunk is structurally low because every off-topic
sentence in the chunk pulls the mean embedding away. We sidestep this by
appending the glossary's own definition of any term we recognise in the
query — the expanded query then shares 10+ content tokens with any passage
that defines the same term, lifting top_score by ~0.10–0.20 on definitional
questions like «что такое стейкхолдер».

No new dependencies — just the YAML files already shipped under
``data/glossary/``.
"""

from __future__ import annotations

import re
from functools import lru_cache

import yaml

from src.rag.config import ROOT

GLOSSARY_DIR = ROOT / "data" / "glossary"
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{2,}")
# Crude Russian stemmer — strips common case/number endings. Wide enough to
# match «стейкхолдера» / «стейкхолдеры» / «надсистемой» / «целей» against
# their lemma forms without dragging in pymorphy2. Order matters: longer
# multi-char endings first.
_RU_ENDING_RE = re.compile(
    r"(ами|ями|ого|ому|ыми|ыми|ого|их|ом|ой|ою|ую|ев|ев|ов|ах|ях|ой|ем|ой|ьми|ам|ям|"
    r"ой|ии|ью|ие|ия|ия|ие|ой|у|ю|ы|и|а|я|е|ь|й)$"
)
# Hard cap on expansions per query — beyond this the prompt bloats and the
# query embedding drifts toward the average of many definitions.
_MAX_EXPANSIONS = 2
# Don't try to match stems shorter than this — too many false positives.
_MIN_STEM_LEN = 4


@lru_cache(maxsize=1)
def _glossary_index() -> dict[str, list[str]]:
    """Map ``content_word_lower → [definition, …]`` across every glossary YAML.

    Indexes **every** content word of multi-word terms (not just the head).
    «Методика Кошарского-Уёмова» and «Методика Волковой-Четверикова» both
    file under «методика» — we keep both definitions, ranked by index order,
    so neither shadows the other.
    """
    out: dict[str, list[str]] = {}
    if not GLOSSARY_DIR.exists():
        return out
    for path in sorted(GLOSSARY_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for entry in data.get("terms", []) or []:
            term = str(entry.get("term", "")).strip()
            defn = str(entry.get("definition", "")).strip()
            if not term or not defn:
                continue
            term_l = term.lower()
            # Full term is the strongest match — register it first.
            out.setdefault(term_l, []).append(defn)
            # Plus every content word inside the term, so «культура» finds
            # «Корпоративная культура», «инициирования» finds «Пространство
            # инициирования целей», etc.
            for word in _WORD_RE.findall(term_l):
                if len(word) >= _MIN_STEM_LEN and word != term_l:
                    bucket = out.setdefault(word, [])
                    if defn not in bucket:
                        bucket.append(defn)
    return out


def _stem(word: str) -> str:
    return _RU_ENDING_RE.sub("", word.lower())


def expand_query(query: str) -> str:
    """Append definitions of any glossary terms found in ``query``.

    Returns the original query unchanged if nothing matches. Expansions are
    deduplicated and capped at ``_MAX_EXPANSIONS`` to keep the encoded
    embedding focused.
    """
    index = _glossary_index()
    if not index:
        return query

    seen: set[str] = set()
    extras: list[str] = []
    for token in _WORD_RE.findall(query):
        stem = _stem(token)
        if len(stem) < _MIN_STEM_LEN:
            continue
        for key, defns in index.items():
            if key.startswith(stem) or stem.startswith(key):
                for defn in defns:
                    if defn not in seen:
                        extras.append(defn)
                        seen.add(defn)
                        if len(extras) >= _MAX_EXPANSIONS:
                            break
                break
        if len(extras) >= _MAX_EXPANSIONS:
            break

    if not extras:
        return query
    return query + " " + " ".join(extras)


__all__ = ["expand_query"]
